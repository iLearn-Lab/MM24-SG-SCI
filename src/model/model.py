import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model.building_blocks import QueryGRUEncoder, VideoSelfAttentionEncoder, PositionwiseFeedForward,\
    QueryVideoCrossModalEncoder, MultiLevelEnhancement
from src.utils.utils import sliding_window


class Model(nn.Module):
    def __init__(self, config, vocab=None, glove=None):
        super(Model, self).__init__()
        self.config = config
        self._read_model_config()
        self.nce_loss = nn.CrossEntropyLoss(reduction="none")

        # build network
        self.query_encoder = QueryGRUEncoder(
            in_dim=300,
            dim=self.dim // 2,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        self.fc_q = PositionwiseFeedForward(dim=self.dim, d_ff=4 * self.dim, dropout=self.dropout)
        self.video_encoder = VideoSelfAttentionEncoder(
            video_len=self.video_feature_len,
            in_dim=config[self.dataset_name]["feature_dim"],
            dim=self.dim,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        self.qv_encoder = QueryVideoCrossModalEncoder(
            dim=self.dim,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        self.CMFE_w = MultiLevelEnhancement(
            alpha=self.alpha,
            dim = self.dim
        )
        self.CMFE_v = MultiLevelEnhancement(
            alpha=self.alpha,
            dim = self.dim
        )

        # create optimizer, scheduler
        self._init_miscs()

        # single GPU assumed
        self.use_gpu = False
        self.device = None
        self.gpu_device = torch.device("cuda:0")
        self.cpu_device = torch.device("cpu")
        self.cpu_mode()

    def pooling(self, x, dim):
        return torch.max(x, dim=dim)[0]

    def max_pooling(self, x, mask, dim):
        return torch.max(x.masked_fill(mask == 0.0, -torch.inf), dim=dim)[0]

    def mean_pooling(self, x, mask, dim):
        return torch.sum(x * mask, dim=dim) / (torch.sum(mask, dim=dim) + 1e-8)

    def forward(self, batch):
        return self.network_forward(batch)
    
    def network_forward(self, batch):
        """ The "Cross-modal Representation Module".

        Returns:
            sentence_feature: (B, dim)
            video_feature: (B, video_feature_len, dim)
            q2v_attn: (B, video_feature_len)
        """
        query_label = batch["query_label"]
        query_mask = batch["query_mask"]
        video = batch["video"]
        video_mask = batch["video_mask"]
        words_feature, _ = self.query_encoder(query_label, query_mask, word_vectors=batch["word_vectors"])
        words_feature = self.fc_q(words_feature)
        video_feature, embeddings = self.video_encoder(video, video_mask)

        words_feature, video_feature, q2v_attn, v2q_attn_m, q2v_attn_m = self.qv_encoder(
            query_feature=words_feature,
            query_mask=query_mask,
            video_feature=video_feature,
            video_mask=video_mask
        )

        words_feature = self.CMFE_w(video_feature, words_feature, v2q_attn_m)
        video_feature = self.CMFE_v(words_feature, video_feature, q2v_attn_m)

        query_mask = batch["query_mask"]
        # max_words_feature = self.pooling(words_feature.masked_fill(query_mask.unsqueeze(2) == 0.0, -torch.inf), dim=1)
        # sentence_feature = self.CMFE(video_feature, words_feature, v2q_attn, max_words_feature)
        sentence_feature = self.pooling_func(words_feature,
                                             query_mask.unsqueeze(2),
                                             dim=1)

        return F.normalize(sentence_feature, dim=1), F.normalize(video_feature, dim=2), q2v_attn

    def forward_train_val(self, batch):
        """ The "Gaussian Alignment Module", use in training.

        Returns:
            loss: single item tensor
        """
        batch = self._prepare_batch(batch)
        sentence_feature, video_feature, attn_weights = self.network_forward(batch)

        def get_gaussian_weight(video_mask, glance_frame):
            """ Get the Gaussian weight of full video feature.
            Args:
                video_mask: (B, L)
                glance_frame: (B)
            Returns:
                weight: (B, L)
            """
            B, L = video_mask.shape

            x = torch.linspace(-1, 1, steps=L, device=self.device).view(1, L).expand(B, L)
            lengths = torch.sum(video_mask, dim=1).to(torch.long)

            # normalize video lengths into range
            sig = lengths / L
            sig = sig.view(B, 1)
            sig *= self.sigma_factor

            # normalize glance frames into range
            u = ((glance_frame - 1) * 2) / (L - 1) - 1
            u = u.view(B, 1)

            weight = torch.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
            weight /= torch.max(weight, dim=1, keepdim=True)[0]  # normalize weight
            weight.masked_fill_(video_mask == 0.0, 0.0)
            return weight

        video_mask = batch["video_mask"]
        glance_frame = batch["glance_frame"]
        weight = get_gaussian_weight(video_mask, glance_frame)  # (B, L)

        # sliding window
        def slice(video_feature, video_mask, weight, glance_frame, attn_weights):
            """ We use the scheme "variational clip frame, fixed stride".

            Args:
                video_feature: (B, L, dim)
                video_mask: (B, L)
                weight: (B, L)
            Returns:
                clips: (B, N, dim)
                clip_masks: (B, N)
                clip_weights: (B, N)
            """
            video_feature = video_feature.masked_fill(video_mask.unsqueeze(2) == 0.0, 0.0)
            B, L, D = video_feature.shape
            clips, clip_masks, clip_weights, clips_in_moment, clips_frame, clips_frame_idx = [], [], [], [], [], []
            for clip_frame in self.clip_frames:
                temp, temp_attn, slice_mask, idx = sliding_window(video_feature, attn_weights, video_mask.unsqueeze(2), clip_frame, self.stride, dim=1)
                # temp_attn 是一个B个元素的list，每个元素是一个(N, window_size)的tensor，代表每个clip的每个frame的attn权重
                # temp_frame_idx 代表每个clip的每个frame的attn权重最大的frame的索引
                temp_frame_idx = torch.stack([torch.argmax(i, 1) for i in temp_attn], dim=1) # (B, N)
                # temp_frame 代表每个clip的每个frame的attn权重最大的frame的特征
                temp_frame = torch.gather(video_feature, 1, temp_frame_idx.unsqueeze(-1).expand(-1, -1, D)) # (B, N, dim)

                temp_frame = torch.stack([self.mean_pooling(x, mask, dim=1) for x,mask in zip(temp,slice_mask)], dim=1) # (B, N, dim)
                temp = torch.stack([self.max_pooling(x, mask, dim=1) for x,mask in zip(temp,slice_mask)], dim=1)  # (B, N, dim)
                temp_mask = video_mask[:, idx[:, 0]]  # (B, N)
                temp.masked_fill_(temp_mask.unsqueeze(2) == 0.0, 0.0)
                # temp_temp_weight = weight[:, idx[:, 0]] * weight[:, idx[:, 1] - 1]weight = weight[:, torch.div(idx[:, 0] + idx[:, 1], 2.0, rounding_mode='floor').to(torch.long)]  # (B, N)
                temp_weight = weight[:, idx[:, 0]] * weight[:, idx[:, 1] - 1]
                temp_start_idx = idx[:,0].unsqueeze(0).expand_as(temp_mask) # (B, N)
                temp_end_idx = idx[:,1].unsqueeze(0).expand_as(temp_mask) 
                temp_glance_frame = glance_frame.unsqueeze(1).expand_as(temp_mask)
                temp_in_moment = torch.logical_and(temp_start_idx <= temp_glance_frame, temp_glance_frame <= temp_end_idx).long() # (B, N)   
                clips.append(temp)
                clip_masks.append(temp_mask)
                clip_weights.append(temp_weight)
                clips_in_moment.append(temp_in_moment)
                clips_frame.append(temp_frame)
                clips_frame_idx.append(temp_frame_idx)
            clips = torch.cat(clips, dim=1)
            clip_masks = torch.cat(clip_masks, dim=1)
            clip_weights = torch.cat(clip_weights, dim=1)
            clips_in_moment = torch.cat(clips_in_moment, dim=1)
            clips_frame = torch.cat(clips_frame, dim=1)
            clips_frame_idx = torch.cat(clips_frame_idx, dim=1)
            return clips, clip_masks, clip_weights, clips_in_moment, clips_frame, clips_frame_idx

        clips, clip_masks, clip_weights, clips_in_moment, clips_frame, clips_frame_idx = slice(video_feature, video_mask, weight, glance_frame, attn_weights)
        # clip_weights = torch.ones_like(clip_weights, device=self.device)  # Ablation 1: Clip-NCE
        # clip_weights.masked_fill_(clip_masks == 0.0, 0.0)  # Ablation 1: Clip-NCE
        scores = torch.matmul(clips, sentence_feature.T.unsqueeze(0))  # (B, N, B)

        # loss
        B, N, _ = scores.shape
        intra_score = torch.zeros(B, N, device=self.device)
        # 取出对角线的值，代表每个clip对应的sentence的分数，此时已经除以温度系数
        for i in range(B):
            intra_score[i, :] = scores[i, :, i] / self.temp

        # intra_score = F.softmax(intra_score, dim=1)
        # # 计算当前video所有clip的分数和
        # video_score = torch.sum(intra_score ,dim=1)
        # # 计算每个clip的分数占当前video所有clip分数的比例
        # intra_score = intra_score / video_score.unsqueeze(1)
        intra_loss = -1.0 * clip_weights * clips_in_moment * F.log_softmax(intra_score, dim=1)
        intra_loss = torch.sum(intra_loss, dim=(0,1)) / torch.sum(clips_in_moment, dim=(0,1))  # mean over positive action moments
        # clips_in_moment中的1代表正样本，计算正样本对应的交叉熵分数
        # intra_loss = self.nce_loss(intra_score , clip_weights * clips_in_moment)
        # intra_loss = torch.sum(intra_loss) / torch.sum(clips_in_moment, dim=(0,1))  # mean over positive action moments
        
        # 计算frame_score，即每个clip的每个frame对应的查询相似度分数
        frame_score = torch.bmm(clips_frame, sentence_feature.unsqueeze(1).transpose(1,2)).squeeze()  # (B, N)        
        frame_score = frame_score / self.temp
        # frame_score = F.softmax(frame_score, dim=1)
        # # 所有得分进行归一化
        # frame_score = frame_score / torch.sum(frame_score, dim=1).unsqueeze(1)
        # # clips_in_moment代表每个clip的每个frame是否在正样本中，weight代表每个frame的权重（高斯分布）
        # frame_loss = -1.0 * weight[torch.arange(B).unsqueeze(1), clips_frame_idx] * clips_in_moment * F.log_softmax(frame_score, dim=1)
        frame_loss = -1.0 * clip_weights * clips_in_moment * F.log_softmax(frame_score, dim=1) 
        frame_loss = torch.sum(frame_loss, dim=(0,1)) / torch.sum(clips_in_moment, dim=(0,1))  # mean over positive action moments

        # batch_score = torch.sum(video_score) 
        # # 除以所有的样本分数
        # inter_score = scores / batch_score
        # label = torch.zeros(B, N, B, device=self.device)
        # for i in range(B):
        #     # 某些clips的权重也应该为0
        #     label[i, :, i] = clip_weights[i, :] * clips_in_moment[i, :]
        #     # 同理
        #     label[i, :, list(range(i)) + list(range(i + 1, B))] = (clip_weights[i, :] * clips_in_moment[i, :]).unsqueeze(1)
        # label.masked_fill_(clip_masks.unsqueeze(2) == 0.0, 0.0)

        # inter_loss = self.nce_loss(inter_score.view(B * N, B) / self.temp, label.view(B * N, B))
        # inter_loss = torch.sum(inter_loss) / torch.sum(clip_masks)  # masked mean

        #attn_loss = F.kl_div(F.log_softmax(attn_weights, dim=1), F.log_softmax(weight, dim=1), reduction="none", log_target=True)
        #attn_loss.masked_fill_(video_mask == 0.0, 0.0)
        #attn_loss = torch.sum(attn_loss) / torch.sum(video_mask) * 10000
        attn_loss = F.cross_entropy(attn_weights, weight)
        
        # glance_loss
        # video_lengths = torch.sum(video_mask, dim=1).to(torch.long) 
        # tempol_dis = torch.abs(torch.argmax(attn_weights, dim=1) - glance_frame) / video_lengths # (B)
        # attn_weights_softmax = F.softmax(attn_weights, dim=1)        
        # semantic_dis = torch.abs(attn_weights_softmax[torch.arange(B), torch.argmax(attn_weights, dim=1)] - attn_weights_softmax[torch.arange(B), glance_frame]) # (B)
        # glance_loss = torch.sum(tempol_dis * semantic_dis) / B

        # loss = self.a * intra_loss + self.b * inter_loss + attn_loss
        # print("intra_loss: {}, inter_loss: {}, attn_loss: {}".format(intra_loss, inter_loss, attn_loss))


        loss = attn_loss + self.a * intra_loss + self.b * frame_loss
        # loss = nce_loss * 2  # Ablation 1: Clip-NCE; Ablation 2: w/o QAG-KL
        return loss

    # def forward_train_val(self, batch):
    #     """ Ablation 1: Video-NCE.
    #
    #     Returns:
    #         loss: single item tensor
    #     """
    #     batch = self._prepare_batch(batch)
    #     sentence_feature, video_feature, attn_weights = self.network_forward(batch)
    #     video_mask = batch["video_mask"]
    #     video_feature = self.pooling(video_feature.masked_fill(video_mask.unsqueeze(2) == 0.0, -torch.inf), dim=1)
    #     scores = torch.matmul(video_feature, sentence_feature.T)  # (B, B)
    #
    #     B, _ = scores.shape
    #     label = torch.zeros(B, B, device=self.device)
    #     for i in range(B):
    #         label[i, i] = 1.0
    #
    #     nce_loss = self.nce_loss(scores / self.temp, label)
    #     nce_loss = torch.mean(nce_loss)
    #     loss = nce_loss * 2
    #     return loss

    def forward_eval(self, batch):
        """ The "Query Attention Guided Inference" module, use in evaluation.

        Returns:
            (B, topk, 2)
                start and end fractions
        """
        batch = self._prepare_batch(batch)
        sentence_feature, video_feature, attn_weights = self.network_forward(batch)

        find_cnt = 0.0
        def generate_proposal(video_feature, video_mask, attn_weight, start_frame, end_frame):
            """ Use attn_weight to generate proposals.

            Returns:
                features: (num_proposals, dim)
                indices: (num_proposals, 2)
            """
            indices = []
            cnt = 0
            video_length = video_feature.shape[0]
            anchor_point = torch.argmax(attn_weight)
            if anchor_point >= start_frame and anchor_point <= end_frame:
                cnt += 1
            for f in self.moment_length_factors:
                l = round(video_length * f)
                if l == 0:
                    continue
                for o in self.overlapping_factors:
                    l_overlap = round(l * o)
                    if l == l_overlap:
                        continue
                    l_rest = l - l_overlap
                    min_index = max(0, anchor_point - l)  # Ablation 3: no anchor point
                    max_index = min(video_length, anchor_point + l)  # Ablation 3: no anchor point
                    starts = range(min_index, anchor_point + 1, l_rest)  # Ablation 3: no anchor point
                    ends = range(min_index + l, max_index + 1, l_rest)  # Ablation 3: no anchor point
                    # starts = range(0, video_length, l_rest)  # Ablation 3: no anchor point
                    # ends = range(l, video_length + l, l_rest)  # Ablation 3: no anchor point
                    indices.append(torch.stack([torch.tensor([start, end]) for start, end in zip(starts, ends)], dim=0))
            indices = torch.cat(indices, dim=0)
            indices = torch.unique(indices, dim=0)  # remove duplicates
           # features = torch.stack(
           #     [self.pooling(video_feature[s: e], dim=0) for s, e in indices], dim=0
           # )
            features = torch.stack(
                [self.pooling_func(video_feature[s: e], video_mask[s: e], dim=0) for s, e in indices], dim=0
            )
            return features, indices, cnt

        B = video_feature.shape[0]
        video_mask = batch["video_mask"]
        start_frame = batch["start_frame"]
        end_frame = batch["end_frame"]
        video_lengths = torch.sum(video_mask, dim=1).to(torch.long)
        res = []
        for i in range(B):
            video_length = video_lengths[i].item()
            video = video_feature[i, :video_length]
            attn_weight = attn_weights[i, :video_length]
            features, indices, cnt = generate_proposal(video, video.new_ones(video.shape), attn_weight, start_frame[i], end_frame[i])
            find_cnt += cnt
            scores = torch.mm(features, sentence_feature[i, :].unsqueeze(1)).squeeze(1)
            res.append(indices[torch.topk(scores, min(self.topk, indices.shape[0]), dim=0)[1].cpu()])
        res = torch.nn.utils.rnn.pad_sequence(res, batch_first=True).to(self.device)
        res = res / video_lengths.view(B, 1, 1)
        return res, find_cnt

    ##### below are helpers #####
    def _read_model_config(self):
        self.dataset_name = self.config["dataset_name"]

        # task independent config
        self.dim = self.config["model"]["dim"]
        self.dropout = self.config["model"]["dropout"]
        self.n_layers = self.config["model"]["n_layers"]
        self.temp = self.config["model"]["temp"]
        self.topk = self.config["model"]["topk"]

        # task dependent config
        self.video_feature_len = self.config[self.dataset_name]["video_feature_len"]
        self.clip_frames = self.config[self.dataset_name]["clip_frames"]
        self.stride = self.config[self.dataset_name]["stride"]
        self.sigma_factor = self.config[self.dataset_name]["sigma_factor"]
        self.moment_length_factors = self.config[self.dataset_name]["moment_length_factors"]
        self.overlapping_factors = self.config[self.dataset_name]["overlapping_factors"]
        self.pooling_func = getattr(self,
                            self.config[self.dataset_name]["pooling_func"])
        self.a = self.config[self.dataset_name]["intra_loss"]
        self.b = self.config[self.dataset_name]["inter_loss"]
        self.alpha = self.config[self.dataset_name]["momentum"]

    def _init_miscs(self):
        """
        Key attributes created here:
            - self.optimizer
            - self.scheduler
        """
        lr = self.config["train"]["init_lr"]
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=3
        )
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #    self.optimizer, milestones=[2,5,10], gamma=0.1                
        #)

    def _prepare_batch(self, batch):
        keys = ["query_label", "query_mask", "video", "video_mask",
                "start_frac", "end_frac", "start_frame", "end_frame",
                "glance_frac", "glance_frame", "word_vectors"]
        for k in keys:
            batch[k] = batch[k].to(self.device)
        return batch

    def optimizer_step(self, loss):
        """ Update the network.
        """
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.config["train"]["clip_norm"])
        self.optimizer.step()

    def scheduler_step(self, valid_loss):
        """
        Args:
            valid_loss: loss on valid set; tensor
        """
        self.scheduler.step(valid_loss)
        #self.scheduler.step()

    def load_checkpoint(self, exp_folder_path, suffix):
        self.load_state_dict(torch.load(os.path.join(exp_folder_path, "model_{}.pt".format(suffix))))
        print("== Checkpoint ({}) is loaded from {}".format(suffix, exp_folder_path))

    def save_checkpoint(self, exp_folder_path, suffix):
        torch.save(self.state_dict(), os.path.join(exp_folder_path, "model_{}.pt".format(suffix)))
        # torch.save(self.optimizer.state_dict(), os.path.join(exp_folder_path, "optimizer_{}.pt".format(suffix)))
        # torch.save(self.scheduler.state_dict(), os.path.join(exp_folder_path, "scheduler_{}.pt".format(suffix)))
        print("== Checkpoint ({}) is saved to {}".format(suffix, exp_folder_path))

    def cpu_mode(self):
        self.use_gpu = False
        self.to(self.cpu_device)
        self.device = self.cpu_device

    def gpu_mode(self):
        self.use_gpu = True
        self.to(self.gpu_device)
        self.device = self.gpu_device

    def train_mode(self):
        self.train()

    def eval_mode(self):
        self.eval()
