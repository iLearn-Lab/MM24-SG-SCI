import os
import yaml
import torch
import random
import argparse
import numpy as np

from pathlib import Path
from tqdm import tqdm

from src.dataset.dataset import prepare_data
from src.utils.utils import load_config, n_params, get_now
from src.experiment.eval import Evaluator
from src.model.model import Model

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def train(config):
    # save log
    exp_folder_path = os.path.join(config["exp_dir"], get_now())
    Path(exp_folder_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(exp_folder_path, "config.yaml"), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # prepare data
    dataset_name = config["dataset_name"]
    data = prepare_data(config, dataset_name)
    train_dl = data["train_dl"]
    valid_dl = data["valid_dl"]
    test_dl = data["test_dl"]

    model = Model(config)
    print("Model has {} parameters.\n".format(n_params(model)))
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.gpu_mode()
    else:
        model.cpu_mode()

    
    
    test_evaluator = Evaluator()
    log_file_path = os.path.join(exp_folder_path, "train.log")
    with open(log_file_path, "w") as log_file:
        log_file.write("{}\n".format(config[dataset_name]["feature_dir"]))
        log_file.write("Model has {} parameters.\n".format(n_params(model)))
        log_file.flush()
        for epoch in range(1, config[dataset_name]["epoch"] + 1):
            for i, batch in tqdm(
                enumerate(train_dl), total=len(train_dl),
                desc="Training epoch {} with lr {}".format(epoch, model.optimizer.param_groups[0]["lr"])
            ):
                model.train_mode()
                loss = model.forward_train_val(batch)
                model.optimizer_step(loss)

                # if use_gpu:
                #     model.batch_to(batch, model.cpu_device)
                #     torch.cuda.empty_cache()

            with torch.no_grad():
                test_loss = test_evaluator.eval_dataloader(model, test_dl, epoch)
                model.scheduler_step(test_loss)

            log_file.write("\n==== epoch {} ====\n".format(epoch))
            log_file.write("        ## test ##\n")
            log_file.write(test_evaluator.report_in_moment() + "\n")
            log_file.write(test_evaluator.report_current() + "\n")
            log_file.write(test_evaluator.report_best() + "\n")
            log_file.flush()

            # save best
            if epoch == test_evaluator.best_epoch:
                model.save_checkpoint(exp_folder_path, "best")
            print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--task", help="Dataset name in {activitynetcaptions, charadessta, tacos}.", required=True)
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = load_config("src/config.yaml")
    config["dataset_name"] = args.task
    config["gpu"] = args.gpu
    train(config)
