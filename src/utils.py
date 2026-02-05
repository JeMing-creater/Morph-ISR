import os
import random
import sys
from collections import OrderedDict
from copy import deepcopy
import math
import numpy as np
import torch
from accelerate import Accelerator
from timm.models.layers import trunc_normal_
from torch import nn
from pathlib import Path
import numpy as np
import shutil


class MetricSaver(nn.Module):
    def __init__(self):
        super().__init__()
        self.best_acc = nn.Parameter(torch.zeros(1), requires_grad=False)


class Logger(object):
    def __init__(self, logdir: str):
        self.console = sys.stdout
        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
            self.log_file = open(logdir + "/log.txt", "w")
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        if sys.stdout is self:
            sys.stdout = self.console
        if sys.stderr is self:
            sys.stderr = self.console
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None


def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
    if download_path.startswith("http"):
        state_dict = torch.hub.load_state_dict_from_url(
            download_path,
            model_dir=save_path,
            check_hash=check_hash,
            map_location=torch.device("cpu"),
        )
    else:
        state_dict = torch.load(download_path, map_location=torch.device("cpu"))
    return state_dict


def resume_train_state(
    model,
    path: str,
    optimizer,
    scheduler,
    train_loader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    seg: bool = True,
):
    if seg != True:
        try:
            # Get the most recent checkpoint
            base_path = os.getcwd() + "/" + "model_store" + "/" + path + "/checkpoint"
            epoch_checkpoint = torch.load(
                base_path + "/epoch.pth.tar",
                map_location="gpu" if accelerator.is_local_main_process else "cpu",
            )
            starting_epoch = epoch_checkpoint["epoch"] + 1
            best_accuracy = epoch_checkpoint["best_accuracy"]
            best_test_accuracy = epoch_checkpoint["best_test_accuracy"]
            best_metrics = epoch_checkpoint["best_metrics"]
            best_test_metrics = epoch_checkpoint["best_test_metrics"]
            step = starting_epoch * len(train_loader)
            # model = load_pretrain_model(
            #     base_path + "/pytorch_model.bin", model, accelerator
            # )
            # optimizer.load_state_dict(torch.load(base_path + "/optimizer.bin"))
            # scheduler.load_state_dict(torch.load(base_path + "/scheduler.bin"))

            accelerator.load_state(base_path)

            accelerator.print(
                f"Loading training state successfully! Start training from {starting_epoch}, Best Acc: {best_accuracy}"
            )
            return (
                model,
                optimizer,
                scheduler,
                starting_epoch,
                step,
                best_accuracy,
                best_test_accuracy,
                best_metrics,
                best_test_metrics,
            )
        except Exception as e:
            accelerator.print(e)
            accelerator.print(f"Failed to load training state!")
            return (
                model,
                optimizer,
                scheduler,
                0,
                0,
                torch.tensor(0),
                torch.tensor(0),
                {},
                {},
            )
    try:
        # Get the most recent checkpoint
        base_path = os.getcwd() + "/" + "model_store" + "/" + path + "/checkpoint"
        epoch_checkpoint = torch.load(base_path + "/epoch.pth.tar", map_location="cpu")
        starting_epoch = epoch_checkpoint["epoch"] + 1
        best_score = epoch_checkpoint["best_score"]
        best_metrics = epoch_checkpoint["best_metrics"]
        best_hd95 = epoch_checkpoint["best_hd95"]
        best_hd95_metrics = epoch_checkpoint["best_hd95_metrics"]
        step = starting_epoch * len(train_loader)
        # model = load_pretrain_model(
        #     base_path + "/pytorch_model.bin", model, accelerator
        # )
        # optimizer.load_state_dict(torch.load(base_path + "/optimizer.bin"))
        # scheduler.load_state_dict(torch.load(base_path + "/scheduler.bin"))
        accelerator.load_state(base_path)
        accelerator.print(
            f"Loading training state successfully! Start training from {starting_epoch}, Best Acc: {best_score}"
        )
        return (
            model,
            optimizer,
            scheduler,
            starting_epoch,
            step,
            best_score,
            best_metrics,
            best_hd95,
            best_hd95_metrics,
        )
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f"Failed to load training state!")
        return (
            model,
            optimizer,
            scheduler,
            0,
            0,
            torch.tensor(0),
            [],
            torch.tensor(1000),
            [],
        )


def load_pretrain_model(pretrain_path: str, model: nn.Module, accelerator: Accelerator):
    try:
        state_dict = load_model_dict(pretrain_path)
        model.load_state_dict(state_dict)
        accelerator.print(f"Successfully loaded the training model for ", pretrain_path)
        return model
    except Exception as e:
        try:
            state_dict = load_model_dict(pretrain_path)
            new_state_dict = {}
            for key in state_dict.keys():
                new_state_dict[key.replace("module.", "")] = state_dict[key]
            model.load_state_dict(new_state_dict)
            accelerator.print(
                f"Successfully loaded the training modelfor ", pretrain_path
            )
            return model
        except Exception as e:
            accelerator.print(e)
            accelerator.print(f"Failed to load the training model！")
            return model


def same_seeds(seed, deterministic: bool = True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(False)  # 需要更严格可设 True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
