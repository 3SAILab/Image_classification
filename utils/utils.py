import numpy as np
import logging
import random
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def log(
    log_path, 
    log_name, 
    param_num, 
    num_epochs, 
    batch_size, 
    init_lr, 
    train_loss_list, 
    eval_loss_list, 
    acc_list, 
    lr_list, 
    ema_train_loss_list, 
    ema_eval_loss_list):
    with open(os.path.join(log_path, log_name), 'w') as f:
        f.write(f"Parameters: {param_num}\n")
        f.write(f"Train Epochs: {num_epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Initial Learning Rate: {init_lr}\n")
        f.write(f"Train Loss List: {train_loss_list}\n")
        f.write(f"Eval Loss List: {eval_loss_list}\n")
        f.write(f"Accuracy List: {acc_list}\n")
        f.write(f"Learning Rate List: {lr_list}\n")
        f.write(f"EMA Train Loss List: {ema_train_loss_list}\n")
        f.write(f"EMA Eval Loss List: {ema_eval_loss_list}\n")

def update_ema(current_value, ema_alpha, last_ema=None):
    if last_ema is None:
        return current_value
    return ema_alpha * current_value + (1 - ema_alpha) * last_ema

def show_model_parameters_num(model):
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"➡️  Model Parameters: {total_params}")
    return total_params