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
    flops,
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
        f.write(f"FLOPs: {flops}\n")
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

def show_model_flops_and_params(model):
    from thop import profile
    import json

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")) as f:
        config = json.load(f)
    device = config['device']

    model.eval()
    input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(input, ), verbose=False)
    logging.info(f"➡️  FLOPs = {str(flops/1000**3)} G")
    logging.info(f"➡️  Params = {str(params/1000**2)} M")
    return flops, params