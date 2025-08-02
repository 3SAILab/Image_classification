import sys
import os
sys.path.append(os.path.dirname(__file__))
from utils.utils import set_seed, show_model_parameters_num, update_ema, log
# from models.wideresnet import WideResNet1 as m
from torchvision.models import resnet50 as m
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import warnings
import logging
import torch
import json

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['SimHei']
logging.basicConfig(level = logging.INFO)

model_path = os.path.join(os.path.dirname(__file__), "model")
log_path = os.path.join(os.path.join(os.path.dirname(__file__), "results"), "log")
config_path = os.path.join(os.path.dirname(__file__), "config.json")

with open(config_path, 'r') as f:
    config = json.load(f)

num_epochs = config['num_epochs']
batch_size = config['batch_size']
lr = config['lr']
num_classes = config['num_classes']
num_workers = config['num_workers']
device = config['device']

log_name = input(
    """
    ❗ 输入此次训练需要记录的日志名\n
    ❗ 输入m表示默认使用模型名称\n
    ❗ 输入x表示此次训练不记录日志\n
    ❗ 输入其他表示 模型名称+自定义日志名 ：
    """
)

need_save_model = input(
    """
    ❗ 是否需要保存模型?\n
    ❗ 输入y表示需要保存模型\n
    ❗ 输入其他表示不需要保存模型\n
    """
)

need_set_seed = input(
    """
    ❗ 是否需要设置并固定config.json中的随机种子?\n
    ❗ 输入y表示需要设置并固定config.json中的随机种子\n
    ❗ 输入其他表示不需要设置并固定config.json中的随机种子\n
    """
)

if log_name != "x":
    log_name = f"{str(m.__name__)}_{log_name}.txt"
elif log_name == "m":
    log_name = f"{str(m.__name__)}.txt"

if need_set_seed == "y":
    seed = config["seed"]
    set_seed(seed)

logging.info(f"✅ 此次训练将保存的日志名为：{log_name}")

if need_save_model == "y":
    logging.info(f"✅ 此次训练将保存模型")
else:
    logging.info(f"❌ 此次训练将不保存模型")

if need_set_seed == "y":
    logging.info(f"✅ 此次训练设置并固定随机种子为{seed}")
else:
    logging.info(f"❌ 此次训练未设置和固定随机种子")

# 在设置好随机种子后，开始导入数据集
from dataset.dataloader import random_train_dataloader, random_eval_dataloader, layer_train_dataloader, layer_eval_dataloader
from visualization.visualization import draw_confusion_matrix #, plot_dataset_distribution, 

def train(
    train_loader, 
    eval_loader, 
    num_epochs, 
    visualizaion=False):

    train_loss = []
    eval_loss = []
    epoch_list = []
    step_list = []
    acc_list = []
    lr_list = []
    acc = 0.0

    model = m(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-4
    )
    scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=10, 
        gamma=0.5
    )

    logging.info(f"❗ Train Started")

    param_num = show_model_parameters_num(model)
    # 开始训练
    for epoch in range(num_epochs):
        model.train()
        epoch_list.append(epoch)
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", colour='green')
        for batch_index, (img, label) in loop:
            img = img.to(device)
            label = label.to(device)

            output = model(img).to(device)
            loss = criterion(output, label)
            # logits, aux_logits2, aux_logits1 = model(img)
            # loss0 = criterion(logits, label)
            # loss1 = criterion(aux_logits1, label)
            # loss2 = criterion(aux_logits2, label)
            # loss = loss0 + 0.3 * loss1 + 0.3 * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            step_list.append(epoch * len(train_loader) + batch_index + 1)

            loop.set_postfix(loss=f'\033[91m{loss.item():.3f}\033[0m', lr=f'{optimizer.param_groups[0]['lr']:.1e}')
        # 验证
        if epoch == num_epochs - 1:
            eval_loss_item, acc = eval(model, eval_loader, criterion, visual_matrix=True)
        else:
            eval_loss_item, acc = eval(model, eval_loader, criterion)
        
        logging.info(f"➡️  Epoch {epoch+1}/{num_epochs}, Val Loss: {eval_loss_item:.2f}, Acc: \033[91m{acc:.2f}\033[0m")

        if (acc > max(acc_list) if acc_list else 0) and need_save_model == "y":
            # 保存最佳模型
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                }, 
                os.path.join(model_path, f"{str(model.__class__.__name__)}_best.pth")
            )
            logging.info(f"✅ Best model saved to {model_path}")

        eval_loss.append(eval_loss_item)
        acc_list.append(acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)
        scheduler.step()
    
    logging.info(f"❗ Train Best Acc: \033[91m{max(acc_list):.2f}\033[0m, Best Epoch: \033[91m{epoch_list[acc_list.index(max(acc_list)) + 1]}\033[0m")
    logging.info('✅ Train Finished')

    # 保存最终模型
    if need_save_model == "y":
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]['lr'],
            }, 
            os.path.join(model_path, f"{str(model.__class__.__name__)}_last.pth")
        )
    # 可视化数据
    if visualizaion:
        ema_alpha = 0.01
        ema_train_loss = []
        for i in range(len(train_loss)):
            if i == 0:
                ema_train_loss.append(train_loss[0])
            else:
                ema_train_loss.append(update_ema(train_loss[i], ema_alpha, ema_train_loss[i-1]))

        ema_eval_loss = []
        for i in range(len(eval_loss)):
            if i == 0:
                ema_eval_loss.append(eval_loss[0])
            else:
                ema_eval_loss.append(update_ema(eval_loss[i], ema_alpha, ema_eval_loss[i-1]))
        # 记录日志
        if log_name != "x":
            log(log_path, log_name, param_num, num_epochs, batch_size, lr, train_loss, eval_loss, acc_list, lr_list, ema_train_loss, ema_eval_loss)
            logging.info(f"✅ Log saved to {os.path.join(log_path, log_name)}")

        plt.figure(figsize=(18,8))

        plt.subplot(1, 6, 1)
        plt.plot(step_list, train_loss)
        plt.title("loss/train")

        plt.subplot(1, 6, 2)
        plt.plot(epoch_list, eval_loss)
        plt.title("loss/eval")

        plt.subplot(1, 6, 3)
        plt.plot(epoch_list, acc_list)
        plt.title("Accuracy")

        plt.subplot(1, 6, 4)
        plt.plot(epoch_list, lr_list)
        plt.title("lr")

        plt.subplot(1, 6, 5)
        plt.plot(step_list, ema_train_loss)
        plt.title("ema_train_loss")

        plt.subplot(1, 6, 6)
        plt.plot(epoch_list, ema_eval_loss)
        plt.title("ema_eval_loss")

        plt.suptitle("Loss and Value")
        plt.show()

def eval(
    model, 
    eval_loader, 
    criterion, 
    visual_matrix=False):
    model.eval()
    total_loss = 0.0
    acc = 0
    total = 0
    pred_list = []
    label_list = []
    with torch.inference_mode():
        for _, (images, label) in enumerate(eval_loader):
            images = images.to(device)
            label = label.to(device)

            outputs = model(images)
            loss = criterion(outputs, label)
            predict_y = torch.max(outputs, dim=1)[1]

            acc += torch.eq(predict_y, label).sum().item()
            total_loss += loss.item()
            total += label.size(0)

            for i in range(len(predict_y)):
                pred_list.append(int(predict_y[i]))
                label_list.append(int(label[i]))

        acc = acc / total
        if visual_matrix:
            draw_confusion_matrix(pred_list, label_list)
            eval_loss = total_loss / max(len(eval_loader), 1)
            return eval_loss, acc
        else:
            eval_loss = total_loss / max(len(eval_loader), 1)
            return eval_loss, acc

def main(spilt_type):
    if spilt_type == "random":
        train(
            train_loader=random_train_dataloader, 
            eval_loader=random_eval_dataloader, 
            num_epochs=num_epochs, 
            visualizaion=True
        )
    elif spilt_type == "layer":
        train(
            train_loader=layer_train_dataloader, 
            eval_loader=layer_eval_dataloader, 
            num_epochs=num_epochs, 
            visualizaion=True
        )

if __name__ == "__main__":
    main("random")