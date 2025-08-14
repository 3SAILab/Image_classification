from utils.utils import show_model_flops_and_params, update_ema, log
from utils.visualization import plot_training_metrics
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn as nn
import logging
import torch
import os

def train(
    m,
    train_loader, 
    eval_loader, 
    num_epochs,
    num_classes,
    device,
    need_save_model,
    model_path,
    log_name,
    log_path,
    batch_size,
    lr,
    draw_confusion_matrix,
    visualization=False
):

    train_loss = []
    eval_loss = []
    epoch_list = []
    step_list = []
    acc_list = []
    lr_list = []
    acc = 0.0
    best_earliest_epoch = 0
    best_acc = 0.0

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

    flops, param = show_model_flops_and_params(model)
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            step_list.append(epoch * len(train_loader) + batch_index + 1)

            loop.set_postfix(loss=f'\033[91m{loss.item():.3f}\033[0m', lr=f'{optimizer.param_groups[0]['lr']:.1e}')
        # 验证
        if epoch == num_epochs - 1:
            eval_loss_item, acc = eval(
                model, 
                eval_loader, 
                criterion, 
                visual_matrix=True,
                draw_confusion_matrix=draw_confusion_matrix
            )
        else:
            eval_loss_item, acc = eval(
                model, 
                eval_loader, 
                criterion
            )
        
        logging.info(f"➡️  Epoch {epoch+1}/{num_epochs}, Val Loss: {eval_loss_item:.2f}, Acc: \033[91m{acc:.2f}\033[0m")
        # 保存最早最高准确率
        if acc > best_acc:
            best_acc = acc
            best_earliest_epoch = epoch + 1

            if need_save_model == "y":
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
    
    logging.info(f"❗ Train Best Acc: \033[91m{best_acc:.2f}\033[0m, Best Epoch: \033[91m{best_earliest_epoch}\033[0m")
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
    if visualization:
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
            log(log_path, log_name, flops, param, num_epochs, batch_size, lr, train_loss, eval_loss, acc_list, lr_list, ema_train_loss, ema_eval_loss)
            logging.info(f"✅ Log saved to {os.path.join(log_path, log_name)}")
        
        # 绘制训练结果图表
        draw_data = {
            "epoch_num" : num_epochs,
            "step_num" : step_list,
            "train_loss" : train_loss,
            "ema_train_loss" : ema_train_loss,
            "eval_loss" : eval_loss,
            "ema_eval_loss" : ema_eval_loss,
            "eval_acc" : acc_list
        }

        plot_training_metrics(**draw_data)

def eval(
    model, 
    eval_loader, 
    criterion, 
    device,
    draw_confusion_matrix=None,
    visual_matrix=False
):
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
            if draw_confusion_matrix is not None:
                draw_confusion_matrix(pred_list, label_list)
            eval_loss = total_loss / max(len(eval_loader), 1)
            return eval_loss, acc
        else:
            eval_loss = total_loss / max(len(eval_loader), 1)
            return eval_loss, acc
