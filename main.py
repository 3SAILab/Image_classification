import sys
import os
sys.path.append(os.path.dirname(__file__))
from utils.utils import set_seed
from models.vgg import VGG as m
# from torchvision.models import densenet121 as m
import warnings
import logging
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level = logging.INFO)

model_path = os.path.join(os.path.dirname(__file__), "model")
log_path = os.path.join(os.path.join(os.path.dirname(__file__), "results"), "log")
config_path = os.path.join(os.path.dirname(__file__), "configs/config.json")

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
from utils.visualization import draw_confusion_matrix, plot_data_distribution
from trainers.train import train

def main(spilt_type):
    # 组织数据
    data = {
        "m" : m,
        "train_loader" : None,
        "eval_loader" : None,
        "num_epochs" : num_epochs,
        "num_classes" : num_classes,
        "device" : device,
        "need_save_model" : need_save_model,
        "model_path" : model_path,
        "log_name" : log_name,
        "log_path" : log_path,
        "batch_size" : batch_size,
        "lr" : lr,
        "draw_confusion_matrix" : draw_confusion_matrix,
        "visualization" : False,
    }

    if spilt_type == "random":
        from dataset.dataloader import random_train_dataloader, random_eval_dataloader, random_train_num, random_eval_num

        plot_data_distribution(
            train_num=random_train_num,
            eval_num=random_eval_num
        )

        data["train_loader"] = random_train_dataloader
        data["eval_loader"] = random_eval_dataloader
        data["visualization"] = True

        train(**data)

    elif spilt_type == "layer":
        from dataset.dataloader import layer_train_dataloader, layer_eval_dataloader, layer_train_num, layer_eval_num

        plot_data_distribution(
            train_num=layer_train_num,
            eval_num=layer_eval_num
        )

        data["train_loader"] = layer_train_dataloader
        data["eval_loader"] = layer_eval_dataloader
        data["visualization"] = True

        train(**data)


if __name__ == "__main__":
    main("random")