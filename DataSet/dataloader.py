from DataSet.dataset import index_path_dict, index_small_dict, small_label_idx_dict
from DataSet.dataset_spilt import spilt
from DataSet.data_transform import transform
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import os

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

batch_size = config['batch_size']

def flatten(nested_list):
    result = []
    for element in nested_list:
        if isinstance(element, list):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result

class MyDataSet(Dataset):
    def __init__(self, num, index, transform=None):
        super().__init__()
        self.num = num
        self.index = index
        self.transform = transform
        self.img_index_list = flatten(list(self.index.values()))

    def __len__(self):
        return len(self.img_index_list)
    
    def __getitem__(self, index):
        i = self.img_index_list[index]
        img_path = index_path_dict[i]
        img = Image.open(img_path).convert("RGB")
        label = small_label_idx_dict[index_small_dict[i]]
        if self.transform:
            img = self.transform(img)
        return img, label

random_train_num, random_eval_num, random_train_index, random_eval_index = spilt(
    spilt_type='random', 
    rate=0.1
)
random_train_dataset = MyDataSet(
    random_train_num, 
    random_train_index, 
    transform=transform["train"]
)
random_train_dataloader = DataLoader(
    random_train_dataset,
    batch_size=batch_size, 
    shuffle=True,
    drop_last=False,
    pin_memory=True
)

random_eval_dataset = MyDataSet(
    random_eval_num, 
    random_eval_index, 
    transform=transform["eval"]
)
random_eval_dataloader = DataLoader(
    random_eval_dataset,
    batch_size=batch_size, 
    shuffle=True,
    drop_last=False,
    pin_memory=True
)

layer_train_num, layer_eval_num, layer_train_index, layer_eval_index = spilt(
    spilt_type='layer', 
    rate=0.1
)
layer_train_dataset = MyDataSet(
    layer_train_num, 
    layer_train_index, 
    transform=transform["train"]
)
layer_train_dataloader = DataLoader(
    layer_train_dataset,
    batch_size=batch_size, 
    shuffle=True,
    drop_last=False,
    pin_memory=True
)
layer_eval_dataset = MyDataSet(
    layer_eval_num, 
    layer_eval_index, 
    transform=transform["eval"]
)
layer_eval_dataloader = DataLoader(
    layer_eval_dataset,
    batch_size=batch_size, 
    shuffle=True,
    drop_last=False,
    pin_memory=True
)