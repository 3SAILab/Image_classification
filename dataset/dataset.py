import os
import json

t_num = 0
big_small_dict = {}
index_path_dict = {}
index_small_dict = {}
t_small_num_dict = {}
t_small_index_dict = {}

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image")

for big in os.listdir(data_path):
    big_path = os.path.join(data_path, big)
    big_small_dict[big] = []
    for small in os.listdir(big_path):
        small_num = 0
        big_small_dict[big].append(small)
        t_small_index_dict[small] = []
        small_path = os.path.join(big_path, small)
        for img in os.listdir(small_path):
            index_path_dict[t_num] = os.path.abspath(os.path.join(small_path, img))
            index_small_dict[t_num] = small
            t_small_index_dict[small].append(t_num)
            t_num += 1
            small_num += 1
        t_small_num_dict[small] = small_num

small_label_idx_dict = {k : v for k, v in zip(t_small_num_dict.keys(), range(len(list(t_small_num_dict.keys()))))}

data = {
    # 总图片数量
    "total_num":t_num, 
    # 大类转小类
    "big_small_dict":big_small_dict, 
    # 图片索引转图片路径
    "index_path_dict":index_path_dict, 
    # 图片索引转小类
    "index_small_dict":index_small_dict, 
    # 小类图片数量
    "t_small_num_dict":t_small_num_dict, 
    # 小类转图片索引
    "t_small_index_dict":t_small_index_dict, 
    # 小类转标签索引
    "small_label_idx_dict":small_label_idx_dict
}

del t_num, big_small_dict, index_path_dict, index_small_dict, t_small_num_dict, t_small_index_dict, small_label_idx_dict

with open("dataset.json", "w") as f:
    json.dump(data, f, indent=4)