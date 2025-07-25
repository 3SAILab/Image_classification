from DataSet.dataset import t_num, index_small_dict, t_small_num_dict, t_small_index_dict
import json
import random
import os

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"),"r") as f:
    config = json.load(f)
seed = config["seed"]

random.seed(seed)

def spilt(spilt_type, rate):
    if spilt_type == "random":
        num = t_num * rate
        range_t_num = range(t_num)

        train_index =  {i : [] for i in t_small_num_dict.keys()}
        eval_index =  {i : [] for i in t_small_num_dict.keys()}
        train_num = {i : 0 for i in t_small_num_dict.keys()}
        eval_num = {k : int(round(v * rate)) for k, v in t_small_num_dict.items()}

        random_eval_list = random.sample(range_t_num, int(round(num)))

        for i in random_eval_list:
            eval_index[index_small_dict[i]].append(i)

        for i in list(set(range_t_num) - set(random_eval_list)):
            train_index[index_small_dict[i]].append(i)

        for i in list(set(range(t_num)) - set(random_eval_list)):
            train_num[index_small_dict[i]] += 1

        return train_num, eval_num, train_index, eval_index
    
    elif spilt_type == "layer":
        eval_num = {k : int(round(v * rate)) for k, v in t_small_num_dict.items()}
        train_num = {k: t_small_num_dict[k] - eval_num[k] for k in t_small_num_dict}
        eval_index = {k : random.sample(v, int(round(eval_num[k]))) for k, v in t_small_index_dict.items()}
        train_index = {k : list(set(t_small_index_dict[k]) - set(v)) for k, v in eval_index.items()}
        return train_num, eval_num, train_index, eval_index

    else:
        return None, None, None, None

if __name__ == "__main__":
    train_num, eval_num, train_index, eval_index = spilt(spilt_type="random", rate=0.1)
    train_num, eval_num, train_index, eval_index = spilt(spilt_type="layer", rate=0.1)