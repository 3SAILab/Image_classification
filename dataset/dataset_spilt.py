from dataset.dataset import data
import random

def spilt(spilt_type, rate):
    train_index = {}
    eval_index = {}
    train_num = {}
    eval_num = {}

    if spilt_type == "random":
        num = data["total_num"] * rate
        range_t_num = range(data["total_num"])

        for k, v in data["t_small_num_dict"].items():
            train_index[k] = []
            eval_index[k] = []
            train_num[k] = 0
            eval_num[k] = int(round(v * rate))

        random_eval_list = random.sample(range_t_num, int(round(num)))

        for i in random_eval_list:
            eval_index[data["index_small_dict"][i]].append(i)

        for i in list(set(range_t_num) - set(random_eval_list)):
            train_index[data["index_small_dict"][i]].append(i)

        for i in list(set(range(data["total_num"])) - set(random_eval_list)):
            train_num[data["index_small_dict"][i]] += 1

        return train_num, eval_num, train_index, eval_index
    
    elif spilt_type == "layer":
        for k, v in data["t_small_num_dict"].items():
            eval_num [k] = int(round(v * rate))
            train_num [k] = data["t_small_num_dict"][k] - eval_num[k]
        
        eval_index = {k : random.sample(v, int(round(eval_num[k]))) for k, v in data["t_small_index_dict"].items()}
        train_index = {k : list(set(data["t_small_index_dict"][k]) - set(v)) for k, v in eval_index.items()}

        return train_num, eval_num, train_index, eval_index

    else:
        return None, None, None, None

if __name__ == "__main__":
    train_num, eval_num, train_index, eval_index = spilt(spilt_type="random", rate=0.1)
    train_num, eval_num, train_index, eval_index = spilt(spilt_type="layer", rate=0.1)