from dataset.dataset import data
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg') 
plt.rcParams['font.family'] = ['SimHei']

def plot_dataset_distribution(
    train_small_num_dicts, 
    eval_small_num_dicts):
    train = list(train_small_num_dicts.values())
    eval = list(eval_small_num_dicts.values())
    all_labels = list(data["t_small_num_dict"].keys())
    t_small_num = list(data["t_small_num_dict"].values())

    _, ax = plt.subplots(figsize=(12, 8))
    
    x = range(len(all_labels))
    width = 0.7
    
    ax.bar(x, train, width, label='训练集', color='#3498db')
    ax.bar(x, eval, width, bottom=train, label='验证集', color='#e74c3c')
    
    ax.set_title('数据集类别分布')
    ax.set_xlabel('类别')
    ax.set_ylabel('样本数量')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.legend()
    for i, j in zip(x, t_small_num):
        ax.text(i, j, str(j), ha='center', va='bottom')

    prev_pos = 0
    _, y_max = ax.get_ylim()
    y_text = y_max * 0.85 
    
    big_label_boundaries = data["big_small_dict"].keys()
    for big_label, position in zip(big_label_boundaries, range(5, len(all_labels) + 1, 5)):
        if position >= 0:
            ax.axvline(x=position - 0.5, color='gray', linestyle='--', alpha=0.7)
            
            mid_pos = (prev_pos + position) / 2
            ax.text(mid_pos, y_text, big_label.upper(), ha='center', va='center', 
                fontsize=10, fontweight='bold', 
                bbox=dict(facecolor='#f9f9f9', alpha=0.8, boxstyle='round,pad=0.5'))
            prev_pos = position
    
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()

def cnn_feature_map_visualization(x,model, num_show_feature_map=12):
    feature_map_list = model(x)
    for feature_map in feature_map_list:
        image = np.squeeze(feature_map.detach().cpu().numpy())
        plt.figure()
        for i in range(num_show_feature_map):
            plt.subplot(3, 4, i+1)
            plt.imshow(image[:,:,i])
        plt.show()

class ConfusionMatrix():
    def __init__(self, num_features):
        self.matrix = np.zeros((num_features,num_features))
        self.num_features = num_features
    
    def update(self, pred_label, true_label):
        for p, t in zip(pred_label, true_label):
            self.matrix[p, t] += 1.0

    def summary(self):
        TP = [0.0] * self.num_features
        TN = [0.0] * self.num_features
        FN = [0.0] * self.num_features
        FP = [0.0] * self.num_features
        Rcall = [0.0] * self.num_features
        Precision = [0.0] * self.num_features
        Specificity = [0.0] * self.num_features

        for i in range(self.num_features):
            TP[i] = self.matrix[i, i]
            FN[i] = np.sum(self.matrix[:,i]) - TP[i]
            FP[i] = np.sum(self.matrix[i,:]) - TP[i]
            TN[i] = np.sum(self.matrix) - FN[i] - FP[i] -TP[i]
        
        for i in range(self.num_features):
            Rcall[i] = TP[i] / (TP[i] + FN[i])
            Precision[i] = TP[i] / (TP[i] + FP[i])
            Specificity[i] = TN[i] / (TN[i] + FP[i])

        Accuracy = sum(TP) / np.sum(self.matrix)
    
        return TP, TN, FN, FP, Rcall, Precision, Specificity, Accuracy

    def plot(self, small_labels):
        if (self.matrix != np.zeros_like(self.matrix)).any():
            plt.figure(figsize=(12, 10))
            plt.imshow(self.matrix)
            plt.xticks(range(self.num_features), small_labels)
            plt.tick_params(axis='x', rotation=45)
            plt.yticks(range(self.num_features), small_labels)
            plt.colorbar()
            plt.xlabel("真实标签")
            plt.ylabel("预测标签")
            plt.title("混淆矩阵")

            thresh = self.matrix.max() / 2
            for x in range(self.num_features):
                for y in range(self.num_features):
                    info = int(self.matrix[y, x])
                    plt.text(x, y, str(info), 
                            horizontalalignment="center", 
                            verticalalignment="center", 
                            color="white" if info > thresh else "black",
                            fontsize=8)
            plt.tight_layout()
            plt.show()

def draw_confusion_matrix(pred_list, true_list):
    confusion_matrix = ConfusionMatrix(len(data["small_label_idx_dict"]))
    confusion_matrix.update(pred_list, true_list)
    confusion_matrix.summary()
    confusion_matrix.plot(list(data["small_label_idx_dict"].keys()))