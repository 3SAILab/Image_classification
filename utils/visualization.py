from dataset.dataset import data
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg') 
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

def plot_training_metrics(
    epoch_num, 
    step_num,
    train_loss, 
    ema_train_loss, 
    eval_loss, 
    ema_eval_loss, 
    eval_acc
):
    """
    绘制训练和评估指标
    
    参数:
    epoch_num: epoch 数 (int)
    step_num: step 数列表 (list)
    train_loss: 训练损失列表
    ema_train_loss: EMA训练损失列表
    eval_loss: 评估损失列表
    ema_eval_loss: EMA评估损失列表
    eval_acc: 评估准确率列表
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    fig.suptitle('Training and Evaluation Metrics', fontsize=16, fontweight='bold')
    
    plt.subplots_adjust(top=0.93, hspace=0.3)
    
    # 处理step_num为list的情况
    total_steps = len(step_num)
    steps_per_epoch = total_steps // epoch_num
    epoch_list = list(range(1, epoch_num + 1))
    step_list = step_num
    
    train_color = '#1f77b4'
    ema_train_color = '#ff7f0e'
    eval_color = '#2ca02c'
    ema_eval_color = '#d62728'
    acc_color = '#9467bd'
    
    # 绘制训练损失
    axes[0].plot(step_list, train_loss, label='Train Loss', alpha=0.5, color=train_color, linewidth=1)
    axes[0].plot(step_list, ema_train_loss, label='EMA Train Loss', linewidth=0.8, color=ema_train_color)
    axes[0].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, color='black', linewidth=0.5, alpha=0.3)
    axes[0].set_facecolor('white')
    
    # 在训练损失图上添加epoch分隔线和标签
    epoch_ticks, epoch_labels = [], []
    for i in range(epoch_num + 1):
        epoch_step_idx = i * steps_per_epoch
        if epoch_step_idx < total_steps:
            step_val = step_list[epoch_step_idx]
            epoch_ticks.append(step_val)
            epoch_labels.append(i)
            # 添加分隔线（除了第一个点）
            if i > 0:
                axes[0].axvline(x=step_val, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    axes[0].set_xticks(epoch_ticks)
    axes[0].set_xticklabels(epoch_labels)
    
    # 绘制评估损失
    axes[1].plot(
        epoch_list, 
        eval_loss, 
        label='Eval Loss', 
        marker='o', 
        color=eval_color, 
        alpha=0.7, 
        markersize=4, 
        linewidth=1
    )
    axes[1].plot(
        epoch_list, 
        ema_eval_loss, 
        label='EMA Eval Loss', 
        marker='s',
        color=ema_eval_color,
        linewidth=0.8, 
        markersize=4
    )
    axes[1].set_title('Evaluation Loss', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, color='black', linewidth=0.5, alpha=0.3)
    axes[1].set_facecolor('white')
    
    # 绘制评估准确率
    axes[2].plot(
        epoch_list, 
        eval_acc, 
        label='Eval Accuracy', 
        color=acc_color, 
        marker='^', 
        markersize=6, 
        linewidth=1.5
    )

    axes[2].set_title('Evaluation Accuracy', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Accuracy', fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, color='black', linewidth=0.5, alpha=0.3)
    axes[2].set_facecolor('white')
    axes[2].set_ylim(0, 1)
    
    fig.patch.set_facecolor('white')
    plt.show()

def plot_data_distribution(train_num, eval_num):
    """
    绘制数据分布图表
    
    参数:
    train_num: 训练数据字典格式 {"label":int,...}
    eval_num: 评估数据字典格式 {"label":int,...}
    """
    # 提取标签和数据
    labels = list(train_num.keys())
    train_data = [train_num[label] for label in labels]
    eval_data = [eval_num[label] for label in labels]
    total_data = [train + eval for train, eval in zip(train_data, eval_data)]
    
    fig, ax = plt.subplots(figsize=(12, 10))

    x = np.arange(len(labels))
    width = 0.6

    train_color, eval_color = '#1f77b4', '#ff7f0e'  # 蓝色和橙色
    
    bars1 = ax.bar(x, train_data, width, label='Train Data', color=train_color, alpha=0.7)
    bars2 = ax.bar(x, eval_data, width, bottom=train_data, label='Eval Data', color=eval_color, alpha=1.0)
    
    # 在每个柱子上方显示总数
    for i, (bar, total) in enumerate(zip(bars1, total_data)):
        height = bar.get_height() + bars2[i].get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(total_data)*0.01,
                f'{total}', ha='center', va='bottom', fontsize=10)
    
    # 设置标签和标题
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('数据数量', fontsize=12)
    ax.set_title('数据分布图表', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    
    # 设置网格
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 启用交互注释
    annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    def update_annot(bar, category, train_count, eval_count, bar_type):
        x, y = bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2
        annot.xy = (x, y)
        # 修改悬浮信息格式，显示颜色预览
        text = f"类别: {category}\n{'Train' if bar_type == 'train' else 'Eval'}: {train_count if bar_type == 'train' else eval_count}"
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(train_color if bar_type == 'train' else eval_color)
        annot.get_bbox_patch().set_edgecolor('black')
        annot.get_bbox_patch().set_linewidth(1)
        annot.get_bbox_patch().set_alpha(0.8)
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.8)
    
    def hover(event):
        if event.inaxes == ax:
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                cont1, _ = bar1.contains(event)
                cont2, _ = bar2.contains(event)
                if cont1:
                    update_annot(bar1, labels[i], train_data[i], eval_data[i], 'train')
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
                elif cont2:
                    update_annot(bar2, labels[i], train_data[i], eval_data[i], 'eval')
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图形
    plt.show()