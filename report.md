# **Inception V1**
## Trick:
- 使用不同大小的卷积核对同一特征图进行卷积或池化后，将各特征图沿通道维度拼接，形成更大的特征图和更丰富的特征表达。 

- 为了减少计算量，在大卷积核卷积之前使用1x1卷积降低通道数。 
> 注：**Inception** 模块：  
> ![Inception V1](some_img\v1.png)

# **Inception V2**
## Trick:
- 提出 **Batch Normlization**  

- 沿用VGG的思想：使用两个3x3卷积核堆叠替代5x5卷积核以减少参数，感受野不变。  

- 在应用 **BN** 的网络中，减少L2正则化，加速LR衰减，移除Dropout。  

> 注：**论文** [**How Does Batch Normalizetion Help Optimization**](https://arxiv.org/pdf/1805.11604)  
>
> 论证了BN的作用并非是因为保证了数据的独立同分布，也没有减少甚至增加了内部协变量偏移，而是：
>    - 让loss曲面变得更光滑
>    - 正常损失曲面上的所有极小值点都在BN的损失曲面上被保留下来
>    - BN可以导致有利的初始化，BN初始化与局部最佳权重的 **L2** 误差有上界。

> 注：**Inception** 模块： 
> ![Inception V2](some_img\v2.png)


# **Inception V3**
## Trick:
- 特征图维度在**12x12-20x20**之间时，用 1xN + Nx1 代替 NxN 卷积核。  

- 根据网络位置和输出矩阵大小使用不同的 **Inception** 模块  

- 引入LSR标签平滑正则化。  

> 注：**Inception V3** 中设计的不同 **Inception** 模块:  
> ![Inception V3I1](some_img\v3i1.png)
> ![Inception V3I2](some_img\v3i2.png)
> ![Inception V3I3](some_img\v3i3.png)
> ![Inception V3R1](some_img\v3r1.png)
> ![Inception V3R1](some_img\v3r2.png)

> 注：**LSR**标签平滑正则化：  
> - 将原来的目标函数转化:设有一个标签分布 **u (k)**，不
依赖于训练样例 **x**，以及一个平滑参数 **ϵ** 。对于具有真实标签 **y** 的训练样例，我们将标签分布**q (k | x) = δk,y**
替换为
> $$  
> q^{\prime}(k|x) = (1 - \varepsilon) \delta_{k,y} + \varepsilon u(k)
> $$  
> - 采用了均匀分布 **u (k) = 1/K** ，真实标签 **k = y**
> $$
> q^{\prime}(k) = (1 - \varepsilon) \delta_{k,y} + \frac{\varepsilon}{k}
> $$
# **Inception V4**
## Trick:
- 将残差连接和 **Inception** 结合  

- 出于计算方面的考量（内存，减少层数）， 在残差相加后不做 **BN**

- 发现在卷积核数量大于 **1000** 后，在Inception-Residual层相加之前对Inception部分进行缩放会使原本训练变得极不稳定的网络训练更稳定
> 注：**Inception V4** 的 **Inception** ：
> ![Inception V4](some_img\1.png)
> ![Inception V4](some_img\2.png)
> ![Inception V4](some_img\3.png)
> ![Inception V4](some_img\4.png)
# **Xception**
## Trick:
- 更改了Inception结构，改为先 **DW3x3** 卷积后 **PW1x1** 卷积，原因是将空间相关性和跨通道相关性充分解耦
> 注：**Inception V3** 中设计的 **Inception** 模块:  
> ![X](some_img\x.png)