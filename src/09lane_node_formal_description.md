# 09lane_node.py 方法的形式化描述

## 1. 问题定义

### 1.1 输入数据结构

**轨迹数据集** $\mathcal{T}$：
$$\mathcal{T} = \{(v_i, t_j, n_k, s_{ij}, c_i) \mid i \in \mathcal{V}, j \in \mathcal{F}, k \in \mathcal{N}\}$$

其中：
- $\mathcal{V}$：车辆ID集合
- $\mathcal{F}$：时间帧集合（单位：秒）
- $\mathcal{N}$：节点ID集合
- $v_i$：车辆ID
- $t_j$：时间帧
- $n_k$：节点ID（FID字段）
- $s_{ij}$：车辆 $v_i$ 在时间 $t_j$ 的速度（单位：km/h）
- $c_i$：车辆 $v_i$ 的类型 $\in \{\text{car}, \text{medium}, \text{heavy}, \text{motorcycle}\}$

**道路图结构** $\mathcal{G}$：
$$\mathcal{G} = (\mathcal{N}, \mathcal{E}, \mathcal{A})$$

其中：
- $\mathcal{N}$：节点集合
- $\mathcal{E}$：边集合
- $\mathcal{A}$：节点属性映射

对于每个节点 $n \in \mathcal{N}$，其属性为：
$$\mathcal{A}(n) = (l_n, p_n, L_n, C_n^d, C_n^r, C_n^c)$$

其中：
- $l_n$：车道ID
- $p_n$：节点在车道中的位置
- $L_n$：节点段长度（单位：米）
- $C_n^d$：直连节点集合（direct connections）
- $C_n^r$：近邻节点集合（near connections）
- $C_n^c$：交叉节点集合（crossing connections）

### 1.2 车辆占用长度映射

定义车辆类型到占用长度的映射函数：
$$\ell: \mathcal{C} \rightarrow \mathbb{R}^+$$

$$\ell(c) = \begin{cases}
4.0 & \text{if } c = \text{car} \\
6.0 & \text{if } c = \text{medium} \\
10.0 & \text{if } c = \text{heavy} \\
2.0 & \text{if } c = \text{motorcycle} \\
4.0 & \text{otherwise (default)}
\end{cases}$$

单位：米

## 2. 多滑块滑动窗口机制

### 2.1 窗口参数定义

定义三种不同的滑动窗口大小：
- $W_s = 1$ 秒（速度窗口）
- $W_f = 10$ 秒（流量窗口）
- $W_o = 4$ 秒（占用率窗口）

窗口半径：
- $r_s = \lfloor W_s / 2 \rfloor = 0$ 秒
- $r_f = \lfloor W_f / 2 \rfloor = 5$ 秒
- $r_o = \lfloor W_o / 2 \rfloor = 2$ 秒

最大窗口半径：
$$r_{\max} = \max(r_s, r_f, r_o) = 5 \text{ 秒}$$

### 2.2 输出时间点集合

给定原始数据的时间范围 $[t_{\min}, t_{\max}]$，输出时间点集合定义为：
$$\mathcal{T}_{\text{out}} = \{t \mid t = t_{\min} + r_{\max} + k, k \in \mathbb{Z}^+, t \leq t_{\max} - r_{\max}\}$$

即：输出时间范围为 $[t_{\min} + r_{\max}, t_{\max} - r_{\max}]$，步长为1秒。

### 2.3 时间窗口定义

对于输出时间点 $t \in \mathcal{T}_{\text{out}}$ 和节点 $n \in \mathcal{N}$：

**速度窗口**：
$$\mathcal{W}_s(t) = [t - r_s, t + r_s) = [t - 0.5, t + 0.5)$$

**流量窗口**：
$$\mathcal{W}_f(t) = [t - r_f, t + r_f) = [t - 5, t + 5)$$

**占用率窗口**：
$$\mathcal{W}_o(t) = [t - r_o, t + r_o) = [t - 2, t + 2)$$

## 3. 交通指标计算

### 3.1 平均速度计算

对于节点 $n$ 和输出时间 $t$，定义速度窗口内的轨迹子集：
$$\mathcal{T}_s(n, t) = \{(v_i, t_j, n_k, s_{ij}, c_i) \in \mathcal{T} \mid n_k = n \land t_j \in \mathcal{W}_s(t)\}$$

平均速度定义为：
$$\bar{S}(n, t) = \begin{cases}
\frac{1}{|\mathcal{T}_s(n, t)|} \sum_{(v_i, t_j, n_k, s_{ij}, c_i) \in \mathcal{T}_s(n, t)} |s_{ij}| & \text{if } \mathcal{T}_s(n, t) \neq \emptyset \\
\text{null} & \text{otherwise}
\end{cases}$$

单位：km/h

### 3.2 流量计算

对于节点 $n$ 和输出时间 $t$，定义流量窗口内的轨迹子集：
$$\mathcal{T}_f(n, t) = \{(v_i, t_j, n_k, s_{ij}, c_i) \in \mathcal{T} \mid n_k = n \land t_j \in \mathcal{W}_f(t)\}$$

定义流量窗口内的唯一车辆集合：
$$\mathcal{V}_f(n, t) = \{v_i \mid (v_i, t_j, n_k, s_{ij}, c_i) \in \mathcal{T}_f(n, t)\}$$

流量定义为：
$$Q(n, t) = |\mathcal{V}_f(n, t)|$$

单位：辆（在10秒窗口内）

### 3.3 占用率计算

#### 3.3.1 单帧占用率

对于节点 $n$ 和特定时间帧 $t_j$，定义该帧内的车辆集合：
$$\mathcal{V}_n(t_j) = \{(v_i, t_j, n, s_{ij}, c_i) \in \mathcal{T} \mid n_k = n\}$$

单帧总占用长度：
$$L_{\text{occ}}(n, t_j) = \sum_{(v_i, t_j, n, s_{ij}, c_i) \in \mathcal{V}_n(t_j)} \ell(c_i)$$

单帧占用率：
$$\rho(n, t_j) = \min\left(\frac{L_{\text{occ}}(n, t_j)}{L_n}, 1.0\right)$$

其中 $L_n$ 是节点 $n$ 的段长度。

#### 3.3.2 平均占用率

对于节点 $n$ 和输出时间 $t$，定义占用率窗口内的时间帧集合：
$$\mathcal{F}_o(n, t) = \{t_j \in \mathcal{F} \mid t_j \in \mathcal{W}_o(t) \land \exists (v_i, t_j, n, s_{ij}, c_i) \in \mathcal{T}\}$$

平均占用率定义为：
$$\bar{\rho}(n, t) = \begin{cases}
\frac{1}{|\mathcal{F}_o(n, t)|} \sum_{t_j \in \mathcal{F}_o(n, t)} \rho(n, t_j) & \text{if } \mathcal{F}_o(n, t) \neq \emptyset \\
0 & \text{otherwise}
\end{cases}$$

无量纲，范围：$[0, 1]$

## 4. 数据归一化

### 4.1 速度归一化

速度保持原始值，不进行归一化：
$$\bar{S}'(n, t) = \bar{S}(n, t)$$

### 4.2 流量归一化

流量采用对数变换后归一化：
$$Q'(n, t) = \frac{\ln(1 + Q(n, t))}{\ln(15)}$$

归一化范围：$[0, 1]$（当 $Q(n, t) = 14$ 时，$Q'(n, t) = 1$）

### 4.3 占用率归一化

占用率已经在 $[0, 1]$ 范围内，不需要额外归一化：
$$\bar{\rho}'(n, t) = \bar{\rho}(n, t)$$

## 5. 输出数据结构

最终输出为一个数据集 $\mathcal{R}$：
$$\mathcal{R} = \{(n, t, \bar{S}'(n, t), \bar{\rho}'(n, t), Q'(n, t)) \mid n \in \mathcal{N}, t \in \mathcal{T}_{\text{out}}\}$$

每条记录包含：
- `node_id`：节点ID $n$
- `start_frame`：输出时间点 $t$
- `avg_speed`：平均速度 $\bar{S}'(n, t)$（km/h，可能为null）
- `avg_occupancy`：平均占用率 $\bar{\rho}'(n, t)$（无量纲，$[0, 1]$）
- `total_vehicles`：归一化流量 $Q'(n, t)$（无量纲，$[0, 1]$）

## 6. 算法复杂度分析

### 6.1 时间复杂度

设：
- $|\mathcal{T}|$：轨迹记录总数
- $|\mathcal{N}|$：节点总数
- $|\mathcal{T}_{\text{out}}|$：输出时间点数量

主循环的时间复杂度：
$$O(|\mathcal{N}| \times |\mathcal{T}_{\text{out}}| \times |\mathcal{T}|)$$

实际实现中，通过预先按节点分组，可以优化为：
$$O(|\mathcal{T}| \times \log|\mathcal{T}| + |\mathcal{N}| \times |\mathcal{T}_{\text{out}}| \times \bar{k})$$

其中 $\bar{k}$ 是每个节点在每个窗口内的平均记录数。

### 6.2 空间复杂度

$$O(|\mathcal{T}| + |\mathcal{N}| + |\mathcal{R}|) = O(|\mathcal{T}| + |\mathcal{N}| \times |\mathcal{T}_{\text{out}}|)$$

## 7. 方法特性与假设

### 7.1 关键假设

1. **时间连续性**：假设轨迹数据在时间上是连续采样的
2. **节点唯一性**：每个时间帧内，一辆车只能在一个节点上
3. **速度非负性**：使用速度的绝对值，忽略方向信息
4. **占用率上界**：占用率被限制在 $[0, 1]$ 范围内，即使理论上可能超过1

### 7.2 方法优势

1. **多尺度时间分析**：不同指标使用不同窗口大小，适应各自的时间特性
2. **数据平滑**：滑动窗口机制减少了随机波动的影响
3. **完整性**：即使某些节点在某些时间段无数据，也会输出记录（值为0或null）

### 7.3 潜在局限

1. **边界效应**：输出时间范围缩减了 $2 \times r_{\max}$ 秒
2. **窗口选择**：窗口大小是预设的，可能不适用于所有交通场景
3. **占用率简化**：假设车辆在节点内均匀分布，未考虑车辆的实际空间分布

## 8. 扩展与改进方向

### 8.1 自适应窗口

可以根据交通流状态动态调整窗口大小：
$$W_f(n, t) = f(\bar{S}(n, t), \bar{\rho}(n, t))$$

### 8.2 加权平均

可以对窗口内的数据点赋予不同权重，例如高斯权重：
$$w(t_j, t) = \exp\left(-\frac{(t_j - t)^2}{2\sigma^2}\right)$$

### 8.3 多车型分类统计

可以分别统计不同车型的流量和占用率：
$$Q_c(n, t) = |\{v_i \in \mathcal{V}_f(n, t) \mid c_i = c\}|$$

## 9. 符号表

| 符号 | 含义 | 单位/类型 |
|------|------|-----------|
| $\mathcal{T}$ | 轨迹数据集 | 集合 |
| $\mathcal{V}$ | 车辆ID集合 | 集合 |
| $\mathcal{F}$ | 时间帧集合 | 秒 |
| $\mathcal{N}$ | 节点ID集合 | 集合 |
| $\mathcal{G}$ | 道路图结构 | 图 |
| $L_n$ | 节点段长度 | 米 |
| $\ell(c)$ | 车辆占用长度 | 米 |
| $W_s, W_f, W_o$ | 窗口大小 | 秒 |
| $r_s, r_f, r_o$ | 窗口半径 | 秒 |
| $\bar{S}(n, t)$ | 平均速度 | km/h |
| $Q(n, t)$ | 流量 | 辆 |
| $\rho(n, t_j)$ | 单帧占用率 | 无量纲 |
| $\bar{\rho}(n, t)$ | 平均占用率 | 无量纲 |
| $\mathcal{T}_{\text{out}}$ | 输出时间点集合 | 秒 |
| $\mathcal{R}$ | 输出结果集 | 集合 |
