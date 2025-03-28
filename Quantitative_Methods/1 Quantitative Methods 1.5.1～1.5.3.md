# 1 Quantitative Methods 量化方法
## 1.5 Portfolio Mathematics 投资组合数学

#### 学习要点

- 计算并解释投资组合收益的期望值、方差、标准差、协方差和相关性
- 使用联合概率函数计算并解释投资组合收益的协方差和相关性
- 定义短缺风险，计算安全第一比率，并使用罗伊的安全第一准则确定最佳投资组合

### 1.5.1 Portfolio Expected Return and Variance of Return 投资组合的期望收益和收益方差

#### 投资组合期望收益（Portfolio Expected Return）

投资组合期望收益是根据投资组合中各项资产的权重和它们的期望收益加权平均值计算得出的。公式为：
$$
E(R_p) = w_1E(R_1) + w_2E(R_2) + \cdots + w_nE(R_n)
$$
其中：
- $E(R_p)$ 是投资组合的期望收益
- $w_1, w_2, \dots, w_n$ 是各资产的权重
- $E(R_1), E(R_2), \dots, E(R_n)$ 是各资产的期望收益

#### 投资组合收益方差（Portfolio Variance）

投资组合收益的方差衡量了投资组合整体收益的波动性，对应投资组合的**风险**，它考虑了资产之间的相关性。公式为：
$$
\begin{align*}
Var(R_p) = w_1^2Var(R_1) + w_2^2Var(R_2) + \cdots \\
+ w_n^2Var(R_n) + 2\sum_{i<j} w_i w_j Cov(R_i, R_j)
\end{align*}
$$
其中：
- $Var(R_p)$ 是投资组合收益的方差
- $Var(R_i)$ 是单一资产 $i$ 的收益方差
- $Cov(R_i, R_j)$ 是资产 $i$ 和资产 $j$ 的协方差
- $w_i, w_j$ 是各资产的权重

这个公式表明，投资组合的方差不仅受各资产方差的影响，还受资产之间**协方差**的影响。通过适当的资产配置，可以降低投资组合的整体波动性。

#### 协方差（Covariance）

协方差是衡量两项资产收益之间关系的统计量，它描述了两项资产收益的共同变动程度。简单来说，协方差告诉我们两项资产的收益是否同时上升或下降，或者它们的收益是否存在反向关系。

##### 协方差公式
$$
Cov(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})
$$
其中：
- $Cov(X, Y)$ 是资产 $X$ 和资产 $Y$ 之间的协方差
- $X_i, Y_i$ 是第 $i$ 期的资产 $X$ 和资产 $Y$ 的收益
- $\bar{X}$ 和 $\bar{Y}$ 分别是资产 $X$ 和资产 $Y$ 的收益平均值
- $n$ 是样本数量

##### 协方差的解释

- 如果协方差为正值，意味着两项资产的收益是**同向变化**的，即当一个资产的收益增加时，另一个资产的收益也倾向于增加。
- 如果协方差为负值，意味着两项资产的收益是**反向变化**的，即当一个资产的收益增加时，另一个资产的收益则倾向于减少。
- 如果协方差为零，表示两项资产的收益变化没有线性关系。

然而，协方差的绝对值通常不易解释，因为它依赖于资产收益的单位。因此，投资者通常更倾向于使用相关系数来衡量资产之间的关系。

#### 相关性（Correlation）

相关性是衡量两项资产收益之间**线性关系**强度和方向的统计量。它是协方差的标准化版本，值的范围从 -1 到 +1。通过相关性，可以更直观地了解两项资产的收益是否同向或反向变化，以及这种关系的强度。

##### 相关性公式
$$
r_{XY} = \frac{Cov(X, Y)}{\sigma_X \sigma_Y}
$$
其中：
- $r_{XY}$ 是资产 $X$ 和资产 $Y$ 之间的相关性系数
- $Cov(X, Y)$ 是资产 $X$ 和资产 $Y$ 之间的协方差
- $\sigma_X$ 和 $\sigma_Y$ 分别是资产 $X$ 和资产 $Y$ 的标准差

##### 相关性的解释

- **正相关（$r_{XY}$ 接近 +1）**：当 $r_{XY}$ 接近 +1 时，表示资产 $X$ 和资产 $Y$ 之间有强烈的正线性关系。也就是说，当一个资产的收益上升时，另一个资产的收益也倾向于上升。
- **负相关（$r_{XY}$ 接近 -1）**：当 $r_{XY}$ 接近 -1 时，表示资产 $X$ 和资产 $Y$ 之间有强烈的负线性关系。也就是说，当一个资产的收益上升时，另一个资产的收益倾向于下降。
- **无相关（$r_{XY}$ 接近 0）**：当 $r_{XY}$ 接近 0 时，表示资产 $X$ 和资产 $Y$ 之间没有显著的线性关系。即它们的收益变化没有明显的同步性。

##### 相关性的应用

投资者通常使用相关性来优化投资组合。通过选择不同相关性的资产，投资者可以降低整体组合的风险。例如，负相关资产可以相互对冲，从而减少整体波动性和风险。

### 1.5.2 Forecasting Correlation of Returns: Covariance Given a Joint Probability Function 预测收益的相关性：基于联合概率函数计算协方差

在投资组合分析中，我们可以使用**联合概率函数**计算两个资产收益的**协方差**，并进一步推导出它们的**相关性**。这种方法在处理离散概率分布时尤为重要。

#### 协方差计算公式（基于联合概率函数）

给定两个资产 $X$ 和 $Y$ 的联合概率分布 $p(x_i, y_j)$，它们的协方差可以计算如下：
$$
Cov(X, Y) = \sum_{i,j} p(x_i, y_j) \left( x_i - E_X \right) \left( y_j - E_Y \right)
$$
其中：
- $x_i$ 和 $y_j$ 是资产 $X$ 和 $Y$ 可能的收益值
- $p(x_i, y_j)$ 是对应的联合概率
- $E_X = \sum_{i} x_i P(X = x_i)$ 是 $X$ 的期望值
- $E_Y = \sum_{j} y_j P(Y = y_j)$ 是 $Y$ 的期望值

#### 计算相关性

计算出协方差后，可以使用以下公式计算相关性：
$$
r_{XY} = \frac{Cov(X, Y)}{\sigma_X \sigma_Y}
$$
其中：
- $\sigma_X = \sqrt{Var(X)}$ 是资产 $X$ 的标准差
- $\sigma_Y = \sqrt{Var(Y)}$ 是资产 $Y$ 的标准差

#### 应用

在投资实践中，这种方法可以用于：
1. **预测资产收益的相关性**，帮助投资者更好地构建分散化投资组合。
2. **计算资产间的协方差**，从而进一步计算投资组合的风险和收益特征。
3. **评估不同市场条件下的相关性变化**，为投资决策提供数据支持。

### 1.5.3 Portfolio Risk Measures: Applications of the Normal Distribution 投资组合风险衡量：正态分布的应用

在投资组合管理中，正态分布被广泛用于衡量和分析投资组合的风险。许多资产回报率被假设为正态分布，这使得投资者能够利用统计方法来量化风险。

#### 投资组合收益的正态分布假设
假设投资组合的收益率 $R_p$ 服从正态分布，即：
$$
R_p \sim N(\mu_p, \sigma_p^2)
$$
其中：
- $\mu_p$ 是投资组合的期望收益率。
- $\sigma_p$ 是投资组合的标准差（风险）。

#### Safety-First Ratio

Safety-First 比率（SFR）衡量投资组合收益低于某个最低可接受收益（Threshold Return, $R_T$）的概率。Roy（1952）提出的 Safety-First 规则建议投资者选择使该概率最小化的投资组合。

Safety-First 比率的计算公式为：
$$
SFRatio = \frac{E(R_p) - R_T}{\sigma_p}
$$
其中：
- $E(R_p)$：投资组合的期望收益率
- $R_T$：最低可接受收益率（通常是无风险利率 $R_f$）
- $\sigma_p$：投资组合收益率的标准差（衡量风险）

##### 解释
- **较高的 SFR** 表示投资组合相对安全，即低于最低可接受收益的概率较低。
- **较低的 SFR** 表示投资组合承担较大风险，可能更频繁地产生低于最低可接受收益的回报。

根据 Roy 的 Safety-First 规则，投资者应选择 SFR **最大** 的投资组合，即：
$$
\max \left( \frac{E(R_p) - R_T}{\sigma_p} \right)
$$

##### 标准正态分布与安全优先规则
假设投资组合收益率服从正态分布，可以计算低于 $R_T$ 的概率：
$$
P(R_p < R_T) = P\left( Z < SFRatio \right)
$$
其中 $Z$ 服从标准正态分布。

根据正态分布表查找该概率，投资者可以选择满足其风险承受能力的最优投资组合。

##### 应用
- 选择最小化破产风险的投资组合
- 评估投资组合是否满足最低收益要求
- 对比不同投资策略的安全性


