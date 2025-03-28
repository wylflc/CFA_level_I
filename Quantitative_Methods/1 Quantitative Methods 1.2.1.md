# 1 Quantitative Methods
## 1.2 Time Value of Money in Finance 时间价值(TVM)在金融中的应用

**要点：**

- 计算和解释基于预期未来现金流的固定收益和股票工具的现值（PV）。
- 计算和解释给定现值（PV）和现金流的固定收益工具的隐含收益率，以及股票工具的要求收益率和隐含增长。
- 解释现金流可加性原理、其对无套利条件的重要性，以及其在计算隐含远期利率、远期汇率和期权价值中的应用。

### 1.2.1 Time Value of Money in Fixed Income and Equity 固定收益和股票中的货币时间价值

#### 学习要点：

- 计算并解释基于预期未来现金流的固定收益和股票工具的现值（PV）

正如在上一节所学的，现在的100元钱和未来的100元钱，并不是等价的，未来的100元钱应当考虑一个折扣率$r$，即利率。因此未来价值(FV)和当前价值(PV)之间有一个换算关系：
$$
FV_t = PV\times (1+r)^t
$$
当$t$足够大时，上述关系，近似在计算连续复利收益，则近似有
$$
FV_t = PV\times e^{rt}
$$
相应的，我们也能通过未来的价值来计算当前价值，即
$$
PV = FV_t\times (1+r)^{-t}
$$
$$
PV = FV_t\times e^{-rt}
$$

#### Fixed-Income Instruments and the Time Value of Money 固定收益工具和货币时间价值

**固定收益工具**是**债务工具**，例如**债券**或**贷款**，它们代表一种合同，其中发行方从投资者那里借款，以换取未来偿还的承诺。固定收益工具的**折现率**是**利率**，而债券或贷款的**回报率**通常被称为**到期收益率**（yield to maturity YTM）。YTM指的是年化收益率。

固定收益工具的现金流通常遵循以下三种一般模式之一：

1. **折扣**：投资者支付债券或贷款的初始价格（现值，PV），并在到期时收到一次性本金现金流（未来值，FV）。差额（FV − PV）代表该工具在其生命周期内赚取的利息。

2. **定期利息**：投资者支付债券或贷款的初始价格（现值，PV），并在工具的生命周期内按预定的间隔收到利息现金流（付款，PMT），最终的利息支付和本金（FV）在到期时偿还。

3. **等额支付**：投资者支付初始价格（现值，PV），并在预定的间隔（A）内收到均匀的现金流，直到到期，这些现金流既包括利息也包括本金偿还。

#### Discount Instruments 折扣工具

这种工具指的是，仅在到期时收到一次性的本金和现金流。也被称作**零息债券**。在已知其到期支付的值FV，以及其规定的利率$r$的情况下，其当前价值PV可以由下式计算：
$$
PV = \frac{FV_t}{(1+r)^t}
$$

#### Coupon Instrument 票息工具

![[Pasted image 20250319161341.png]]
如上图所示，每半年付一次利息，并且付的金额相同，到期之后归还本金。

为了计算票息工具的现值PV，我们需要使用一个非常类似现金流公式的式子。
$$
PV = \frac{PMT_1}{1+r}+\frac{PMT_2}{(1+r)^2}+\cdots+\frac{PMT_N+FV_N}{(1+r)^N}
$$
永续债券（**Perpetual Bond**）是一种较为少见的**票息债券（Coupon Bond）**，其特点是**没有固定的到期日**。大多数永续债券由公司发行，以获得类似权益融资（equity-like financing），并且通常包含赎回条款（**Redemption Features**）。
当 $N\rightarrow \infty$（即无限期）时，我们可以简化计算**永续固定周期现金流（且无提前赎回）的现值（PV）**，前提是 $r > 0$，计算公式如下：
$$
PV = \frac{PMT}{r}
$$
上面的公式根据**等比数列求和**不难得到。

#### Annuity Instruments 年金工具

![[Pasted image 20250319170546.png]]
简单来说，就是初始投资一笔钱，然后之后的每个**固定时间周期**，每个周期收到**等量的钱(PMT)**。最贴近我们的例子，就是**房贷**：初始买房时，我们相当于收到了银行投资给我们的一笔钱，然后我们在接下来的30年里面，需要每个月还给银行固定金额的钱。

在以上的模式下，现值PV的计算如下：
$$
\begin{align*}
PV &= \frac{PMT}{(1+r)}+\frac{PMT}{(1+r)^2}+\cdots + \frac{PMT}{(1+r)^n}\\
&= PMT\left(\frac{\frac{1}{1+r}\times \left(1-\left(\frac{1}{1+r}\right)^n\right)}{1-\frac{1}{1+r}}\right)\\
&= PMT\frac{1-(1+r)^{-n}}{r}
\end{align*}
$$
##### 例题

一个投资者寻求一笔固定利率的30年期按揭贷款，以融资购买价值1,000,000美元的住宅建筑的80%。

1. 如果年按揭利率为5.25%，请计算投资者的月度还款额。
2. 前两个月的现金流中，本金摊销和利息的分解是多少？

##### 答案

1. 投资者现在借了800,000美元，因此PV=800,000。按月还款，$r=\frac{5.25\%}{12}=0.4375\%$。基于之前的公式，可以计算得到月付4417.63美元。
2. 第一个月中，全部800,000美元都需要支付利息，因此利息为$800,000\times 0.4375\%=3500$，则本金部分为$4417.63-3500=917.63$；第二个月，仅使用了$800,000-917.63=799082.37$的本金，因此需要支付$799082.37\times 0.4375\% = 3495.99$的利息，进而可计算得到还了$4417.63-3495.99=921.64$的本金。

#### Equity Instruments and the Time Value of Money 股权工具与时间价值

**股权投资**，如优先股或普通股，代表公司中的**所有权份额**，赋予投资者以股息形式接收任何自由支配现金流的权利。与固定收益工具不同，股权投资没有到期日，通常被认为会一直存在，直到公司被**出售**、**重组**或**清算**为止。评估公司股票的一种方法是通过使用预期的**回报率**（$r$）折现预期的**未来现金流**。这些现金流包括**所有定期收到的股息**以及**投资期末预期获得的价格**。

与基于**股息现金流**估值股权工具相关的常见假设通常遵循以下三种一般方法之一：

1. **恒定股息（Constant Dividends）**：投资者支付初始价格（PV）购买优先股或普通股，并定期收到固定的股息（D）。

2. **恒定股息增长率（Constant Dividend Growth Rate）**：投资者支付初始价格（PV）购买股票，并在一个时期内收到初始股息（Dt+1），预计股息将以恒定的增长率（g）增长。

3. **变化股息增长率（Changing Dividend Growth Rate）**：投资者支付初始价格（PV）购买股票，并在一个时期内收到初始股息（Dt+1）。股息预计会以变化的增长率增长，即公司从初期的高速增长期过渡到成熟期的较慢增长期。

对于最简单的**恒定股息**的情况，类似之前的等比数列求和的思路，股权的现值应当为
$$
\begin{align*}
PV &= \sum_{i=1}^{\infty}\frac{D}{(1+r)^i}\\
&= \frac{D}{r}
\end{align*}
$$
对于**恒定股息增长率**的情况，假设股息增长率为$g$，则其股权现值应当为：
$$
\begin{align*}
PV &= \sum_{i=1}^{\infty}\frac{D(1+g)^i}{(1+r)^i}\\
&= \frac{D(1+g)}{r-g}
\end{align*}
$$
对于变化股息增长率的股票，实质上也是类似的计算方法。

##### 例题

回顾之前的计算，基于每年恒定的 GBP1.50 股息和 15% 的必要回报率，我们计算出 Shipline PLC 预期的股票价格为 GBP10.00。  
假设现在一位投资分析师认为 Shipline 每年的股息将以 6% 的速度无限期增长。
1. 在分析师假设股息以恒定 6% 的增长率增长的情况下，Shipline 预期的股价会如何变化？
2. 如果我们假设 Shipline 在最初三年内股息增长 6%，随后股息以 2% 的恒定增长率增长，那么其预期股价会如何变化？

##### 答案

1. 直接套用恒定股息增长率情况下的公式，$D = 1.5,~r = 0.15,~g = 0.06$，可以得到结果为$17.67$.
2. 首先需要计算前三年的现金流，按照$g_1 = 0.06$进行计算，之后需要按照$g_2=0.02$计算后续的永续现金流，具体公式如下：
$$
\begin{align*}
PV &= \sum_{i=1}^{3}\frac{D(1+g_1)^i}{(1+r)^i} + \sum_{i=4}^{\infty}\frac{D(1+g_2)^i}{(1+r)^i}\\
&= 13.05
\end{align*}
$$
