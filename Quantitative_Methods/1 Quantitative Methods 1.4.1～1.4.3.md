# 1 Quantitative Methods 量化方法
## 1.4 Probability Trees and Conditional Expectations 概率树与条件期望

**学习要点**

- 计算期望值、方差和标准差，并展示其在投资问题中的应用  
- 将投资问题构建为概率树，并解释条件期望在投资中的应用  
- 在投资环境中使用贝叶斯公式计算和解释更新后的概率  

### 1.4.1 Expected Value and Variance 期望值和方差

#### 期望值 (Expected Value)  

期望值衡量随机变量的平均值，定义如下：  
$$ E(X) = \sum_{i=1}^{n} P(x_i) x_i $$
其中，$X$ 是随机变量，$x_i$ 是可能的取值，$P(x_i)$ 是对应的概率。对于连续随机变量，期望值的计算公式为：  
$$ E(X) = \int_{-\infty}^{\infty} x f(x) \,dx $$
其中，$f(x)$ 是概率密度函数。  

#### 方差 (Variance)  

方差衡量随机变量的离散程度，定义如下：  
$$ Var(X) = E[(X - E(X))^2] = \sum_{i=1}^{n} P(x_i) (x_i - E(X))^2 $$
对于连续随机变量，方差计算公式为：  
$$ Var(X) = \int_{-\infty}^{\infty} (x - E(X))^2 f(x) \,dx $$
方差的平方根称为标准差（Standard Deviation）：  
$$ \sigma_X = \sqrt{Var(X)} $$
标准差是衡量数据波动性的常用指标，在投资分析中广泛用于衡量资产的风险。

### 1.4.2 Probability Trees and Conditional Expectations 概率树与条件期望

#### 概率树 (Probability Trees)  

概率树是一种用于表示多个可能结果及其概率的图形工具，通常用于决策分析和投资评估。概率树的每个分支代表一个可能的结果及其概率，最终可以计算总体期望收益或风险。  

#### 条件期望 (Conditional Expectation)  

条件期望衡量在给定某一事件发生的情况下，随机变量的期望值。定义如下：  
$$ E(X | Y) = \sum_{i} x_i P(X = x_i | Y) $$
对于连续变量，条件期望计算公式为：  
$$ E(X | Y) = \int_{-\infty}^{\infty} x f(x | y) \,dx $$
其中，$f(x | y)$ 是 $X$ 在给定 $Y$ 条件下的条件概率密度函数。  

在投资分析中，条件期望用于调整预测值，例如在不同市场条件下的资产回报预期计算。

#### Total Probability Rule for Expected Value 期望值的全概率公式 

全概率公式用于计算随机变量的期望值，当其取决于某一分解事件的概率分布时。公式如下：  

**离散型随机变量**

若随机变量 $X$ 依赖于一组互斥且穷尽的事件 $B_1, B_2, \dots, B_n$，则 $X$ 的期望值可以表示为：  
$$ E(X) = \sum_{i} E(X | B_i) P(B_i) $$
**连续型随机变量**

若 $X$ 依赖于连续随机变量 $Y$，则其期望值为：  
$$ E(X) = \int_{-\infty}^{\infty} E(X | Y=y) f_Y(y) \, dy $$
其中，$f_Y(y)$ 是随机变量 $Y$ 的概率密度函数。  

**应用**

在投资分析中，全概率公式用于计算资产回报的期望值，考虑不同市场状态（如牛市、熊市）下的条件期望和状态概率。

### 1.4.3 Bayes' Formula and Updating Probability Estimates 贝叶斯公式与概率更新

#### 要点

在投资环境中使用贝叶斯公式计算和解释更新概率  

#### 贝叶斯公式  

贝叶斯定理（Bayes' Theorem）公式：
$$
P(A | B) = \frac{P(B | A) P(A)}{P(B)}
$$
其中$B$指Information，$A$指Event

贝叶斯公式是一个看起来略显复杂，但是实际用起来极为复杂的公式，先验概率和后验概率的转化在许多应用中都极为重要，尤其在机器学习和深度学习中也应用广泛，是一个非常重要的结果。

- **$P(A | B)$ ：后验概率（Posterior Probability）**。即在观测到 **B 发生之后**，事件 **A 发生的概率**。
- **$P(A)$ ：先验概率（Prior Probability）**。表示在没有新证据前，对事件 A 发生的原始估计概率。
- **$P(B | A)$ ：似然（Likelihood）**。表示 **如果 A 发生，B 发生的概率**。
- **$P(B)$ ：全概率（Total Probability）**。B 发生的总概率，计算方式：  $$
   P(B) = P(B | A) P(A) + P(B | \neg A) P(\neg A)
   $$其中 **$\neg A$** 代表 A 不发生。

#### 例题

**推断 DriveMed 的每股收益 (EPS) 是否符合市场预期**

你是 DriveMed 股票的投资者。已知你的先验概率如下：  
- $P(\text{EPS 超出预期}) = 0.45$  
- $P(\text{EPS 符合预期}) = 0.30$  
- $P(\text{EPS 低于预期}) = 0.25$  

此外，你还掌握以下条件概率：  
- $P(\text{DriveMed 扩张} | \text{EPS 超出预期}) = 0.75$  
- $P(\text{DriveMed 扩张} | \text{EPS 符合预期}) = 0.20$  
- $P(\text{DriveMed 扩张} | \text{EPS 低于预期}) = 0.05$  

请估算DriveMed 扩张情况下EPS 超出预期，符合预期，和低于预期的先验概率。  

#### 解答

首先计算DriveMed 扩张的概率：
$$
\begin{align*}
P(\text{扩张}) = &P(\text{扩张} | \text{超出预期})\times P(\text{超出预期}) + \\
&P(\text{扩张} | \text{符合预期})\times P(\text{符合预期}) + \\
&P(\text{扩张} | \text{低于预期})\times P(\text{低于预期})
\end{align*}
$$
计算得到$P(\text{扩张}) = 0.75\times 0.45 + 0.20\times 0.30 + 0.05\times 0.25 = 0.41$。
- 超出预期的先验概率为$P(\text{超出预期}|\text{ 扩张}) = \frac{P(\text{扩张} | \text{超出预期})\times P(\text{超出预期})}{P(\text{扩张})}=0.8232$。
- 符合预期的先验概率类似可得为0.1463。
- 低于预期的先验概率为0.0305。



