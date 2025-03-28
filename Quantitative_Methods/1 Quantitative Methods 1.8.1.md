# 1 Quantitative Methods 量化方法
## 1.8 Hypothesis Testing 假设检验

#### 学习要点

- 解释**假设检验**及其组成部分，包括**统计显著性**、**第一类错误（Type I error）** 和**第二类错误（Type II error）**，以及**检验的统计功效（power of a test）**。
- 构建**假设检验**，并确定其统计显著性、相关的第一类错误和第二类错误，以及在给定显著性水平下的检验功效。
- 比较和对比**参数检验（parametric tests）** 与**非参数检验（nonparametric tests）**，并描述在何种情况下，每种检验方法更为合适。

### 1.8.1 Hypothesis Tests for Finance 金融中的假设检验

假设检验是一种统计推断方法，用于评估数据是否提供足够的证据来拒绝某个假设。其过程通常包括以下几个关键步骤：

![[Pasted image 20250326151449.png]]
#### 设定假设（Stating the Hypotheses）

在每个假设检验中，我们需要设定两个假设：
- **原假设（Null Hypothesis, $H_0$）**：关于总体参数的假设，除非有足够证据表明其错误，否则认为它成立。**原假设需要永远包括等号**。
- **备择假设（Alternative Hypothesis, $H_a$）**：与原假设相对立的假设，通常是研究者希望证明的结论。

**关键点：**
1. **原假设是希望被拒绝的**。如果样本数据提供了足够的证据，我们可以拒绝原假设，接受备择假设。
2. **假设是针对总体参数提出的**，但我们使用样本统计量来检验这些假设。
3. **$H_0$ 和 $H_a$ 必须是互斥且穷尽的**，即所有可能的情况都包含在二者之一中。

#### 确定合适的检验统计量与分布（Identify the Appropriate Test Statistic and Distribution）

##### 检验统计量（Test Statistic）
检验统计量是基于样本计算出的值，在决策规则的配合下，用于判断是否拒绝原假设（$H_0$）。

##### 选择检验统计量的依据
检验统计量的选择取决于具体的检验目标，即我们正在检验的内容。不同的检验问题需要使用不同的检验统计量。

##### 检验统计量的分布
在确定适当的检验统计量后，还需要关注其分布。不同类型的检验统计量具有不同的分布。例如：
- **$z$ 统计量**：服从标准正态分布，用于大样本或已知总体方差的情况。
- **$t$ 统计量**：服从 $t$ 分布，用于小样本或总体方差未知的情况。
- **$\chi^2$ 统计量**：服从卡方分布，常用于方差分析或适配度检验。
- **$F$ 统计量**：服从 $F$ 分布，常用于方差齐性检验或回归分析。

![[Pasted image 20250326153800.png]]
$\overline{s_d}$是标准误差，$\overline{s_d} = \frac{s}{\sqrt{n}}$。

#### 指定显著性水平（Specify the Level of Significance）

##### 显著性水平的含义
显著性水平（significance level）反映了我们拒绝原假设（$H_0$）所需的样本证据强度。根据假设的性质及错误决策可能带来的后果，所需的证据标准可能有所不同。

##### 假设检验的四种可能结果
在假设检验中，可能出现四种情况：
- **正确接受**：原假设 $H_0$ 为真，且未拒绝 $H_0$。
- **正确拒绝**：原假设 $H_0$ 为假，且拒绝 $H_0$。
- **I 类错误（Type I error）**：误拒 $H_0$（错误地拒绝了一个真实的原假设），也称为“假阳性”。
- **II 类错误（Type II error）**：误受 $H_0$（未能拒绝一个错误的原假设），也称为“假阴性”。

I 类错误和 II 类错误是相互排斥的：如果误拒了真实的 $H_0$，则犯的是 I 类错误；如果未能拒绝错误的 $H_0$，则犯的是 II 类错误。

##### 显著性水平与错误概率
- **I 类错误的概率（$\alpha$）**：显著性水平，表示错误拒绝 $H_0$ 的概率。例如，显著性水平设定为 5%（$\alpha = 0.05$），意味着有 5% 的概率误拒了真实的 $H_0$，对应的置信水平（$1 - \alpha$）为 95%。
- **II 类错误的概率（$\beta$）**：表示错误接受 $H_0$ 的概率。
- **检验的统计功效（power of a test）**：正确拒绝 $H_0$ 的概率，即 $1 - \beta$，表示在 $H_0$ 为假时成功拒绝它的能力。

##### I 类与 II 类错误的权衡
降低 I 类错误的概率（如将显著性水平从 5% 降至 1%）会增加 II 类错误的概率，因为我们会更少地拒绝 $H_0$，即使它实际上是错误的。因此，决定接受哪种错误以及容忍程度通常取决于错误带来的成本和影响。

减少 I 类和 II 类错误的**唯一方法**是增加样本量 $n$。样本量越大，检验的统计功效（power）就越高，从而提高检验的准确性。

#### 设定决策规则（State the Decision Rule）

##### 决策规则的定义
假设检验的第四步是设定决策规则，即：
- **何时拒绝原假设（$H_0$）**
- **何时不拒绝原假设（$H_0$）**

决策的依据是 **计算得到的样本检验统计量（test statistic）** 与 **指定的临界值（critical value）** 进行比较。

##### 临界值的确定
临界值的选择取决于：
1. **显著性水平（$\alpha$）**：表示我们愿意接受的 I 类错误的概率。
2. **检验统计量的概率分布**：根据假设检验的类型，决定使用正态分布、$t$ 分布等。

##### 检验规则
- **如果计算出的检验统计量值比临界值更极端**，则拒绝原假设 $H_0$，即结果在统计上显著。
- **如果计算出的检验统计量值未超过临界值**，则无法拒绝原假设 $H_0$，即没有足够的证据支持拒绝 $H_0$。

##### p 值（p-value）与显著性
p 值是 **在给定原假设为真时，观测到当前检验统计量或更极端结果的概率**。它表示在概率分布中超出计算检验统计量的区域面积。

- **如果 p 值小于显著性水平 $\alpha$**（如 0.05），则拒绝 $H_0$。
- **如果 p 值大于等于 $\alpha$**，则无法拒绝 $H_0$。

换句话说，p 值是能否拒绝原假设的最小显著性水平。

#### 例题：假设检验结论（Hypothesis Test Conclusions）

对于以下关于总体均值 $\mu$ 的假设检验，根据计算得到的检验统计量（t-statistic）、临界值（critical value）和 p 值（p-value）得出结论。

##### 1. $H_0: \mu = 10$ vs. $H_a: \mu \neq 10$
- 计算得到的 $t$ 统计量：$t = 2.05$
- 临界值：$\pm1.984$
- **结论**：拒绝原假设 $H_0$，因为计算得到的 $t$ 统计量 $2.05$ 超出了临界值范围（$\pm1.984$）。

##### 2. $H_0: \mu \leq 10$ vs. $H_a: \mu > 10$
- 计算得到的 $t$ 统计量：$t = 2.35$
- 临界值：$+1.679$
- **结论**：拒绝原假设 $H_0$，因为计算得到的 $t$ 统计量 $2.35$ 大于临界值 1.679。

##### 3. $H_0: \mu = 10$ vs. $H_a: \mu \neq 10$
- 计算得到的 $t$ 统计量对应的 $p$ 值：$4.6352\%$
- 显著性水平：$5\%$
- **结论**：拒绝原假设 $H_0$，因为 $p$ 值（$4.6352\%$）小于显著性水平（$5\%$）。

##### 4. $H_0: \mu \leq 10$ vs. $H_a: \mu > 10$
- 计算得到的检验统计量对应的 $p$ 值：$3\%$
- 显著性水平：$2\%$
- **结论**：无法拒绝原假设 $H_0$，因为 $p$ 值（$3\%$）大于显著性水平（$2\%$）。


