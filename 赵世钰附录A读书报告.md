## <center>《强化学习中的数学基础》附录A（结合Sutton第三章）读书笔记：概率论基础与强化学习应用
##### <center> 智能科学与技术 2213530 张禹豪
#### **一、核心概念总结**
1. **随机变量（Random Variable）**  
   - **定义**：表示可能取多个值的变量，取值服从概率分布。术语“变量”表明随机变量可以从一组数值中取值。术语“随机”表明取值的具体结果必须遵循某种概率分布。 
   - **表示方法**：大写字母（如 \(X\)）表示变量，小写字母（如 \(x\)）表示具体取值。  
   - **关键点**：  
     - 本书主要讨论有限取值的随机变量（标量或向量）。  
     - 同普通变量类似，随机变量也可以进行常规的数学运算，如求和、乘积和绝对值。（如 \(X+Y\)、\(XY\)）。  

2. **随机序列（Stochastic Sequence）**  
   - **定义**：由随机变量构成的序列（如掷骰子 \(n\) 次的序列 \(\{x_1, x_2, \ldots, x_n\}\))。  
   - **区分**：确定性序列（如 \(\{1,6,3,5,\ldots\}\)）与随机变量序列（\(x_i\) 是未确定的随机变量）。  

3. **概率（Probability）**  
   - **符号**：\(p(X=x)\) 或 \(p(x)\) 表示 \(X\) 取值为 \(x\) 的概率。  

---

#### **二、联合概率与条件概率**
1. **联合概率**  
   - **符号**：\(p(x,y)\) 表示 \(X=x\) 且 \(Y=y\) 的概率。  
   - **性质**：\(\sum_{y} p(x,y) = p(x)\)（边缘化）。  

2. **条件概率**  
   - **符号**：符号$p(X=x∣A=a)$表示在随机变量$A$已取值为$a$的条件下，随机变量$X$取值为$x$的概率。通常简写为 $p(x∣a)$。
   - **公式**：  
     \[
     p(x,a) = p(x \mid a) p(a) \quad且\quad p(x \mid a) = \frac{p(x,a)}{p(a)}.
     \]
   - **全概率定律**：  
     \[
     p(x) = \sum_{a} p(x \mid a) p(a).
     \]

3. **独立性与条件独立性**  
   - **独立性**：若一个随机变量的取值不影响另一个随机变量，则称二者独立。数学上，若 \(X\) 和 \(Y\) 独立，则 \(p(x,y) = p(x)p(y)\)，且由于$p(x,y) = p(x|y)p(y)$，所以 \(p(x \mid y) = p(x)\)。  
   - **条件独立性**：设 $X,A,B $为三个随机变量。若在给定 $B=b $的条件下，$X $的取值与 $A$ 无关，则称 $X $条件独立于$A$（给定$B$）。公式如下：给定 \(B=b\) 时，\(X\) 与 \(A\) 独立的条件为 \(p(x \mid a,b) = p(x \mid b)\)。 
   - **应用**：在马尔可夫过程中，下一状态仅依赖当前状态（无记忆性）：  
     \[
     p(s_{t+2} \mid s_{t+1}, s_t) = p(s_{t+2} \mid s_{t+1}).
     \]
4. **条件概率与联合概率的链式法则**
&emsp;&emsp;根据条件概率的定义，有$$
p(a,b) = p(a \mid b) p(b) 
$$
&emsp;&emsp;该法则可扩展至多变量情形，例如：$$
p(a,b,c) = p(a \mid b,c) p(b,c) = p(a \mid b,c) p(b \mid c) p(c) 
$$
&emsp;&emsp;因此可得：$$
\frac{p(a,b,c)}{p(c)} = p(a,b \mid c) =p(a \mid b,c) p(b \mid c)
$$
&emsp;&emsp;类似地，条件联合概率满足以下性质$$
p(x \mid a) = \sum_bp(x,b \mid a) = \sum_bp(x \mid b,a) p(b \mid a)
$$
---

#### **三、期望**
**期望（Expectation）**  
   - **定义**：\(\mathbb{E}[X] = \sum_{x} p(x) x\)。  
   - **线性性**：  
     \[
     \mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y], \quad \mathbb{E}[aX] = a \mathbb{E}[X].
     \]
   - **线性性质证明**： $$\begin{aligned}
\mathbb{E}[X+Y] &= \sum_{x}\sum_{y}(x+y)p(x,y) \\
&= \sum_{x}xp(x) + \sum_{y}yp(y) \\
&= \mathbb{E}[X] + \mathbb{E}[Y]
\end{aligned}$$


---

#### **四、梯度与期望的结合**
- **期望的梯度** ：设\(f(X, \beta)\)为随机变量$X$和确定性参数向量$\beta$的标量函数，则期望的梯度满足：  
  \[
  \nabla_\beta \mathbb{E}[f(X, \beta)] = \mathbb{E}[\nabla_\beta f(X, \beta)].
  \]
  - **应用场景**：强化学习中的策略梯度方法，通过梯度优化期望回报。  
  - **证明**：$$\begin{aligned}
\mathbb{E}[f(X,\beta)] &= \sum_{x}f(x,\beta)p(x) \\
\nabla_{\beta}\mathbb{E}[f(X,\beta)] &= \nabla_{\beta}\sum_{x}f(x,\beta)p(x) \\
&= \sum_{x} \nabla_{\beta}f(x,\beta)p(x) \\
&= \mathbb{E}[\nabla_{\beta}f(X,\beta)]
\end{aligned}$$

---


#### **五、方差、协方差与协方差矩阵**
- **方差（单个随机变量）**
$$
\operatorname{var}(X) = \mathbb{E}[(X-\bar{X})^2], \quad \text{其中} \ \bar{X} = \mathbb{E}[X]
$$
- **协方差（两个随机变量）**
$$
\operatorname{cov}(X,Y) = \mathbb{E}[(X-\bar{X})(Y-\bar{Y})]
$$
- **协方差矩阵（随机向量）**
&emsp;&emsp;设随机向量$X = [X_1,...,X_n]^T$，则协方差矩阵定义为：
$$
\operatorname{var}(X) \doteq \Sigma = \mathbb{E}[(X-\bar{X})(X-\bar{X})^T] \in \mathbb{R}^{n \times n}
$$
&emsp;&emsp;其中矩阵元素满足：
$$
[\Sigma]_{ij} = \operatorname{cov}(X_i,X_j) = \mathbb{E}[(X_i-\bar{X}_i)(X_j-\bar{X}_j)]
$$
- **性质**
   - **确定性量的方差为零**：若$a$为确定性量，则$var(a)=0$ 
   - **线性变换性质：**：$$var(AX+a) = A var(X)A^T = A\sum A^T$$

---

#### **六、重要性质总结**
- **协方差与期望的关系**$$\operatorname{cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$
   - **证明**$$\begin{aligned}
\mathbb{E}[(X-\bar{X})(Y-\bar{Y})] &= \mathbb{E}[XY - X\bar{Y} - Y\bar{X} + \bar{X}\bar{Y}] \\
&= \mathbb{E}[XY] - \mathbb{E}[X]\bar{Y} - \bar{X}\mathbb{E}[Y] + \bar{X}\bar{Y} \\
&= \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
\end{aligned}$$

- **独立随机变量的期望乘积**$$X,Y \text{独立} \implies \mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$$
   - **证明**$$\mathbb{E}[XY] = \sum_{x}\sum_{y}p(x,y)xy = \sum_{x}p(x)x \sum_{y}p(y)y = \mathbb{E}[X]\mathbb{E}[Y]$$

- **独立随机变量的协方差为零**$$X,Y \text{独立} \implies \operatorname{cov}(X,Y) = 0$$
   - **证明**$$\operatorname{cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = 0$$

---

#### **七、强化学习中的核心应用**
1. **马尔可夫过程**：依赖条件独立性实现状态转移的无记忆性。  
2. **贝尔曼方程**：利用全期望定律分解状态值函数。  
3. **策略梯度定理**：通过梯度与期望的交换性优化策略参数。  

---

#### **八、总结**
概率论为强化学习提供了建模不确定性的数学工具：  
- **随机变量**描述环境与智能体的交互；  
- **条件概率与独立性**简化状态转移的建模；  
- **期望与方差**量化策略的长期收益与风险；  
- **梯度与期望的结合**支撑策略优化算法（如 Policy Gradient）。  

掌握这些基础概念，是理解强化学习核心理论（如 MDP、贝尔曼方程）的关键前提。