# Think bayesian & Statistics review

## Main principles

1. Use prior konwledge
2. Chose answer that explains observations the most
3. Avoid extra assumptions

### example
A main is running, why?

1. He is in a hurry 
2. He is doing exports (use principle 2 to exclude, does not waer a sports suit, contradicts the data)
3. He always runs  (use principle 3 to exclude)
4. He saw a dragon  (use principle 1 to exclude)


## Probability

for throw a dice, the probability of one side is 1/6 

## Random variable

### Discrete

Probability Mass Function(PMF)
$$P(X) = \begin{equation}\left\{\begin{array}{**lr**}
    & 0.2 & x = 1 \\
    & 0.5 & x = 3 \\
    & 0.3 & x = 7 \\
    & 0 & otherwise
\end{array}\right.\end{equation}  
$$

### Continuous
Probability Density Function(PDF)

$$
P(x \in [a,b] = \lmoustache_{a}^{b} p(x)dx )
$$

### Independence

X and Y are independent if:
$$P(X,Y) = P(X)P(Y)$$

- P(x,y) -> Joint
- P(x)   -> Marinals

## Conditional probability
Probability of X given that Y happened:

$$P(X|Y) = \frac{P(X,Y)}{P(Y)}$$

### Chain rule

$$\begin{equation}\begin{split}
& P(X,Y) = P(X|Y)P(Y) \\
& P(X,Y,Z) = P(X|Y,Z)P(Y|Z)P(Z) \\
& P(X_1,\cdots,X_N) = \prod_{i=1}^{N}P(X_i|X_1,\cdots,X_{i-1})
\end{split} \end{equation}$$

### Sum rule 
$$P(X) = \lmoustache_{-\infty}^{\infty}P(X,Y)dy $$

## Total probability

1. $B_1, B_2 \cdots $ 两两互斥，即 $B_i \cap B_j = \emptyset$ ，$i \neq j$, i,j=1，2，....，且$P(B_i)>0$,i=1,2,....;
2. $B_1 \cup B_2 \cdots = \Omega$ ，则称事件组 $B_1 \cup B_2 \cdots$ 是样本空间 $\Omega$ 的一个划分

$$P(A) = \sum_{i=1}^{\infty}P(B_i)(A|B_i)$$

## Bayes theorem

- $\theta$: parameters
- $X$: observations
- $P(\theta|X)$: Posterior
- $P(X)$: Evidence
- $P(X|\theta)$: Likelyhood
- $P(\theta)$: Prior


$$P(\theta|X) = \frac{P(X,\theta)}{P(X)} = \frac{P(X|\theta)P(\theta)}{P(X)}$$

## Bayesian approach to statistics

### Frequentist
- Objective
- $\theta$ is fixed, X is random
- training  
    Maximum Likelyhood (they try to find the parameters theta that maximize the likelihood, the probability of their data given parameters)
    $$\hat{\theta} = argmax_{\theta}P(x|\theta)$$


### Bayesian
- Subjective
- X is random, $\theta$ is fixed
- Training(Bayes theorem)  
    what Bayesians will try to do is they would try to compute the posterior, the probability of the parameters given the data.
    $$P(\theta|x) = \frac{P(X|\theta)P(\theta)}{P(X)}$$
- Classification
    - Training:
    $$P(\theta|x_tr,y_tr) = \frac{P(y_tr|\theta,x_tr)P(\theta)}{P(y_tr|x_tr)}$$
    - Prediction:
    $$P(y_ts|x_ts,x_tr,y_tr) = \lmoustache{P(y_ts|x_ts,\theta)P(\theta|x_tr,y_tr)}d\theta$$
- On-line learning (get posterior)
    $$P_k{\theta} = P(\theta|x_k) = \frac{P(x|\theta)P_{k-1}(\theta)}{P_{(x_k)}}$$
    

> reference https://zhuanlan.zhihu.com/p/72506771

## 介绍

在概率论与数理统计领域中，对于一个未知参数的分布我们往往可以采用生成一批观测数据、通过这批观测数据做参数估计的做法来估计参数。最常用的有最大似然估计(MLP)、矩估计、最大后验估计(MAP)、贝叶斯估计等。
MLP通过最大化似然函数 $L(\theta|D)$ 从而找出参数$\theta$ ，思想在于找出能最大概率生成这批数据的参数。但是这种做法完全依赖于数据本身，当数据量大的时候，最大似然估计往往能很好的估计出参数$\theta$ ；但是当数据量小的时候，估计出来的结果并不会很好。就例如丢硬币问题，当我们投掷了5次，得出了正正正正正的结果，极大似然估计会得出投出正面的概率为100%！这显然是不符常理的。
贝叶斯派的人认为，被估计的参数同样服从一种分布，即参数也为一个随机变量。他们在估计参数前会先带来先验知识，例如参数在[0.5,0.6]的区域内出现的概率最大，在引入了先验知识后在数据量小的情况下估计出来的结果往往会更合理。


## MAP 与贝叶斯估计

MLP认为参数是一个常数，希望能找出最大化产生观测数据的参数，即：
$$\theta^{\prime} = argmax_{\theta}L(\theta|D) = argmax_{\theta}\theta^{m_h}(1-\theta)^{m_t}$$

贝叶斯派认为参数是一个随机变量，对它做估计就是计算其后验概率分布 [公式] ，我们借助贝叶斯公式展开有：

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$

其中 $P(D)$ 可当成是常数，因此可以有：

$$P(\theta|D) \propto \frac{P(D|\theta)}{P(\theta)}$$

其中 P( \theta) 为参数服从的分布，即先验知识。 接着我们可以有两种做法：
1. 最大后验估计(MAP)：找出最大化后验概率的参数 
$$\theta^{\prime} = argmax_{\theta}P(D|\theta)P(\theta)$$
预测阶段，借助参数做预测：
$$P(X^{new}|\theta^{\prime},D)$$
2. 贝叶斯估计：借助先验分布 $P(\theta)$, 与观测数据得到的$P(D|\theta)$ 得出后验分布，预测阶段借助后验分布有:
$$P(X^{new}|D) = \int{P(X^{new},\theta|D)d\theta} = \int{P(X^{new}|\theta,D)P(\theta|D)d\theta}$$


## 共轭分布与共轭先验
现在有了先验分布、似然，就可以接着做贝叶斯估计了。我们根据以往数据，给出先验知识，例如在以前的数据中，硬币出现了$\alpha_{h}$次正面，$\alpha_{t}$次背面，代入beta分布后有：

$$P(\theta|\alpha_{h},\alpha_{t}) = \frac{1}{B(\alpha_{h},\alpha_{t})}\theta^{\alpha_{h}-1}(1-\theta)^{\alpha_{t}-1}$$

$$P(\theta|\alpha_{h},\alpha_{t}) \propto \theta^{\alpha_{h}-1}(1-\theta)^{\alpha_{t}-1} $$

接着贝叶斯估计有:


$$\begin{equation}\begin{split}
P(\theta|D) & \propto \theta^{m_h}(1-\theta)^{m_t}\theta^{\alpha_h-1}(1-\theta)^{\alpha_{t}-1}
            & \propto \theta^{m_h+\alpha_h-1}(1-\theta)^{m_t+\alpha_t-1}
\end{split}\end{equation}$$

得出后验分布同样服从Beta分布 $Be(m_h + \alpha_h,m_t + \alpha_t )$,我们加上标准化函数后可以得到：

$$P(\theta|D) = \frac{1}{B(m_h + \alpha_h,m_t + \alpha_t)}\theta^{m_h+\alpha_h-1}(1-\theta)^{m_t+\alpha_t-1} $$

像这种先验分布和后验分布同分布时，先验分布和后验分布称之为共轭分布，**此时先验被称为似然函数的共轭先验**.

