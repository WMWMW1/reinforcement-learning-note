# Exploration Methods

In this chapter we overview several major classes of exploration in reinforcement learning, focusing first on the Bernoulli bandit model, then defining regret, and finally presenting a number of exploration algorithms.

---

## 1. Bernoulli Bandit

A *$K$-armed Bernoulli bandit* is defined by $K$ unknown probabilities $\{\theta_i\}_{i=1}^K$, where each arm $i$ yields a reward of 1 with probability $\theta_i$ and 0 otherwise. At each time step $t=1,2,\dots$, the agent selects an arm $A_t\in\{1,\dots,K\}$ and observes reward  
$$
R_t \sim \mathrm{Bernoulli}(\theta_{A_t})\,.  
$$

---

## 2. Regret

The *cumulative regret* after $T$ steps measures the total difference in expected reward between always selecting the best possible action and following the actual actions chosen by the algorithm. Let the optimal arm be:

$$
i^* = \arg\max_i \theta_i
$$

Then the cumulative regret is defined as:

$$
R(T) = \sum_{t=1}^T \left( \theta_{i^*} - \theta_{A_t} \right)
$$

This can also be written by grouping terms by action:

$$
R(T) = \sum_{i=1}^K \Delta_i \cdot \mathbb{E}[N_i(T)]
$$

where:

- $\Delta_i = \theta_{i^*} - \theta_i$ is the suboptimality gap for arm $i$
- $N_i(T)$ is the number of times arm $i$ was played up to time $T$

This form helps analyze algorithms: we want suboptimal arms (with $\Delta_i > 0$) to be selected as few times as possible.

---

### Example

Suppose we have a 3-armed Bernoulli bandit problem with the following true reward probabilities:

- $\theta_1 = 0.5$
- $\theta_2 = 0.7$  ← *optimal arm*
- $\theta_3 = 0.6$

Then:

- $i^* = 2$
- $\Delta_1 = 0.7 - 0.5 = 0.2$
- $\Delta_3 = 0.7 - 0.6 = 0.1$

Assume that after $T=10$ steps, the algorithm chose the arms in this sequence:  
$$
A_{1:10} = [1, 1, 3, 2, 3, 2, 2, 3, 2, 2]
$$

So the count of arm pulls is:

- $N_1(10) = 2$
- $N_2(10) = 5$
- $N_3(10) = 3$

Then the cumulative regret is:

$$
R(10) = \Delta_1 \cdot N_1(10) + \Delta_3 \cdot N_3(10)  
= 0.2 \cdot 2 + 0.1 \cdot 3 = 0.4 + 0.3 = 0.7
$$

This means the algorithm has lost 0.7 expected reward compared to always playing arm 2.

---
In well-designed algorithms, we aim for **sublinear regret**, which means that the regret grows slower than linearly with respect to time $T$:

$$
\lim_{T \to \infty} \frac{R(T)}{T} = 0
$$

This implies that, as $T$ becomes very large, the **average regret per time step** goes to zero:

$$
\frac{R(T)}{T} \to 0
$$

In other words, the algorithm "learns" which arm is optimal and gradually stops choosing suboptimal arms. Over time, it behaves more and more like the optimal policy.

---

### Why Sublinear Regret Matters

- **Linear regret** (e.g., $R(T) = c \cdot T$) would mean the algorithm never learns and continues to make a constant number of mistakes per step.
- **Sublinear regret** ensures that the proportion of mistakes vanishes as $T$ grows — even if the total number of mistakes still increases slowly.
- **Logarithmic regret** (e.g., $R(T) = O(\log T)$) is particularly desirable and is the best possible in many settings.

---

### Regret Rates for Common Algorithms

| Algorithm              | Regret Bound                  | Notes                               |
|------------------------|-------------------------------|-------------------------------------|
| Greedy                 | $O(T)$                        | No exploration; linear regret       |
| $\varepsilon$-Greedy   | $O(\log T)$ (with decay)      | Needs proper $\varepsilon_t$ decay |
| UCB                    | $O(\log T)$                   | Optimism in face of uncertainty     |
| Thompson Sampling      | $O(\log T)$ (in expectation)  | Bayesian; empirically strong        |
| Random Policy          | $O(T)$                        | Pure exploration; worst case        |

---

Thus, sublinear regret is a formal guarantee that an algorithm is **converging toward optimal behavior**, even if it takes time to explore and learn.



## 3. Exploration Algorithms

### 3.1 Greedy

The simplest strategy always picks the arm with highest empirical mean:
$$
\hat\theta_i(t) = \frac{\sum_{s=1}^{t-1} \mathbb{I}\{A_s=i\} R_s}{\max(1,\,N_i(t-1))},
\quad
A_t = \arg\max_i \hat\theta_i(t)\,,
$$
where $N_i(t-1)$ is the count of times arm $i$ has been played before step $t$.  
*Drawback:* Zero exploration leads to poor performance if initial estimates are wrong.

### 3.2 $\varepsilon$-Greedy

This strategy chooses a random arm with probability $\varepsilon$ (exploration), and the greedy arm with probability $1 - \varepsilon$ (exploitation):

- With probability $1 - \varepsilon$:  
  $$
  A_t = \arg\max_i \hat\theta_i(t)
  $$
- With probability $\varepsilon$:  
  $$
  A_t \sim \text{Uniform}(\{1, 2, \dots, K\})
  $$

A common approach is to decay $\varepsilon$ over time (e.g., $\varepsilon_t = \frac{1}{t}$) to shift from exploration to exploitation.

---

### 3.3 Upper Confidence Bound (UCB)

Upper Confidence Bound methods choose actions by adding an uncertainty bonus to each estimated value. Formally, at each time step $t$:

1. **Estimate the action value**  
   $$
   Q_t(a) = \frac{1}{N_t(a)} \sum_{s=1}^{t-1} \mathbb{I}\{A_s = a\}\,R_s,
   $$  

   $$
   Q_t(a) \;=\; \frac{\text{Cumulative reward for action }a}{\text{Number of times action }a\text{ was selected}}
   $$

   where  
   - $A_s$ is the action taken at time $s$,  
   - $R_s$ is the reward observed at time $s$,  
   - $N_t(a) = \sum_{s=1}^{t-1} \mathbb{I}\{A_s = a\}$ is the number of times action $a$ was selected before time $t$.  


2. **Compute an uncertainty bonus** $U_t(a)$ so that with high probability  
   $$
   Q_t(a) + U_t(a) \ge q(a),
   $$  
   where $q(a)$ is the true expected value of action $a$.  A common choice via Hoeffding’s inequality is  
   $$
   U_t(a) = \sqrt{\frac{\ln t}{2\,N_t(a)}}\,.  
   $$  
   Intuitively:  
   - If $N_t(a)$ is small, $U_t(a)$ is large (high uncertainty)  
   - If $N_t(a)$ is large, $U_t(a)$ is small (value estimate is accurate)  

3. **Select the action** that maximizes the upper confidence bound:  
   $$
   A_t = \arg\max_{a \in \mathcal{A}} \Bigl[\,Q_t(a) + U_t(a)\Bigr].
   $$

---

#### Derivation via Hoeffding’s Inequality

Before applying Hoeffding’s inequality, note that:

1. **Bounded rewards**  
   In the bandit setting we assume each reward $R_s$ lies in $[0,1]$.  
2. **Independence**  
   Given an arm $a$, the sequence of rewards $\{R_s : A_s = a\}$ are i.i.d. with true mean  
   $$
   q(a) = \mathbb{E}[R_s \mid A_s = a].
   $$
3. **Empirical average**  
   We estimate $q(a)$ at time $t$ by  
   $$
   Q_t(a) = \frac{1}{N_t(a)} \sum_{s=1}^{t-1} \mathbb{I}\{A_s = a\}\,R_s,
   $$
   which is the sample mean of the rewards observed so far for arm $a$.

Because the $R_s$ are bounded in $[0,1]$ and i.i.d., Hoeffding’s inequality gives a high-probability bound on the deviation between the empirical mean $Q_t(a)$ and the true mean $q(a)$.

---



## 1. Hoeffding's Inequality (Bound on Deviation)

Suppose we have i.i.d. random variables $X_1, \dots, X_n$ such that $X_i \in [0,1]$ for all $i$. Let the true mean be $\mu = \mathbb{E}[X_i]$ and the empirical mean be:

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i.
$$

Then **Hoeffding’s inequality** states that, for any $u > 0$:

$$
\Pr\left( \bar{X}_n \le \mu - u \right) \le e^{-2n u^2}
\quad \text{and} \quad
\Pr\left( \bar{X}_n \ge \mu + u \right) \le e^{-2n u^2}.
$$

This tells us that the probability that the sample mean deviates from the true mean by more than $u$ decreases exponentially with the number of samples $n$.

---

## 2. Turning Hoeffding’s Inequality into a Confidence Interval

We can use the inequality to construct a **$(1 - \delta)$ confidence interval** for $\mu$. For a given failure probability $\delta$, we solve:

$$
e^{-2n u^2} = \delta \quad \Rightarrow \quad u = \sqrt{\frac{\ln(1/\delta)}{2n}}.
$$

Thus, with probability at least $1 - \delta$:

$$
\mu \in \left[ \bar{X}_n - \sqrt{\frac{\ln(1/\delta)}{2n}},\; \bar{X}_n + \sqrt{\frac{\ln(1/\delta)}{2n}} \right].
$$

This is a **distribution-free confidence interval**, meaning it does not require knowledge of the distribution of $X_i$, only that $X_i \in [0,1]$.

---

## 3. Applying to the UCB Algorithm

In the UCB algorithm, we estimate the expected reward $q(a)$ of arm $a$ by its empirical mean $Q_t(a)$, and add an exploration bonus $U_t(a)$:

$$
\text{UCB}_t(a) = Q_t(a) + U_t(a).
$$

We want to ensure that, with high probability:

$$
q(a) \le Q_t(a) + U_t(a).
$$

So we set the failure probability for each time step to $\delta_t = 1/t$, and solve:

$$
e^{-2 N_t(a) \cdot u^2} = \frac{1}{t}
\quad \Rightarrow \quad
u = \sqrt{ \frac{\ln t}{2 N_t(a)} }.
$$

Thus, with probability at least $1 - \tfrac{1}{t}$:

$$
q(a) \le Q_t(a) + \sqrt{ \frac{\ln t}{2 N_t(a)} }.
$$

This motivates setting the **exploration bonus** in UCB as:

$$
U_t(a) = \sqrt{ \frac{\ln t}{2 N_t(a)} }.
$$

---

## 4. Relationship Between $1 - \delta$ and $1 - \alpha$

The expressions $1 - \delta$ and $1 - \alpha$ both represent **confidence levels**, though used in slightly different contexts:

- In **machine learning and bandit literature**, $\delta$ typically represents the total allowed failure probability (i.e., the chance that a confidence interval does **not** contain the true value). So $1 - \delta$ is the confidence level.
- In **classical statistics**, $\alpha$ denotes the **significance level**, especially in hypothesis testing. Again, $1 - \alpha$ is the confidence level.

So conceptually, $1 - \delta$ and $1 - \alpha$ mean the same thing: the probability that your interval or estimate **correctly** captures the truth.

---

## 5. Summary

- Hoeffding’s inequality gives a high-probability bound on how far the sample mean can deviate from the true mean.
- By inverting this inequality, we can derive a confidence interval for the true mean.
- The width of this interval becomes the **exploration bonus** in the UCB algorithm.
- The expressions $1 - \delta$ and $1 - \alpha$ both represent confidence levels, and are interchangeable depending on the context (machine learning vs statistics).

---

#### UCB1 Algorithm (Auer et al., 2002)

Putting it all together yields the classic UCB1 rule:

$$
A_t = \arg\max_{a\in\mathcal{A}} \Bigl[\,Q_t(a) + \sqrt{\tfrac{2\ln t}{N_t(a)}}\Bigr].
$$

> **Theorem (Auer et al., 2002):**  
> UCB1 achieves an expected cumulative regret bound  
> $$
> \mathbb{E}[R(T)] \le \sum_{a:\,\Delta_a>0} \!\Bigl(\tfrac{8\ln T}{\Delta_a} + O(1)\Bigr),
> $$  
> where $\Delta_a = q(a^*) - q(a)$ is the gap to the optimal action.

---

In summary, UCB balances exploitation (high $Q_t(a)$) and exploration (high $U_t(a)$) in a principled, theoretically grounded way.  

---

### 3.4 Policy Gradient Exploration

Rather than adding bonuses to value estimates, policy gradient methods maintain a stochastic policy $\pi_\phi(a)$ and adjust the parameters $\phi$ to maximize expected return:

$$
\nabla_\phi J(\phi) = \mathbb{E}_{a \sim \pi_\phi} \bigl[\,G_t \,\nabla_\phi \log \pi_\phi(a)\bigr],
$$

where $G_t$ is the return (e.g., cumulative future rewards). Exploration arises naturally from the policy’s stochasticity.

#### Parameter Update (REINFORCE)

In practice we use a Monte Carlo estimate of the gradient and perform stochastic gradient ascent with learning rate $\alpha$:

1. Sample action $A_t \sim \pi_{\phi_t}(\cdot)$ and observe return $G_t$.
2. Compute the gradient estimate:
   $$
   \hat g_t \;=\; G_t \,\nabla_\phi \log \pi_{\phi_t}(A_t).
   $$
3. Update the parameters:
   $$
   \phi_{t+1} = \phi_t + \alpha\,\hat g_t
   \;=\;
   \phi_t + \alpha\,G_t\,\nabla_\phi \log \pi_{\phi_t}(A_t).
   $$

Optionally, subtract a baseline $b$ (e.g., an average return) to reduce variance:

$$
\phi_{t+1} = \phi_t + \alpha\,(G_t - b)\,\nabla_\phi \log \pi_{\phi_t}(A_t).
$$

---

### 3.5 Thompson Sampling

Thompson Sampling (Bayesian approach) maintains a posterior over the reward distribution for each arm. For Bernoulli rewards, we can use a Beta distribution:

1. For each arm $i$, sample  
   $$
   \tilde\theta_i \sim \text{Beta}(\alpha_i, \beta_i)
   $$
2. Select the arm with the highest sampled value:  
   $$
   A_t = \arg\max_i \tilde\theta_i
   $$
3. Update $(\alpha_i, \beta_i)$ based on the observed reward.

This method balances exploration and exploitation via posterior sampling.

---
