# Heston Stochastic Volatility Model

### Abstract of the article:
---
The present text shows the performance and provides a complete calibration of the Heston stochastic volatility model (1993). The model proposed by Heston is more general than the Black-Scholes model (1973) and includes it as a special case. We applied a weighted least squared error fit, choosing the deterministic trust region reflective optimization method in combination with the stochastic simulated annealing algorithm to calibrate the model. Then, we compared the efficiency of the Heston and Black-Scholes models on real data. <\br>

Keywords: Options pricing, Martingale method, Heston model, Black-Scholes model, Trust Region Reflective, Simulated Annealing, Least-Squares Monte Carlo method (LSM) <\br>


To calibrate the parameters of the Heston model I proposed and implemented a methodology which applies both deterministic (trust region reflective) and stochastic optimization methods (dual simulated annealing) and leverages the closed-form solution the model provides for European options. The project has resulted in one article, which has recently passed the peer-review process and is set to be published in the proceedings of the conference I presented it, as well as a more comprehensive thesis that expands on the article by including additional theory, rigorous proofs, and various implementations of models, in topics such as market forecasting, and numerical methods such as Monte Carlo in derivatives pricing (see Special_Topic_Latex repository). <\br>

I used data from https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022 for the S&P 500 American options to calibrate the model.

An introduction to the Heston SV model:
---
One of the most popular stochastic volatility models is the so-called **Heston model**, which is an extension of the **Black-Scholes-Merton model**. It is defined by the following system of stochastic differential equations, which describe the movement of an asset's price when both the price and its volatility follow random stochastic processes based on **Brownian motion**:
  
$$
dS(t) = \mu S(t) dt + \sqrt{v(t)} S(t)\, dW_S(t)  
$$

$$
dv(t) = \kappa(\theta - v(t)) dt + \sigma \sqrt{v(t)}\, dW_v(t)  
$$

where the parameters are:

- $\mu$: average rate of return for the underlying asset price.
- $\theta$: the limit of the expected value of $v(t)$ as $t \to \infty$, i.e. $\lim_{t\to\infty}\mathbb{E}[v(t)]=\theta$.
- $\kappa$: rate of mean reversion of $v(t)$ towards $\theta$.
- $\sigma$: volatility of volatility $v(t)$.
- $W_S(t)$: Brownian motion of the underlying asset price.
- $W_v(t)$: Brownian motion of the asset's price volatility.
- $\rho$: correlation between $W_S(t)$ and $W_v(t)$, i.e., $dW_S(t)\, dW_v(t)=\rho dt$.

Additionally, when simulating the above stochastic differential equations, we also need the parameter $v_0 = dv(0)$, which denotes the initial volatility.

Under the **risk-neutral** probability measure $\widehat{\mathbb{P}}$ and applying **Girsanov's theorem**, it is shown that the equations become:

$$
dS(t) = r S(t) dt + \sqrt{v(t)} S(t)\, d\widehat{W}_S(t)
$$

$$
dv(t) = \widehat{\kappa} (\widehat{\theta} - v(t)) dt + \sigma \sqrt{v(t)}\, d\widehat{W}_v(t)
$$

with

$$
d\widehat{W}_S(t) = dW_S(t) + \alpha_S \,dt \quad\text{where}\quad \alpha_S=\frac{\mu-r}{\sqrt{v(t)}}
$$

$$
d\widehat{W}_v(t) = dW_v(t) + \alpha_v \,dt \quad\text{where}\quad \alpha_v=\frac{\lambda}{\sigma}\sqrt{v(t)}
$$

and

$$
\widehat{\kappa} = \kappa + \lambda\,, \quad \widehat{\theta} = \frac{\kappa \theta}{\kappa + \lambda} \,, \quad \widehat{\rho}= \rho
$$

where $$\lambda$$ is the **risk premium** parameter, which can be estimated using expected returns from positions in options hedged against the risk of changes in the underlying asset.

One of the advantages of this model is that it provides an analytical formula for pricing simple European options using the characteristic function. However, for more complex options, this is not the case, and to price them, we will need to use numerical methods, such as the **Monte Carlo** method. In practice, we estimate the parameters of the **Heston model** (under the **risk-neutral** measure) based on observed option prices in the market and then apply the model, with the same parameters, to calculate the value of simple or exotic options, all for the same underlying asset.

Let us first see how we can discretize the stochastic differential equations using the **Euler method**.

We integrate the stochastic differential equations from $t$ to $t+dt$ and approximate them using the left-point rule.

Following this process, we have,

$$
v(t+dt) = v(t) + \int_{t}^{t+dt}\kappa(\theta - v(u))du + \int_{t}^{t+dt}\sigma \sqrt{v(u)}\, dW_v(u)
$$

$$
\approx v(t) + \kappa(\theta - v(t))dt + \sigma \sqrt{v(t)}(W_v(t+dt) - W_v(t))
$$

$$
= v(t) + \kappa(\theta - v(t))dt + \sigma \sqrt{v(t) dt}Z_v
$$

where $Z_v$ is the standard normal distribution. It becomes apparent that the above discrete process for $v(t)$ can become negative with non-zero probability, making the calculation of $\sqrt{v(t)}$ impossible. Therefore, to avoid negative values, we replace $v(t)$ with $v^+(t)=\max(0,v(t))$ (the volatility $v(t)$ here is a **square root process**).

Similarly, 

$$
S(t+dt) = S(t) + \int_{t}^{t+dt} \mu S(u)\, du + \int_{t}^{t+dt} \sqrt{v(u)} S(u)\, dW_S(u)
$$

$$
\approx S(t) + \mu S(t) dt + \sqrt{v(t)} S(t)(W_S(t+dt) - W_S(t))
$$

$$
= S(t) + \mu S(t) dt + \sqrt{v(t) dt} S(t) Z_S
$$

where $Z_S$ is the standard normal distribution correlated with $Z_v$ by $\rho$.

Following Cholensky's decomposition, to construct $Z_S$ and $Z_v$, we first generate two independent $Z_1$ and $Z_2$ following $\mathcal{N}(0,1)$ and then set $Z_v = Z_1$ and $Z_S = \rho Z_1 + \sqrt{1-\rho^2}Z_2$.

Solving the equation for the price of the underlying asset, leads to the solution:

$$
S(t+dt) = S(t)\exp \left[ \int_{t}^{t+dt}\left(\mu-\frac{1}{2}v(u)\right)\, du + \int_{t}^{t+dt}\sqrt{v(u)}\, dW(u)\right]
$$

and applying the **Euler discretization method**, we get:

$$
S(t+dt) = S(t)\exp\\left[ \left(\mu-\frac{1}{2}v(t)\right)\, dt + \sqrt{v(t) dt}Z_S\right]
$$
