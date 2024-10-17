import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
# Heston Model Simulation and Plot
########################################################################################################################

def HestonModelSim(S0, v0, rho, kappa, theta, sigma, r, T, n, M):
    """
    Inputs:
     - S0, v0: initial parameters for asset and variance
     - rho   : correlation between asset returns and variance
     - kappa : rate of mean reversion in variance process
     - theta : long-term mean of variance process
     - sigma : vol of vol / volatility of variance process
     - r     : risk free rate
     - T     : expiry time of simulation
     - n     : number of time steps
     - M     : number of simulations

    Outputs:
    - asset prices over time (numpy array)
    - variance over time (numpy array)
    """
    # Define time interval
    dt = T / n
    # Arrays for storing prices and variances
    S = np.full(shape=(n + 1, M), fill_value=S0)
    v = np.full(shape=(n + 1, M), fill_value=v0)
    # Generate correlated brownian motions
    Z_v = np.random.normal(0, 1, size=(n, M))
    Z_s = rho * Z_v + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, size=(n, M))

    for i in range(1, n + 1):
        S[i] = S[i - 1] * np.exp((r - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z_s[i - 1,:])
        v[i] = np.maximum(v[i - 1] + kappa * (theta - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * Z_v[i - 1,:], 0)

    return S, v


def PlotHestonModel(S, v, T, n):
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    time = np.linspace(0, T, n + 1)

    ax1.plot(time, S)
    ax1.set_title('Heston Model Asset Prices', fontsize=16)
    ax1.set_xlabel('Time', fontsize=16)
    ax1.set_ylabel('Asset Prices', fontsize=16)

    ax2.plot(time, v)
    ax2.set_title('Heston Model Variance Process', fontsize=16)
    ax2.set_xlabel('Time', fontsize=16)
    ax2.set_ylabel('Variance', fontsize=16)

    plt.show()


def PriceOption(S, M, K, call_or_put='c', knockin=None, knockout=None):
    # K= strike, knockin= barrier level for knockin option, knockout= barrier level for knockout option
    if knockin and knockout:
        raise Exception("Can't have 2 barriers!")

    cp = 1 if call_or_put == 'c' else -1
    paths = S

    # Make a copy for plotting
    paths_copy = np.copy(paths)
    barrier_crossed = None

    if knockout and knockout > S0: # Up and Out
        barrier_crossed = np.any(paths > knockout, axis=0)      # checked for put (call is off)
        paths[:, barrier_crossed] = 0
    elif knockout and knockout < S0: # Down and Out
        barrier_crossed = np.any(paths < knockout, axis=0)      # checked for call (put is off)
        paths[:, barrier_crossed] = 0
    elif knockin and knockin > S0: # Up and In
        barrier_crossed = np.any(paths > knockin, axis=0)       # checked for put (0.01 diff) (call is off)
        paths[:, ~barrier_crossed] = 0
    elif knockin and knockin < S0: # Down and In
        barrier_crossed = np.any(paths < knockin, axis=0)       # checked for call (put is off)
        paths[:, ~barrier_crossed] = 0

    payoff = np.maximum(0, cp * (paths[-1] - K) [paths[-1] != 0]) # use [paths[-1] != 0] for put (cp=-1)
    option_price = np.exp(-r * T) *  np.sum(payoff) / M

    return option_price


# Example
S0 = 100.0             # asset price
T = 1.0                # time in years
r = 0.02               # risk-free rate
n = 360                # number of time steps in simulation
M = 100                # number of simulations
# Heston dependent parameters
kappa = 2               # rate of mean reversion of variance under risk-neutral dynamics
theta = 0.05            # long-term mean of variance under risk-neutral dynamics
v0 = 0.08               # initial variance under risk-neutral dynamics
rho = 0.5               # correlation between returns and variances under risk-neutral dynamics
sigma = 0.2             # volatility of volatility


S,v = HestonModelSim(S0, v0, rho, kappa, theta, sigma, r, T, n, M)
PlotHestonModel(S, v, T, n)

Price = PriceOption(S, M, K=110, call_or_put='c')
print("Estimated European Option Price (HESTON):", Price)
Price_knockout = value = PriceOption(S, M, K=110, call_or_put='c', knockout= 120)
print("Estimated Barrier Option (knockout) Price (HESTON):", Price_knockout)