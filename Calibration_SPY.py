import numpy as np
import pandas as pd
# Set display options
pd.set_option('display.max_rows', None)    # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

from dataclasses import dataclass
from abc import ABC, abstractmethod
import QuantLib as ql

from scipy.integrate import quad
from scipy import optimize
from scipy.stats import norm
from statsmodels.api import OLS, add_constant

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

'''
Some basic data analysis in order to integrate the dataset in python.
'''

"""Load data"""
np.random.seed(123)
atm_call_data = pd.read_csv('atm_call_options.csv', low_memory=False)
market_datas = pd.read_csv('market_datas.csv', low_memory=False)
PriceSurface = pd.read_csv('PriceSurface.csv', low_memory=False) # Needed for the plots

class StochasticProcess(ABC):
    """Represents a Stochastic process"""

    @abstractmethod
    def simulate(self):
        ...

@dataclass
class GeometricBrownianMotion(StochasticProcess):

    mu: float
    sigma: float

    def simulate(
        self, s0: float, T: int, N: int, M: int, v0: float = None
    ) -> pd.DataFrame:  # M = number of paths, N = number of discretization points

        dt = T / N
        S = np.exp(
            (self.mu - self.sigma ** 2 / 2) * dt
            + self.sigma * np.sqrt(dt) * np.random.normal(0, 1, size=(N, M))
        )
        S = np.vstack([np.ones(M), S])
        S = s0 * S.cumprod(axis=0)

        return S


@dataclass
class HestonProcess(StochasticProcess):

    mu: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float

    def simulate(
        self, s0: float, T: int, N: int, M: int
    ) -> pd.DataFrame:  # M = number of paths, N = number of discretization points

        # Define time interval
        dt = T / N
        # Arrays for storing prices and variances
        S = np.full(shape=(N + 1, M), fill_value=s0)
        v = np.full(shape=(N + 1, M), fill_value=self.v0)
        # Generate correlated brownian motions
        Z_v = np.random.normal(0, 1, size=(N, M))
        Z_s = self.rho * Z_v + np.sqrt(1 - self.rho ** 2) * np.random.normal(0, 1, size=(N, M))

        for i in range(1, N + 1):
            S[i] = S[i - 1] * np.exp((self.mu - 0.5 * v[i - 1]) * dt + np.sqrt(v[i - 1] * dt) * Z_s[i - 1, :])
            v[i] = np.maximum(
                v[i - 1] + self.kappa * (self.theta - v[i - 1]) * dt + self.sigma * np.sqrt(v[i - 1] * dt) * Z_v[i - 1, :], 0)

        return S


@dataclass
class Option:
    """
    Representation of an option derivative
    """

    s0: float
    T: int
    K: int
    v0: float = None
    call: bool = True

    def payoff(self, s: np.ndarray) -> np.ndarray:
        payoff = np.maximum(0, s - self.K) if self.call else np.maximum(0, self.K - s)
        return payoff


def PriceOption(
        option: Option, process: StochasticProcess, N: int, M: int,
) -> float:     # M = number of paths, N = number of discretization points
    """
    Given an option and a process followed by the underlying asset, calculate the price estimator with classic monte carlo
    """

    # For European Option:
    s = process.simulate(s0=option.s0, T=option.T, N=N, M=M)                                              # use [s[-1] != 0] for pricing barrier options
    st = s[-1]
    payoffs = option.payoff(s=st)

    discount = np.exp(-process.mu * option.T)
    price = np.mean(payoffs) * discount

    return np.round(price, 2)

def PriceOption_LS(option: Option, process: StochasticProcess, N: int, M: int, k=3) -> float:
    """
    Price an American option using the Longstaff-Schwartz method.

    Parameters:
    option : Option
        An instance of the Option dataclass.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    N : int
        Number of time steps.
    M : int
        Number of simulated paths.

    Returns:
    float
        Estimated price of the American option.
    """
    dt = option.T / N
    discount_factor = np.exp(-process.mu * dt) # process.mu=r

    paths = process.simulate(s0=option.s0, T=option.T, N=N, M=M)
    payoffs = option.payoff(paths[-1])

    for t in range(N - 1, 0, -1):
        X = paths[t, :]
        Y = discount_factor * payoffs
        # Use in-the-money paths for regression
        in_the_money = option.payoff(X) > 0
        X_in_the_money = X[in_the_money]
        Y_in_the_money = Y[in_the_money]

        if len(X_in_the_money) > 0:  # Ensure there are points for regression
            A = basis_functions(X_in_the_money, k)
            beta = np.linalg.lstsq(A, Y_in_the_money, rcond=None)[0]
            continuation_value = np.dot(basis_functions(X, k), beta)
        else:
            continuation_value = np.zeros(X.shape)

        exercise_value = option.payoff(X)
        # Update only the in-the-money paths
        payoffs[in_the_money] = np.where(exercise_value[in_the_money] > continuation_value[in_the_money], exercise_value[in_the_money], discount_factor * payoffs[in_the_money])

    option_price = np.mean(discount_factor * payoffs)

    '''
    for t in range(N - 1, 0, -1):
        X = paths[t, :]
        Y = discount_factor * payoffs
        A = basis_functions(X, k)
        beta = np.linalg.lstsq(A, Y, rcond=None)[0]
        continuation_value = np.dot(A, beta)

        exercise_value = option.payoff(X)
        payoffs = np.where(exercise_value > continuation_value, exercise_value, discount_factor * payoffs)

    # Discount the payoff back to present value
    option_price = np.mean(discount_factor * payoffs)
    '''

    return option_price

def basis_functions(X, k):
    """
    Generate basis functions for the regression.

    Parameters:
    X : array
        Asset prices
    k : int
        Number of basis functions
    Returns:
    numpy.ndarray
        Basis functions evaluated at X
    """
    if k == 1:
        A = np.vstack([np.ones(X.shape), 1 - X]).T
    elif k == 2:
        A = np.vstack([np.ones(X.shape), 1 - X, 0.5 * (2 - 4 * X + X**2)]).T
    elif k == 3:
        A = np.vstack([np.ones(X.shape), 1 - X, 0.5 * (2 - 4 * X + X**2),
                       1/6 * (6 - 18 * X + 9 * X**2 - X**3)]).T
    elif k == 4:
        A = np.vstack([np.ones(X.shape), 1 - X, 0.5 * (2 - 4 * X + X**2),
                       1/6 * (6 - 18 * X + 9 * X**2 - X**3),
                       1/24 * (24 - 96 * X + 72 * X**2 - 16 * X**3 + X**4)]).T
    elif k == 5:
        A = np.vstack([np.ones(X.shape), 1 - X, 0.5 * (2 - 4 * X + X**2),
                       1/6 * (6 - 18 * X + 9 * X**2 - X**3),
                       1/24 * (24 - 96 * X + 72 * X**2 - 16 * X**3 + X**4),
                       1/120 * (120 - 600 * X + 600 * X**2 - 200 * X**3 + 25 * X**4 - X**5)]).T
    else:
        raise ValueError('Too many basis functions requested')
    return A

def longstaff_schwartz_american_option(option: Option, r: float, sigma: float, N: int, M: int, k=3) -> float:
    """
    Price an American option using the Longstaff-Schwartz method.

    Parameters:
    option : Option
        An instance of the Option dataclass.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    N : int
        Number of time steps.
    M : int
        Number of simulated paths.

    Returns:
    float
        Estimated price of the American option.
    """
    dt = option.T / N
    discount_factor = np.exp(-r * dt)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(option.s0))
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(r)), ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(sigma)), ql.Actual365Fixed()))
    process = ql.GeneralizedBlackScholesProcess(spot_handle, flat_ts, flat_ts, vol_ts)

    rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(N, ql.UniformRandomGenerator()))
    seq = ql.GaussianPathGenerator(process, option.T, N, rng, False)

    paths = np.zeros((M, N + 1))
    for i in range(M):
        sample_path = seq.next()
        values = sample_path.value()
        path = np.array([values[j] for j in range(len(values))])
        paths[i, :] = path

    payoffs = option.payoff(paths[:, -1])

    for t in range(N - 1, 0, -1):
        X = paths[:, t]
        Y = discount_factor * payoffs
        # Use in-the-money paths for regression
        in_the_money = option.payoff(X) > 0
        X_in_the_money = X[in_the_money]
        Y_in_the_money = Y[in_the_money]

        if len(X_in_the_money) > 0:  # Ensure there are points for regression
            A = basis_functions(X_in_the_money, k)
            beta = np.linalg.lstsq(A, Y_in_the_money, rcond=None)[0]
            continuation_value = np.dot(basis_functions(X, k), beta)
        else:
            continuation_value = np.zeros(X.shape)

        exercise_value = option.payoff(X)
        # Update only the in-the-money paths
        payoffs[in_the_money] = np.where(exercise_value[in_the_money] > continuation_value[in_the_money],
                                         exercise_value[in_the_money], discount_factor * payoffs[in_the_money])

    option_price = np.mean(discount_factor * payoffs)

    return option_price


def longstaff_schwartz_american_option_reg(option: Option, r: float, sigma: float, N: int, M: int, k=3) -> float:
    """
    Price an American option using the Longstaff-Schwartz method.

    Parameters:
    option : Option
        An instance of the Option dataclass.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    N : int
        Number of time steps.
    M : int
        Number of simulated paths.

    Returns:
    float
        Estimated price of the American option.
    """
    # Set up the parameters for the process
    dt = option.T / N
    discount_factor = np.exp(-r * dt)

    # Setup QuantLib objects
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(option.s0))
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(r)), ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(sigma)), ql.Actual365Fixed()))
    process = ql.GeneralizedBlackScholesProcess(spot_handle, flat_ts, flat_ts, vol_ts)

    # Simulate paths
    rng = ql.GaussianRandomSequenceGenerator(
            ql.UniformRandomSequenceGenerator(
                N, ql.UniformRandomGenerator()))
    seq = ql.GaussianPathGenerator(process, option.T, N, rng, False)

    # Generate paths
    paths = np.zeros((M, N + 1))
    for i in range(M):
        sample_path = seq.next()
        values = sample_path.value()
        path = np.array([values[j] for j in range(len(values))])
        paths[i, :] = path

    # Calculate the payoff at each node
    payoffs = option.payoff(paths[:, -1])

    # Perform regression to estimate continuation value
    for t in range(N - 1, 0, -1):
        X = paths[:, t]
        Y = discount_factor * payoffs
        A = basis_functions(X, k)
        beta = np.linalg.lstsq(A, Y, rcond=None)[0]
        continuation_value = np.dot(A, beta)

        exercise_value = option.payoff(X)
        payoffs = np.where(exercise_value > continuation_value, exercise_value, discount_factor * payoffs)

    # Discount the payoff back to present value
    option_price = np.mean(discount_factor * payoffs)

    return option_price

def black_scholes_merton(r, sigma, option: Option):
    """
    Calculate the price of vanilla options using BSM formula
    """
    d1 = (np.log(option.s0 / option.K) + (r + sigma**2 / 2) * option.T) / (sigma * np.sqrt(option.T))
    d2 = d1 - sigma * np.sqrt(option.T)

    price = option.s0 * norm.cdf(d1) - option.K * np.exp(-r * option.T) * norm.cdf(d2)
    price = price if option.call else price - option.s0 + option.K * np.exp(-r * option.T)

    return np.round(price, 2)

def price_european_call_heston(option: Option, r, kappa, theta, sigma, rho, v0):
    # Setup Heston process
    spot = option.s0
    K = option.K
    T = option.T

    # Convert T to QuantLib Date object
    today = ql.Date().todaysDate()
    maturity_date = today + int(T * 365)  # Convert years to days and create a Date object

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(r)), ql.Actual360()))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(0)), ql.Actual360()))
    heston_process = ql.HestonProcess(flat_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)

    # Define the option
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, exercise)

    # Set up the Heston model and engine
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)
    european_option.setPricingEngine(engine)

    # Calculate the option price
    price = european_option.NPV()
    return price

# First way to price call from Heston
def HestonCallQuad(kappa, theta, sigma, rho, v0, r, option: Option):
    """
    Computes the price of a European option using the Heston model, based on the Option class.
    """
    s0, T, K = option.s0, option.T, option.K

    call = s0 * HestonP(kappa, theta, sigma, rho, v0, r, T, s0, K, 1) \
           - K * np.exp(-r * T) * HestonP(kappa, theta, sigma, rho, v0, r, T, s0, K, 2)

    if not option.call:  # Convert to put option price if required using put-call parity
        call = call - s0 + K * np.exp(-r * T)

    return call


def HestonP(kappa, theta, sigma, rho, v0, r, T, s0, K, option_type):
    """Computes the Heston characteristic function using numerical integration."""
    integral_result = quad(HestonPIntegrand, 0, 100, args=(kappa, theta, sigma, rho, v0, r, T, s0, K, option_type))[0]
    return 0.5 + (1 / np.pi) * integral_result


def HestonPIntegrand(phi, kappa, theta, sigma, rho, v0, r, T, s0, K, option_type):
    """Evaluates the integrand for the Heston characteristic function."""
    return np.real(np.exp(-1j * phi * np.log(K)) *
                   HestonCharfun(phi, kappa, theta, sigma, rho, v0, r, T, s0, option_type) / (1j * phi))


def HestonCharfun(phi, kappa, theta, sigma, rho, v0, r, T, s0, option_type):
    """Computes the Heston characteristic function."""
    if option_type == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    x = np.log(s0)
    d = np.sqrt((rho * sigma * phi * 1j - b) ** 2 - sigma ** 2 * (2 * u * phi * 1j - phi ** 2))
    g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)

    C = r * phi * 1j * T + (a / sigma ** 2) * ((b - rho * sigma * phi * 1j + d) * T -
                                               2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
    D = (b - rho * sigma * phi * 1j + d) / sigma ** 2 * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

    return np.exp(C + D * v0 + 1j * phi * x)

# Second way to price call option with Heston
def heston_charfunc(phi, kappa, theta, sigma, rho, lambd, r, option: Option):

    # constants
    a = kappa*theta
    b = kappa+lambd

    # common terms w.r.t phi
    rspi = rho*sigma*phi*1j

    # define d parameter given phi and b
    d = np.sqrt( (rho*sigma*phi*1j - b)**2 + (phi*1j+phi**2)*sigma**2 )

    # define g parameter given phi, b and d
    g = (b-rspi+d)/(b-rspi-d)

    # calculate characteristic function by components
    exp1 = np.exp(r*phi*1j*option.T)
    term2 = option.s0**(phi*1j) * ( (1-g*np.exp(d*option.T))/(1-g) )**(-2*a/sigma**2)
    exp2 = np.exp(a*option.T*(b-rspi+d)/sigma**2 + option.v0*(b-rspi+d)*( (1-np.exp(d*option.T))/(1-g*np.exp(d*option.T)) )/sigma**2)

    return exp1*term2*exp2

def integrand(phi, kappa, theta, sigma, rho, lambd, r, option: Option):
    S0 = option.s0
    v0 = option.v0
    tau = option.T
    args = (kappa, theta, sigma, rho, lambd, r, option)
    numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - option.K*heston_charfunc(phi,*args)
    denominator = 1j*phi*option.K**(1j*phi)
    return numerator/denominator

def heston_price_rec(kappa, theta, sigma, rho, lambd, r, option: Option):
    S0 = option.s0
    v0 = option.v0
    tau = option.T
    args = (kappa, theta, sigma, rho, lambd, r, option)

    P, umax, N = 0, 100, 10000
    dphi=umax/N #dphi is width

    for i in range(1,N):
        # rectangular integration
        phi = dphi * (2*i + 1)/2 # midpoint to calculate height
        numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - K * heston_charfunc(phi,*args)
        denominator = 1j*phi*option.K**(1j*phi)

        P += dphi * numerator/denominator

    return np.real((S0 - K*np.exp(-r*tau))/2 + P/np.pi)

def analytical_heston_price(kappa, theta, sigma, rho, lambd, r, option: Option):
    S0 = option.s0
    v0 = option.v0
    tau = option.T
    args = (kappa, theta, sigma, rho, lambd, r, option)

    real_integral, err = np.real( quad(integrand, 0, 100, args=args) )

    return (S0 - option.K*np.exp(-r*tau))/2 + real_integral/np.pi

# Third way to price call option with Heston
def HestonCallFFT(option: Option, r, kappa, theta, sigma, rho, v0):
    """
    Computes the European call option price using the Heston model and FFT.

    Parameters:
    kappa  - Rate of reversion
    theta  - Long-run variance
    sigma  - Volatility of variance
    rho    - Correlation
    r      - Risk-free rate
    v0     - Initial variance
    s0     - Initial asset price
    strike - Strike price
    T      - Time to maturity

    Returns:
    Call option price
    """
    #start_time = time.time()  # Start timer for the whole function

    x0 = np.log(option.s0)  # Initial log price
    alpha = 1.25
    N = 4096
    c = 600
    eta = c / N
    b = np.pi / eta
    u = np.arange(0, N) * eta
    lambd = 2 * b / N
    position = int(np.round((np.log(option.K) + b) / lambd))  # Position of call value in FFT

    # Complex numbers for characteristic function
    v = u - (alpha + 1) * 1j
    zeta = -0.5 * (v ** 2 + 1j * v)
    gamma = kappa - rho * sigma * v * 1j
    PHI = np.sqrt(gamma ** 2 - 2 * sigma ** 2 * zeta)
    A = 1j * v * (x0 + r * option.T)
    B = v0 * ((2 * zeta * (1 - np.exp(-PHI * option.T))) /
              (2 * PHI - (PHI - gamma) * (1 - np.exp(-PHI * option.T))))
    C = -(kappa * theta / sigma ** 2) * (
            2 * np.log((2 * PHI - (PHI - gamma) * (1 - np.exp(-PHI * option.T))) / (2 * PHI)) +
            (PHI - gamma) * option.T
    )

    # Characteristic function
    char_func = np.exp(A + B + C)

    # Modified characteristic function
    modified_char_func = (char_func * np.exp(-r * option.T) /
                          (alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u))

    # Simpson weights for integration
    simpson_w = (1 / 3) * (3 + (-1) ** np.arange(1, N + 1) - np.append(1, np.zeros(N - 1)))

    # FFT computation
    fft_func = np.exp(1j * b * u) * modified_char_func * eta * simpson_w
    payoff = np.real(np.fft.fft(fft_func))

    # Extract call value
    call_value_m = np.exp(-np.log(option.K) * alpha) * payoff / np.pi
    call_value = call_value_m[position]

    #print(f"Total execution time: {time.time() - start_time:.4f} seconds")
    return call_value


def calibrate(init_val, market_datas):
    def error(x):
        kappa, theta, sigma, rho, v0, l= x
        print('x=', kappa, theta, sigma, rho, v0, l)
        result = 0.0
        avg_rel_err = 0.0
        for i in range(0, len(market_datas)):
            s0, k, market_price, r, T, ivol = market_datas.iloc[i]
            #print(s0, k, market_price, r, T)

            heston = HestonProcess(mu=r, kappa=kappa + l, theta= (kappa * theta) / (kappa + l), sigma=sigma, rho=rho, v0=v0)
            opt = Option(s0=s0, v0=v0, T=T, K=k, call=True)

            heston_price = PriceOption_LS(option=opt, process=heston, N=360, M=2000)
            result += (heston_price - market_price) ** 2
            rel_err = (heston_price / market_price - 1)
            avg_rel_err += abs(rel_err)

        avg_rel_err = avg_rel_err * 100 / len(market_datas)
        if avg_rel_err < 1.5:
            print('-- Resulting average abs error: ', avg_rel_err)
            print('-- Resulting squared error: ', result, '\n')

        return result

    # Define bounds for each parameter
    bounds = ([0.001, 0.001, 0.001, -1, 0.001, -2], [8, 2, 5, 1, 2, 2])
    opt = optimize.least_squares(error, init_val, bounds=bounds, method='trf')
    return opt

def calibrate_SA(init_val, market_datas):
    def error(x):
        kappa, theta, sigma, rho, v0, l = x
        print('x=', kappa, theta, sigma, rho, v0, l)

        # Check if parameters are valid
        if kappa + l <= 0 or (kappa * theta) / (kappa + l) <= 0 or sigma <= 0 or v0 < 0 or not (-1 <= rho <= 1):
            return float('inf')  # Penalize invalid parameter sets

        result = 0.0
        avg_rel_err = 0.0
        for i in range(0, len(market_datas)):
            s0, k, market_price, r, T, ivol = market_datas.iloc[i]

            #heston = HestonProcess(mu=r, kappa=kappa + l, theta=(kappa * theta) / (kappa + l), sigma=sigma, rho=rho, v0=v0)
            opt = Option(s0=s0, v0=v0, T=T, K=k, call=True)

            heston_price = price_european_call_heston(option=opt, r=r, kappa=kappa + l, theta=(kappa * theta) / (kappa + l), sigma=sigma, rho=rho, v0=v0) # PriceOption_LS(option=opt, process=heston, N=500, M=5000)
            result += (heston_price - market_price) ** 2
            rel_err = (heston_price / market_price - 1)
            avg_rel_err += abs(rel_err)

        avg_rel_err = avg_rel_err * 100 / len(market_datas)
        if avg_rel_err < 0.5: #or result < 230:
            print('-- Resulting average abs error: ', avg_rel_err)
            print('-- Resulting squared error: ', result, '\n')

        return result

    # Define bounds for each parameter
    bounds = [(0.001, 8), (0.001, 2), (0.001, 5), (-0.99, 0.99), (0.001, 2), (-2, 2)]

    # Perform simulated annealing
    result = optimize.dual_annealing(error, bounds=bounds, x0=init_val)

    return result

def calibrate_DE(init_val, market_datas):
    def error(x):
        kappa, theta, sigma, rho, v0, l = x
        print('x=', kappa, theta, sigma, rho, v0, l)
        result = 0.0
        avg_rel_err = 0.0
        for i in range(len(market_datas)):
            s0, k, market_price, r, T, ivol = market_datas.iloc[i]

            # Assuming HestonProcess and Option classes are defined
            heston = HestonProcess(mu=r, kappa=kappa + l, theta=(kappa * theta) / (kappa + l), sigma=sigma, rho=rho, v0=v0)
            opt = Option(s0=s0, v0=v0, T=T, K=k, call=True)

            heston_price = PriceOption(option=opt, process=heston, N=360, M=1000)
            result += (heston_price - market_price) ** 2
            rel_err = (heston_price / market_price - 1)
            avg_rel_err += abs(rel_err)

        avg_rel_err = avg_rel_err * 100 / len(market_datas)
        print('-- Resulting average abs error: ', avg_rel_err)
        print('-- Resulting squared error: ', result, '\n')

        return result

    # Define bounds for each parameter
    bounds = [(0, 4), (0, 1), (0, 1), (-1, 1), (0, 1), (-2, 2)]

    # Perform differential evolution
    result = optimize.differential_evolution(error, bounds=bounds)

    return result

def CalibrationReport(market_datas, kappa, theta, sigma, rho, v0, l):
    print("Heston model parameters:\nkappa = {0}\ntheta = {1}\nsigma = {2}\nrho = {3}\nv0 = {4}"
          .format(kappa, theta, sigma, rho, v0))
    print(
        "%20s %10s %14s %18s %15s %20s %13s %20s %20s %34s" % (
            "Underlying Price", "Strike", "Expiry",
            "Market Value", "Heston Value", "Relative Error(%)", "BSM Value", "Relative Error(%)", "True BSM", "True Heston"))
    print(
        "=" * 210)
    avg1 = 0.0
    avg2 = 0.0
    avg3 = 0.0
    avg4 = 0.0

    se1 = 0.0
    se2 = 0.0
    se3 = 0.0
    se4 = 0.0

    model_prices = []
    model_errors = []

    gbm_prices = []
    gbm_errors = []

    bsm_prices = []
    bsm_errors = []

    hes_prices = []
    hes_errors = []

    kappaRN = kappa + l
    thetaRN = (kappa * theta) / (kappa + l)

    for i in range(0, len(market_datas)):
        s0, k, market_price, r, T, ivol = market_datas.iloc[i]

        heston = HestonProcess(mu=r, kappa=kappaRN, theta=thetaRN, sigma=sigma, rho=rho, v0=v0) # under risk neutral measure
        #gbm = GeometricBrownianMotion(mu=r, sigma=ivol)
        opt = Option(s0=s0, v0=ivol, T=T, K=k, call=True) #v0=v0

        heston_price = PriceOption_LS(option=opt, process=heston, N=36, M=50, k=4) # N=int(np.ceil(T * 365)) * 10
        gbm_price = longstaff_schwartz_american_option_reg(option=opt, r=r, sigma=ivol, N=30, M=10, k=3) #PriceOption_LS(option=opt, process=gbm, N=600, M=20000, k=4) #       #PriceOption(option=opt, process=gbm, N=500, M=10000)
        bsm_price = black_scholes_merton(r=r, sigma=ivol,option=opt)
        hes_price = price_european_call_heston(option=opt, r=r, kappa=kappaRN, theta=thetaRN, sigma=sigma, rho=rho, v0=v0)  # analytical_heston_price(kappa=kappa, theta=theta, sigma=sigma, rho=rho, lambd=l, r=r, option=opt)
                                                                                                                            # HestonCallQuad(kappa=kappaRN, theta=thetaRN, sigma=sigma, rho=rho, v0=v0, r=r, option=opt)
        # relative error                                                                                                    #  HestonCallFFT(option=opt, r=r, kappa=kappaRN, theta=thetaRN, sigma=sigma, rho=rho, v0=v0)
        err1 = (heston_price / market_price - 1)
        err2 = (gbm_price / market_price - 1)
        err3 = (bsm_price / market_price - 1)
        err4 = (hes_price / market_price - 1)

        # squared error
        se1 += (heston_price - market_price) ** 2
        se2 += (gbm_price - market_price) ** 2
        se3 += (bsm_price - market_price) ** 2
        se4 += (hes_price - market_price) ** 2

        model_prices.append(heston_price)
        model_errors.append(err1)
        gbm_prices.append(gbm_price)
        gbm_errors.append(err2)
        bsm_prices.append(bsm_price)
        bsm_errors.append(err3)
        hes_prices.append(hes_price)
        hes_errors.append(err4)

        print(
            "%15.2f %15.2f %15.7f %13.2f %16.2f %17.7f %16.2f %17.7f %16.2f %17.7f %16.2f %17.7f" % (
                s0, k, T , market_price, heston_price,
                100.0 * err1, gbm_price, 100.0 * err2, bsm_price, 100.0 * err3, hes_price, 100.0 * err4))
        avg1 += abs(err1)
        avg2 += abs(err2)
        avg3 += abs(err3)
        avg4 += abs(err4)

    # mean squared error
    mse1 = se1 / len(market_datas)
    mse2 = se2 / len(market_datas)
    mse3 = se3 / len(market_datas)
    mse4 = se4 / len(market_datas)
    # average abs error
    avg1 = avg1 * 100.0 / len(market_datas)
    avg2 = avg2 * 100.0 / len(market_datas)
    avg3 = avg3 * 100.0 / len(market_datas)
    avg4 = avg4 * 100.0 / len(market_datas)
    print("-" * 210)
    print("Average Abs Error for Heston model (%%) : %5.9f, %5.9f\n"
          "Mean Squared Error (MSE) for Heston model: %5.9f, %5.9f\n"
          "Total Squared Error for Heston model: %10.9f, %10.9f\n"
          "Average Abs Error for BSM model (%%) : %5.9f, %5.9f\n"
          "Mean Squared Error (MSE) for BSM model: %5.9f, %5.9f\n"
          "Total Squared Error for BSM model: %10.9f, %10.9f\n"% (avg1, avg4, mse1, mse4, se1, se4, avg2, avg3, mse2, mse3, se2, se3))
    return hes_prices, hes_errors, gbm_prices, gbm_errors
