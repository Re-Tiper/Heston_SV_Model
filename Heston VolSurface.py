import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import brentq
#from scipy.fft import fft, ifft


def BSimpvol(S, K, r, T, C):
    """
    Calculate the implied volatility using the Black-Scholes model.

    Parameters:
    S : float
        Current stock price.
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate (annualized).
    T : float
        Time to maturity in years.
    C : float
        Price of the European call option.

    Returns:
    float
        Implied volatility.
    """

    # Initial guess for volatility
    sigma = 0.2

    # Define the function to minimize
    def f(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        C_calc = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return C_calc - C

    # Use a numerical method to find the root
    from scipy.optimize import brentq

    try:
        implied_vol = brentq(f, 1e-6, 5.0)  # Find the root in the interval [1e-6, 5.0]
        return implied_vol
    except ValueError:
        # If brentq fails, return NaN
        return np.nan

def HestonCallQuad(kappa, theta, sigma, rho, v0, r, T, s0, K):
    """Computes the price of a European call option using the Heston model."""
    call = s0 * HestonP(kappa, theta, sigma, rho, v0, r, T, s0, K, 1) \
         - K * np.exp(-r * T) * HestonP(kappa, theta, sigma, rho, v0, r, T, s0, K, 2)
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
    d = np.sqrt((rho * sigma * phi * 1j - b)**2 - sigma**2 * (2 * u * phi * 1j - phi**2))
    g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)

    C = r * phi * 1j * T + (a / sigma**2) * ((b - rho * sigma * phi * 1j + d) * T -
                                                2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
    D = (b - rho * sigma * phi * 1j + d) / sigma**2 * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

    return np.exp(C + D * v0 + 1j * phi * x)

# Define parameters
kappa = 2
theta = 0.05
sigma = 0.2
v0 = 0.08
r = 0.02
s0 = 100

K = 110
T = 1
rho = 0.5
price = HestonCallQuad(kappa, theta, sigma, rho, v0, r, T, s0, K)
print(f"The option price is: {price:.4f}")


"""Plot volatility surface with changing rho"""

# Define strikes and maturities
strikes = np.linspace(80, 120, 20)
mats = np.linspace(0.3, 2, 20)  # maturities

# Initialize price and volatility matrices
prices = np.zeros((20, 20))
Volatility = np.zeros((20, 20))


# Separate plots

# First surface for rho = 0.5
for i in range(20):
    for j in range(20):
        price = HestonCallQuad(kappa, theta, sigma, 0.5, v0, r, T=mats[i], s0=s0, K=strikes[j])
        prices[i, j] = price
        Volatility[i, j] = BSimpvol(s0, strikes[j], r, mats[i], price)

# Create meshgrid for strikes and maturities
strike, mat = np.meshgrid(strikes, mats)

# Plot the first surface (rho = 0.5)
plt.figure()
surf1 = plt.axes(projection='3d')
surf1.plot_surface(mat, strike, Volatility, cmap='coolwarm')
surf1.set_xlabel('Maturity (years)')
surf1.set_ylabel('Strike')
surf1.set_title(r'$\rho = 0.5$')
surf1.set_zlabel('Implied Volatility')

# Second surface for rho = 0
for i in range(20):
    for j in range(20):
        price = HestonCallQuad(kappa, theta, sigma, 0, v0, r, T=mats[i], s0=s0, K=strikes[j])
        prices[i, j] = price
        Volatility[i, j] = BSimpvol(s0, strikes[j], r, mats[i], price)

# Plot the second surface (rho = 0)
plt.figure()
surf2 = plt.axes(projection='3d')
surf2.plot_surface(mat, strike, Volatility, cmap='coolwarm')
surf2.set_xlabel('Maturity (years)')
surf2.set_ylabel('Strike')
surf2.set_title(r'$\rho = 0$')
surf2.set_zlabel('Implied Volatility')

# Third surface for rho = -0.5
for i in range(20):
    for j in range(20):
        price = HestonCallQuad(kappa, theta, sigma, -0.5, v0, r, T=mats[i], s0=s0, K=strikes[j])
        prices[i, j] = price
        Volatility[i, j] = BSimpvol(s0, strikes[j], r, mats[i], price)

# Plot the third surface (rho = -0.5)
plt.figure()
surf3 = plt.axes(projection='3d')
surf3.plot_surface(mat, strike, Volatility, cmap='coolwarm')
surf3.set_xlabel('Maturity (years)')
surf3.set_ylabel('Strike')
surf3.set_title(r'$\rho = -0.5$')
surf3.set_zlabel('Implied Volatility')

plt.show()



# All together plots
# Create a single figure for all surfaces
fig = plt.figure(figsize=(18, 12))

# First surface for rho = 0.5
ax1 = fig.add_subplot(131, projection='3d')
for i in range(20):
    for j in range(20):
        price = HestonCallQuad(kappa, theta, sigma, 0.5, v0, r, T=mats[i], s0=s0, K=strikes[j])
        prices[i, j] = price
        Volatility[i, j] = BSimpvol(s0, strikes[j], r, mats[i], price)

# Create meshgrid for strikes and maturities
strike, mat = np.meshgrid(strikes, mats)

ax1.plot_surface(mat, strike, Volatility, cmap='coolwarm')
ax1.set_xlabel('Maturity (years)')
ax1.set_ylabel('Strike')
ax1.set_title(r'$\rho = 0.5$')
ax1.set_zlabel('Implied Volatility')

# Second surface for rho = 0
ax2 = fig.add_subplot(132, projection='3d')
for i in range(20):
    for j in range(20):
        price = HestonCallQuad(kappa, theta, sigma, 0, v0, r, T=mats[i], s0=s0, K=strikes[j])
        prices[i, j] = price
        Volatility[i, j] = BSimpvol(s0, strikes[j], r, mats[i], price)

ax2.plot_surface(mat, strike, Volatility, cmap='coolwarm')
ax2.set_xlabel('Maturity (years)')
ax2.set_ylabel('Strike')
ax2.set_title(r'$\rho = 0$')
ax2.set_zlabel('Implied Volatility')

# Third surface for rho = -0.5
ax3 = fig.add_subplot(133, projection='3d')
for i in range(20):
    for j in range(20):
        price = HestonCallQuad(kappa, theta, sigma, -0.5, v0, r, T=mats[i], s0=s0, K=strikes[j])
        prices[i, j] = price
        Volatility[i, j] = BSimpvol(s0, strikes[j], r, mats[i], price)

ax3.plot_surface(mat, strike, Volatility, cmap='coolwarm')
ax3.set_xlabel('Maturity (years)')
ax3.set_ylabel('Strike')
ax3.set_title(r'$\rho = -0.5$')
ax3.set_zlabel('Implied Volatility')

# Adjust layout and show
plt.tight_layout()
plt.show()



"""See how sigma (vol of vol) affects the volatility surface"""

strikes = np.linspace(70, 130, 20)
volvols = np.arange(0.1, 0.5, 0.1) # Volatility of volatility
mats = np.linspace(0.3, 2, 20)  # maturities
styles = ['-', '--', '-.', ':']
colors = ['k', 'b', 'r', 'm']

# Initialize price and volatility arrays
prices = np.zeros((4, 20))
Volatility = np.zeros((4, 20))

# Plot for rho = 0.5
plt.figure()
for i, sigma in enumerate(volvols):
    for j, strike in enumerate(strikes):
        price = HestonCallQuad(kappa, theta, sigma, 0.5, v0, r, T=mats[i], s0=s0, K=strikes[j])
        prices[i, j] = price
        Volatility[i, j] = BSimpvol(s0, strikes[j], r, mats[i], price)
    plt.plot(strikes, Volatility[i, :], color=colors[i], linestyle=styles[i], label=f'σ = {sigma:.1f}')
plt.ylabel('Implied Volatility')
plt.xlabel('Strike')
plt.title('ρ = 0.5')
plt.legend()

# Plot for rho = 0
plt.figure()
for i, sigma in enumerate(volvols):
    for j, strike in enumerate(strikes):
        price = HestonCallQuad(kappa, theta, sigma, 0, v0, r, T=mats[i], s0=s0, K=strikes[j])
        prices[i, j] = price
        Volatility[i, j] = BSimpvol(s0, strikes[j], r, mats[i], price)
    plt.plot(strikes, Volatility[i, :], color=colors[i], linestyle=styles[i], label=f'σ = {sigma:.1f}')
plt.ylabel('Implied Volatility')
plt.xlabel('Strike')
plt.title('ρ = 0')
plt.legend()

# Plot for rho = -0.5
plt.figure()
for i, sigma in enumerate(volvols):
    for j, strike in enumerate(strikes):
        price = HestonCallQuad(kappa, theta, sigma, -0.5, v0, r, T=mats[i], s0=s0, K=strikes[j])
        prices[i, j] = price
        Volatility[i, j] = BSimpvol(s0, strikes[j], r, mats[i], price)
    plt.plot(strikes, Volatility[i, :], color=colors[i], linestyle=styles[i], label=f'σ = {sigma:.1f}')
plt.ylabel('Implied Volatility')
plt.xlabel('Strike')
plt.title('ρ = -0.5')
plt.legend()
plt.show()
