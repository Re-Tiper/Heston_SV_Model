'''
We provide a second method to compute the closed-form solution of the Heston model using the 
adaptive quadrature method implemented by the quad function from SciPy’s integrate module
'''
from scipy import quad

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
