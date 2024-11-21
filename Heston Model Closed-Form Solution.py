def HestonCallFFT(kappa, theta, sigma, rho, r, v0, s0, strike, T):
    """
    Computes the European call option price using the Heston model and Fast Fourier Transform.

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

    x0 = np.log(s0)  # Initial log price
    alpha = 1.25
    N = 4096
    c = 600
    eta = c / N
    b = np.pi / eta
    u = np.arange(0, N) * eta
    lambd = 2 * b / N
    position = int(np.round((np.log(strike) + b) / lambd))  # Position of call value in FFT

    # Complex numbers for characteristic function
    v = u - (alpha + 1) * 1j
    zeta = -0.5 * (v ** 2 + 1j * v)
    gamma = kappa - rho * sigma * v * 1j
    PHI = np.sqrt(gamma ** 2 - 2 * sigma ** 2 * zeta)
    A = 1j * v * (x0 + r * T)
    B = v0 * ((2 * zeta * (1 - np.exp(-PHI * T))) /
              (2 * PHI - (PHI - gamma) * (1 - np.exp(-PHI * T))))
    C = -(kappa * theta / sigma ** 2) * (
            2 * np.log((2 * PHI - (PHI - gamma) * (1 - np.exp(-PHI * T))) / (2 * PHI)) +
            (PHI - gamma) * T
    )

    # Characteristic function
    char_func = np.exp(A + B + C)

    # Modified characteristic function
    modified_char_func = (char_func * np.exp(-r * T) /
                          (alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u))

    # Simpson weights for integration
    simpson_w = (1 / 3) * (3 + (-1) ** np.arange(1, N + 1) - np.append(1, np.zeros(N - 1)))

    # FFT computation
    fft_func = np.exp(1j * b * u) * modified_char_func * eta * simpson_w
    payoff = np.real(np.fft.fft(fft_func))

    # Extract call value
    call_value_m = np.exp(-np.log(strike) * alpha) * payoff / np.pi
    call_value = call_value_m[position]

    #print(f"Total execution time: {time.time() - start_time:.4f} seconds")
    return call_value

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
call_price = HestonCallFFT(kappa, theta, sigma, rho, r, v0, s0, K, T)
print(f"European Call Price: {call_price:.4f}")
