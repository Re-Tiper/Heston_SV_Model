function CallValue = HestonCallFftTimeValue(kappa, theta, sigma, rho, r, v0, s0, strike, T)
    % HestonCallFftTimeValue: Prices a European call option using the Heston model via FFT,
    % accounting for time value.
    %
    % Inputs:
    % kappa  - Rate of mean reversion
    % theta  - Long-run variance
    % sigma  - Volatility of volatility
    % rho    - Correlation between asset and volatility
    % r      - Risk-free interest rate
    % v0     - Initial variance
    % s0     - Initial asset price
    % strike  - Strike price of the option
    % T      - Time until maturity
    %
    % Output:
    % CallValue - Price of the European call option

    % Log of the initial asset price
    x0 = log(s0);
    alpha = 1.25; % Fourier transform parameter
    N = 4096; % Number of FFT points
    c = 600; % Domain size
    eta = c / N; % Spacing of the grid
    b = pi / eta; % Scaling factor

    % Create frequency grid
    u = (0:N-1) * eta; 
    lambda = 2 * b / N; % Step size for integration
    position = (log(strike) + b) / lambda + 1; % Position of the call option in the FFT

    % Value in the FFT for the first part
    w1 = u - 1i * alpha;
    v1 = u - 1i * alpha - 1i;
    zeta1 = -0.5 * (v1.^2 + 1i * v1);
    gamma1 = kappa - rho * sigma * v1 * 1i;
    PHI1 = sqrt(gamma1.^2 - 2 * sigma^2 * zeta1);
    A1 = 1i * v1 * (x0 + r * T);
    B1 = v0 * ((2 * zeta1 .* (1 - exp(-PHI1 .* T))) ./ (2 * PHI1 - (PHI1 - gamma1) .* (1 - exp(-PHI1 * T))));
    C1 = -kappa * theta / sigma^2 * (2 * log((2 * PHI1 - (PHI1 - gamma1) .* (1 - exp(-PHI1 * T))) ./ (2 * PHI1)) + (PHI1 - gamma1) * T);
    charFunc1 = exp(A1 + B1 + C1);

    % Modified characteristic function for the first part
    ModifiedCharFunc1 = exp(-r * T) * (1 ./ (1 + 1i * w1) - exp(r * T) ./ (1i * w1) - charFunc1 ./ (w1.^2 - 1i * w1));

    % Value in the FFT for the second part
    w2 = u + 1i * alpha;
    v2 = u + 1i * alpha - 1i;
    zeta2 = -0.5 * (v2.^2 + 1i * v2);
    gamma2 = kappa - rho * sigma * v2 * 1i;
    PHI2 = sqrt(gamma2.^2 - 2 * sigma^2 * zeta2);
    A2 = 1i * v2 * (x0 + r * T);
    B2 = v0 * ((2 * zeta2 .* (1 - exp(-PHI2 .* T))) ./ (2 * PHI2 - (PHI2 - gamma2) .* (1 - exp(-PHI2 * T))));
    C2 = -kappa * theta / sigma^2 * (2 * log((2 * PHI2 - (PHI2 - gamma2) .* (1 - exp(-PHI2 * T))) ./ (2 * PHI2)) + (PHI2 - gamma2) * T);
    charFunc2 = exp(A2 + B2 + C2);

    % Modified characteristic function for the second part
    ModifiedCharFunc2 = exp(-r * T) * (1 ./ (1 + 1i * w2) - exp(r * T) ./ (1i * w2) - charFunc2 ./ (w2.^2 - 1i * w2));

    % Combine the modified characteristic functions
    ModifiedCharFuncCombo = (ModifiedCharFunc1 - ModifiedCharFunc2) / 2;

    % Simpson's rule weights for integration
    SimpsonW = 1/3 * (3 + (-1).^(1:N) - [1, zeros(1, N-1)]);
    
    % FFT function
    FftFunc = exp(1i * b * u) .* ModifiedCharFuncCombo * eta .* SimpsonW;
    
    % Calculate the payoff
    payoff = real(fft(FftFunc)); % Inverse FFT
    
    % Final call value
    CallValueM = payoff / pi / sinh(alpha * log(strike));
    
    % Return the final call value
    CallValue = CallValueM(round(position));
end