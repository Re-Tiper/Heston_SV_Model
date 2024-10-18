function CallValue = HestonCallFft(kappa, theta, sigma, rho, r, v0, s0, strike, T)
    % HestonCallFft: Prices a European call option using the Heston model via FFT.
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

    % Value in the FFT
    v = u - (alpha + 1) * 1i; % Adjusted frequency
    zeta = -0.5 * (v.^2 + 1i * v); % Auxiliary variable
    gamma = kappa - rho * sigma * v * 1i; % Gamma calculation
    PHI = sqrt(gamma.^2 - 2 * sigma^2 * zeta); % Characteristic function part

    % Calculate terms A, B, and C for the characteristic function
    A = 1i * v * (x0 + r * T);
    B = v0 * ((2 * zeta .* (1 - exp(-PHI * T))) ./ (2 * PHI - (PHI - gamma) .* (1 - exp(-PHI * T))));
    C = -kappa * theta / sigma^2 * (2 * log((2 * PHI - (PHI - gamma) .* (1 - exp(-PHI * T))) ./ (2 * PHI)) + (PHI - gamma) * T);

    % Characteristic function
    charFunc = exp(A + B + C);
    ModifiedCharFunc = charFunc * exp(-r * T) ./ (alpha^2 + alpha - u.^2 + 1i * (2 * alpha + 1) * u);

    % Simpson's rule weights for integration
    SimpsonW = 1/3 * (3 + (-1i).^(1:N) - [1, zeros(1, N-1)]);
    
    % FFT function
    FftFunc = exp(1i * b * u) .* ModifiedCharFunc * eta .* SimpsonW;
    
    % Calculate the payoff
    payoff = real(fft(FftFunc)); % Inverse FFT
    
    % Final call value
    CallValueM = exp(-log(strike) * alpha) * payoff / pi;
    
    % Return the final call value
    CallValue = CallValueM(round(position));
end