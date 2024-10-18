function call = HestonCallQuad(kappa, theta, sigma, rho, v0, r, T, s0, K)
    % HestonCallQuad: Computes the price of a European call option using
    % the Heston model with the Quadrature method.
    %
    % Inputs:
    % kappa  - Rate of mean reversion
    % theta  - Long-run variance
    % sigma  - Volatility of volatility
    % rho    - Correlation between asset price and volatility
    % v0     - Initial variance
    % r      - Risk-free interest rate
    % T      - Time until maturity
    % s0     - Initial asset price
    % K      - Strike price
    %
    % Output:
    % call   - Price of the European call option

    % Suppress warnings
    warning off;

    % Calculate the call price using the Heston pricing formula
    call = s0 * HestonP(kappa, theta, sigma, rho, v0, r, T, s0, K, 1) ...
         - K * exp(-r * T) * HestonP(kappa, theta, sigma, rho, v0, r, T, s0, K, 2);
end

function ret = HestonP(kappa, theta, sigma, rho, v0, r, T, s0, K, type)
    % HestonP: Computes the Heston characteristic function using numerical integration.
    %
    % Inputs:
    % kappa  - Rate of mean reversion
    % theta  - Long-run variance
    % sigma  - Volatility of volatility
    % rho    - Correlation between asset price and volatility
    % v0     - Initial variance
    % r      - Risk-free interest rate
    % T      - Time until maturity
    % s0     - Initial asset price
    % K      - Strike price
    % type   - 1 for call price, 2 for put price
    %
    % Output:
    % ret    - Value of the Heston characteristic function

    % Perform numerical integration
    ret = 0.5 + (1/pi) * quadl(@HestonPIntegrand, 0, 100, [], [], ...
        kappa, theta, sigma, rho, v0, r, T, s0, K, type);
end

function ret = HestonPIntegrand(phi, kappa, theta, sigma, rho, v0, r, T, s0, K, type)
    % HestonPIntegrand: Evaluates the integrand for the Heston characteristic function.
    %
    % Inputs:
    % phi    - Integration variable
    % kappa  - Rate of mean reversion
    % theta  - Long-run variance
    % sigma  - Volatility of volatility
    % rho    - Correlation
    % v0     - Initial variance
    % r      - Risk-free interest rate
    % T      - Time until maturity
    % s0     - Initial asset price
    % K      - Strike price
    % type   - 1 for call price, 2 for put price
    %
    % Output:
    % ret    - Value of the integrand

    ret = real(exp(-1i * phi * log(K)) .* ...
        Hestf(phi, kappa, theta, sigma, rho, v0, r, T, s0, type) ./ (1i * phi));
end

function f = Hestf(phi, kappa, theta, sigma, rho, v0, r, T, s0, type)
    % Hestf: Computes the Heston characteristic function.
    %
    % Inputs:
    % phi    - Frequency variable
    % kappa  - Rate of mean reversion
    % theta  - Long-run variance
    % sigma  - Volatility of volatility
    % rho    - Correlation
    % v0     - Initial variance
    % r      - Risk-free interest rate
    % T      - Time until maturity
    % s0     - Initial asset price
    % type   - 1 for call price, 2 for put price
    %
    % Output:
    % f      - Value of the characteristic function

    % Define parameters based on the option type
    if type == 1
        u = 0.5; % For call options
        b = kappa - rho * sigma;
    else
        u = -0.5; % For put options
        b = kappa;
    end

    % Precompute variables
    a = kappa * theta;
    x = log(s0);
    d = sqrt((rho * sigma * phi * 1i - b).^2 - sigma^2 * (2 * u * phi * 1i - phi.^2));
    g = (b - rho * sigma * phi * 1i + d) ./ (b - rho * sigma * phi * 1i - d);
    
    % Calculate the terms in the characteristic function
    C = r * phi * 1i * T + (a / sigma^2) * ((b - rho * sigma * phi * 1i + d) * T - ...
        2 * log((1 - g .* exp(d * T)) ./ (1 - g)));
    D = (b - rho * sigma * phi * 1i + d) ./ sigma^2 .* ((1 - exp(d * T)) ./ (1 - g .* exp(d * T)));

    % Final characteristic function
    f = exp(C + D * v0 + 1i * phi * x);
end