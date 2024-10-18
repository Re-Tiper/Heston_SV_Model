% Define parameters
strikes = linspace(0.8, 1.2, 11);  % Strike prices
volvols = 0.1:0.1:0.4;  % Volatility of volatility
styleV = {'-', '--', '-.', ':'};  % Line styles
colourV = {'k', 'b', 'r', 'm'};  % Line colors

% Initialize price and volatility matrices
prices = zeros(4, 11);
Volatility = zeros(4, 11);

% Plot for rho = 0.5
figure;
for i = 1:4
    for j = 1:11
        % Calculate price using HestonCallQuad
        price = HestonCallQuad(2, 0.04, volvols(i), 0.5, 0.04, 0.01, 1, 1, strikes(j));
        prices(i,j) = price;
        
        % Calculate implied volatility using Black-Scholes
        Volatility(i,j) = blsimpv(1, strikes(j), 0.01, 1, price);
    end
    % Plot implied volatility
    plot(strikes, Volatility(i,:), 'Color', colourV{i}, 'LineStyle', styleV{i});
    hold on;
end
ylabel('Implied Volatility');
xlabel('Strike');
title('\rho = 0.5');
legend('\sigma = 0.1', '\sigma = 0.2', '\sigma = 0.3', '\sigma = 0.4');
hold off;

% Plot for rho = 0
figure;
for i = 1:4
    for j = 1:11
        % Calculate price using HestonCallQuad
        price = HestonCallQuad(2, 0.04, volvols(i), 0, 0.04, 0.01, 1, 1, strikes(j));
        prices(i,j) = price;
        
        % Calculate implied volatility using Black-Scholes
        Volatility(i,j) = blsimpv(1, strikes(j), 0.01, 1, price);
    end
    % Plot implied volatility
    plot(strikes, Volatility(i,:), 'Color', colourV{i}, 'LineStyle', styleV{i});
    hold on;
end
ylabel('Implied Volatility');
xlabel('Strike');
title('\rho = 0');
legend('\sigma = 0.1', '\sigma = 0.2', '\sigma = 0.3', '\sigma = 0.4');
hold off;

% Plot for rho = -0.5
figure;
for i = 1:4
    for j = 1:11
        % Calculate price using HestonCallQuad
        price = HestonCallQuad(2, 0.04, volvols(i), -0.5, 0.04, 0.01, 1, 1, strikes(j));
        prices(i,j) = price;
        
        % Calculate implied volatility using Black-Scholes
        Volatility(i,j) = blsimpv(1, strikes(j), 0.01, 1, price);
    end
    % Plot implied volatility
    plot(strikes, Volatility(i,:), 'Color', colourV{i}, 'LineStyle', styleV{i});
    hold on;
end
ylabel('Implied Volatility');
xlabel('Strike');
title('\rho = -0.5');
legend('\sigma = 0.1', '\sigma = 0.2', '\sigma = 0.3', '\sigma = 0.4');
hold off;