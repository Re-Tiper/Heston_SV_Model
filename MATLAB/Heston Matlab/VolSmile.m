% Define strikes and maturities
strikes = linspace(0.8, 1.2, 11);
mats = linspace(0.3, 3, 11); % maturities

% Initialize price and volatility matrices
prices = zeros(11, 11);
Volatility = zeros(11, 11);

% First surface for rho = 0.5
for i = 1:11
    for j = 1:11
        price = HestonCallQuad(2, 0.04, 0.1, 0.5, 0.04, 0.01, mats(i), 1, strikes(j));
        prices(i,j) = price;
        Volatility(i,j) = blsimpv(1, strikes(j), 0.01, mats(i), price);
    end
end

% Create meshgrid for strikes and maturities
[strike, mat] = meshgrid(strikes, mats);

% Plot the first surface (rho = 0.5)
figure;
surf(mat, strike, Volatility);
xlabel('Maturity (years)');
ylabel('Strike');
title('\rho = 0.5');
zlabel('Implied Volatility');

% Set the colormap 
colormap("hot"); 
%colorbar; % Optional: display colorbar to indicate the scale of values

% Second surface for rho = 0
for i = 1:11
    for j = 1:11
        price = HestonCallQuad(2, 0.04, 0.1, 0, 0.04, 0.01, mats(i), 1, strikes(j));
        prices(i,j) = price;
        Volatility(i,j) = blsimpv(1, strikes(j), 0.01, mats(i), price);
    end
end

% Plot the second surface (rho = 0)
figure;
surf(mat, strike, Volatility);
xlabel('Maturity (years)');
ylabel('Strike');
title('\rho = 0');
zlabel('Implied Volatility');

% Set the colormap 
colormap("hot"); 

% Third surface for rho = -0.5
for i = 1:11
    for j = 1:11
        price = HestonCallQuad(2, 0.04, 0.1, -0.5, 0.04, 0.01, mats(i), 1, strikes(j));
        prices(i,j) = price;
        Volatility(i,j) = blsimpv(1, strikes(j), 0.01, mats(i), price);
    end
end

% Plot the third surface (rho = -0.5)
figure;
surf(mat, strike, Volatility);
xlabel('Maturity (years)');
ylabel('Strike');
title('\rho = -0.5');
zlabel('Implied Volatility');

% Set the colormap 
colormap("hot"); 
