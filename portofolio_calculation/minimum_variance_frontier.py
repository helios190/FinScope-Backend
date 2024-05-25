import numpy as np
from scipy.optimize import minimize

def calculate_minimum_variance_frontier(dreturns):
    daily_returns = dreturns.drop(columns=['S&P500','^IRX'])
    n_assets = daily_returns.shape[1]
    returns_mean = daily_returns.mean()
    returns_cov = daily_returns.cov()

    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(returns_cov, weights))

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))

    min_var_portfolios = []
    target_returns = np.linspace(returns_mean.min(), returns_mean.max(), 100)

    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: np.dot(weights, returns_mean) - target_return}
        )

        result = minimize(portfolio_variance, n_assets * [1. / n_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            min_var_portfolios.append((target_return, result.fun))
        else:
            print("Optimization failed for target return:", target_return)

    return min_var_portfolios