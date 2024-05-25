import numpy as np
from scipy.optimize import minimize

def calculate_max_sharpe_ratio(dreturns, risk_free_rate):
    daily_returns = dreturns.drop(columns=['S&P500','^IRX'])
    n_assets = daily_returns.shape[1]
    returns_mean = daily_returns.mean()
    returns_cov = daily_returns.cov()

    def sharpe_ratio(weights):
        portfolio_return = np.dot(weights, returns_mean)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_cov, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))

    result = minimize(sharpe_ratio, n_assets * [1. / n_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x, -result.fun
    else:
        raise ValueError("Optimization failed for the Sharpe Ratio")