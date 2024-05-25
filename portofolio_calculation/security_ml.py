import statsmodels.api as sm

def calculate_alpha_beta(excess_returns):
    market_excess_return = excess_returns['S&P500']
    alpha_beta_dict = {}

    for stock in excess_returns.columns:
        if stock not in ['S&P500']:
            Y = excess_returns[stock]
            X = sm.add_constant(market_excess_return)
            model = sm.OLS(Y, X).fit()
            alpha, beta = model.params
            alpha_beta_dict[stock] = (alpha, beta)

    return alpha_beta_dict