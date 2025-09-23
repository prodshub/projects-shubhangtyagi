import yfinance as yf
import numpy as np
import pandas as pd

def fetch_data(tickers, start_date, end_date, interval='1d'):
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval)['Close']
    return data

def compute_returns(data):
    return data.pct_change().dropna()

def compute_sharpe_ratio(weights, returns, risk_free_rate):
    expected_return = np.dot(weights, returns.mean())
    volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
    return expected_return, volatility, sharpe_ratio

def steepest_descent_sharpe(returns, risk_free_rate, alpha=0.01, max_iters=1000, tol=1e-6, max_weight=0.2):
    n_assets = returns.shape[1]
    weights = np.ones(n_assets) / n_assets 
    iterations = 0
    for _ in range(max_iters):
        expected_return, volatility, sharpe_ratio_value = compute_sharpe_ratio(weights, returns, risk_free_rate)
        gradient = (returns.mean() - risk_free_rate) / volatility - (expected_return - risk_free_rate) * (returns.cov() @ weights) / (volatility ** 3)
        alpha = np.argmax([compute_sharpe_ratio(weights + step * gradient, returns, risk_free_rate)[2] for step in np.linspace(0.001, 0.1, 10)]) * 0.001
        new_weights = weights + alpha * gradient
        new_weights = np.clip(new_weights, 0, max_weight)  
        new_weights /= new_weights.sum()  
        iterations += 1
        if np.linalg.norm(new_weights - weights) < tol:
            break
        weights = new_weights
    return weights, compute_sharpe_ratio(weights, returns, risk_free_rate)

tickers = ['AAPL','NVDA', 'META', 'SNAP', 'GME', 'ZM', 'AMZN'] 
start_date = '2015-03-14'
end_date = '2025-03-14'
risk_free_rate = 0.04/252  
stock_data = fetch_data(tickers, start_date, end_date)
returns = compute_returns(stock_data)
optimal_weights, (opt_return, opt_volatility, opt_sharpe) = steepest_descent_sharpe(returns, risk_free_rate, max_weight=0.3)
print("Optimal Portfolio Weights:", (optimal_weights*100).round(3))
print("Optimal Expected Daily Return:", (opt_return*100).round(3))
print("Optimal Daily Volatility:", (opt_volatility*100).round(3))
print("Optimal Sharpe Ratio:", (opt_sharpe*100).round(3))
