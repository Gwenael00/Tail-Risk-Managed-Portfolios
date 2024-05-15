
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, chi2
import cvxpy as cp

def get_RNDs(df):
    
    lower_moneyness = 0.85
    upper_moneyness = 1.15
    
    def calc_MEM(S, call_strikes, call_prices, put_strikes, put_prices, r):
    
        P = np.hstack([
            np.maximum(put_strikes[None, :] - w[:, None]*S, 0)*np.exp(-r*30/365),
            np.maximum(w[:, None]*S - call_strikes[None, :], 0)*np.exp(-r*30/365)
        ])
        
        d = np.concatenate([
            put_prices,
            call_prices
        ])
        
        m, n = P.shape
        
        q_var = cp.Variable(m)
        
        entr = cp.sum(cp.entr(q_var))
        
        cons = [P.T @ q_var == d, cp.sum(q_var) == 1, q_var >= 0]
        
        problem = cp.Problem(cp.Maximize(entr), cons)
        problem.solve(solver=cp.MOSEK)
        print("Status:", problem.status)
    
        return q_var.value
    
    def sorter(df2):
        
        df2 = pd.DataFrame(df2)
        target_range = np.arange(lower_moneyness, upper_moneyness + 0.01, 0.025)
        closest_rows = pd.DataFrame()
        
        for value in target_range:
            # Calculate the absolute difference between moneyness values and the current target value
            df2['diff'] = abs(df2['moneyness'] - value)
            # Append the row with the smallest difference to the result DataFrame
            closest_rows = closest_rows.append(df2.loc[df2['diff'].idxmin()])
        # Remove the temporary 'diff' column and drop duplicates if any
        closest_rows = closest_rows.drop(columns=['diff']).drop_duplicates()
        
        return closest_rows
    
    
    #execute
    w = 0.5 + 0.002 * np.arange(501)
    distr = np.zeros((len(w),len(df['date'].unique())))
    
    stats = np.zeros((7,len(df['date'].unique())))
    dates = df['date'].unique() 

    i = 0
    for date in dates:
        try:
            rows_for_date = df[df['date'] == date]
            
            call_data = rows_for_date[rows_for_date['cp_flag'] == 'C']
            call_data = call_data.sort_values(by='moneyness')
            call_data = sorter(call_data)
            
            put_data = rows_for_date[rows_for_date['cp_flag'] == 'P']
            put_data = put_data.sort_values(by='moneyness')
            put_data = sorter(put_data)
            
            call_data.reset_index(drop=True, inplace=True)
            put_data.reset_index(drop=True, inplace=True)
            
            S = rows_for_date['spindx'].iloc[0]
            r = rows_for_date['rate'].iloc[0]
            call_strikes = call_data['strike_price'].values
            call_prices = call_data['price'].values
            put_strikes = put_data['strike_price'].values
            put_prices = put_data['price'].values
            
            distr[:, i] = calc_MEM(S, call_strikes, call_prices, put_strikes, put_prices,r)
            stats[0, i] = len(call_prices)
            stats[1, i] = len(put_prices)
            stats[2, i] = r
            stats[3, i] = rows_for_date['returns'].iloc[0]
            
            print(f"{i+1} iteration - Success")
        except Exception as e:
            print(f"{i+1} iteration - Failed")
        
        i += 1
        print(f"Total options: {len(call_prices) + len(put_prices)}")
    
    distr = pd.DataFrame(distr)  
    stats = pd.DataFrame(stats)
    
    return distr, stats



def fit_RWDs(distr, df):
    
    distr_a = distr.iloc[:,:71].copy()
    w = 0.5 + 0.002 * np.arange(501)
    prices = df.drop_duplicates(subset=['date'], keep='first')
    prices_new = prices['spindxex'].iloc[:71].reset_index(drop=True)

    prices_old = prices['spindx'].iloc[:71].reset_index(drop=True)

    def calculate_inverse_probability_transformations(distrb):

        y_ts = np.zeros(len(distrb.T))
        for i in range(len(distrb.T)):
            price_new= prices_new.iloc[i]
            price_old= w*prices_old.iloc[i]
            # Find the index of the closest price point less than or equal to X_t
            index = np.searchsorted(price_old, price_new, side='right') - 1
            # Integrate the PDF up to this index (cumulative sum of probabilities)
            y_t = np.sum(distrb[i][:index+1])
            y_ts[i] = y_t
        
        return y_ts

    def transform_to_standard_normal_quantiles(y_ts):
        z_ts = norm.ppf(y_ts)
        return z_ts

    def log_likelihood(params, data):

        mu, rho, sigma = params
        T = len(data)

        term1 = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma**2 / (1 - rho**2)) - \
                (((data[0] - mu / (1 - rho))**2) / (2*sigma**2 / (1 - rho**2)))
        errors = data[1:] - mu - rho * data[:-1]
        sum_term = np.sum((errors**2) / (2 * sigma**2))
        log_likelihood = term1 - (T - 1)/2 * np.log(2 * np.pi) - \
                         (T - 1)/2 * np.log(sigma**2) - sum_term
        
        return -log_likelihood  # Negative log-likelihood for minimization

    def estimate_ar1_model(z_ts):
        # Initial parameter guesses
        initial_params = np.array([np.mean(z_ts), 0.5, np.std(z_ts)])
        # Minimize the negative log-likelihood
        result = minimize(log_likelihood, initial_params, args=(z_ts,),method="BFGS")
        print(result.success)
        
        if result.success:
            mu, rho, sigma = result.x
            print(f"Estimated parameters: mu = {mu}, rho = {rho}, sigma = {sigma}")
        else:
            raise ValueError("Model fitting did not converge")
        
        return result.x

    def compute_lr_test(fitted_log_likelihood, null_log_likelihood, df):

        lr_stat = -2 * (null_log_likelihood - fitted_log_likelihood)
        p_value = chi2.sf(lr_stat, df)
        return lr_stat, p_value
    
    y_ts = calculate_inverse_probability_transformations(distrb=distr_a)
    z_ts = transform_to_standard_normal_quantiles(y_ts)    
    estimated_params = estimate_ar1_model(z_ts)

    fitted_ll = -log_likelihood(estimated_params, z_ts)  # Fitted model log-likelihood
    null_ll_lr3 = -log_likelihood([0, 0, 1], z_ts)  # Null model log-likelihood for LR_3
    null_ll_lr1 = -log_likelihood([estimated_params[0], 0, estimated_params[2]], z_ts)   

    # Compute LR_3
    lr_3_stat, lr_3_p_value = compute_lr_test(fitted_ll, null_ll_lr3, df=3)
    print(f"LR_3 Statistic: {lr_3_stat}, p-value: {lr_3_p_value}")

    # Compute LR_1
    lr_1_stat, lr_1_p_value = compute_lr_test(fitted_ll, null_ll_lr1, df=1)
    print(f"LR_1 Statistic: {lr_1_stat}, p-value: {lr_1_p_value}")

    def rwd(rnd,gamma,S):
        denom = 0
        rwd = np.zeros(len(S))
        for s in range(len(S)):
            rwd[s] = rnd[s]/(S[s]**(-gamma))
            denom += rwd[s]
        rwd = rwd/denom
        return rwd
        
    def estimate_rwd_model(gamma):
        
        rw_distr = distr_a.copy()
        for col in range(len(rw_distr.T)):
            rw_distr[col] = rwd(rnd=distr_a[col],gamma=gamma,S=prices_old.iloc[col]*w)
        
        y_ts = calculate_inverse_probability_transformations(distrb=rw_distr)
        z_ts = transform_to_standard_normal_quantiles(y_ts)
        estimated_params = estimate_ar1_model(z_ts)
        fitted_ll = -log_likelihood(estimated_params, z_ts)  # Fitted model log-likelihood
        null_ll_lr3 = -log_likelihood([0, 0, 1], z_ts)  # Null model log-likelihood for LR_3
        lr_3_stat, lr_3_p_value = compute_lr_test(fitted_ll, null_ll_lr3, df=3)
        print(f"LR_3 Statistic: {lr_3_stat}, p-value: {lr_3_p_value}")
        null_ll_lr1 = -log_likelihood([estimated_params[0], 0, estimated_params[2]], z_ts)   
        lr_1_stat, lr_1_p_value = compute_lr_test(fitted_ll, null_ll_lr1, df=1)
        print(f"LR_1 Statistic: {lr_1_stat}, p-value: {lr_1_p_value}")

        return -lr_3_p_value

    initial_gamma = -5
    result = minimize(estimate_rwd_model, initial_gamma,method='BFGS')
    
    rw_distr = distr_a.copy()
    for col in range(len(rw_distr.T)):
        rw_distr[col] = rwd(rnd=distr_a[col],gamma=result.x,S=prices_old.iloc[col]*w)
        
    y_ts_rw = calculate_inverse_probability_transformations(distrb=rw_distr)
    z_ts_rw = transform_to_standard_normal_quantiles(y_ts_rw)
        
    return rw_distr, z_ts , z_ts_rw
    

def run_strategies(distr,rw_distr,df,stats):
    
    distr_a = distr.iloc[:,:71].copy()
    stats_a = stats.iloc[:,:71].copy()
    dates = df['date'].unique() 
    date_differences = np.diff(dates[71:]).astype('timedelta64[D]')

    # Convert to an array of integers representing the number of days
    date_differences = date_differences / np.timedelta64(1, 'D')

    date_differences.tolist()
    date_differences = date_differences/365

    returns = df['returns_strat'].unique()[2:]
    w = 0.5 + 0.002 * np.arange(501)
    
    def calculate_VaR_for_dataframe( p_df, alpha):
        
        def calculate_VaR( p, alpha):    
            cumulative_probabilities = []
            cumulative_sum = 0
            for probability in p:
                cumulative_sum += probability
                cumulative_probabilities.append(cumulative_sum)
            
            # Find the VaR
            for i in range(len(cumulative_probabilities)):
                if cumulative_probabilities[i] >= alpha:
                    return w[i]-1
            
            # If the loop completes without returning, then the VaR is the maximum loss (worst outcome)
            return w[-1]-1
        
        VaR_results = {}
        for column in p_df.columns:
            p = p_df[column].tolist()
            VaR_results[column] = calculate_VaR( p, 1 - alpha)

        VaR_df = pd.DataFrame(VaR_results, index=["VaR"])
        
        return np.array(VaR_df).tolist()[0]

    def calculate_CVaR_for_dataframe( p_df, alpha):
        
        def calculate_CVaR(p, alpha):
            cumulative_probabilities = []
            cumulative_sum = 0
            for probability in p:
                cumulative_sum += probability
                cumulative_probabilities.append(cumulative_sum)
        
            index = 0
            for i, cum_prob in enumerate(cumulative_probabilities):
                if cum_prob >= alpha:
                    index = i
                    break
        
            total_loss = 0
            total_probability = 0
            for i in range(0, index+1):
                total_loss += p[i] * (w[i]-1)
                total_probability += p[i]
        
            if total_probability > 0:
                return (total_loss) / total_probability
            else:
                return w[-1]-1 
        
        CVaR_results = {}
        for column in p_df.columns:
            p = p_df[column].tolist()
            CVaR_results[column] = calculate_CVaR( p, 1 - alpha)

        CVaR_df = pd.DataFrame(CVaR_results, index=["CVaR"])
        
        return np.array(CVaR_df).tolist()[0]

    def calculate_w_t_risky(target, r_f_series, measure_series):
        w_t_risky_series = []
        for r_f, alpha in zip(r_f_series, measure_series):
            w_t_risky_series.append((target + r_f) / (alpha + r_f)) #here add adjustment of 0.7 for adjusten TMPs
        
        return np.clip(w_t_risky_series, 0, 1) # Ensure weights are between 0 and 1

    #target calculated from insample
    targets = {
        'VaR': {'RN': {'90%': -6.9944, '95%': -9.9070},
                'RW': {'90%': -6.3662, '95%': -9.1831}},
        'CVaR': {'RN': {'90%': -10.8011, '95%': -13.3856},
                 'RW': {'90%': -10.0566, '95%': -12.5714}}
    }

    strategies = {}
    for measure_type, conf_levels in targets.items():
        for density_type, targets_conf in conf_levels.items():
            for conf_level, target in targets_conf.items():
                alpha = float(conf_level.strip('%')) / 100
                # Select appropriate distribution
                p_df = distr_a if density_type == 'RN' else rw_distr
                # Calculate measure and weight
                measure_series = calculate_VaR_for_dataframe(p_df, alpha) if measure_type == 'VaR' else calculate_CVaR_for_dataframe(p_df, alpha)
                r_f_series = np.exp(stats_a.iloc[2] * 30 / 365) - 1
                weight_series = calculate_w_t_risky((target / 100), r_f_series, measure_series)
                strategies[(measure_type, density_type, conf_level)] = weight_series

    strategy_performance = {}

    for strategy, weights in strategies.items():
        measure_type, density_type, conf_level = strategy
        aligned_weights = weights[:-1]
        risk_free_returns = np.exp(stats_a.iloc[2, :-1] * date_differences) - 1
        
        # Calculate the portfolio returns based on the weights
        portfolio_returns = aligned_weights* returns[71:] + (1 - aligned_weights) * risk_free_returns
        cumulative_strategy_return = np.cumprod(1 + portfolio_returns) - 1
        strategy_performance[strategy] = cumulative_strategy_return


    strategy_performance_non_cum = {}
    for strategy, weights in strategies.items():
        measure_type, density_type, conf_level = strategy
        aligned_weights = weights[:-1]
        risk_free_returns = np.exp(stats_a.iloc[2, :-1] * date_differences) - 1
        
        # Calculate the portfolio returns based on the weights
        portfolio_returns = aligned_weights* returns[71:] + (1 - aligned_weights) * risk_free_returns
        strategy_performance_non_cum[strategy] = portfolio_returns
        
        
        
    def sharpe_ratio(returns, risk_free_rate):
        per_rf = np.exp(risk_free_rate*date_differences)-1
        excess_returns = returns - per_rf
        return np.mean(excess_returns/date_differences) / np.std(excess_returns*np.sqrt(1/date_differences))

    def sortino_ratio(returns, risk_free_rate):
        ann_ret = (1+returns)**(1/date_differences)-1
        downside_returns = ann_ret[ann_ret < risk_free_rate]
        downside_risk = np.std(downside_returns)
        excess_returns = ann_ret - risk_free_rate
        return np.mean(excess_returns) / downside_risk if downside_risk != 0 else np.nan

    def max_drawdown(returns):
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def average_drawdown(returns):
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return drawdown.mean()

    # Prepare a dictionary to store the performance metrics
    performance_metrics = {
        'Average Return': {},
        'Volatility': {},
        'Sharpe Ratio': {},
        'Sortino Ratio': {},
        'Max Drawdown': {},
        'Average Drawdown': {}
    }

    annual_risk_free_rate = stats_a.iloc[2, :-1]

    # Calculate performance metrics for each strategy
    for strategy, returnss in strategy_performance_non_cum.items():
        returns_array = np.array(returnss)
        
        performance_metrics['Sharpe Ratio'][strategy] = sharpe_ratio(returns_array, annual_risk_free_rate)
        performance_metrics['Sortino Ratio'][strategy] = sortino_ratio(returns_array, annual_risk_free_rate)
        performance_metrics['Max Drawdown'][strategy] = max_drawdown(returns_array)*100
        performance_metrics['Average Drawdown'][strategy] = average_drawdown(returns_array)*100
        performance_metrics['Average Return'][strategy] = np.mean(returns_array/date_differences)*100
        performance_metrics['Volatility'][strategy] = np.std(returns_array*np.sqrt(1/date_differences))*100

    # Convert the performance metrics dictionary into a DataFrame
    performance_metrics_df = pd.DataFrame(performance_metrics)
    print(performance_metrics_df)

    print(max_drawdown(0.7*returns[71:]+0.3*(np.exp(annual_risk_free_rate*date_differences)-1)))
    print(np.mean((0.7*returns[71:]+0.3*(np.exp(annual_risk_free_rate*date_differences)-1))/date_differences)*100) 
    print(np.std((0.7*returns[71:]+0.3*(np.exp(annual_risk_free_rate*date_differences)-1))*np.sqrt(1/date_differences))*100)
    
    return strategies, strategy_performance, strategy_performance_non_cum
