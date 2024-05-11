# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 12:19:19 2024

@author: gwena
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from scipy.stats import norm, chi2
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import skew, kurtosis
import cvxpy as cp


plt.style.use('seaborn-whitegrid')
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern'

#get option data
os.chdir("D:\Master\Thesis")
df = pd.read_csv("jvossrlke6ddmel8.csv")
df = df.drop(columns=["optionid","index_flag","issuer"])
df['strike_price'] = df['strike_price']/1000
df['price'] = (df['best_bid'] + df['best_offer'])/2

#get rf
rf = pd.read_csv('xl4ajfq0nrwzgqq6.csv')
rf = rf.drop(rf[rf.days>100].index)
rf['rate'] = rf['rate']/100

# Function to interpolate the "rate" for 30-day maturity
def interpolate_30_day_rate(group):
    # Ensure the group is sorted by "days" for proper interpolation
    group = group.sort_values(by='days')
    
    # Check for an exact 30-day match
    if (group['days'] == 30).any():
        return group[group['days'] == 30]
    
    # Identify the closest "days" below and above 30
    below = group[group['days'] < 30].iloc[-1:] if not group[group['days'] < 30].empty else None
    above = group[group['days'] > 30].iloc[:1] if not group[group['days'] > 30].empty else None
    
    # Perform interpolation if both "below" and "above" are present
    if below is not None and above is not None:
        distance = above['days'].values[0] - below['days'].values[0]
        weight_below = (30 - below['days'].values[0]) / distance
        weight_above = 1 - weight_below
        
        interpolated_rate = below['rate'].values[0] * weight_below + above['rate'].values[0] * weight_above
        interpolated_row = pd.DataFrame({'date': below['date'].values, 'days': [30], 'rate': [interpolated_rate]})
        
        return interpolated_row
    else:
        # If interpolation isn't possible, return an empty dataframe
        return pd.DataFrame()

# Group the second dataframe by 'date' and apply the interpolation function
rf_grouped = rf.groupby('date')
interpolated_rates = pd.concat([interpolate_30_day_rate(group) for _, group in rf_grouped])

# Merge the interpolated rates with the first dataframe
df = pd.merge(df, interpolated_rates, on='date', how='left')

#get sp500 data
sp500 = pd.read_csv('golvjl9eyxaeqz4r.csv')
sp500.rename(columns={"caldt": "date"},inplace=True)

#merge
df = pd.merge(df,sp500,on='date',how='left')

#get price at maturity
sp500ex = sp500.copy()
sp500ex.rename(columns={"date": "exdate","spindx":"spindxex"},inplace=True)
sp500ex['exdate'] =  pd.to_datetime(sp500ex['exdate'])
df['exdate'] = pd.to_datetime(df['exdate'])
df = pd.merge_asof(df,sp500ex,on='exdate',direction='forward')
df['date'] = pd.to_datetime(df['date'])


def filter_dates(df):
    df_dates = pd.DataFrame(df['date'].copy())
    
    # Now, let's count how often each date occurs
    date_counts = df_dates['date'].value_counts()
    
    # Convert date_counts Series to a DataFrame
    date_counts_df = date_counts.reset_index()
    date_counts_df.columns = ['date', 'count']
    
    date_counts_df.sort_values(by=['count'], ascending=False, inplace=True)
    
    # Add year and month columns for grouping
    date_counts_df['year'] = date_counts_df['date'].dt.year
    date_counts_df['month'] = date_counts_df['date'].dt.month
    
    # Now, group by year and month, and select the first row in each group (the one with the highest count)
    highest_counts_per_month = date_counts_df.groupby(['year', 'month']).first().reset_index()
    
    dates_with_highest_counts = highest_counts_per_month['date'].dt.strftime('%Y-%m-%d').tolist()
    
    # Filter the original DataFrame to keep only rows with dates in the list
    return df[df['date'].isin(dates_with_highest_counts)]

df = filter_dates(df=df)

df['moneyness'] = df['strike_price']/df['spindx']

lower_moneyness = 0.85
upper_moneyness = 1.15

df = df.drop(df[(df.best_bid==0) | (df.open_interest==0)].index)
df = df.drop(df[(df.moneyness < lower_moneyness) | (df.moneyness > upper_moneyness)].index)
df = df.drop(df[((df.cp_flag == 'C') & (df.moneyness > 1)) | 
                ((df.cp_flag == 'P') & (df.moneyness < 1))].index)    
   
 
def plot_sp500():
    plt.figure(figsize=(8, 4))  # Adjusted for better fit in documents
    
    
    plt.plot(df['date'], df['spindx'], linestyle='-', color='black')  # Black for a classic look

    plt.xlabel('Date', fontsize=10)
    
    # Updated ylabel to reflect that we're now directly showing percentages
    plt.ylabel('Index Level', fontsize=10)
    
    # Adjusting the x-axis to show every third month to reduce clutter
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, interval=15))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    
    # Beautify the x-labels for dates
    plt.gcf().autofmt_xdate()
    
    # Saving the plot with high resolution
    plt.savefig('sp500.jpg', dpi=300, bbox_inches='tight')
    
    # Show the plot (optional here since we're mainly saving it)
    plt.show()
    
#plot_sp500()   

df['returns'] = df['spindxex']/df['spindx']-1 
df['returns_strat'] = df['spindx'].pct_change() 

def plot_sp500_ret():
    plt.figure(figsize=(8, 4))  # Adjusted for better fit in documents
    
    
    plt.plot(df['date'], df['returns']*100, linestyle='-', color='black')  # Black for a classic look
    
    plt.xlabel('Date', fontsize=10)
    
    # Updated ylabel to reflect that we're now directly showing percentages
    plt.ylabel('Return (\%)', fontsize=10)
    
    # Adjusting the x-axis to show every third month to reduce clutter
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, interval=15))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    
    # Beautify the x-labels for dates
    plt.gcf().autofmt_xdate()
    
    # Saving the plot with high resolution
    plt.savefig('sp500_rets.jpg', dpi=300, bbox_inches='tight')
    
    # Show the plot (optional here since we're mainly saving it)
    plt.show()

#plot_sp500_ret()

def plot_dates():
    #visualize dates
    sorted_dates = np.sort(df['date'].unique())
    
    date_range = pd.date_range(start=sorted_dates.min(), end=sorted_dates.max(), freq='D')
    date_df = pd.DataFrame(date_range, columns=['Date'])
    date_df['EventCount'] = date_df['Date'].apply(lambda x: np.sum(sorted_dates == x))
    date_df['CumulativeEvents'] = date_df['EventCount'].cumsum()
    
    # Plotting
    plt.figure(figsize=(8, 4))  # Adjusted for better fit in documents
    plt.plot(date_df['Date'], date_df['CumulativeEvents'], marker=',', linestyle='-', color='black')  # Black for a classic look
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Number of Dates', fontsize=10)
    
    # Adjusting the x-axis to show every third month to reduce clutter
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, interval=15))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    
    # Beautify the x-labels for dates
    plt.gcf().autofmt_xdate()
    
    # Saving the plot with high resolution
    plt.savefig('cum_event.jpg', dpi=300, bbox_inches='tight')
    
    # Show the plot (optional here since we're mainly saving it)
    plt.show()
    
#plot_dates()

def plot_rf():
    plt.figure(figsize=(8, 4))  # Adjusted for better fit in documents
    
    # Multiply df['rate'] by 100 to convert to percent
    plt.plot(df['date'], df['rate']*100, linestyle='-', color='black')  # Black for a classic look
    

    plt.xlabel('Date', fontsize=10)
    
    # Updated ylabel to reflect that we're now directly showing percentages
    plt.ylabel('Risk-Free Rate (\%)', fontsize=10)
    
    # Adjusting the x-axis to show every third month to reduce clutter
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, interval=15))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    
    # Beautify the x-labels for dates
    plt.gcf().autofmt_xdate()
    
    # Saving the plot with high resolution
    plt.savefig('rf.jpg', dpi=300, bbox_inches='tight')
    
    # Show the plot (optional here since we're mainly saving it)
    plt.show()

#plot_rf()

def calc(S, call_strikes, call_prices, put_strikes, put_prices, r):

    P = np.hstack([
        np.maximum(put_strikes[None, :] - w[:, None]*S, 0)*np.exp(-r*30/365),
        np.maximum(w[:, None]*S - call_strikes[None, :], 0)*np.exp(-r*30/365)
    ])
    
    d = np.concatenate([
        put_prices,
        call_prices
    ])
    
    m, n = P.shape
    
    lambda_var = cp.Variable(m)
    
    entr = cp.sum(cp.entr(lambda_var))
    
    cons = [P.T @ lambda_var == d, cp.sum(lambda_var) == 1, lambda_var >= 0]
    
    problem = cp.Problem(cp.Maximize(entr), cons)
    problem.solve(solver=cp.MOSEK)
    print("Status:", problem.status)

    return lambda_var.value

    
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
        
        call_payoff = np.zeros((len(w), len(call_strikes)))
        put_payoff = np.zeros((len(w), len(put_strikes)))
        call_payoffs = np.zeros((len(w), len(call_strikes)))
        put_payoffs = np.zeros((len(w), len(put_strikes)))
        
        #distr[:, i] = calc(S, call_strikes, call_prices, put_strikes, put_prices,r)
        stats[0, i] = len(call_prices)
        stats[1, i] = len(put_prices)
        stats[2, i] = r
        stats[3, i] = rows_for_date['returns'].iloc[0]
        
        print(f"{i+1} iteration - Success")
    except Exception as e:
        print(f"{i+1} iteration - Failed")
    
    i += 1
    print(f"Total options: {len(call_prices) + len(put_prices)}")

#distr = pd.DataFrame(distr)
#distr.to_csv('distr_final.csv', index=False)  
#stats = pd.DataFrame(stats)
#stats.to_csv('stats_final.csv',index=True)
    
distr = pd.read_csv('distr_final.csv',header=0)
#distr = pd.read_csv('distr4.csv',header=0)
distr.columns = range(len(distr.T))
stats = pd.read_csv('stats_final.csv',header=0)
stats = stats.iloc[:,1:]



nan_cols = distr.columns[distr.isna().all()]

for col in nan_cols:
    left_col_index = distr.columns.get_loc(col) - 1  # Index of the column immediately to the left
    if left_col_index >= 0:  # Ensure it does not go out of bounds
        distr[col] = distr.iloc[:, left_col_index]


def plot_options_over_time():
    plt.figure(figsize=(8, 4))  # Slightly larger figure for better clarity

    calls = stats.iloc[0].values  # Calls data
    puts = stats.iloc[1].values  # Puts data
    total = calls + puts  # Total volume for area plot base

    # Using the specified custom colors
    plt.fill_between(dates, 0, calls, color=(0.2, 0.4, 0.8), alpha=0.7, label='Call Options')
    plt.fill_between(dates, calls, total, color=(0.8, 0.2, 0.4), alpha=0.7, label='Put Options')

    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Number of Options', fontsize=10)

    plt.ylim(0, 17)  # Setting the maximum y-axis value to 30

    # Improving the x-axis to handle a large number of dates
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())

    plt.gcf().autofmt_xdate()  # Beautify the x-labels for dates
    plt.legend(fontsize=10)  # Add legend

    plt.tight_layout()  # Adjust layout to make room for the legend
    plt.savefig('options_area_plot_adjusted.jpg', dpi=300, bbox_inches='tight')
    plt.show()

#plot_options_over_time()

distr_a = distr.iloc[:,71:].copy()
distr_a.columns = [int(col) - 71 for col in distr_a.columns]  # Convert to int and then subtract
stats_a = stats.iloc[:,71:].copy()
stats_a.columns = [int(col) - 71 for col in stats_a.columns]  # Convert to int and then subtract

def density_surface(dens,name):
    datess = pd.Series(dates[71:])
    # Generate date labels with only month and year
    date_labels = datess.dt.strftime('%m-%Y').tolist()
    time_numeric = np.arange(len(date_labels))
    
    Z = (dens*100).values
    Y, X = np.meshgrid(time_numeric, (w-1)*100)  # Now Y is time, X is return percentages
    
    # Mask Z values where X (now return percentages) is outside the desired range (-30 to 30)
    Z_masked = np.where((X >= -40) & (X <= 40), Z, np.nan)
    
    # Define custom colormap
    colors = [(0.2, 0.4, 0.8), (0.8, 0.2, 0.4)]  # Example colors, adjust as needed
    cmap_name = 'custom_cmap'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
    
    # Setting up the 3D plot using the masked Z values
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Creating the surface plot with switched axes and custom colormap
    surf = ax.plot_surface(X, Y, Z_masked, cmap=custom_cmap, edgecolor='none')
    
    # Customizing the axes and adding a color bar
    ax.set_xlabel('Return (\%)', fontsize=14)
    ax.set_ylabel('Date', fontsize=14)
    ax.set_zlabel('Density Difference (\%)', fontsize=14)
    
    # Setting the X-axis limits to match the desired return percentage range
    ax.set_xlim(-40, 40)
    
    # Adjusting the y-axis ticks to display time labels
    tick_interval = 24
    filtered_ticks = np.linspace(0, len(date_labels)-1, len(date_labels))[::tick_interval]
    filtered_labels = date_labels[::tick_interval]

    ax.set_yticks(filtered_ticks)
    ax.set_yticklabels(filtered_labels, fontsize=12)
    
    ax.yaxis.labelpad = 20
    # Rotate the plot (example: azimuth=220, elevation=20)
    ax.view_init(azim=230, elev=20)

    
    plt.savefig(name, dpi=300, bbox_inches='tight')  # Save the plot
    plt.show()

density_surface(distr_a,"3dRWD_out_dif")


def sumstats(df):
    
    metrics_df = pd.DataFrame(columns=['Distribution', 'Mean', 'Volatility', 'Skewness', 'Kurtosis'])
    log_w = (w-1)*100
    
    for column in df.columns:
        
        P = df[column]
        
        # Implied mean (mu)
        mean = np.sum(P * log_w)
        
        # Implied volatility (sigma)
        volatility = np.sqrt(np.sum(P * (log_w - mean)**2))
        
        # Implied skewness
        skewness = (np.sum(P * (log_w - mean)**3)) / (volatility**3)
        
        # Implied kurtosis
        kurtosis = (np.sum(P * (log_w - mean)**4)) / (volatility**4)
        
        # Append to DataFrame
        metrics_df = metrics_df.append({
            'Distribution': column,
            'Mean': mean,
            'Volatility': volatility,
            'Skewness': skewness,
            'Kurtosis': kurtosis
        }, ignore_index=True)

    return metrics_df

sumstat = sumstats(df=distr_a)
print(np.mean(sumstat))
print(np.std(sumstat))
print(np.mean(np.exp(stats_a.iloc[2]*30/365)-1)*100)
print(np.std(np.exp(stats_a.iloc[2]*30/365)-1)*100)

prices = df.drop_duplicates(subset=['date'], keep='first')
prices_new = prices['spindxex'].iloc[71:].reset_index(drop=True)

# Select the column 'spindx' and extract data starting from the 72nd entry
# Reset the index to start from 0
prices_old = prices['spindx'].iloc[71:].reset_index(drop=True)

def calculate_inverse_probability_transformations(distrb):
    """
    Calculate the inverse probability transformations (y_t) for observed prices.

    :param estimated_pdfs: A list of arrays, each representing the estimated PDF (hat_f_t(u)) at a time t.
    :param observed_prices: An array of observed prices (X_t) corresponding to each time t.
    :param price_points: An array of price points corresponding to the discretized values over which the PDFs are defined.
    :return: An array of y_t values.
    """
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

y_ts = calculate_inverse_probability_transformations(distrb=distr_a)


def transform_to_standard_normal_quantiles(y_ts):
    """
    Transform y_t values to standard normal quantiles (z_t).

    :param y_ts: An array of y_t values, resulting from the inverse probability transformations.
    :return: An array of z_t values, transformed to standard normal quantiles.
    """
    z_ts = norm.ppf(y_ts)
    return z_ts

# Assuming y_ts from the previous example

z_ts = transform_to_standard_normal_quantiles(y_ts)

    
def log_likelihood(params, data):
    """
    The log-likelihood function for the AR(1) model with mean adjustment.
    """
    mu, rho, sigma = params
    T = len(data)
    
    # Correcting the first term calculation
    term1 = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma**2 / (1 - rho**2)) - \
            (((data[0] - mu / (1 - rho))**2) / (2*sigma**2 / (1 - rho**2)))

    # Calculation of errors for t=2 to T
    errors = data[1:] - mu - rho * data[:-1]
    
    # Correcting the sum term calculation
    sum_term = np.sum((errors**2) / (2 * sigma**2))

    # Complete log-likelihood calculation
    log_likelihood = term1 - (T - 1)/2 * np.log(2 * np.pi) - \
                     (T - 1)/2 * np.log(sigma**2) - sum_term
    
    return -log_likelihood  # Negative log-likelihood for minimization


def estimate_ar1_model(z_ts):
    """
    Estimate the AR(1) model with mean adjustment for the provided z_t values.

    :param z_ts: An array of z_t values, transformed to standard normal quantiles.
    :return: Estimated parameters mu, rho, and sigma (standard deviation of eps_t).
    """
    # Initial parameter guesses
    initial_params = np.array([np.mean(z_ts), 0.5, np.std(z_ts)])
    # Minimize the negative log-likelihood
    result = minimize(log_likelihood, initial_params, args=(z_ts,),method="SLSQP")
    print(result.success)
    
    if result.success:
        mu, rho, sigma = result.x
        print(f"Estimated parameters: mu = {mu}, rho = {rho}, sigma = {sigma}")
    else:
        raise ValueError("Model fitting did not converge")
    
    return result.x

# Example usage
# Assuming z_ts from the previous step


# Estimate the AR(1) model
estimated_params = estimate_ar1_model(z_ts)

def compute_lr_test(fitted_log_likelihood, null_log_likelihood, df):
    """
    Compute the likelihood ratio test statistic and p-value.

    :param fitted_log_likelihood: Log-likelihood of the fitted model.
    :param null_log_likelihood: Log-likelihood under the null hypothesis.
    :param df: Degrees of freedom (number of parameters tested).
    :return: LR statistic and p-value.
    """
    lr_stat = -2 * (null_log_likelihood - fitted_log_likelihood)
    p_value = chi2.sf(lr_stat, df)
    return lr_stat, p_value

# Example calculations (you need to compute the log-likelihood values accordingly)
fitted_ll = -log_likelihood(estimated_params, z_ts)  # Fitted model log-likelihood
null_ll_lr3 = -log_likelihood([0, 0, 1], z_ts)  # Null model log-likelihood for LR_3
null_ll_lr1 = -log_likelihood([estimated_params[0], 0, estimated_params[2]], z_ts)    # Simplified for demonstration; adjust for your null model

# Compute LR_3
lr_3_stat, lr_3_p_value = compute_lr_test(fitted_ll, null_ll_lr3, df=3)
print(f"LR_3 Statistic: {lr_3_stat}, p-value: {lr_3_p_value}")

# Compute LR_1
# Adjust null_ll_lr1 as needed based on your specific model for independence
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
    """
    Estimate the AR(1) model with mean adjustment for the provided z_t values.

    :param z_ts: An array of z_t values, transformed to standard normal quantiles.
    :return: Estimated parameters mu, rho, and sigma (standard deviation of eps_t).
    """
    
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
    null_ll_lr1 = -log_likelihood([estimated_params[0], 0, estimated_params[2]], z_ts)    # Simplified for demonstration; adjust for your null model
    lr_1_stat, lr_1_p_value = compute_lr_test(fitted_ll, null_ll_lr1, df=1)
    print(f"LR_1 Statistic: {lr_1_stat}, p-value: {lr_1_p_value}")

    return -lr_3_p_value

initial_gamma = -5
result = minimize(estimate_rwd_model, initial_gamma,method='Powell')

rw_distr = distr_a.copy()
for col in range(len(rw_distr.T)):
    rw_distr[col] = rwd(rnd=distr_a[col],gamma=1.1813,S=prices_old.iloc[col]*w)

def plot_standard_normal_quantiles_refined():
    datess = pd.Series(dates[71:])  # Replace this with your actual dates series
    fig, ax1 = plt.subplots(figsize=(8, 4))

    color1 = (0.2, 0.4, 0.8)  # Custom blue-like color for RWD Quantiles
    color2 = (0.8, 0.2, 0.4)  # Custom red-like color for the difference

    # Plot RWD Quantiles
    line1, = ax1.plot(datess, z_ts_rw, color=color1, label='RWD Quantiles')

    ax1.set_xlabel('Date', color='black', fontsize=10)
    ax1.set_ylabel('RWD Quantiles', color='black', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=10)
    ax1.tick_params(axis='x', labelcolor='black', labelsize=10)

    # Adjust the major and minor x-axis locators and formatters
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=1, interval=10))

    # Plot Difference (z_ts_rw - z_ts)
    ax2 = ax1.twinx()
    line2, = ax2.plot(datess, z_ts_rw - z_ts, linestyle='dotted', color=color2, linewidth=3, label='Difference RWD vs. RND Quantiles')
    ax2.set_ylabel('Quantile Difference', color='black', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=10)

    # Add legend
    plt.legend(handles=[line1, line2], loc='lower left', fontsize=10)

    fig.tight_layout()
    
    # Rotate the x-axis labels for better readability
    plt.setp(ax1.xaxis.get_minorticklabels(), rotation=45)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    plt.savefig("out_of_sample_error", dpi=300, bbox_inches='tight')
    plt.show()



y_ts_rw = calculate_inverse_probability_transformations(distrb=rw_distr)
z_ts_rw = transform_to_standard_normal_quantiles(y_ts_rw)
plot_standard_normal_quantiles_refined()

estimated_params_rwd = estimate_ar1_model(z_ts_rw)
fitted_ll = -log_likelihood(estimated_params_rwd, z_ts_rw)  # Fitted model log-likelihood
null_ll_lr3 = -log_likelihood([0, 0, 1], z_ts_rw)  # Null model log-likelihood for LR_3
null_ll_lr1 = -log_likelihood([estimated_params_rwd[0], 0, estimated_params_rwd[2]], z_ts_rw)    # Simplified for demonstration; adjust for your null model

# Compute LR_3
lr_3_stat, lr_3_p_value = compute_lr_test(fitted_ll, null_ll_lr3, df=3)
print(f"LR_3 Statistic: {lr_3_stat}, p-value: {lr_3_p_value}")

# Compute LR_1
# Adjust null_ll_lr1 as needed based on your specific model for independence
lr_1_stat, lr_1_p_value = compute_lr_test(fitted_ll, null_ll_lr1, df=1)
print(f"LR_1 Statistic: {lr_1_stat}, p-value: {lr_1_p_value}")

density_surface(rw_distr,"3dRWD")
density_surface(rw_distr-distr_a,"3dDIF")


from scipy.integrate import cumtrapz
from matplotlib.ticker import PercentFormatter


def plot_pdf_cdf(pdf1, pdf2, x_values, highlight_x):
    # Ensure the PDFs are normalized

    # Calculate the CDFs using cumulative trapezoidal integration
    cdf1 = cumtrapz(pdf1, x_values, initial=0)
    cdf2 = cumtrapz(pdf2, x_values, initial=0)

    # Changed subplot configuration to 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(6, 10))

    color1 = (0.2, 0.4, 0.8)  # Custom blue-like color
    color2 = (0.8, 0.2, 0.4)  # Custom red-like color
    highlight_color = 'green'  # Color for highlighting the specific x value

    # Plot PDFs
    axes[0].plot(x_values, pdf1*100, label='Risk-Neutral', color=color1)
    axes[0].plot(x_values, pdf2*100, label='Real-World', color=color2)
    axes[0].axvline(highlight_x, color=highlight_color, linestyle='--', label=f'{highlight_x}\%')
    axes[0].set_title('Probability Density Functions', fontsize=14)
    axes[0].set_xlabel(r'Return (\%)', fontsize=12)
    axes[0].set_ylabel(r'Density (\%)', fontsize=12)
    axes[0].tick_params(labelsize=12)
    axes[0].legend(fontsize=12)

    # Plot CDFs
    axes[1].plot(x_values, cdf1*500, label='Risk-Neutral', color=color1)
    axes[1].plot(x_values, cdf2*500, label='Real-World', color=color2)
    axes[1].axvline(highlight_x, color=highlight_color, linestyle='--', label=f'{highlight_x}\%')
    axes[1].set_title('Cumulative Distribution Functions', fontsize=14)
    axes[1].set_xlabel(r'Return (\%)', fontsize=12)
    axes[1].set_ylabel(r'Cumulative Density (\%)', fontsize=12)
    axes[1].tick_params(labelsize=12)
    axes[1].legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('Density_appendix.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
plot_pdf_cdf(distr_a[81], rw_distr[81], (w-1)*100,  15.8405)

prices = df.drop_duplicates(subset=['date'], keep='first')
prices_new = prices['spindxex'].iloc[:71]
prices_old = prices['spindx'].iloc[:71]
    
rn_distr_out = distr.iloc[:,71:].copy()
rw_distr_out = distr.iloc[:,71:].copy()
for col in range(len(rw_distr_out.T)):
    rw_distr_out.iloc[:,col] = rwd(rnd=rn_distr_out.iloc[:,col],gamma=result.x,S=prices_old.iloc[col]*w)

#############################################################################################
################################## Strategy implementation ##################################
#############################################################################################

date_differences = np.diff(dates[71:]).astype('timedelta64[D]')

# Convert to an array of integers representing the number of days
date_differences = date_differences / np.timedelta64(1, 'D')

date_differences.tolist()
date_differences = date_differences/365

returns = df['returns_strat'].unique()[2:]

def calculate_VaR_for_dataframe( p_df, alpha):
    # Ensure S_df is sorted from worst to best for each column
    
    def calculate_VaR( p, alpha):    
        # Calculate cumulative probabilities
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
    
    # Calculate the VaR for each column
    VaR_results = {}
    for column in p_df.columns:
        p = p_df[column].tolist()
        VaR_results[column] = calculate_VaR( p, 1 - alpha)

    # Create a new DataFrame for VaR results
    VaR_df = pd.DataFrame(VaR_results, index=["VaR"])
    
    return np.array(VaR_df).tolist()[0]

def calculate_CVaR_for_dataframe( p_df, alpha):
    # Ensure S_df is sorted from worst to best for each column
    
    def calculate_CVaR(p, alpha):
        # Calculate cumulative probabilities
        cumulative_probabilities = []
        cumulative_sum = 0
        for probability in p:
            cumulative_sum += probability
            cumulative_probabilities.append(cumulative_sum)
    
        # Find the index where the cumulative probability exceeds the confidence level
        index = 0
        for i, cum_prob in enumerate(cumulative_probabilities):
            if cum_prob >= alpha:
                index = i
                break
    
        # Calculate CVaR as the average of losses that are as bad or worse than VaR
        total_loss = 0
        total_probability = 0
        for i in range(0, index+1):
            total_loss += p[i] * (w[i]-1)
            total_probability += p[i]
    
        # Handle the case where total_probability is 0 to avoid division by zero
        if total_probability > 0:
            return (total_loss) / total_probability
        else:
            return w[-1]-1  # If no probability mass is left after VaR, return the worst case
    
    # Calculate the VaR for each column
    CVaR_results = {}
    for column in p_df.columns:
        p = p_df[column].tolist()
        CVaR_results[column] = calculate_CVaR( p, 1 - alpha)

    # Create a new DataFrame for VaR results
    CVaR_df = pd.DataFrame(CVaR_results, index=["CVaR"])
    
    return np.array(CVaR_df).tolist()[0]

def calculate_w_t_risky(target, r_f_series, measure_series):
    w_t_risky_series = []
    for r_f, alpha in zip(r_f_series, measure_series):
        w_t_risky_series.append((target*0.7 + r_f) / (alpha + r_f))
    
    return np.clip(w_t_risky_series, 0, 1) # Ensure weights are between 0 and 1

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
            # Convert string percentage to alpha
            alpha = float(conf_level.strip('%')) / 100
            # Select appropriate distribution
            p_df = distr_a if density_type == 'RN' else rw_distr
            # Calculate measure and weight
            measure_series = calculate_VaR_for_dataframe(p_df, alpha) if measure_type == 'VaR' else calculate_CVaR_for_dataframe(p_df, alpha)
            r_f_series = np.exp(stats_a.iloc[2] * 30 / 365) - 1
            weight_series = calculate_w_t_risky((target / 100), r_f_series, measure_series)
            # Store the results
            strategies[(measure_type, density_type, conf_level)] = weight_series



strategy_performance = {}

for strategy, weights in strategies.items():
    measure_type, density_type, conf_level = strategy
    # Shift weights to align with returns (weights calculated at start of period for next period's returns)
    aligned_weights = weights[:-1]
    # Calculate the returns of the risky asset for the time periods
    # Calculate the risk-free returns for the time periods
    risk_free_returns = np.exp(stats_a.iloc[2, :-1] * date_differences) - 1
    
    # Calculate the portfolio returns based on the weights
    portfolio_returns = aligned_weights* returns[71:] + (1 - aligned_weights) * risk_free_returns
    # Calculate the cumulative return of the strategy
    cumulative_strategy_return = np.cumprod(1 + portfolio_returns) - 1
    # Store the cumulative return
    strategy_performance[strategy] = cumulative_strategy_return


strategy_performance_non_cum = {}
for strategy, weights in strategies.items():
    measure_type, density_type, conf_level = strategy
    # Shift weights to align with returns (weights calculated at start of period for next period's returns)
    aligned_weights = weights[:-1]
    # Calculate the returns of the risky asset for the time periods
    # Calculate the risk-free returns for the time periods
    risk_free_returns = np.exp(stats_a.iloc[2, :-1] * date_differences) - 1
    
    # Calculate the portfolio returns based on the weights
    portfolio_returns = aligned_weights* returns[71:] + (1 - aligned_weights) * risk_free_returns
    # Store the cumulative return
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

# Assuming `returns` is a Series of the risky asset returns
# And assuming `stats_a.iloc[2, :-1]` contains the log returns of the risk-free rate
# And `date_differences` represents the time periods between returns

# Prepare a dictionary to store the performance metrics
performance_metrics = {
    'Average Return': {},
    'Volatility': {},
    'Sharpe Ratio': {},
    'Sortino Ratio': {},
    'Max Drawdown': {},
    'Average Drawdown': {}
}

# Calculate the risk-free rate over the returns period
annual_risk_free_rate = stats_a.iloc[2, :-1]

# Calculate performance metrics for each strategy
for strategy, returnss in strategy_performance_non_cum.items():
    # Convert the strategy's non-cumulative returns into a numpy array
    returns_array = np.array(returnss)
    
    # Calculate and store each metric
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

custom_blue = (0.2, 0.4, 0.8)  # Blue-like color
custom_red = (0.8, 0.2, 0.4)  # Red-like color



import matplotlib.patches as mpatches

import matplotlib.lines as mlines

# Create a figure with 4 rows and 2 columns of subplots
fig, axes = plt.subplots(4, 2, figsize=(15, 20))  # Adjust figsize to fit a full page in your thesis
axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing
datess = pd.Series(dates[72:])  # Replace this with your actual dates series

for i, (strategy, cumulative_returns) in enumerate(strategy_performance.items()):
    strategy_title = str(strategy).replace('%', '\\%')
    ax = axes[i]
    ax.plot(datess, cumulative_returns*100, label="Strategy", color=custom_blue)
    ax.plot(datess, 100*(np.cumprod(1*returns[71:]+0*(np.exp(annual_risk_free_rate*date_differences)-1)+1)-1), label="SPX", color='black')  # Market return with neutral black
    ax.plot(datess, 100*(np.cumprod(0.7*returns[71:]+0.3*(np.exp(annual_risk_free_rate*date_differences)-1)+1)-1), label="Equal-Weight", color=custom_red)
    ax.set_title(f"{strategy_title}", fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(r'Cumulative Return (\%)', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

# Create custom handles for the shared legend
strategy_line = mlines.Line2D([], [], color=custom_blue, label='Strategy')
spx_line = mlines.Line2D([], [], color='black', label='SPX')
equal_weight_line = mlines.Line2D([], [], color=custom_red, label='Constant-Mix')


# Place a shared legend at the bottom center of the figure
fig.legend(handles=[strategy_line, spx_line, equal_weight_line], loc='lower center', ncol=3, fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to make room for the suptitle and the shared legend
plt.savefig("cumulative_returns_subplot.jpg", dpi=300, bbox_inches='tight')  # Save as high-resolution for inclusion in thesis
plt.show()



custom_blue = (0.2, 0.4, 0.8)  # Blue-like color for strategy weights
custom_red = (0.8, 0.2, 0.4)  # Red-like color for riskless asset weights

# Create a figure with 4 rows and 2 columns of subplots for the weights
fig, axes = plt.subplots(4, 2, figsize=(15, 20))  # Adjust figsize to fit a full page in your thesis
axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing
datess = pd.Series(dates[71:])  # Replace this with your actual dates series

for i, ((measure_type, density_type, conf_level), weight_series) in enumerate(strategies.items()):
    ax = axes[i]
    # Plot the weight of the risky asset
    ax.fill_between(datess, 0, weight_series*100, color=custom_blue, alpha=0.5)
    # Plot the weight of the riskless asset
    ax.fill_between(datess, weight_series*100, 100, color=custom_red, alpha=0.5)
    strategy_title = str(strategy).replace('%', '\\%')    
    ax.set_title(f"{strategy_title}", fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Weight (\%)', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

# Create custom handles for the shared legend
risky_patch = mpatches.Patch(color=custom_blue, alpha=0.5, label='SPX')
riskless_patch = mpatches.Patch(color=custom_red, alpha=0.5, label='Risk-Free Asset')


# Place a shared legend at the bottom center of the figure
fig.legend(handles=[risky_patch, riskless_patch], loc='lower center', ncol=2, fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to make room for the suptitle and the shared legend
plt.savefig("strategy_weights_subplot.jpg", dpi=300, bbox_inches='tight')  # Save as high-resolution for inclusion in thesis
plt.show()


custom_blue = (0.2, 0.4, 0.8)  # Blue-like color for strategy weights
custom_red = (0.8, 0.2, 0.4)  # Red-like color for riskless asset weights

# Extract the specific strategy data
specific_key = ('CVaR', 'RW', '95%')
weight_series = strategies[specific_key]

# Convert the dates to a Pandas Series with datetime type
datess = pd.to_datetime(dates[71:])  # Assuming 'dates' is already defined

# Create a single plot for the specific strategy
plt.figure(figsize=(8, 4))
plt.fill_between(datess, 0, weight_series * 100, color=custom_blue, alpha=0.5, label='SPX')
plt.fill_between(datess, weight_series * 100, 100, color=custom_red, alpha=0.5, label='Risk-Free Asset')

# Formatting the plot
plt.title("('CVaR', 'RW', '95\%')", fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Weight (\%)', fontsize=12)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

# Place the legend below the plot
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, fontsize=10)

plt.tight_layout()
plt.savefig("singleweight2.jpg", dpi=300, bbox_inches='tight')  # Save as high-resolution for inclusion in thesis
plt.show()



custom_blue = (0.2, 0.4, 0.8)  # Blue-like color
custom_red = (0.8, 0.2, 0.4)  # Red-like color

# Extract the specific strategy data
specific_key = ('CVaR', 'RW', '95%')
cumulative_returns = strategy_performance[specific_key]

# Convert the dates to a Pandas Series with datetime type
datess = pd.to_datetime(dates[72:])  # Assuming 'dates' is already defined

# Create a single plot for the specific strategy
plt.figure(figsize=(8, 4))
plt.plot(datess, cumulative_returns*100, label=f"Strategy {specific_key}", color=custom_blue)
plt.plot(datess, 100*(np.cumprod(1*returns[71:]+0*(np.exp(annual_risk_free_rate*date_differences)-1)+1)-1), label="SPX", color='black')  # Market return
plt.plot(datess, 100*(np.cumprod(0.7*returns[71:]+0.3*(np.exp(annual_risk_free_rate*date_differences)-1)+1)-1), label="Constant-Mix", color=custom_red)

# Formatting the plot
strategy_title = ', '.join(specific_key).replace('%', '\\%')  # Create the title from the key tuple
plt.title(f"({strategy_title})", fontsize=12)
plt.xlabel('Date', fontsize=10)
plt.ylabel('Cumulative Return (\%)', fontsize=10)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

# Create custom handles for the legend
strategy_line = mlines.Line2D([], [], color=custom_blue, label='Strategy')
spx_line = mlines.Line2D([], [], color='black', label='SPX')
equal_weight_line = mlines.Line2D([], [], color=custom_red, label='Constant-Mix')

# Place the legend below the plot
plt.legend(handles=[strategy_line, spx_line, equal_weight_line], loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3, fontsize=10)

plt.tight_layout()  # Adjust the rect to make room for the title and the legend
plt.savefig("singlePerf2.jpg", dpi=300, bbox_inches='tight')  # Save as high-resolution for inclusion in thesis
plt.show()



# Function to calculate drawdown
def calculate_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown * 100  # Convert to percentage

# Custom colors for the plot
custom_blue = (0.2, 0.4, 0.8)  # Blue-like color
custom_red = (0.8, 0.2, 0.4)  # Red-like color

# Extract the specific strategy data and calculate drawdown
specific_key = ('CVaR', 'RW', '95%')
strategy_title = ', '.join(specific_key).replace('%', '\\%') 
strategy_cumulative = strategy_performance[specific_key]
strategy_drawdown = calculate_drawdown(strategy_cumulative+1)

# Calculate drawdown for SPX and Equal-Weight
spx_cumulative = np.cumprod(1*returns[71:] + 0*(np.exp(annual_risk_free_rate*date_differences) - 1) + 1) - 1
spx_drawdown = calculate_drawdown(spx_cumulative+1)

equal_weight_cumulative = np.cumprod(0.7*returns[71:] + 0.3*(np.exp(annual_risk_free_rate*date_differences) - 1) + 1) - 1
equal_weight_drawdown = calculate_drawdown(equal_weight_cumulative+1)

# Convert the dates to a Pandas Series with datetime type
datess = pd.to_datetime(dates[72:])  # Assuming 'dates' is already defined

# Determine the common y-axis limits
min_drawdown = min(strategy_drawdown.min(), spx_drawdown.min(), equal_weight_drawdown.min())
max_drawdown = max(strategy_drawdown.max(), spx_drawdown.max(), equal_weight_drawdown.max())

# Create subplots for drawdowns
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

# Plot drawdowns
axes[0].fill_between(datess, strategy_drawdown, 0, color=custom_blue, alpha=0.5, label=f"({strategy_title})")
axes[1].fill_between(datess, spx_drawdown, 0, color='black', alpha=0.5, label="SPX")
axes[2].fill_between(datess, equal_weight_drawdown, 0, color=custom_red, alpha=0.5, label="Constant-Mix")

# Formatting the subplots
for ax in axes:
    ax.set_ylim([min_drawdown, max_drawdown])  # Set the same y-axis scale for all subplots
    ax.set_ylabel('Drawdown (\%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax.legend(loc='lower left')

axes[0].set_title('TMP Drawdown')
axes[1].set_title('SPX Drawdown')
axes[2].set_title('Constant-Mix Drawdown')

# Set common xlabel
axes[-1].set_xlabel('Date')

plt.tight_layout()  # Adjust the layout
plt.savefig("drawdowns_subplot2.jpg", dpi=300, bbox_inches='tight')  # Save as high-resolution for inclusion in thesis
plt.show()














def calculate_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown * 100  # Convert to percentage

# Custom colors for the plot
custom_blue = (0.2, 0.4, 0.8)  # Blue-like color
custom_red = (0.8, 0.2, 0.4)  # Red-like color

# SPX and Equal-Weight Drawdown Calculation
spx_cumulative = np.cumprod(1*returns[71:] + 0*(np.exp(annual_risk_free_rate*date_differences) - 1) + 1) - 1
spx_drawdown = calculate_drawdown(spx_cumulative+1)
equal_weight_cumulative = np.cumprod(0.7*returns[71:] + 0.3*(np.exp(annual_risk_free_rate*date_differences) - 1) + 1) - 1
equal_weight_drawdown = calculate_drawdown(equal_weight_cumulative+1)

# Convert the dates to a Pandas Series with datetime type
datess = pd.to_datetime(dates[72:])  # Assuming 'dates' is already defined

# Set up subplots - adjust the subplot grid based on the number of strategies
num_strategies = len(strategy_performance)
fig, axes = plt.subplots(num_strategies + 2, 1, figsize=(15, num_strategies * 2 + 4), sharex=True)

# Iterate through each strategy and plot its drawdown
for i, (strategy, performance) in enumerate(strategy_performance.items()):
    strategy_drawdown = calculate_drawdown(performance + 1)
    axes[i].fill_between(datess, strategy_drawdown, 0, color=custom_blue, alpha=0.5, label=strategy)
    axes[i].set_title(f"{strategy}", fontsize=14)
    axes[i].set_ylabel('Drawdown (%)')
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    axes[i].legend(loc='lower left')

# Plot SPX and Constant Mix in the last subplots
axes[-2].fill_between(datess, spx_drawdown, 0, color='black', alpha=0.5, label="SPX")
axes[-2].set_title('SPX Drawdown')
axes[-2].set_ylabel('Drawdown (%)')
axes[-2].legend(loc='lower left')

axes[-1].fill_between(datess, equal_weight_drawdown, 0, color=custom_red, alpha=0.5, label="Constant-Mix")
axes[-1].set_title('Constant-Mix Drawdown')
axes[-1].set_ylabel('Drawdown (%)')
axes[-1].legend(loc='lower left')

# Set common xlabel for the last subplot
axes[-1].set_xlabel('Date')

plt.tight_layout()  # Adjust the layout
plt.savefig("all_strategy_drawdowns_subplot.jpg", dpi=300, bbox_inches='tight')  # Save as high-resolution for inclusion in thesis
plt.show()