
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# To match LaTeX style of thesis
plt.style.use('seaborn-whitegrid')
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern'

def plot_datasets(df, rf, sp500):

    def plot_sp500():
        plt.figure(figsize=(8, 4))  
        plt.plot(df['date'], df['spindx'], linestyle='-', color='black')
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Index Level', fontsize=10)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, interval=15))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        plt.gcf().autofmt_xdate()
        plt.savefig('sp500.jpg', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_sp500_ret():
        plt.figure(figsize=(8, 4))  
        plt.plot(df['date'], df['returns']*100, linestyle='-', color='black')  
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Return (\%)', fontsize=10)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, interval=15))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        plt.gcf().autofmt_xdate()
        plt.savefig('sp500_rets.jpg', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_dates():
        sorted_dates = np.sort(df['date'].unique())
        date_range = pd.date_range(start=sorted_dates.min(), end=sorted_dates.max(), freq='D')
        date_df = pd.DataFrame(date_range, columns=['Date'])
        date_df['EventCount'] = date_df['Date'].apply(lambda x: np.sum(sorted_dates == x))
        date_df['CumulativeEvents'] = date_df['EventCount'].cumsum()
        plt.figure(figsize=(8, 4))
        plt.plot(date_df['Date'], date_df['CumulativeEvents'], marker=',', linestyle='-', color='black') 
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Number of Dates', fontsize=10)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, interval=15))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        plt.gcf().autofmt_xdate()
        plt.savefig('cum_event.jpg', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_rf():
        plt.figure(figsize=(8, 4))
        plt.plot(df['date'], df['rate']*100, linestyle='-', color='black') 
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Risk-Free Rate (\%)', fontsize=10)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, interval=15))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        plt.gcf().autofmt_xdate()
        plt.savefig('rf.jpg', dpi=300, bbox_inches='tight')
        plt.show()

    # Generate Plots
    plot_sp500() 
    plot_sp500_ret()
    plot_dates()
    plot_rf()

def plot_distrs(df,distr,stats):
    
    dates = df['date'].unique()
    w = 0.5 + 0.002 * np.arange(501)
    
    def plot_options_over_time():
        plt.figure(figsize=(8, 4))  
        calls = stats.iloc[0].values  
        puts = stats.iloc[1].values  # 
        total = calls + puts  # Total volume for area plot base
        plt.fill_between(dates, 0, calls, color=(0.2, 0.4, 0.8), alpha=0.7, label='Call Options')
        plt.fill_between(dates, calls, total, color=(0.8, 0.2, 0.4), alpha=0.7, label='Put Options')
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Number of Options', fontsize=10)
        plt.ylim(0, 17)  
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()  
        plt.legend(fontsize=10) 
        plt.tight_layout()  
        plt.savefig('options_area_plot_adjusted.jpg', dpi=300, bbox_inches='tight')
        plt.show()

    # adjust for out-of-sample
    distr_a = distr.iloc[:,:71].copy()
    #distr_a.columns = [int(col) - 71 for col in distr_a.columns]  # Convert to int and then subtract
    stats_a = stats.iloc[:,:71].copy()
    #stats_a.columns = [int(col) - 71 for col in stats_a.columns]  # Convert to int and then subtract

    def density_surface(dens,name):
        datess = pd.Series(dates[71:])
        # Generate date labels with only month and year
        date_labels = datess.dt.strftime('%m-%Y').tolist()
        time_numeric = np.arange(len(date_labels))
        Z = (dens*100).values
        Y, X = np.meshgrid(time_numeric, (w-1)*100)  # Now Y is time, X is return percentages
        Z_masked = np.where((X >= -40) & (X <= 40), Z, np.nan)
        colors = [(0.2, 0.4, 0.8), (0.8, 0.2, 0.4)]  # Example colors, adjust as needed
        cmap_name = 'custom_cmap'
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z_masked, cmap=custom_cmap, edgecolor='none')
        ax.set_xlabel('Return (\%)', fontsize=14)
        ax.set_ylabel('Date', fontsize=14)
        ax.set_zlabel('Density (\%)', fontsize=14)
        ax.set_xlim(-40, 40)
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


    def sumstats(df):
        metrics_df = pd.DataFrame(columns=['Distribution', 'Mean', 'Volatility', 'Skewness', 'Kurtosis'])
        log_w = (w-1)*100
        for column in df.columns: 
            P = df[column]
            mean = np.sum(P * log_w)
            volatility = np.sqrt(np.sum(P * (log_w - mean)**2))
            skewness = (np.sum(P * (log_w - mean)**3)) / (volatility**3)
            kurtosis = (np.sum(P * (log_w - mean)**4)) / (volatility**4)
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

    plot_options_over_time()
    density_surface(distr_a,"3dRND")
    
    
    def plot_errors(distr, rw_distr, df,  z_ts , z_ts_rw):
        
        def plot_standard_normal_quantiles_refined():
            datess = pd.Series(dates[:71])  
            fig, ax1 = plt.subplots(figsize=(8, 4))
            color1 = (0.2, 0.4, 0.8) 
            color2 = (0.8, 0.2, 0.4)  
            # Plot RWD Quantiles
            line1, = ax1.plot(datess, z_ts_rw, color=color1, label='RWD Quantiles')
            ax1.set_xlabel('Date', color='black', fontsize=10)
            ax1.set_ylabel('RWD Quantiles', color='black', fontsize=10)
            ax1.tick_params(axis='y', labelcolor='black', labelsize=10)
            ax1.tick_params(axis='x', labelcolor='black', labelsize=10)
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
            ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=1, interval=10))
            # Plot Difference (z_ts_rw - z_ts)
            ax2 = ax1.twinx()
            line2, = ax2.plot(datess, z_ts_rw - z_ts, linestyle='dotted', color=color2, linewidth=3, label='Difference RWD vs. RND Quantiles')
            ax2.set_ylabel('Quantile Difference', color='black', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='black', labelsize=10)
            plt.legend(handles=[line1, line2], loc='upper center', fontsize=10, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)
            fig.tight_layout()
            plt.setp(ax1.xaxis.get_minorticklabels(), rotation=45)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.savefig("in_of_sample_error", dpi=300, bbox_inches='tight')
            plt.show()
        
        plot_standard_normal_quantiles_refined()


def plot_strats(strategies, strategy_performance, strategy_performance_non_cum,df,stats):
    
    stats_a = stats.iloc[:,:71].copy()
    dates = df['date'].unique() 
    date_differences = np.diff(dates[71:]).astype('timedelta64[D]')
    # Convert to an array of integers representing the number of days
    date_differences = date_differences / np.timedelta64(1, 'D')
    date_differences.tolist()
    date_differences = date_differences/365
    annual_risk_free_rate = stats_a.iloc[2, :-1]
    
    returns = df['returns_strat'].unique()[2:]
    custom_blue = (0.2, 0.4, 0.8)  # Blue-like color
    custom_red = (0.8, 0.2, 0.4)  # Red-like color
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
        strategy_title = str((measure_type, density_type, conf_level)).replace('%', '\\%')    
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

