from data_processing import load_data, preprocess_data
from analysis import get_RNDs, fit_RWDs, run_strategies
from visualization import plot_datasets, plot_distrs, plot_errors, plot_strats

def main():
    df, rf, sp500 = load_data()
    df, rf, sp500 = preprocess_data(df, rf, sp500)
    plot_datasets(df, rf, sp500)
    
    distr, stats = get_RNDs(df)
    plot_distrs(df,distr,stats)
    
    rw_distr, z_ts , z_ts_rw = fit_RWDs(distr, df)
    plot_errors(distr, rw_distr, df,  z_ts , z_ts_rw)
    
    strategies, strategy_performance, strategy_performance_non_cum = run_strategies(distr,rw_distr,df,stats)
    plot_strats(strategies, strategy_performance, strategy_performance_non_cum,df,df)

    
if __name__ == "__main__":
    main()