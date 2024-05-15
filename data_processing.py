
import pandas as pd
import numpy as np

def load_data():
    """Load data."""
    
    # Load Option Data
    df = pd.read_csv("jvossrlke6ddmel8.csv")
    df = df.drop(columns=["optionid","index_flag","issuer"])
    df['strike_price'] = df['strike_price']/1000
    df['price'] = (df['best_bid'] + df['best_offer'])/2

    # Load Zero Coupon Data
    rf = pd.read_csv('xl4ajfq0nrwzgqq6.csv')
    rf = rf.drop(rf[rf.days>100].index)
    rf['rate'] = rf['rate']/100

    # Load SPX price data
    sp500 = pd.read_csv('golvjl9eyxaeqz4r.csv')
    sp500.rename(columns={"caldt": "date"},inplace=True)

    return df, rf, sp500

def preprocess_data(df, rf, sp500):
    """Preprocess data by interpolating rates and merging datasets."""
    # Interpolate 30-day maturity rate
    def interpolate_30_day_rate(group):
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


    rf_grouped = rf.groupby('date')
    interpolated_rates = pd.concat([interpolate_30_day_rate(group) for _, group in rf_grouped])
    
    df = pd.merge(df, interpolated_rates, on='date', how='left')
    df = pd.merge(df,sp500,on='date',how='left')
    
    #get price at maturity
    sp500ex = sp500.copy()
    sp500ex.rename(columns={"date": "exdate","spindx":"spindxex"},inplace=True)
    sp500ex['exdate'] =  pd.to_datetime(sp500ex['exdate'])
    df['exdate'] = pd.to_datetime(df['exdate'])
    df = pd.merge_asof(df,sp500ex,on='exdate',direction='forward')
    df['date'] = pd.to_datetime(df['date'])

    # Adjust for maturities
    sp500ex = sp500.copy()
    sp500ex.rename(columns={"date": "exdate", "spindx": "spindxex"}, inplace=True)
    sp500ex['exdate'] = pd.to_datetime(sp500ex['exdate'])
    df['exdate'] = pd.to_datetime(df['exdate'])
    df = pd.merge_asof(df, sp500ex, on='exdate', direction='forward')
    df['date'] = pd.to_datetime(df['date'])
    
    def filter_dates(df):
        df_dates = pd.DataFrame(df['date'].copy())
        
        date_counts = df_dates['date'].value_counts()
        
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
    
    df['returns'] = df['spindxex']/df['spindx']-1 
    df['returns_strat'] = df['spindx'].pct_change() 

    return df, rf, sp500
