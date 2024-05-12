from data_processing import load_data, preprocess_data
from analysis import compute_statistics, run_models
from visualization import plot_data

def main():
    df, rf, sp500 = load_data()
    df = preprocess_data(df, rf, sp500)
    stats = compute_statistics(df)
    run_models(df)
    plot_data(df)

if __name__ == "__main__":
    main()