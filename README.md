# Tail Risk-Managed Portfolios

This GitHub repository contains the Python code accompanying the master's thesis titled "Tail Risk-Managed Portfolios" by Gwenael-Theo Münker. The thesis explores the construction and performance evaluation of portfolios managed to mitigate tail risks using Value at Risk (VaR) and Conditional Value at Risk (CVaR) metrics derived from option-implied probability measures.

## Repository Structure

- `README.md` - This file, explaining the project and navigation.
- `risk_analysis.py` - Main Python script containing all the functions and logic for data processing, RND calculation, and portfolio management strategies.
- `data/` - Directory containing the datasets used (note: due to privacy and data sharing policies, actual data files are not included in this repository).

## Prerequisites

To run the scripts, ensure you have Python 3.8 or later installed, along with the following packages:
- `pandas` for data manipulation,
- `numpy` for numerical calculations,
- `matplotlib` for plotting graphs,
- `scipy` for optimization functions,
- `cvxpy` along with the `MOSEK` solver for optimization problems.

You can install the necessary Python packages using pip:

```bash
pip install pandas numpy matplotlib scipy cvxpy mosek
