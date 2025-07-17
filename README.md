# Stock Market Data Collector

A comprehensive Python tool for collecting, analyzing, and visualizing stock market data. Features include historical data collection, daily updates, correlation analysis, autocorrelation studies, and interactive visualizations.

## 🚀 Initial Setup

###1irtual Environment
```bash
# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Test Installation
```bash
# Verify Python and dependencies are working
python -cimport yfinance, plotly, pandas, numpy, scipy; print('✅ All dependencies installed successfully!')"
```

### 4. Prepare Your Data
```bash
# Edit tickers.txt to add your stock symbols (one per line)
# Example tickers.txt:
AAPL
MSFT
GOOGL
AMZN
TSLA
```

## 📊 Available Actions

### 1. **Initialize Historical Data** (`init`)
Downloads 2 years of historical data for all tickers in `tickers.txt`.

```bash
# Download historical data for all tickers
python stock_data.py --action init

# Custom tickers file and results directory
python stock_data.py --action init --tickers-file my_tickers.txt --results-dir my_results
```

**Output**: Creates individual `.txt` files in `results/` directory with format:
```
timestamp_ms    MM-DD-YYYY    closing_price
```

### 2 **Daily Updates** (`update`)
Appends today's stock data to existing files. If a ticker file doesn't exist, it downloads historical data first.

```bash
# Update all tickers with today's data
python stock_data.py --action update

# Custom parameters
python stock_data.py --action update --date-delimiter "," --row-delimiter "\r\n
```

**Output**: Appends new daily data to existing files in `results/` directory.

### 3. **Daily Price Differences** (`diff`)
Calculates daily price differences and percentage changes for all tickers.

```bash
# Generate daily differences report
python stock_data.py --action diff

# Custom parameters
python stock_data.py --action diff --date-delimiter |" --years-back 1*Output**: Creates `diffday.txt` with columns:
```
Ticker    Today_Price    Yesterday_Price    Difference    Percent_Change
```

### 4. **Interactive Visualizations** (`visualize`)
Creates interactive HTML charts for price trends and rate of change analysis.

```bash
# Generate all visualizations
python stock_data.py --action visualize

# Custom parameters
python stock_data.py --action visualize --results-dir custom_results
```

**Output Structure**:
```
results/visualizations/
├── stock_price_over_time/
│   ├── AAPL_price_chart.html
│   ├── MSFT_price_chart.html
│   └── ...
├── rate_of_change/
│   ├── AAPL_rate_of_change.html
│   ├── MSFT_rate_of_change.html
│   └── ...
├── all_tickers_comparison.html
└── all_tickers_rate_comparison.html
```

**Features**:
- Interactive zoom and pan
- Hover tooltips with exact values
- Export options (PNG, SVG, HTML)
- Statistical summaries on charts

### 5. **Correlation Analysis** (`correlate`)
Performs comprehensive correlation analysis including autocorrelations and cross-correlations.

```bash
# Run full correlation analysis
python stock_data.py --action correlate

# Custom parameters
python stock_data.py --action correlate --years-back 3
```

**Output Structure**:
```
results/correlation_analysis/
├── AAPL_autocorrelation.html
├── MSFT_autocorrelation.html
├── AAPL_MSFT_crosscorrelation.html
├── correlation_matrix.html
├── correlation_matrix.csv
└── correlation_summary.txt
```

**Analysis Includes**:
- **Autocorrelation**: How today's returns correlate with past returns
- **Cross-correlation**: Lead-lag relationships between stock pairs
- **Correlation Matrix**: Heatmap of all stock pair correlations
- **Statistical Summary**: Max, min, average correlations

### 6*Time-Windowed Autocorrelation** (`autocorr`)
Analyzes autocorrelation for a specific stock over the last k months.

```bash
# Analyze AAPL over last6 months (default)
python stock_data.py --action autocorr --ticker AAPL

# Analyze TSLA over last 3 months
python stock_data.py --action autocorr --ticker TSLA --months-back3 Analyze MSFT over last 12 months
python stock_data.py --action autocorr --ticker MSFT --months-back 12
```

**Output Structure**:
```
results/autocorrelation_windows/
├── AAPL_6months_autocorrelation.html
├── AAPL_6months_autocorrelation.csv
└── ...
```

**Features**:
- Interactive plots with confidence intervals
- Statistical interpretation (momentum vs. mean reversion)
- CSV export for further analysis
- Console summary with insights

## 🔧 Command Line Options

### Global Options
```bash
--tickers-file FILE     # File containing ticker symbols (default: tickers.txt)
--results-dir DIR       # Directory to store results (default: results)
--date-delimiter DELIM  # Delimiter between date and price (default: tab)
--row-delimiter DELIM   # Delimiter between rows (default: newline)
--years-back YEARS      # Years of historical data to fetch (default: 2
```

### Action-Specific Options
```bash
--action ACTION         # Action to perform (init|update|diff|visualize|correlate|autocorr)
--ticker SYMBOL        # Specific ticker for autocorrelation analysis
--months-back MONTHS   # Number of months for autocorrelation (default: 6 📁 File Structure

```
stock-example/
├── README.md
├── requirements.txt
├── stock_data.py
├── tickers.txt
├── results/
│   ├── AAPL.txt
│   ├── MSFT.txt
│   ├── visualizations/
│   │   ├── stock_price_over_time/
│   │   ├── rate_of_change/
│   │   └── ...
│   ├── correlation_analysis/
│   └── autocorrelation_windows/
├── diffday.txt
└── stock_data.log
```

## 📈 Data Format

### Stock Data Files (`results/TICKER.txt`)
```
timestamp_ms    MM-DD-YYYY    closing_price
170467200 1/12024150.251704153600 1/224151.3...
```

### Daily Differences (`diffday.txt`)
```
Ticker    Today_Price    Yesterday_Price    Difference    Percent_Change
AAPL      190.1218850    10.62      0.86%
MSFT      420.541875    10.75       042...
```

## 🎯 Use Cases

### **Portfolio Management**
```bash
# Initialize data for your portfolio
python stock_data.py --action init

# Daily updates
python stock_data.py --action update

# Weekly correlation analysis
python stock_data.py --action correlate
```

### **Trading Strategy Research**
```bash
# Analyze momentum vs. mean reversion
python stock_data.py --action autocorr --ticker AAPL --months-back 3re stock behaviors
python stock_data.py --action visualize
```

### **Market Analysis**
```bash
# Daily market summary
python stock_data.py --action diff

# Sector correlation study
python stock_data.py --action correlate
```

## 🔍 Troubleshooting

### Common Issues

1. **No module named 'yfinance"**
   ```bash
   pip install -r requirements.txt
   ```

2*"Tickers file not found** - Ensure `tickers.txt` exists in the project directory
   - Check file permissions

3Insufficient data for autocorrelation"**
   - Ensure you have at least30 days of data
   - Run `--action init` first to download historical data4*Interactive plots not opening**
   - Open HTML files in a web browser
   - Ensure JavaScript is enabled

### Logging
All actions create detailed logs in `stock_data.log` for debugging.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different tickers and time periods
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Stock Analysis! 📊📈** 