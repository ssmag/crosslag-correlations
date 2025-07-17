#!/usr/bin/env python3
"""
Stock Market Data Collector

This script pulls historical and daily stock market data for tickers specified in tickers.txt.
It creates files in the results folder with date (milliseconds) and closing price data.

Author: Stock Data Collector
"""

import os
import sys
import yfinance as yf
from datetime import datetime, timedelta
import argparse
from typing import List, Tuple, Optional
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import correlate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_data.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class StockDataCollector:
    def __init__(self, 
                 tickers_file: str = "tickers.txt",
                 results_dir: str = "results",
                 date_delimiter: str = "\t",
                 row_delimiter: str = "\n",
                 years_back: int = 2):
        """
        Initialize the StockDataCollector.
        Args:
            tickers_file: Path to file containing ticker symbols
            results_dir: Directory to store results
            date_delimiter: Delimiter between date and price in each row
            row_delimiter: Delimiter between rows
            years_back: Number of years of historical data to fetch
        """
        self.tickers_file = tickers_file
        self.results_dir = results_dir
        self.date_delimiter = date_delimiter
        self.row_delimiter = row_delimiter
        self.years_back = years_back
        os.makedirs(self.results_dir, exist_ok=True)

    def read_tickers(self) -> List[str]:
        """Read ticker symbols from tickers.txt file."""
        try:
            with open(self.tickers_file, 'r') as f:
                tickers = [line.strip().upper() for line in f if line.strip()]
            logger.info(f"Loaded {len(tickers)} tickers from {self.tickers_file}")
            return tickers
        except FileNotFoundError:
            logger.error(f"Tickers file {self.tickers_file} not found!")
            return []
        except Exception as e:
            logger.error(f"Error reading tickers file: {e}")
            return []

    def get_historical_data(self, ticker: str) -> Optional[List[Tuple[int, str, float]]]:
        """
        Fetch historical data for a ticker.
        Args:
            ticker: Stock ticker symbol
        Returns:
            List of tuples (timestamp_ms, date_str, closing_price) or None if error
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * self.years_back)
            logger.info(f"Fetching historical data for {ticker} from {start_date.date()} to {end_date.date()}")
            stock = yf.Ticker(ticker)
            hist_data = stock.history(start=start_date, end=end_date)
            if hist_data.empty:
                logger.warning(f"No data found for ticker {ticker}")
                return None
            data = []
            for date, row in hist_data.iterrows():
                timestamp_ms = int(date.timestamp() * 1000)
                date_str = date.strftime("%m-%d-%Y")
                closing_price = row['Close']
                data.append((timestamp_ms, date_str, closing_price))
            logger.info(f"Retrieved {len(data)} data points for {ticker}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    def get_daily_data(self, ticker: str) -> Optional[Tuple[int, str, float]]:
        """
        Fetch today's data for a ticker.
        Args:
            ticker: Stock ticker symbol
        Returns:
            Tuple (timestamp_ms, date_str, closing_price) or None if error
        """
        try:
            logger.info(f"Fetching daily data for {ticker}")
            stock = yf.Ticker(ticker)
            today_data = stock.history(period="1d")
            if today_data.empty:
                logger.warning(f"No daily data found for ticker {ticker}")
                return None
            latest_date = today_data.index[-1]
            timestamp_ms = int(latest_date.timestamp() * 1000)
            date_str = latest_date.strftime("%m-%d-%Y")
            closing_price = today_data.iloc[-1]['Close']
            logger.info(f"Retrieved daily data for {ticker}: {closing_price}")
            return (timestamp_ms, date_str, closing_price)
        except Exception as e:
            logger.error(f"Error fetching daily data for {ticker}: {e}")
            return None

    def write_data_to_file(self, ticker: str, data: list, mode: str = 'w'):
        """
        Write data to a file for the given ticker.
        Args:
            ticker: Stock ticker symbol
            data: List of (timestamp_ms, date_str, closing_price) tuples
            mode: File write mode ('w' for overwrite, 'a' for append)
        """
        filename = os.path.join(self.results_dir, f"{ticker}.txt")
        try:
            with open(filename, mode) as f:
                for timestamp_ms, date_str, closing_price in data:
                    line = f"{timestamp_ms}{self.date_delimiter}{date_str}{self.date_delimiter}{closing_price}{self.row_delimiter}"
                    f.write(line)
            logger.info(f"Wrote {len(data)} data points to {filename}")
        except Exception as e:
            logger.error(f"Error writing data for {ticker}: {e}")

    def append_daily_data(self, ticker: str, daily_data: Tuple[int, str, float]):
        """
        Append a single daily data point to a ticker's file.
        Args:
            ticker: Stock ticker symbol
            daily_data: Tuple (timestamp_ms, date_str, closing_price)
        """
        self.write_data_to_file(ticker, [daily_data], mode='a')

    def file_exists(self, ticker: str) -> bool:
        """Check if a file exists for the given ticker."""
        filename = os.path.join(self.results_dir, f"{ticker}.txt")
        return os.path.exists(filename)

    def initialize_all_tickers(self):
        """
        Download historical data for all tickers in tickers.txt.
        """
        tickers = self.read_tickers()
        if not tickers:
            logger.error("No tickers found. Please check your tickers.txt file.")
            return
        logger.info(f"Initializing historical data for {len(tickers)} tickers...")
        for ticker in tickers:
            if self.file_exists(ticker):
                logger.info(f"File already exists for {ticker}, skipping...")
                continue
            data = self.get_historical_data(ticker)
            if data:
                self.write_data_to_file(ticker, data)
            else:
                logger.error(f"Failed to get data for {ticker}")
        logger.info("Initialization complete!")

    def update_all_tickers(self):
        """
        Update all tickers with today's data. If a ticker file does not exist, download historical data first.
        """
        tickers = self.read_tickers()
        if not tickers:
            logger.error("No tickers found. Please check your tickers.txt file.")
            return
        logger.info(f"Updating daily data for {len(tickers)} tickers...")
        for ticker in tickers:
            daily_data = self.get_daily_data(ticker)
            if daily_data:
                if self.file_exists(ticker):
                    self.append_daily_data(ticker, daily_data)
                else:
                    logger.info(f"File not found for {ticker}, getting historical data first...")
                    hist_data = self.get_historical_data(ticker)
                    if hist_data:
                        self.write_data_to_file(ticker, hist_data)
                        self.append_daily_data(ticker, daily_data)
            else:
                logger.error(f"Failed to get daily data for {ticker}")
        logger.info("Daily update complete!")

    def calculate_daily_differences(self):
        """
        Calculate daily price differences for all tickers and write to diffday.txt.
        Format: ticker, today_price, yesterday_price, difference, percent_change
        """
        tickers = self.read_tickers()
        if not tickers:
            logger.error("No tickers found. Please check your tickers.txt file.")
            return
        
        logger.info(f"Calculating daily differences for {len(tickers)} tickers...")
        
        differences = []
        for ticker in tickers:
            filename = os.path.join(self.results_dir, f"{ticker}.txt")
            if not os.path.exists(filename):
                logger.warning(f"No data file found for {ticker}, skipping...")
                continue
            
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) < 2:
                    logger.warning(f"Insufficient data for {ticker}, need at least 2 days")
                    continue
                
                # Get the last two lines (most recent data)
                last_line = lines[-1].strip()
                second_last_line = lines[-2].strip()
                
                # Parse the data (timestamp, date, price)
                last_parts = last_line.split(self.date_delimiter)
                second_last_parts = second_last_line.split(self.date_delimiter)
                
                if len(last_parts) < 3 or len(second_last_parts) < 3:
                    logger.warning(f"Invalid data format for {ticker}")
                    continue
                
                today_price = float(last_parts[2])
                yesterday_price = float(second_last_parts[2])
                difference = today_price - yesterday_price
                percent_change = (difference / yesterday_price * 100) if yesterday_price != 0 else 0.0
                
                differences.append((ticker, today_price, yesterday_price, difference, percent_change))
                logger.info(f"{ticker}: Today=${today_price:.2f}, Yesterday=${yesterday_price:.2f}, Diff=${difference:.2f}, %Change={percent_change:.2f}%")
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
        
        # Write differences to diffday.txt
        try:
            with open('diffday.txt', 'w') as f:
                f.write(f"Ticker{self.date_delimiter}Today_Price{self.date_delimiter}Yesterday_Price{self.date_delimiter}Difference{self.date_delimiter}Percent_Change{self.row_delimiter}")
                for ticker, today_price, yesterday_price, difference, percent_change in differences:
                    line = f"{ticker}{self.date_delimiter}{today_price:.2f}{self.date_delimiter}{yesterday_price:.2f}{self.date_delimiter}{difference:.2f}{self.date_delimiter}{percent_change:.2f}%{self.row_delimiter}"
                    f.write(line)
            
            logger.info(f"Wrote {len(differences)} daily differences to diffday.txt")
            
        except Exception as e:
            logger.error(f"Error writing diffday.txt: {e}")

    def visualize_all_tickers(self):
        """
        Create matplotlib visualizations for all tickers in the results directory.
        Generates price charts and rate of change charts, saved in separate subdirectories.
        """
        tickers = self.read_tickers()
        if not tickers:
            logger.error("No tickers found. Please check your tickers.txt file.")
            return
        
        logger.info(f"Creating visualizations for {len(tickers)} tickers...")
        
        # Create directories for different types of visualizations
        viz_dir = os.path.join(self.results_dir, "visualizations")
        price_dir = os.path.join(viz_dir, "stock_price_over_time")
        rate_dir = os.path.join(viz_dir, "rate_of_change")
        
        os.makedirs(price_dir, exist_ok=True)
        os.makedirs(rate_dir, exist_ok=True)
        
        for ticker in tickers:
            filename = os.path.join(self.results_dir, f"{ticker}.txt")
            if not os.path.exists(filename):
                logger.warning(f"No data file found for {ticker}, skipping...")
                continue
            
            try:
                # Read data from file
                data = []
                with open(filename, 'r') as f:
                    for line in f:
                        parts = line.strip().split(self.date_delimiter)
                        if len(parts) >= 3:
                            timestamp_ms = int(parts[0])
                            date_str = parts[1]
                            price = float(parts[2])
                            # Convert date string to datetime
                            date_obj = datetime.strptime(date_str, "%m-%d-%Y")
                            data.append((date_obj, price))
                
                if not data:
                    logger.warning(f"No valid data found for {ticker}")
                    continue
                
                # Sort data by date
                data.sort(key=lambda x: x[0])
                dates = [item[0] for item in data]
                prices = [item[1] for item in data]
                
                # Create the price over time plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers', name=ticker))
                fig.update_layout(
                    title=f'{ticker} Stock Price Over Time',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    hovermode='x unified'
                )
                fig.update_xaxes(
                    tickformat='%m/%d/%Y',
                    tickangle=45
                )
                fig.update_yaxes(
                    tickformat='$.2f'
                )
                
                # Add price statistics
                min_price = min(prices)
                max_price = max(prices)
                current_price = prices[-1]
                price_change = prices[-1] - prices[0]
                price_change_pct = (price_change / prices[0] * 100) if prices[0] != 0 else 0
                
                stats_text = f'Current: ${current_price:.2f}\nMin: ${min_price:.2f}\nMax: ${max_price:.2f}\nChange: ${price_change:.2f} ({price_change_pct:.1f}%)'
                fig.add_annotation(
                    x=0.02, y=0.98,
                    text=stats_text,
                    showarrow=False,
                    xref='paper', yref='paper',
                    xanchor='left', yanchor='top',
                    bgcolor='wheat', bordercolor='wheat', borderwidth=1,
                    font=dict(size=12)
                )
                
                # Save the price plot
                plot_filename = os.path.join(price_dir, f"{ticker}_price_chart.html")
                fig.write_html(plot_filename)
                logger.info(f"Created price chart for {ticker}: {plot_filename}")
                
                # Calculate rate of change (derivative)
                if len(prices) > 1:
                    # Calculate daily rate of change
                    rate_of_change = []
                    rate_dates = []
                    
                    for i in range(1, len(prices)):
                        # Calculate daily percentage change
                        daily_change = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
                        rate_of_change.append(daily_change)
                        rate_dates.append(dates[i])
                    
                    # Create rate of change plot
                    fig_rate = go.Figure()
                    fig_rate.add_trace(go.Scatter(x=rate_dates, y=rate_of_change, mode='lines+markers', name=ticker, line=dict(color='red')))
                    fig_rate.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.3)
                    fig_rate.update_layout(
                        title=f'{ticker} Rate of Change (Daily % Change)',
                        xaxis_title='Date',
                        yaxis_title='Daily % Change',
                        hovermode='x unified'
                    )
                    fig_rate.update_xaxes(
                        tickformat='%m/%d/%Y',
                        tickangle=45
                    )
                    fig_rate.update_yaxes(
                        tickformat='.2f'
                    )
                    
                    # Add rate of change statistics
                    max_change = max(rate_of_change)
                    min_change = min(rate_of_change)
                    avg_change = sum(rate_of_change) / len(rate_of_change)
                    current_rate = rate_of_change[-1] if rate_of_change else 0
                    
                    stats_text = f'Current: {current_rate:.2f}%\nMax: {max_change:.2f}%\nMin: {min_change:.2f}%\nAvg: {avg_change:.2f}%'
                    fig_rate.add_annotation(
                        x=0.02, y=0.98,
                        text=stats_text,
                        showarrow=False,
                        xref='paper', yref='paper',
                        xanchor='left', yanchor='top',
                        bgcolor='lightcoral', bordercolor='lightcoral', borderwidth=1,
                        font=dict(size=12)
                    )
                    
                    # Save the rate of change plot
                    rate_filename = os.path.join(rate_dir, f"{ticker}_rate_of_change.html")
                    fig_rate.write_html(rate_filename)
                    logger.info(f"Created rate of change chart for {ticker}: {rate_filename}")
                
            except Exception as e:
                logger.error(f"Error creating visualization for {ticker}: {e}")
                continue
        
        # Create summary comparison charts
        try:
            # Price comparison chart
            fig_comparison = make_subplots(rows=1, cols=1)
            for ticker in tickers:
                filename = os.path.join(self.results_dir, f"{ticker}.txt")
                if not os.path.exists(filename):
                    continue
                
                data = []
                with open(filename, 'r') as f:
                    for line in f:
                        parts = line.strip().split(self.date_delimiter)
                        if len(parts) >= 3:
                            date_str = parts[1]
                            price = float(parts[2])
                            date_obj = datetime.strptime(date_str, "%m-%d-%Y")
                            data.append((date_obj, price))
                
                if data:
                    data.sort(key=lambda x: x[0])
                    dates = [item[0] for item in data]
                    prices = [item[1] for item in data]
                    
                    # Normalize prices to start at 100
                    if prices:
                        normalized_prices = [p / prices[0] * 100 for p in prices]
                        fig_comparison.add_trace(go.Scatter(x=dates, y=normalized_prices, mode='lines+markers', name=ticker, line=dict(color=px.colors.qualitative.Plotly[tickers.index(ticker)])))
            
            fig_comparison.update_layout(
                title='Stock Price Comparison (Normalized to 100)',
                xaxis_title='Date',
                yaxis_title='Normalized Price (Base=100)',
                hovermode='x unified'
            )
            fig_comparison.update_xaxes(
                tickformat='%m/%d/%Y',
                tickangle=45
            )
            fig_comparison.update_yaxes(
                tickformat='.2f'
            )
            fig_comparison.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            
            # Save the comparison chart
            comparison_filename = os.path.join(viz_dir, "all_tickers_comparison.html")
            fig_comparison.write_html(comparison_filename)
            logger.info(f"Created comparison chart: {comparison_filename}")
            
            # Rate of change comparison chart
            fig_rate_comparison = make_subplots(rows=1, cols=1)
            for ticker in tickers:
                filename = os.path.join(self.results_dir, f"{ticker}.txt")
                if not os.path.exists(filename):
                    continue
                
                data = []
                with open(filename, 'r') as f:
                    for line in f:
                        parts = line.strip().split(self.date_delimiter)
                        if len(parts) >= 3:
                            date_str = parts[1]
                            price = float(parts[2])
                            date_obj = datetime.strptime(date_str, "%m-%d-%Y")
                            data.append((date_obj, price))
                
                if data and len(data) > 1:
                    data.sort(key=lambda x: x[0])
                    dates = [item[0] for item in data]
                    prices = [item[1] for item in data]
                    
                    # Calculate rate of change
                    rate_of_change = []
                    rate_dates = []
                    for i in range(1, len(prices)):
                        daily_change = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
                        rate_of_change.append(daily_change)
                        rate_dates.append(dates[i])
                    
                    if rate_of_change:
                        fig_rate_comparison.add_trace(go.Scatter(x=rate_dates, y=rate_of_change, mode='lines+markers', name=ticker, line=dict(color=px.colors.qualitative.Plotly[tickers.index(ticker)])))
            
            fig_rate_comparison.update_layout(
                title='Rate of Change Comparison (Daily % Change)',
                xaxis_title='Date',
                yaxis_title='Daily % Change',
                hovermode='x unified'
            )
            fig_rate_comparison.update_xaxes(
                tickformat='%m/%d/%Y',
                tickangle=45
            )
            fig_rate_comparison.update_yaxes(
                tickformat='.2f'
            )
            fig_rate_comparison.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.3)
            fig_rate_comparison.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            
            # Save the rate comparison chart
            rate_comparison_filename = os.path.join(viz_dir, "all_tickers_rate_comparison.html")
            fig_rate_comparison.write_html(rate_comparison_filename)
            logger.info(f"Created rate comparison chart: {rate_comparison_filename}")
            
        except Exception as e:
            logger.error(f"Error creating comparison charts: {e}")
        
        logger.info("Visualization complete! Check the 'visualizations' directory for charts.")

    def compute_autocorrelation(self, ticker: str, max_lag: int = 30) -> Optional[Tuple[List[int], List[float]]]:
        """
        Compute autocorrelation for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            max_lag: Maximum lag to compute (default: 30)
        
        Returns:
            Tuple of (lags, autocorrelations) or None if error
        """
        filename = os.path.join(self.results_dir, f"{ticker}.txt")
        if not os.path.exists(filename):
            logger.warning(f"No data file found for {ticker}")
            return None
        
        try:
            # Read price data
            prices = []
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split(self.date_delimiter)
                    if len(parts) >= 3:
                        prices.append(float(parts[2]))
            if len(prices) < max_lag + 1:
                logger.warning(f"Insufficient data for {ticker} autocorrelation")
                return None
            
            # Calculate returns (percentage change)
            returns = np.diff(prices) / prices[:-1] *100      
            # Compute autocorrelation
            lags = list(range(1, min(max_lag + 1, len(returns))))
            autocorrs = []
            
            for lag in lags:
                if lag < len(returns):
                    # Compute correlation between returns and lagged returns
                    corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                    autocorrs.append(corr)
            
            return lags, autocorrs
            
        except Exception as e:
            logger.error(f"Error computing autocorrelation for {ticker}: {e}")
            return None
    
    def compute_cross_correlation(self, ticker1: str, ticker2: str, max_lag: int = 30) -> Optional[Tuple[List[int], List[float]]]:
        """
        Compute cross-lag correlation between two tickers.
        
        Args:
            ticker1: First stock ticker symbol
            ticker2: Second stock ticker symbol
            max_lag: Maximum lag to compute (default: 30)
        
        Returns:
            Tuple of (lags, cross_correlations) or None if error
        """
        filename1 = os.path.join(self.results_dir, f"{ticker1}.txt")
        filename2 = os.path.join(self.results_dir, f"{ticker2}.txt")
        
        if not os.path.exists(filename1) or not os.path.exists(filename2):
            logger.warning(f"Data files not found for {ticker1} and {ticker2}")
            return None
        
        try:
            # Read price data for both tickers
            prices1 = []
            prices2 = []
            
            with open(filename1, 'r') as f:
                for line in f:
                    parts = line.strip().split(self.date_delimiter)
                    if len(parts) >= 3:
                        prices1.append(float(parts[2]))
            with open(filename2, 'r') as f:
                for line in f:
                    parts = line.strip().split(self.date_delimiter)
                    if len(parts) >= 3:
                        prices2.append(float(parts[2]))
            # Ensure same length
            min_length = min(len(prices1), len(prices2))
            if min_length < max_lag + 1:
                logger.warning(f"Insufficient data for cross-correlation between {ticker1} and {ticker2}")
                return None
            
            prices1 = prices1[:min_length]
            prices2 = prices2[:min_length]
            
            # Calculate returns
            returns1 = np.diff(prices1) / prices1[:-1] * 100
            returns2 = np.diff(prices2) / prices2[:-1] *100      
            # Compute cross-correlation
            lags = list(range(-max_lag, max_lag + 1))
            cross_corrs = []
            
            for lag in lags:
                if lag < 0:
                    # ticker1 leads ticker2
                    if abs(lag) < len(returns1) and len(returns2) > abs(lag):
                        corr = np.corrcoef(returns1[lag:], returns2[:-abs(lag)])[0, 1]
                        cross_corrs.append(corr)
                    else:
                        cross_corrs.append(np.nan)
                elif lag == 0:
                    # Same time
                    corr = np.corrcoef(returns1, returns2)[0, 1]
                    cross_corrs.append(corr)
                else:
                    # ticker2 leads ticker1
                    if lag < len(returns2) and len(returns1) > lag:
                        corr = np.corrcoef(returns2[lag:], returns1[:-lag])[0, 1]
                        cross_corrs.append(corr)
                    else:
                        cross_corrs.append(np.nan)
            
            return lags, cross_corrs
            
        except Exception as e:
            logger.error(f"Error computing cross-correlation between {ticker1} and {ticker2}: {e}")
            return None
    
    def analyze_correlations(self):
        """
        Analyze autocorrelations and cross-correlations for all tickers.
        Creates interactive plots and saves results to files.
        """
        tickers = self.read_tickers()
        if not tickers:
            logger.error("No tickers found. Please check your tickers.txt file.")
            return
        
        logger.info(f"Analyzing correlations for {len(tickers)} tickers...")
        
        # Create directory for correlation analysis
        corr_dir = os.path.join(self.results_dir, "correlation_analysis")
        os.makedirs(corr_dir, exist_ok=True)
        
        # Analyze autocorrelations
        autocorr_results = {}
        for ticker in tickers:
            result = self.compute_autocorrelation(ticker)
            if result:
                lags, autocorrs = result
                autocorr_results[ticker] = (lags, autocorrs)
                
                # Create autocorrelation plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=lags, y=autocorrs, mode='lines+markers', name=ticker))
                fig.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.3)
                fig.update_layout(
                    title=f'{ticker} Autocorrelation of Returns',
                    xaxis_title='Lag (days)',
                    yaxis_title='Autocorrelation',
                    hovermode='x unified',
                    yaxis=dict(range=[-1, 1])
                )
                fig.update_xaxes(tickformat='d')
                fig.update_yaxes(tickformat='.3f', range=[-1, 1])
                
                # Save plot
                plot_filename = os.path.join(corr_dir, f"{ticker}_autocorrelation.html")
                fig.write_html(plot_filename)
                logger.info(f"Created autocorrelation plot for {ticker}")
        
        # Analyze cross-correlations for all pairs
        cross_corr_results = {}
        ticker_pairs = []
        
        for i, ticker1 in enumerate(tickers):
            for ticker2 in tickers[i+1:]:
                result = self.compute_cross_correlation(ticker1, ticker2)
                if result:
                    lags, cross_corrs = result
                    cross_corr_results[(ticker1, ticker2)] = (lags, cross_corrs)
                    ticker_pairs.append((ticker1, ticker2))
                    
                    # Create cross-correlation plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=lags, y=cross_corrs, mode='lines+markers', name=f'{ticker1} vs {ticker2}'))
                    fig.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.3)
                    fig.update_layout(
                        title=f'Cross-Correlation: {ticker1} vs {ticker2}',
                        xaxis_title='Lag (days)',
                        yaxis_title='Cross-Correlation',
                        hovermode='x unified'
                    )
                    fig.update_xaxes(tickformat='d')
                    fig.update_yaxes(tickformat='.3f')
                    
                    # Save plot
                    plot_filename = os.path.join(corr_dir, f"{ticker1}_{ticker2}_crosscorrelation.html")
                    fig.write_html(plot_filename)
                    logger.info(f"Created cross-correlation plot for {ticker1} vs {ticker2}")
        
        # Create summary correlation matrix
        if len(tickers) > 1:
            try:
                # Read all price data
                price_data = {}
                for ticker in tickers:
                    filename = os.path.join(self.results_dir, f"{ticker}.txt")
                    if os.path.exists(filename):
                        prices = []
                        with open(filename, 'r') as f:
                            for line in f:
                                parts = line.strip().split(self.date_delimiter)
                                if len(parts) >= 3:
                                    prices.append(float(parts[2]))
                        if prices:
                            price_data[ticker] = prices
                
                # Create correlation matrix
                if len(price_data) > 1:
                    # Align data to same length
                    min_length = min(len(prices) for prices in price_data.values())
                    aligned_data = {}
                    for ticker, prices in price_data.items():
                        aligned_data[ticker] = prices[:min_length]
                    
                    # Calculate returns
                    returns_data = {}
                    for ticker, prices in aligned_data.items():
                        returns = np.diff(prices) / prices[:-1] * 100
                        returns_data[ticker] = returns
                    
                    # Create correlation matrix
                    ticker_list = list(returns_data.keys())
                    corr_matrix = np.zeros((len(ticker_list), len(ticker_list)))
                    
                    for i, ticker1 in enumerate(ticker_list):
                        for j, ticker2 in enumerate(ticker_list):
                            if i == j:
                                corr_matrix[i, j] = 1.0
                            else:
                                corr = np.corrcoef(returns_data[ticker1], returns_data[ticker2])[0, 1]
                                corr_matrix[i, j] = corr
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix,
                        x=ticker_list,
                        y=ticker_list,
                        colorscale='RdBu',
                        zmid=0,
                        text=np.round(corr_matrix, 3),
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        hoverongaps=False
                    ))
                    
                    fig.update_layout(
                        title='Correlation Matrix of Stock Returns',
                        xaxis_title='Ticker',
                        yaxis_title='Ticker'
                    )
                    
                    # Save correlation matrix
                    matrix_filename = os.path.join(corr_dir, "correlation_matrix.html")
                    fig.write_html(matrix_filename)
                    logger.info(f"Created correlation matrix: {matrix_filename}")
                    
                    # Save correlation data to CSV
                    import csv
                    csv_filename = os.path.join(corr_dir, "correlation_matrix.csv")
                    with open(csv_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Ticker'] + ticker_list)
                        for i, ticker in enumerate(ticker_list):
                            writer.writerow([ticker] + list(corr_matrix[i]))
                    logger.info(f"Saved correlation matrix to CSV: {csv_filename}")
                    
            except Exception as e:
                logger.error(f"Error creating correlation matrix: {e}")
        
        # Save summary statistics
        try:
            with open(os.path.join(corr_dir, "correlation_summary.txt"), 'w') as f:
                f.write("CORRELATION ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("AUTOCORRELATION SUMMARY:\n")
                f.write("-" * 30 + "\n")
                for ticker, (lags, autocorrs) in autocorr_results.items():
                    max_autocorr = max(autocorrs) if autocorrs else 0
                    min_autocorr = min(autocorrs) if autocorrs else 0
                    avg_autocorr = np.mean(autocorrs) if autocorrs else 0
                    f.write(f"{ticker}: Max={max_autocorr:0.3f}, Min={min_autocorr:0.3f}, Avg={avg_autocorr:.3f}\n")
                
                f.write("\nCROSS-CORRELATION SUMMARY:\n")
                f.write("-" * 30 + "\n")
                for (ticker1, ticker2), (lags, cross_corrs) in cross_corr_results.items():
                    max_cross = max(cross_corrs) if cross_corrs else 0
                    min_cross = min(cross_corrs) if cross_corrs else 0
                    avg_cross = np.mean(cross_corrs) if cross_corrs else 0
                    f.write(f"{ticker1} vs {ticker2}: Max={max_cross:.3f}, Min={min_cross:.3f}, Avg={avg_cross:.3f}\n")
            
            logger.info(f"Saved correlation summary to {corr_dir}/correlation_summary.txt")
            
        except Exception as e:
            logger.error(f"Error saving correlation summary: {e}")
        
        logger.info("Correlation analysis complete! Check the 'correlation_analysis' directory for results.")

    def compute_autocorrelation_window(self, ticker: str, months_back: int = 6, max_lag: int = 30) -> Optional[Tuple[List[int], List[float], str]]:
        """
        Compute autocorrelation for a specific stock over the last k months.
        
        Args:
            ticker: Stock ticker symbol
            months_back: Number of months to look back (default: 6)
            max_lag: Maximum lag to compute (default: 30)
        
        Returns:
            Tuple of (lags, autocorrelations, date_range) or None if error
        """
        filename = os.path.join(self.results_dir, f"{ticker}.txt")
        if not os.path.exists(filename):
            logger.warning(f"No data file found for {ticker}")
            return None
        
        try:
            # Read all data with dates
            data = []
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split(self.date_delimiter)
                    if len(parts) >= 3:
                        timestamp_ms = int(parts[0])
                        date_str = parts[1]
                        price = float(parts[2])
                        date_obj = datetime.strptime(date_str, "%m-%d-%Y")
                        data.append((date_obj, price))
            
            if not data:
                logger.warning(f"No valid data found for {ticker}")
                return None
            
            # Sort by date
            data.sort(key=lambda x: x[0])
            # Calculate cutoff date (months_back from now)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30 * months_back)
            
            # Filter data to last k months
            filtered_data = [(date, price) for date, price in data if date >= start_date]
            
            if len(filtered_data) < max_lag + 1:
                logger.warning(f"Insufficient data for {ticker} autocorrelation over last {months_back} months")
                return None
            
            # Extract prices and dates
            dates = [item[0] for item in filtered_data]
            prices = [item[1] for item in filtered_data]
            
            # Calculate returns (percentage change)
            returns = np.diff(prices) / prices[:-1] *100      
            # Compute autocorrelation
            lags = list(range(1, min(max_lag + 1, len(returns))))
            autocorrs = []
            
            for lag in lags:
                if lag < len(returns):
                    # Compute correlation between returns and lagged returns
                    corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                    autocorrs.append(corr)
            
            # Create date range string
            date_range = f"{dates[0].strftime('%m/%d/%Y')} to {dates[-1].strftime('%m/%d/%Y')}"      
            return lags, autocorrs, date_range
            
        except Exception as e:
            logger.error(f"Error computing autocorrelation for {ticker}: {e}")
            return None
    
    def analyze_autocorrelation_window(self, ticker: str, months_back: int = 6):
        """
        Analyze autocorrelation for a specific stock over the last k months.
        Creates interactive plot and saves results.
        
        Args:
            ticker: Stock ticker symbol
            months_back: Number of months to look back
        """
        result = self.compute_autocorrelation_window(ticker, months_back)
        if not result:
            logger.error(f"Could not compute autocorrelation for {ticker}")
            return
        
        lags, autocorrs, date_range = result
        
        # Create directory for window analysis
        window_dir = os.path.join(self.results_dir, "autocorrelation_windows")
        os.makedirs(window_dir, exist_ok=True)
        
        # Create autocorrelation plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lags, y=autocorrs, mode='lines+markers', name=ticker))
        fig.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.3)
        
        # Add confidence intervals (approximate 95%)
        n = len(autocorrs)
        if n > 0:
            ci_upper = 1.96 * np.sqrt(n)  # Approximate 95 CI
            ci_lower = -1.96 * np.sqrt(n)
            fig.add_hline(y=ci_upper, line_dash='dot', line_color='red', opacity=0.5, 
                         annotation_text="95% CI Upper")
            fig.add_hline(y=ci_lower, line_dash='dot', line_color='red', opacity=0.5,
                         annotation_text="95% CI Lower")
        
        fig.update_layout(
            title=f'{ticker} Autocorrelation of Returns (Last {months_back} Months)<br><sub>{date_range}</sub>',
            xaxis_title='Lag (days)',
            yaxis_title='Autocorrelation',
            hovermode='x unified',
            yaxis=dict(range=[-1, 1])
        )
        fig.update_xaxes(tickformat='d')
        fig.update_yaxes(tickformat='.3f', range=[-1, 1])
        
        # Add statistics annotation
        max_autocorr = max(autocorrs) if autocorrs else 0
        min_autocorr = min(autocorrs) if autocorrs else 0
        avg_autocorr = np.mean(autocorrs) if autocorrs else 0    
        stats_text = f'Max: {max_autocorr:.3f}<br>Min: {min_autocorr:.3f}<br>Avg: {avg_autocorr:.3f}'
        fig.add_annotation(
            x=0.02, y=0.98,
            text=stats_text,
            showarrow=False,
            xref='paper', yref='paper',
            xanchor='left', yanchor='top',
            bgcolor='lightblue', bordercolor='lightblue', borderwidth=1,
            font=dict(size=12)
        )
        
        # Save plot
        plot_filename = os.path.join(window_dir, f"{ticker}_{months_back}months_autocorrelation.html")
        fig.write_html(plot_filename)
        logger.info(f"Created autocorrelation plot for {ticker} (last {months_back} months): {plot_filename}")
        
        # Save data to CSV
        import csv
        csv_filename = os.path.join(window_dir, f"{ticker}_{months_back}months_autocorrelation.csv")
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Lag (days)', 'Autocorrelation'])
            for lag, autocorr in zip(lags, autocorrs):
                writer.writerow([lag, autocorr])
        
        logger.info(f"Saved autocorrelation data to CSV: {csv_filename}")
        
        # Print summary
        print(f"\n=== {ticker} Autocorrelation Analysis (Last {months_back} Months) ===")
        print(f"DateRange: {date_range}")
        print(f"Data Points: {len(autocorrs)}")
        print(f"Max Autocorrelation: {max_autocorr:.3f}")
        print(f"Min Autocorrelation: {min_autocorr:.3f}")
        print(f"Average Autocorrelation: {avg_autocorr:.3f}")
        
        # Interpretation
        if avg_autocorr > 0.1:
            print("Interpretation: Strong positive autocorrelation - momentum effects present")
        elif avg_autocorr < -0.1:
            print("Interpretation: Strong negative autocorrelation - mean reversion effects present")
        else:
            print("Interpretation: Weak autocorrelation - near random walk behavior")

    def compute_cross_correlation_window(self, ticker1: str, ticker2: str, start_date_str: str, max_lag: int = 30) -> Optional[Tuple[List[int], List[float], str]]:
        """
        Compute cross-lag correlation between two tickers from a specific start date to today.
        """
        filename1 = os.path.join(self.results_dir, f"{ticker1}.txt")
        filename2 = os.path.join(self.results_dir, f"{ticker2}.txt")
        if not os.path.exists(filename1) or not os.path.exists(filename2):
            logger.warning(f"Data files not found for {ticker1} and {ticker2}")
            return None
        try:
            start_date = datetime.strptime(start_date_str, "%m-%d-%Y")
            end_date = datetime.now()
            if start_date >= end_date:
                logger.error(f"Start date {start_date_str} must be before today")
                return None
            # Read price data for both tickers
            data1 = {}
            data2 = {}
            with open(filename1, 'r') as f:
                for line in f:
                    parts = line.strip().split(self.date_delimiter)
                    if len(parts) >= 3:
                        date_str = parts[1]
                        price = float(parts[2])
                        date_obj = datetime.strptime(date_str, "%m-%d-%Y")
                        if start_date <= date_obj <= end_date:
                            data1[date_obj] = price
            with open(filename2, 'r') as f:
                for line in f:
                    parts = line.strip().split(self.date_delimiter)
                    if len(parts) >= 3:
                        date_str = parts[1]
                        price = float(parts[2])
                        date_obj = datetime.strptime(date_str, "%m-%d-%Y")
                        if start_date <= date_obj <= end_date:
                            data2[date_obj] = price
            # Find common dates and sort
            common_dates = sorted(set(data1.keys()) & set(data2.keys()))
            if len(common_dates) < max_lag + 2:
                logger.warning(f"Insufficient overlapping data for cross-correlation")
                return None
            # Align prices by common dates
            prices1 = [data1[d] for d in common_dates]
            prices2 = [data2[d] for d in common_dates]
            # Calculate returns
            returns1 = np.diff(prices1) / prices1[:-1] * 100
            returns2 = np.diff(prices2) / prices2[:-1] * 100
            # Now returns1 and returns2 are the same length
            lags = list(range(-max_lag, max_lag + 1))
            cross_corrs = []
            for lag in lags:
                if lag < 0:
                    if abs(lag) < len(returns1):
                        corr = np.corrcoef(returns1[:lag], returns2[-lag:])[0, 1]
                        cross_corrs.append(corr)
                    else:
                        cross_corrs.append(np.nan)
                elif lag == 0:
                    corr = np.corrcoef(returns1, returns2)[0, 1]
                    cross_corrs.append(corr)
                else:
                    if lag < len(returns2):
                        corr = np.corrcoef(returns1[lag:], returns2[:-lag])[0, 1]
                        cross_corrs.append(corr)
                    else:
                        cross_corrs.append(np.nan)
            date_range = f"{common_dates[0].strftime('%m/%d/%Y')} to {common_dates[-1].strftime('%m/%d/%Y')}"
            return lags, cross_corrs, date_range
        except Exception as e:
            logger.error(f"Error computing cross-correlation between {ticker1} and {ticker2}: {e}")
            return None

    def analyze_cross_correlation_window(self, ticker1: str, ticker2: str, start_date: str):
        """
        Analyze cross-lag correlation between two specific stocks from a start date to today.
        Creates interactive plot and saves results.
        """
        result = self.compute_cross_correlation_window(ticker1, ticker2, start_date)
        if not result:
            logger.error(f"Could not compute cross-correlation between {ticker1} and {ticker2}")
            return
        lags, cross_corrs, date_range = result
        cross_dir = os.path.join(self.results_dir, "cross_correlation_windows")
        os.makedirs(cross_dir, exist_ok=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lags, y=cross_corrs, mode='lines+markers', name=f'{ticker1} vs {ticker2}', line=dict(color='blue')))
        fig.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.3)
        valid_corrs = [x for x in cross_corrs if not np.isnan(x)]
        n = len(valid_corrs)
        if n > 0:
            ci_upper = 1.96 * np.sqrt(n)
            ci_lower = -1.96 * np.sqrt(n)
            fig.add_hline(y=ci_upper, line_dash='dot', line_color='red', opacity=0.5, annotation_text="95% CI Upper")
            fig.add_hline(y=ci_lower, line_dash='dot', line_color='red', opacity=0.5, annotation_text="95% CI Lower")
        fig.update_layout(
            title=f'Cross-Correlation: {ticker1} vs {ticker2}<br><sub>{date_range}</sub>',
            xaxis_title='Lag (days)',
            yaxis_title='Cross-Correlation',
            hovermode='x unified',
            yaxis=dict(range=[-1, 1])
        )
        fig.update_xaxes(tickformat='d')
        fig.update_yaxes(tickformat='.3f', range=[-1, 1])
        if valid_corrs:
            max_cross = max(valid_corrs)
            min_cross = min(valid_corrs)
            avg_cross = np.mean(valid_corrs)
            max_idx = np.argmax(valid_corrs)
            max_lag = lags[max_idx]
            stats_text = f'Max: {max_cross:0.3f} (lag {max_lag})<br>Min: {min_cross:.3f}<br>Avg: {avg_cross:.3f}'
            fig.add_annotation(
                x=0.02, y=0.98,
                text=stats_text,
                showarrow=False,
                xref='paper', yref='paper',
                xanchor='left', yanchor='top',
                bgcolor='lightgreen', bordercolor='lightgreen', borderwidth=1,
                font=dict(size=12)
            )
        plot_filename = os.path.join(cross_dir, f"{ticker1}_{ticker2}_{start_date.replace('-', '')}_crosscorrelation.html")
        fig.write_html(plot_filename)
        logger.info(f"Created cross-correlation plot for {ticker1} vs {ticker2}: {plot_filename}")
        import csv
        csv_filename = os.path.join(cross_dir, f"{ticker1}_{ticker2}_{start_date.replace('-', '')}_crosscorrelation.csv")
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Lag (days)', 'Cross-Correlation'])
            for lag, corr in zip(lags, cross_corrs):
                writer.writerow([lag, corr if not np.isnan(corr) else ''])
        logger.info(f"Saved cross-correlation data to CSV: {csv_filename}")
        print(f"\n=== Cross-Correlation Analysis: {ticker1} vs {ticker2} ===\n")
        print(f"DateRange: {date_range}")
        print(f"Data Points: {len(valid_corrs)}")
        print(f"Max Cross-Correlation: {max_cross:0.3f} at lag {max_lag}")
        print(f"Min Cross-Correlation: {min_cross:.3f}")
        print(f"Average Cross-Correlation: {avg_cross:.3f}")
        if max_lag > 0:
            print(f"Interpretation: {ticker2} leads {ticker1} by {max_lag} days")
        elif max_lag < 0:
            print(f"Interpretation: {ticker1} leads {ticker2} by {abs(max_lag)} days")
        else:
            print("Interpretation: Stocks move together simultaneously")
        if abs(max_cross) > 0.5:
            print("Strong correlation detected")
        elif abs(max_cross) > 0.3:
            print("Moderate correlation detected")
        else:
            print("Weak correlation detected")

def main():
    """
    Handle command line arguments and run the appropriate action.
    """
    parser = argparse.ArgumentParser(description='Stock Market Data Collector')
    parser.add_argument('--tickers-file', default='tickers.txt', 
                       help='File containing ticker symbols (default: tickers.txt)')
    parser.add_argument('--results-dir', default='results', 
                       help='Directory to store results (default: results)')
    parser.add_argument('--date-delimiter', default='\t',
                       help='Delimiter between date and price (default: tab)')
    parser.add_argument('--row-delimiter', default='\n',
                       help='Delimiter between rows (default: newline)')
    parser.add_argument('--years-back', type=int, default=2, 
                       help='Years of historical data to fetch (default: 2)')
    parser.add_argument('--action', choices=['init', 'update', 'diff', 'visualize', 'correlate', 'autocorr', 'crosscorr'], required=True,
                       help='Action to perform: init (historical data), update (daily data), diff (daily differences), visualize (create charts), correlate (correlation analysis), autocorr (autocorrelation analysis), or crosscorr (cross-correlation analysis)')
    parser.add_argument('--ticker', type=str, 
                       help='Specific ticker for autocorrelation analysis (required for autocorr action)')
    parser.add_argument('--tickers', nargs=2, metavar=('TICKER1', 'TICKER2'),
                       help='Two tickers for cross-correlation analysis (required for crosscorr action, e.g., --tickers AAPL GOOGL)')
    parser.add_argument('--start-date', type=str, 
                       help='Start date in MM-DD-YYYY format for cross-correlation analysis (required for crosscorr action)')
    parser.add_argument('--months-back', type=int, default=6,
                       help='Number of months to look back for autocorrelation (default: 6)')
    args = parser.parse_args()
    
    collector = StockDataCollector(
        tickers_file=args.tickers_file,
        results_dir=args.results_dir,
        date_delimiter=args.date_delimiter,
        row_delimiter=args.row_delimiter,
        years_back=args.years_back
    )
    
    if args.action == 'init':
        collector.initialize_all_tickers()
    elif args.action == 'update':
        collector.update_all_tickers()
    elif args.action == 'diff':
        collector.calculate_daily_differences()
    elif args.action == 'visualize':
        collector.visualize_all_tickers()
    elif args.action == 'correlate':
        collector.analyze_correlations()
    elif args.action == 'autocorr':
        if not args.ticker:
            logger.error("--ticker is required for autocorrelation analysis")
            return
        collector.analyze_autocorrelation_window(args.ticker, args.months_back)
    elif args.action == 'crosscorr':
        if not args.tickers or not args.start_date:
            logger.error("--tickers (two tickers) and --start-date are required for cross-correlation analysis")
            return
        ticker1, ticker2 = args.tickers
        collector.analyze_cross_correlation_window(ticker1, ticker2, args.start_date)

if __name__ == "__main__":
    main() 