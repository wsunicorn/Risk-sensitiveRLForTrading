# Risk-Sensitive RL Trading Platform

A Django-based web application for testing and comparing reinforcement learning models for algorithmic trading.

## Features

- **Real-time Trading Signals**: Get live predictions from PPO, CVaR-PPO, and Sortino-PPO models
- **Backtesting**: Historical performance analysis with comprehensive metrics
- **Model Comparison**: Side-by-side comparison of all models
- **Risk-Sensitive RL**: Advanced algorithms that consider risk in trading decisions

## Models

1. **PPO (Proximal Policy Optimization)**: Risk-neutral baseline
2. **CVaR-PPO**: Incorporates Conditional Value at Risk for downside protection
3. **Sortino-PPO**: Focuses on downside volatility using Sortino ratio

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run database migrations:
```bash
python manage.py migrate
```

3. Start the development server:
```bash
python manage.py runserver
```

4. Open your browser and go to `http://127.0.0.1:8000`

## API Endpoints

- `GET /`: Main application interface
- `POST /api/realtime_predict/`: Get real-time trading signals
- `POST /api/backtest/`: Run historical backtesting
- `GET /api/compare_models/`: Compare model performance
- `GET /api/status/`: Check system status

## Project Structure

```
demo/
├── trading_platform/          # Django project settings
├── trading/                   # Main Django app
│   ├── views.py              # API endpoints and logic
│   ├── urls.py               # URL routing
│   └── ...
├── templates/                # HTML templates
├── static/                   # Static files (CSS, JS)
├── cvar_ppo_model.pth        # Trained CVaR-PPO model
├── ppo_model.pth            # Trained PPO model
├── sortino_ppo_model.pth    # Trained Sortino-PPO model
├── norm_params.json         # Normalization parameters
├── results.json             # Test results
└── requirements.txt         # Python dependencies
```

## Usage

### Real-time Prediction
1. Select a stock symbol (e.g., SPY, AAPL)
2. Choose a model (PPO, CVaR-PPO, or Sortino-PPO)
3. Click "Get Prediction" to receive trading signals

### Backtesting
1. Enter stock symbol and date range
2. Set initial balance
3. Run backtest to see historical performance
4. Compare all models visually

### Model Comparison
View comprehensive performance metrics and charts comparing all models.

## Technical Details

- **Framework**: Django 4.2 with Django REST Framework
- **ML Framework**: PyTorch
- **Data Source**: Yahoo Finance (yfinance)
- **Technical Indicators**: TA-Lib
- **Visualization**: Plotly.js
- **Frontend**: Bootstrap 5, Custom CSS

## Risk Disclaimer

This platform is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.

## License

This project is for educational use.

4. **Verify directory structure:**
```
demo/
├── app.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── ppo_model.pth
└── cvar_ppo_model.pth
```

## 🏃 Running the Application

1. **Start the Flask server:**
```bash
cd demo
python3 app.py
```

2. **Access the web interface:**
Open your browser and navigate to:
```
http://localhost:5000
```

## 📊 Usage Guide

### 1. Real-time Prediction

- Enter a stock symbol (e.g., SPY, AAPL, MSFT)
- Click "Get Prediction"
- View current price and trading signals from both models
- Signals are interpreted as:
  - **BUY**: Action > 0.1 (bullish signal)
  - **SELL**: Action < -0.1 (bearish signal)
  - **HOLD**: -0.1 ≤ Action ≤ 0.1 (neutral)

### 2. Backtest

- Enter stock symbol and date range
- Set initial balance
- Click "Run Backtest"
- View comprehensive performance metrics and charts
- Compare all three strategies side-by-side

### 3. Model Comparison

- Click "Load Comparison"
- View pre-computed test set results
- Analyze radar chart showing normalized metrics
- Compare risk-return profiles

## 🔧 Configuration

Edit the `CONFIG` dictionary in `app.py` to customize:

```python
CONFIG = {
    'INITIAL_BALANCE': 10000,      # Starting capital
    'TRANSACTION_COST': 0.001,     # 0.1% per trade
    'SLIPPAGE': 0.0005,            # 0.05%
    'PPO_HIDDEN_DIM': 256,         # Neural network size
    'DEVICE': 'cpu'                # Use 'cuda' if GPU available
}
```

## 📈 Model Information

### PPO (Proximal Policy Optimization)
- **Type**: Risk-neutral baseline
- **Focus**: Maximize expected returns
- **Architecture**: Actor-Critic with 256 hidden units
- **Training**: 100 episodes on SPY (2015-2022)

### CVaR-PPO (Conditional Value at Risk PPO)
- **Type**: Risk-sensitive with CVaR constraint
- **Focus**: Manage tail risk and downside protection
- **Alpha**: 0.15 (penalizes worst 15% returns)
- **Lambda**: Adaptive 0.16 → 0.10
- **Training**: 100 episodes with L2 regularization

## 📊 Performance Metrics

### Test Set Results (SPY 2023-2024)

| Metric | PPO | CVaR-PPO | Buy & Hold |
|--------|-----|----------|------------|
| Total Return | 8.62% | 1.66% | 33.53% |
| Sharpe Ratio | 0.7926 | 0.1994 | 1.6689 |
| Max Drawdown | -7.69% | -6.56% | -9.97% |
| Win Rate | 54.35% | 49.46% | 57.99% |
| Volatility | 7.50% | 6.82% | 12.29% |

### Key Observations
- **PPO**: Best risk-adjusted returns among RL methods
- **CVaR-PPO**: Lowest max drawdown, better risk management
- **Buy & Hold**: Highest absolute return in bull market

## 🔌 API Endpoints

### GET /api/status
Check system status and model availability

### POST /api/realtime_predict
Get real-time prediction for a stock
```json
{
  "symbol": "SPY"
}
```

### POST /api/backtest
Run backtest on historical data
```json
{
  "symbol": "SPY",
  "start_date": "2023-01-01",
  "end_date": "2024-12-31",
  "initial_balance": 10000
}
```

### GET /api/compare_models
Get pre-computed model comparison data

## 🛠️ Troubleshooting

### "Required packages not installed"
- Install missing packages: `pip install <package> --break-system-packages`
- Check if all packages are importable in Python

### "Models not loaded"
- Ensure `ppo_model.pth` and `cvar_ppo_model.pth` are in the demo directory
- Check file permissions
- Verify model files are not corrupted

### "No data for symbol"
- Check internet connection
- Verify stock symbol is valid
- Try a different date range

### Charts not displaying
- Check browser console for errors
- Ensure Plotly and Chart.js CDN links are accessible
- Try refreshing the page

## ⚠️ Disclaimer

**This is an educational project.** Do not use these models for real trading without:
- Proper validation on different market conditions
- Comprehensive risk management
- Understanding of model limitations
- Consideration of transaction costs and slippage

Past performance does not guarantee future results.

## 📚 Technical Details

### Technical Indicators Used
- SMA (Simple Moving Average): 10, 20, 50 days
- RSI (Relative Strength Index): 14 days
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Returns and log returns
- Volume SMA

### State Space
- 16 technical indicators
- 3 portfolio features (balance, shares value, portfolio value)
- Total: 19 dimensions

### Action Space
- Continuous: [-1, 1]
- Positive: Buy (amount proportional to value)
- Negative: Sell (amount proportional to value)
- Near zero: Hold

## 🔄 Updates and Future Improvements

Potential enhancements:
1. Multi-asset portfolio trading
2. News sentiment integration
3. Ensemble model predictions
4. Real-time streaming data
5. Advanced risk metrics (VaR, Expected Shortfall)
6. Walk-forward optimization
7. Live trading integration (paper trading)

## 📝 License

Educational use only. Not for commercial trading.

## 👥 Contact

For questions or issues, please check:
- Model training notebook for implementation details
- Flask documentation for server issues
- PyTorch documentation for model loading issues

---

**Built with Flask, PyTorch, and Plotly | February 2026**
