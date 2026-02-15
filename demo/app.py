"""
RL Trading Model Testing Platform
Flask web application for testing PPO and CVaR-PPO trading models
"""

from flask import Flask, render_template, jsonify, request, send_file
import json
import os
from datetime import datetime, timedelta
import io
import base64

# Note: These imports might not work without packages installed
# The app will still run and show the UI structure
try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.distributions import Normal
    import yfinance as yf
    import ta
    PACKAGES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some packages not available: {e}")
    PACKAGES_AVAILABLE = False
    # Create dummy classes for structure
    class nn:
        class Module: pass
        class Sequential: pass
        class Linear: pass
        class ReLU: pass
        class Parameter: pass

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Configuration
CONFIG = {
    'INITIAL_BALANCE': 10000,
    'TRANSACTION_COST': 0.001,
    'SLIPPAGE': 0.0005,
    'PPO_HIDDEN_DIM': 256,
    'DEVICE': 'cpu'
}

# ============================================================================
# MODEL DEFINITIONS (from your notebook)
# ============================================================================

class ActorCritic(nn.Module):
    """Actor-Critic Network for PPO"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        return self.shared(state)
    
    def act(self, state):
        shared_features = self.forward(state)
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        action = torch.tanh(action)
        return action, action_log_prob


class TradingEnvironment:
    """Custom Trading Environment"""
    def __init__(self, df, initial_balance=10000, transaction_cost=0.001, slippage=0.0005):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        self.feature_columns = [col for col in df.columns if col not in ['date', 'open', 'high', 'low']]
        self.feature_means = df[self.feature_columns].mean()
        self.feature_stds = df[self.feature_columns].std() + 1e-8
        
        self.current_step = 0
        self.balance = initial_balance
        self.shares = 0
        self.portfolio_value = initial_balance
        self.history = []
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.portfolio_value = self.initial_balance
        self.history = []
        return self._get_observation()
    
    def _get_observation(self):
        market_features_raw = self.df.loc[self.current_step, self.feature_columns].values
        market_features = (market_features_raw - self.feature_means.values) / self.feature_stds.values
        
        current_price = self.df.loc[self.current_step, 'close']
        portfolio_features = np.array([
            self.balance / self.initial_balance,
            self.shares * current_price / self.initial_balance,
            self.portfolio_value / self.initial_balance
        ])
        
        observation = np.concatenate([market_features, portfolio_features]).astype(np.float32)
        return observation
    
    def step(self, action):
        """Execute one trading step"""
        current_price = self.df.loc[self.current_step, 'close']
        
        # Ensure action is a scalar - handle all possible input types
        try:
            if isinstance(action, (list, tuple)):
                action = float(action[0]) if len(action) > 0 else 0.0
            elif isinstance(action, np.ndarray):
                action = float(action.flatten()[0]) if action.size > 0 else 0.0
            elif hasattr(action, 'item'):  # PyTorch tensor or numpy scalar
                action = float(action.item())
            else:
                action = float(action)
        except (ValueError, TypeError, IndexError) as e:
            print(f"Warning: Could not convert action {action} to float: {e}")
            action = 0.0
        
        old_portfolio_value = self.balance + self.shares * current_price
        
        # Execute trading
        if action > 0.01:  # Buy
            amount_to_invest = self.balance * abs(action)
            if amount_to_invest > 0:
                execution_price = current_price * (1 + self.slippage)
                shares_to_buy = (amount_to_invest * (1 - self.transaction_cost)) / execution_price
                self.shares += shares_to_buy
                self.balance -= amount_to_invest
        elif action < -0.01:  # Sell
            shares_to_sell = self.shares * abs(action)
            if shares_to_sell > 0:
                execution_price = current_price * (1 - self.slippage)
                proceeds = shares_to_sell * execution_price * (1 - self.transaction_cost)
                self.shares -= shares_to_sell
                self.balance += proceeds
        
        self.current_step += 1
        
        if self.current_step < len(self.df):
            new_price = self.df.loc[self.current_step, 'close']
            new_portfolio_value = self.balance + self.shares * new_price
        else:
            new_portfolio_value = self.balance + self.shares * current_price
        
        reward = (new_portfolio_value - old_portfolio_value) / (old_portfolio_value + 1e-10)
        self.portfolio_value = new_portfolio_value
        done = self.current_step >= len(self.df) - 1
        
        self.history.append({
            'step': self.current_step,
            'date': str(self.df.loc[self.current_step-1, 'date']),
            'action': float(action),
            'balance': float(self.balance),
            'shares': float(self.shares),
            'price': float(current_price),
            'portfolio_value': float(self.portfolio_value),
            'reward': float(reward)
        })
        
        observation = self._get_observation()
        return observation, reward, done, {}


# ============================================================================
# MODEL LOADING AND EVALUATION
# ============================================================================

class ModelEvaluator:
    """Class to handle model loading and evaluation"""
    
    def __init__(self):
        self.ppo_model = None
        self.cvar_model = None
        self.models_loaded = False
    
    def load_models(self, ppo_path='ppo_model.pth', cvar_path='cvar_ppo_model.pth'):
        """Load trained models"""
        if not PACKAGES_AVAILABLE:
            return False, "Required packages not installed"
        
        try:
            # Load PPO model
            if os.path.exists(ppo_path):
                checkpoint = torch.load(ppo_path, map_location='cpu')
                state_dim = 19  # From your config
                action_dim = 1
                self.ppo_model = ActorCritic(state_dim, action_dim, CONFIG['PPO_HIDDEN_DIM'])
                
                if 'model_state_dict' in checkpoint:
                    self.ppo_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.ppo_model.load_state_dict(checkpoint)
                
                self.ppo_model.eval()
                print("PPO model loaded successfully")
            
            # Load CVaR-PPO model
            if os.path.exists(cvar_path):
                checkpoint = torch.load(cvar_path, map_location='cpu')
                state_dim = 19
                action_dim = 1
                self.cvar_model = ActorCritic(state_dim, action_dim, CONFIG['PPO_HIDDEN_DIM'])
                
                if 'model_state_dict' in checkpoint:
                    self.cvar_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.cvar_model.load_state_dict(checkpoint)
                
                self.cvar_model.eval()
                print("CVaR-PPO model loaded successfully")
            
            self.models_loaded = True
            return True, "Models loaded successfully"
        
        except Exception as e:
            return False, f"Error loading models: {str(e)}"
    
    def evaluate_model(self, model, env):
        """Evaluate a model on an environment"""
        if not PACKAGES_AVAILABLE or model is None:
            return None, None
        
        state = env.reset()
        done = False
        
        with torch.no_grad():
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_tensor, _ = model.act(state_tensor)
                # Convert tensor to numpy array then flatten to get scalar
                action = action_tensor.cpu().numpy().flatten()
                state, reward, done, _ = env.step(action)
        
        return env.history, env.portfolio_value


# Global evaluator instance
evaluator = ModelEvaluator()


# ============================================================================
# DATA PROCESSING
# ============================================================================

def download_and_process_data(symbol, start_date, end_date):
    """Download and add technical indicators to stock data"""
    if not PACKAGES_AVAILABLE:
        return None, "Required packages not available"
    
    try:
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            return None, f"No data for {symbol}"
        
        df.reset_index(inplace=True)
        df.columns = df.columns.str.lower()
        
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        # Add technical indicators
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
        
        df.dropna(inplace=True)
        
        return df, None
    
    except Exception as e:
        return None, str(e)


def calculate_metrics(history, initial_balance=10000):
    """Calculate trading performance metrics"""
    if not PACKAGES_AVAILABLE:
        return {}
    
    portfolio_values = [h['portfolio_value'] for h in history]
    
    total_return = (portfolio_values[-1] - initial_balance) / initial_balance
    returns = np.diff(portfolio_values) / (np.array(portfolio_values[:-1]) + 1e-10)
    
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / (peak + 1e-10)
    max_drawdown = np.min(drawdown)
    
    win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
    
    return {
        'total_return': float(total_return),
        'final_value': float(portfolio_values[-1]),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'volatility': float(volatility),
        'num_trades': len(history)
    }


def buy_and_hold_strategy(df, initial_balance=10000):
    """Buy and Hold baseline strategy"""
    if not PACKAGES_AVAILABLE:
        return None, None
    
    first_price = float(df['close'].iloc[0])
    last_price = float(df['close'].iloc[-1])
    shares = float(initial_balance / first_price)
    final_value = float(shares * last_price)
    
    history = []
    for i, (idx, row) in enumerate(df.iterrows()):
        portfolio_value = float(shares * row['close'])
        history.append({
            'step': i,
            'date': str(row['date']),
            'action': 1.0 if i == 0 else 0.0,
            'balance': 0.0,
            'shares': float(shares),
            'price': float(row['close']),
            'portfolio_value': portfolio_value,
            'reward': 0.0
        })
    
    return history, final_value


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/test_models', methods=['POST'])
def test_models():
    """Test models on specified data"""
    if not PACKAGES_AVAILABLE:
        return jsonify({'error': 'Required packages not installed'}), 500
    
    try:
        data = request.json
        symbol = data.get('symbol', 'SPY')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Download and process data
        df, error = download_and_process_data(symbol, start_date, end_date)
        if error:
            return jsonify({'error': error}), 400
        
        # Load models if not already loaded
        if not evaluator.models_loaded:
            success, message = evaluator.load_models()
            if not success:
                return jsonify({'error': message}), 500
        
        # Create environment
        env = TradingEnvironment(df, CONFIG['INITIAL_BALANCE'], CONFIG['TRANSACTION_COST'], CONFIG['SLIPPAGE'])
        
        # Evaluate PPO
        ppo_history, ppo_value = evaluator.evaluate_model(evaluator.ppo_model, env)
        ppo_metrics = calculate_metrics(ppo_history, CONFIG['INITIAL_BALANCE'])
        
        # Evaluate CVaR-PPO
        env.reset()
        cvar_history, cvar_value = evaluator.evaluate_model(evaluator.cvar_model, env)
        cvar_metrics = calculate_metrics(cvar_history, CONFIG['INITIAL_BALANCE'])
        
        # Evaluate Buy & Hold
        bh_history, bh_value = buy_and_hold_strategy(df, CONFIG['INITIAL_BALANCE'])
        bh_metrics = calculate_metrics(bh_history, CONFIG['INITIAL_BALANCE'])
        
        return jsonify({
            'success': True,
            'ppo': {
                'metrics': ppo_metrics,
                'history': ppo_history[:100]  # Limit for response size
            },
            'cvar': {
                'metrics': cvar_metrics,
                'history': cvar_history[:100]
            },
            'buy_hold': {
                'metrics': bh_metrics,
                'history': bh_history[:100]
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest', methods=['POST'])
def backtest():
    """Run backtest on models"""
    if not PACKAGES_AVAILABLE:
        return jsonify({'error': 'Required packages not installed'}), 500
    
    try:
        data = request.json
        symbol = data.get('symbol', 'SPY')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        initial_balance = data.get('initial_balance', 10000)
        
        # Download data
        df, error = download_and_process_data(symbol, start_date, end_date)
        if error:
            return jsonify({'error': error}), 400
        
        # Load models
        if not evaluator.models_loaded:
            success, message = evaluator.load_models()
            if not success:
                return jsonify({'error': message}), 500
        
        # Run backtest for each model
        results = {}
        
        for model_name, model in [('ppo', evaluator.ppo_model), ('cvar', evaluator.cvar_model)]:
            env = TradingEnvironment(df, initial_balance, CONFIG['TRANSACTION_COST'], CONFIG['SLIPPAGE'])
            history, final_value = evaluator.evaluate_model(model, env)
            metrics = calculate_metrics(history, initial_balance)
            
            results[model_name] = {
                'metrics': metrics,
                'history': history,
                'final_value': final_value
            }
        
        # Buy and Hold
        bh_history, bh_value = buy_and_hold_strategy(df, initial_balance)
        bh_metrics = calculate_metrics(bh_history, initial_balance)
        
        results['buy_hold'] = {
            'metrics': bh_metrics,
            'history': bh_history,
            'final_value': bh_value
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'data_info': {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'num_days': len(df)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/realtime_predict', methods=['POST'])
def realtime_predict():
    """Get real-time prediction from a specific model"""
    if not PACKAGES_AVAILABLE:
        return jsonify({'error': 'Required packages not installed'}), 500
    
    try:
        data = request.json
        symbol = data.get('symbol', 'SPY')
        requested_model = data.get('model', 'ppo')  # 'ppo' or 'cvar'
        
        # Get recent data (last 90 days for technical indicators)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        df, error = download_and_process_data(symbol, start_date, end_date)
        if error:
            return jsonify({'error': error}), 400
        
        # Load models if not already loaded
        if not evaluator.models_loaded:
            success, message = evaluator.load_models()
            if not success:
                return jsonify({'error': message}), 500
        
        # Get current state
        env = TradingEnvironment(df, CONFIG['INITIAL_BALANCE'], CONFIG['TRANSACTION_COST'], CONFIG['SLIPPAGE'])
        state = env.reset()
        
        # Move to last time step
        for _ in range(len(df) - 1):
            env.current_step += 1
        
        state = env._get_observation()
        current_price = float(df['close'].iloc[-1])
        
        # Get prediction from requested model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            if requested_model == 'ppo':
                action_tensor, _ = evaluator.ppo_model.act(state_tensor)
                action = float(action_tensor.cpu().numpy().flatten()[0])
            else:  # cvar
                action_tensor, _ = evaluator.cvar_model.act(state_tensor)
                action = float(action_tensor.cpu().numpy().flatten()[0])
        
        # Interpret action
        def interpret_action(action):
            if action > 0.1:
                return 'BUY', abs(action)
            elif action < -0.1:
                return 'SELL', abs(action)
            else:
                return 'HOLD', 0.0
        
        signal, strength = interpret_action(action)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            requested_model: {
                'action': action,
                'signal': signal,
                'strength': strength
            }
        })
    
    except Exception as e:
        import traceback
        print(f"Error in realtime_predict: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare_models', methods=['GET'])
def compare_models():
    """Compare model performance"""
    try:
        # Return pre-computed results from your training
        comparison = {
            'models': ['PPO', 'CVaR-PPO', 'Buy & Hold'],
            'test_results': {
                'PPO': {
                    'total_return': 0.0862,
                    'final_value': 10861.76,
                    'sharpe_ratio': 0.7926,
                    'max_drawdown': -0.0769,
                    'win_rate': 0.5435,
                    'volatility': 0.0750
                },
                'CVaR-PPO': {
                    'total_return': 0.0166,
                    'final_value': 10165.96,
                    'sharpe_ratio': 0.1994,
                    'max_drawdown': -0.0656,
                    'win_rate': 0.4946,
                    'volatility': 0.0682
                },
                'Buy & Hold': {
                    'total_return': 0.3353,
                    'final_value': 13352.85,
                    'sharpe_ratio': 1.6689,
                    'max_drawdown': -0.0997,
                    'win_rate': 0.5799,
                    'volatility': 0.1229
                }
            }
        }
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check system status"""
    return jsonify({
        'packages_available': PACKAGES_AVAILABLE,
        'models_loaded': evaluator.models_loaded,
        'config': CONFIG
    })


if __name__ == '__main__':
    print("="*80)
    print("RL TRADING MODEL TESTING PLATFORM")
    print("="*80)
    print(f"Packages available: {PACKAGES_AVAILABLE}")
    print(f"Starting Flask server...")
    print("="*80)
    app.run(host='0.0.0.0', port=5000, debug=True)
