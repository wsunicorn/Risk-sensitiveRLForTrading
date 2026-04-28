import json
import os
from datetime import datetime, timedelta
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import django
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import required packages
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
    # Create dummy classes
    class nn:
        class Module: pass
        class Sequential: pass
        class Linear: pass
        class ReLU: pass
        class Parameter: pass

# Configuration
CONFIG = {
    'INITIAL_BALANCE': 10000,
    'TRANSACTION_COST': 0.001,
    'SLIPPAGE': 0.0005,
    'PPO_HIDDEN_DIM': 256,
    'DEVICE': 'cpu'
}

ARTIFACT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NormalizationConfigError(ValueError):
    """Raised when training-time normalization parameters are missing or invalid."""


def load_norm_params(artifact_dir=ARTIFACT_DIR):
    """Load feature normalization parameters saved by the training notebook."""
    norm_path = os.path.join(artifact_dir, 'norm_params.json')
    if not os.path.exists(norm_path):
        raise NormalizationConfigError(f"Missing normalization parameters: {norm_path}")

    try:
        with open(norm_path, 'r') as f:
            params = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise NormalizationConfigError(f"Invalid normalization parameters file {norm_path}: {e}") from e

    feature_columns = params.get('feat_cols') or params.get('feature_columns')
    means = params.get('feat_mean') or params.get('feature_means') or params.get('mean')
    stds = params.get('feat_std') or params.get('feature_stds') or params.get('std')

    if means is None or stds is None:
        raise NormalizationConfigError(
            "norm_params.json must contain feat_mean/feat_std or feature_means/feature_stds"
        )

    if isinstance(means, dict) and isinstance(stds, dict):
        if feature_columns is None:
            feature_columns = list(means.keys())
        missing_std = [col for col in feature_columns if col not in stds]
        missing_mean = [col for col in feature_columns if col not in means]
        if missing_mean or missing_std:
            raise NormalizationConfigError(
                f"norm_params.json missing mean/std for features: {missing_mean + missing_std}"
            )
        means = [means[col] for col in feature_columns]
        stds = [stds[col] for col in feature_columns]
    elif isinstance(means, dict) or isinstance(stds, dict):
        raise NormalizationConfigError("norm_params.json mean/std must both be lists or both be objects")

    if feature_columns is not None:
        if isinstance(feature_columns, str) or not isinstance(feature_columns, (list, tuple)):
            raise NormalizationConfigError("norm_params.json feature columns must be a list")
        feature_columns = list(feature_columns)

    try:
        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise NormalizationConfigError("norm_params.json mean/std values must be numeric") from e

    if means.ndim != 1 or stds.ndim != 1 or len(means) != len(stds):
        raise NormalizationConfigError("norm_params.json mean/std must be one-dimensional arrays of equal length")
    if feature_columns is not None and len(feature_columns) != len(means):
        raise NormalizationConfigError("norm_params.json feature column count does not match mean/std length")
    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise NormalizationConfigError("norm_params.json mean/std contain non-finite values")
    if (stds <= 0).any():
        raise NormalizationConfigError("norm_params.json std values must be positive")

    state_dim = params.get('state_dim')
    if state_dim is None:
        state_dim = len(means) + 3
    else:
        try:
            state_dim = int(state_dim)
        except (TypeError, ValueError) as e:
            raise NormalizationConfigError("norm_params.json state_dim must be an integer") from e
        if feature_columns is not None and state_dim != len(feature_columns) + 3:
            raise NormalizationConfigError("norm_params.json state_dim does not match feature count + portfolio state")

    try:
        action_dim = int(params.get('action_dim', 1))
    except (TypeError, ValueError) as e:
        raise NormalizationConfigError("norm_params.json action_dim must be an integer") from e
    if action_dim != 1:
        raise NormalizationConfigError("norm_params.json action_dim does not match backend model action size")

    return {
        'path': norm_path,
        'feature_columns': feature_columns,
        'feature_means': means,
        'feature_stds': stds,
        'state_dim': state_dim,
        'action_dim': action_dim,
    }

# ============================================================================
# MODEL DEFINITIONS (from notebook)
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
    def __init__(self, df, initial_balance=10000, transaction_cost=0.001, slippage=0.0005, norm_params=None):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        if norm_params is None:
            raise NormalizationConfigError("Training-time norm_params are required for model inference")

        available_feature_columns = [col for col in df.columns if col not in ['date', 'open', 'high', 'low']]
        if norm_params['feature_columns'] is None:
            print("Warning: norm_params.json has no feature columns; using backend feature order")
            self.feature_columns = available_feature_columns
        else:
            self.feature_columns = norm_params['feature_columns']

        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise NormalizationConfigError(f"Input data missing normalized features: {missing_features}")

        self.feature_means = norm_params['feature_means']
        self.feature_stds = norm_params['feature_stds']
        if len(self.feature_columns) != len(self.feature_means):
            raise NormalizationConfigError("Feature column count does not match normalization parameters")
        if norm_params['state_dim'] != len(self.feature_columns) + 3:
            raise NormalizationConfigError("Model state_dim does not match normalized feature count")

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
        market_features_raw = self.df.loc[self.current_step, self.feature_columns].values.astype(np.float64)
        market_features = (market_features_raw - self.feature_means) / self.feature_stds

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

        # Ensure action is a scalar
        try:
            if isinstance(action, (list, tuple)):
                action = float(action[0]) if len(action) > 0 else 0.0
            elif isinstance(action, np.ndarray):
                action = float(action.flatten()[0]) if action.size > 0 else 0.0
            elif hasattr(action, 'item'):
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
        self.sortino_model = None
        self.norm_params = None
        self.models_loaded = False
        # Load models on initialization
        self.load_models()

    def load_models(self):
        """Load trained models"""
        if not PACKAGES_AVAILABLE:
            return False, "Required packages not installed"

        try:
            self.norm_params = load_norm_params(ARTIFACT_DIR)

            # Load PPO model
            ppo_path = os.path.join(ARTIFACT_DIR, 'ppo_model.pth')
            if os.path.exists(ppo_path):
                checkpoint = torch.load(ppo_path, map_location='cpu')
                state_dim = self.norm_params['state_dim']
                action_dim = 1
                self.ppo_model = ActorCritic(state_dim, action_dim, CONFIG['PPO_HIDDEN_DIM'])

                if 'model_state_dict' in checkpoint:
                    self.ppo_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.ppo_model.load_state_dict(checkpoint)

                self.ppo_model.eval()
                print("PPO model loaded successfully")

            # Load CVaR-PPO model
            cvar_path = os.path.join(ARTIFACT_DIR, 'cvar_ppo_model.pth')
            if os.path.exists(cvar_path):
                checkpoint = torch.load(cvar_path, map_location='cpu')
                state_dim = self.norm_params['state_dim']
                action_dim = 1
                self.cvar_model = ActorCritic(state_dim, action_dim, CONFIG['PPO_HIDDEN_DIM'])

                if 'model_state_dict' in checkpoint:
                    self.cvar_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.cvar_model.load_state_dict(checkpoint)

                self.cvar_model.eval()
                print("CVaR-PPO model loaded successfully")

            # Load Sortino-PPO model
            sortino_path = os.path.join(ARTIFACT_DIR, 'sortino_ppo_model.pth')
            if os.path.exists(sortino_path):
                checkpoint = torch.load(sortino_path, map_location='cpu')
                state_dim = self.norm_params['state_dim']
                action_dim = 1
                self.sortino_model = ActorCritic(state_dim, action_dim, CONFIG['PPO_HIDDEN_DIM'])

                if 'model_state_dict' in checkpoint:
                    self.sortino_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.sortino_model.load_state_dict(checkpoint)

                self.sortino_model.eval()
                print("Sortino-PPO model loaded successfully")

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
            'balance': float(0.0),
            'shares': float(shares),
            'price': float(row['close']),
            'portfolio_value': portfolio_value,
            'reward': 0.0
        })

    return history, final_value


# ============================================================================
# DJANGO VIEWS
# ============================================================================

def index(request):
    """Main page"""
    return render(request, 'index.html')


@method_decorator(csrf_exempt, name='dispatch')
class TestModelsView(View):
    """Test models on specified data"""

    def post(self, request):
        if not PACKAGES_AVAILABLE:
            return JsonResponse({'error': 'Required packages not installed'}, status=500)

        try:
            data = json.loads(request.body)
            symbol = data.get('symbol', 'SPY')
            start_date = data.get('start_date')
            end_date = data.get('end_date')

            # Download and process data
            df, error = download_and_process_data(symbol, start_date, end_date)
            if error:
                return JsonResponse({'error': error}, status=400)

            # Load models if not already loaded
            if not evaluator.models_loaded:
                success, message = evaluator.load_models()
                if not success:
                    return JsonResponse({'error': message}, status=500)

            # Create environment
            env = TradingEnvironment(
                df,
                CONFIG['INITIAL_BALANCE'],
                CONFIG['TRANSACTION_COST'],
                CONFIG['SLIPPAGE'],
                norm_params=evaluator.norm_params
            )

            # Evaluate PPO
            ppo_history, ppo_value = evaluator.evaluate_model(evaluator.ppo_model, env)
            ppo_metrics = calculate_metrics(ppo_history, CONFIG['INITIAL_BALANCE'])

            # Evaluate CVaR-PPO
            env.reset()
            cvar_history, cvar_value = evaluator.evaluate_model(evaluator.cvar_model, env)
            cvar_metrics = calculate_metrics(cvar_history, CONFIG['INITIAL_BALANCE'])

            # Evaluate Sortino-PPO
            env.reset()
            sortino_history, sortino_value = evaluator.evaluate_model(evaluator.sortino_model, env)
            sortino_metrics = calculate_metrics(sortino_history, CONFIG['INITIAL_BALANCE'])

            # Evaluate Buy & Hold
            bh_history, bh_value = buy_and_hold_strategy(df, CONFIG['INITIAL_BALANCE'])
            bh_metrics = calculate_metrics(bh_history, CONFIG['INITIAL_BALANCE'])

            return JsonResponse({
                'success': True,
                'ppo': {
                    'metrics': ppo_metrics,
                    'history': ppo_history[:100]
                },
                'cvar': {
                    'metrics': cvar_metrics,
                    'history': cvar_history[:100]
                },
                'sortino': {
                    'metrics': sortino_metrics,
                    'history': sortino_history[:100]
                },
                'buy_hold': {
                    'metrics': bh_metrics,
                    'history': bh_history[:100]
                }
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class BacktestView(View):
    """Run backtest on models"""

    def post(self, request):
        if not PACKAGES_AVAILABLE:
            return JsonResponse({'error': 'Required packages not installed'}, status=500)

        try:
            data = json.loads(request.body)
            symbol = data.get('symbol', 'SPY')
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            initial_balance = data.get('initial_balance', 10000)

            # Download data
            df, error = download_and_process_data(symbol, start_date, end_date)
            if error:
                return JsonResponse({'error': error}, status=400)

            # Load models
            if not evaluator.models_loaded:
                success, message = evaluator.load_models()
                if not success:
                    return JsonResponse({'error': message}, status=500)

            # Run backtest for each model
            results = {}

            for model_name, model in [('ppo', evaluator.ppo_model), ('cvar', evaluator.cvar_model), ('sortino', evaluator.sortino_model)]:
                env = TradingEnvironment(
                    df,
                    initial_balance,
                    CONFIG['TRANSACTION_COST'],
                    CONFIG['SLIPPAGE'],
                    norm_params=evaluator.norm_params
                )
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

            return JsonResponse({
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
            return JsonResponse({'error': str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class RealtimePredictView(View):
    """Get real-time prediction from a specific model"""

    def post(self, request):
        if not PACKAGES_AVAILABLE:
            return JsonResponse({'error': 'Required packages not installed'}, status=500)

        try:
            data = json.loads(request.body)
            symbol = data.get('symbol', 'SPY')
            requested_model = data.get('model', 'ppo')  # 'ppo', 'cvar', or 'sortino'

            # Get recent data (last 90 days for technical indicators)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

            df, error = download_and_process_data(symbol, start_date, end_date)
            if error:
                return JsonResponse({'error': error}, status=400)

            # Load models if not already loaded
            if not evaluator.models_loaded:
                success, message = evaluator.load_models()
                if not success:
                    return JsonResponse({'error': message}, status=500)

            # Get current state
            env = TradingEnvironment(
                df,
                CONFIG['INITIAL_BALANCE'],
                CONFIG['TRANSACTION_COST'],
                CONFIG['SLIPPAGE'],
                norm_params=evaluator.norm_params
            )
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
                elif requested_model == 'cvar':
                    action_tensor, _ = evaluator.cvar_model.act(state_tensor)
                    action = float(action_tensor.cpu().numpy().flatten()[0])
                elif requested_model == 'sortino':
                    action_tensor, _ = evaluator.sortino_model.act(state_tensor)
                    action = float(action_tensor.cpu().numpy().flatten()[0])
                else:
                    return JsonResponse({'error': 'Invalid model specified'}, status=400)

            # Interpret action
            def interpret_action(action):
                if action > 0.1:
                    return 'BUY', abs(action)
                elif action < -0.1:
                    return 'SELL', abs(action)
                else:
                    return 'HOLD', 0.0

            signal, strength = interpret_action(action)

            return JsonResponse({
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
            return JsonResponse({'error': str(e)}, status=500)


def compare_models(request):
    """Compare model performance"""
    try:
        # Load pre-computed results from results.json
        results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results_data = json.load(f)
            comparison = {
                'models': ['PPO', 'CVaR-PPO', 'Sortino-PPO', 'Buy & Hold'],
                'test_results': results_data.get('test_metrics', {})
            }
        else:
            # Fallback to hardcoded values
            comparison = {
                'models': ['PPO', 'CVaR-PPO', 'Sortino-PPO', 'Buy & Hold'],
                'test_results': {
                    'PPO': {
                        'total_return': 0.1483,
                        'final_value': 11483.01,
                        'sharpe_ratio': 1.0476,
                        'max_drawdown': -0.0544,
                        'win_rate': 0.5451,
                        'volatility': 0.0690
                    },
                    'CVaR-PPO': {
                        'total_return': 0.1954,
                        'final_value': 11953.64,
                        'sharpe_ratio': 1.0746,
                        'max_drawdown': -0.0837,
                        'win_rate': 0.5571,
                        'volatility': 0.0872
                    },
                    'Sortino-PPO': {
                        'total_return': 0.1414,
                        'final_value': 11414.24,
                        'sharpe_ratio': 0.9124,
                        'max_drawdown': -0.0700,
                        'win_rate': 0.5471,
                        'volatility': 0.0753
                    },
                    'Buy & Hold': {
                        'total_return': 0.5882,
                        'final_value': 15882.05,
                        'sharpe_ratio': 1.8844,
                        'max_drawdown': -0.0997,
                        'win_rate': 0.5760,
                        'volatility': 0.1281
                    }
                }
            }

        return JsonResponse({
            'success': True,
            'comparison': comparison
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def status(request):
    """Check system status"""
    return JsonResponse({
        'packages_available': PACKAGES_AVAILABLE,
        'models_loaded': evaluator.models_loaded,
        'config': CONFIG
    })


def ppo_page(request):
    """Render PPO model page"""
    return render(request, 'ppo.html')


def cvar_page(request):
    """Render CVaR-PPO model page"""
    return render(request, 'cvar.html')


def sortino_page(request):
    """Render Sortino-PPO model page"""
    return render(request, 'sortino.html')


def compare_page(request):
    """Render model comparison page"""
    return render(request, 'compare.html')


def dashboard_page(request):
    """Render project dashboard page"""
    return render(request, 'dashboard.html')
