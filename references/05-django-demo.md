# 05. Django Demo Và Luồng Hệ Thống

File này giải thích web demo trong thư mục [../demo](../demo): cấu trúc project Django, model artifacts, luồng realtime signal, backtest, compare models, và cách Django tái sử dụng logic từ notebook để chạy inference.

## Mục Tiêu Của Demo

Demo không train model mới. Demo dùng model đã train trong notebook để:

- tạo realtime trading signal cho ticker người dùng nhập;
- chạy backtest trên khoảng thời gian do người dùng chọn;
- so sánh PPO, CVaR-PPO, Sortino-PPO và Buy & Hold;
- trực quan hóa equity curve, drawdown, risk metrics và regime analysis.

Nói ngắn gọn:

```text
Notebook train model -> lưu .pth + norm_params.json -> Django load artifact -> inference/backtest
```

## Cấu Trúc Chính

```text
demo/
├── manage.py
├── requirements.txt
├── db.sqlite3
├── trading/
│   ├── urls.py
│   ├── views.py
│   ├── models.py
│   └── apps.py
├── trading_platform/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── ppo.html
│   ├── cvar.html
│   ├── sortino.html
│   └── compare.html
├── static/
│   ├── css/style.css
│   └── js/main.js
├── weights/
│   ├── ppo_model.pth
│   ├── cvar_ppo_model.pth
│   ├── sortino_ppo_model.pth
│   ├── norm_params.json
│   ├── results.json
│   └── test_results.csv
├── model_v1/
└── model_v2/
```

## Chạy Demo

Từ thư mục gốc project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r demo\requirements.txt
python demo\manage.py check
python demo\manage.py runserver
```

Mở:

```text
http://127.0.0.1:8000/
```

## Dependencies

[../demo/requirements.txt](../demo/requirements.txt) gồm:

```text
Django
djangorestframework
torch
numpy
pandas
yfinance
ta
plotly
seaborn
```

Vai trò:

| Package | Vai trò |
|---|---|
| Django | Web framework |
| djangorestframework | Hỗ trợ API nếu cần mở rộng |
| torch | Load Actor-Critic weights và chạy inference |
| numpy/pandas | Xử lý dữ liệu, metric, timeseries |
| yfinance | Tải dữ liệu thị trường |
| ta | Tính technical indicators |
| plotly | Frontend chart |
| seaborn | Phục vụ phân tích/visualization nếu mở rộng |

## URL Routes

Trong [../demo/trading/urls.py](../demo/trading/urls.py):

| Route | View | Ý nghĩa |
|---|---|---|
| `/` | `index` | Landing page |
| `/dashboard/` | `dashboard_page` | Dashboard tổng quan |
| `/ppo/` | `ppo_page` | Trang PPO |
| `/cvar/` | `cvar_page` | Trang CVaR-PPO |
| `/sortino/` | `sortino_page` | Trang Sortino-PPO |
| `/compare/` | `compare_page` | Trang so sánh model |
| `/api/status/` | `status` | Kiểm tra package/model loaded |
| `/api/realtime_predict/` | `RealtimePredictView` | Sinh tín hiệu realtime |
| `/api/backtest/` | `BacktestView` | Chạy backtest |
| `/api/compare_models/` | `compare_models` | So sánh nhiều chiến lược |
| `/api/test_models/` | `TestModelsView` | API test model cũ/tiện ích |

## Model Artifacts

Django active artifact directory:

```text
demo/weights/
```

Các file:

| File | Vai trò |
|---|---|
| `ppo_model.pth` | Trọng số Actor-Critic của PPO |
| `cvar_ppo_model.pth` | Trọng số Actor-Critic của CVaR-PPO |
| `sortino_ppo_model.pth` | Trọng số Actor-Critic của Sortino-PPO |
| `norm_params.json` | Feature columns, train mean/std, state/action dim |
| `results.json` | Config và metric đã lưu |
| `test_results.csv` | Bảng test metrics |

Thư mục `model_v1` và `model_v2` lưu các phiên bản artifact khác. `model_v2` khớp với nhiều output trong notebook/report, còn `weights` là bản app load mặc định.

## `norm_params.json` Quan Trọng Thế Nào?

Model `.pth` chỉ chứa trọng số. Nó không biết feature nào đứng ở vị trí nào, mean/std ra sao.

`norm_params.json` cung cấp:

```text
feat_cols
feat_mean
feat_std
state_dim
action_dim
```

Backend dùng nó để đảm bảo state trong demo giống state lúc train.

Nếu thiếu hoặc sai:

- feature order có thể lệch;
- scale input có thể sai;
- model vẫn chạy nhưng action vô nghĩa;
- hoặc backend raise `NormalizationConfigError`.

## Các Class Và Hàm Quan Trọng Trong `views.py`

File trung tâm: [../demo/trading/views.py](../demo/trading/views.py)

| Thành phần | Vai trò |
|---|---|
| `load_norm_params()` | Đọc và validate `norm_params.json` |
| `NormalizationConfigError` | Lỗi cấu hình normalization |
| `ActorCritic` | Kiến trúc mạng giống notebook |
| `TradingEnvironment` | Môi trường backtest/inference trong Django |
| `ModelEvaluator` | Load models và evaluate policy |
| `download_and_process_data()` | Tải OHLCV và thêm indicators |
| `download_realtime_market_data()` | Tải dữ liệu đủ dài cho realtime signal |
| `add_technical_indicators()` | Tính SMA, RSI, MACD, Bollinger, ATR, returns |
| `calculate_risk_metrics()` | Tính return, Sharpe, Sortino, drawdown, VaR/CVaR |
| `classify_market_regimes()` | Phân loại bull/sideways/bear-high-vol trong compare |
| `buy_and_hold_strategy()` | Baseline Buy & Hold |
| `run_single_backtest()` | Backtest một model |
| `ranking_summary()` | Tóm tắt model thắng từng metric |

## Luồng Load Model

Khi Django import `views.py`, global evaluator được tạo:

```text
evaluator = ModelEvaluator()
```

`ModelEvaluator.load_models()`:

1. Đọc `norm_params.json`.
2. Khởi tạo `ActorCritic(state_dim, action_dim, hidden_dim=256)`.
3. Load `ppo_model.pth`.
4. Load `cvar_ppo_model.pth`.
5. Load `sortino_ppo_model.pth`.
6. Set `model.eval()`.

Backend dùng CPU:

```text
CONFIG['DEVICE'] = 'cpu'
```

Điều này hợp lý cho demo vì inference nhẹ.

## Luồng Realtime Signal

Endpoint:

```text
POST /api/realtime_predict/
```

Input JSON điển hình:

```json
{
  "symbol": "SPY",
  "model": "ppo"
}
```

Luồng xử lý:

1. Nhận `symbol` và `model`.
2. Tải dữ liệu gần đây bằng `yfinance`.
3. Làm sạch OHLCV.
4. Tính technical indicators.
5. Khởi tạo `TradingEnvironment` với `norm_params`.
6. Di chuyển environment đến timestep cuối.
7. Tạo observation 19 chiều.
8. Load model tương ứng.
9. Chạy `act_deterministic()`.
10. Map action thành signal:

```text
action > 0.1   -> BUY
action < -0.1  -> SELL
otherwise      -> HOLD
```

Lưu ý: threshold realtime signal là `0.1`, lớn hơn threshold execution trong environment (`0.01`). Điều này giúp UI tránh gọi BUY/SELL khi action rất nhỏ và không đáng kể.

Output JSON gồm:

```json
{
  "success": true,
  "symbol": "SPY",
  "current_price": 500.0,
  "timestamp": "...",
  "ppo": {
    "action": 0.25,
    "signal": "BUY",
    "strength": 0.25
  }
}
```

## Luồng Backtest

Endpoint:

```text
POST /api/backtest/
```

Input JSON:

```json
{
  "symbol": "SPY",
  "start_date": "2023-01-01",
  "end_date": "2024-12-31",
  "initial_balance": 10000
}
```

Luồng xử lý:

1. Tải dữ liệu theo symbol/date.
2. Tính indicators.
3. Load models nếu chưa load.
4. Phân loại market regimes.
5. Chạy PPO, CVaR-PPO, Sortino-PPO bằng cùng môi trường.
6. Chạy Buy & Hold baseline.
7. Tính metrics cho từng strategy.
8. Trả về history, portfolio values, drawdowns, action distribution, regime metrics.

Output có dạng:

```text
results.ppo
results.cvar
results.sortino
results.buy_hold
regimes
data_info
```

Frontend dùng dữ liệu này để vẽ chart và metric cards.

## Luồng Compare Models

Endpoint:

```text
GET/POST /api/compare_models/
```

Khác với backtest riêng từng model, compare endpoint luôn chạy cả:

- PPO
- CVaR-PPO
- Sortino-PPO
- Buy & Hold

Sau đó thêm:

- ranking summary;
- regime metrics;
- data info.

Trang [../demo/templates/compare.html](../demo/templates/compare.html) có logic frontend riêng để:

- vẽ equity curve so sánh;
- vẽ metric bar chart;
- vẽ drawdown comparison;
- phân tích regime;
- sinh kết luận tự động từ metric.

## Market Regime Analysis Trong Demo

Hàm `classify_market_regimes(df)` phân loại từng ngày thành:

```text
bull
sideways
bear_high_vol
```

Django dùng các feature như return, rolling trend/volatility để tạo regime label. Đây là phân tích diễn giải từ dữ liệu backtest, không phải một mô hình regime-switching riêng được train trong notebook.

Ý nghĩa:

- Bull: ưu tiên đọc return và upside capture.
- Sideways: đọc turnover, volatility và khả năng tránh nhiễu.
- Bear/high-vol: đọc Max Drawdown, CVaR, Sortino nhiều hơn return.

## Frontend Pages

### `index.html`

Landing page giải thích ý tưởng Risk-Sensitive RL Lab, pipeline từ dữ liệu đến policy/backtest.

### `dashboard.html`

Dashboard tổng quan cho project.

### `ppo.html`, `cvar.html`, `sortino.html`

Mỗi trang model có:

- mô tả mô hình;
- realtime signal;
- form backtest;
- chart equity curve;
- metric cards;
- bảng chi tiết.

### `compare.html`

Trang phân tích chính để so sánh bốn chiến lược.

## Frontend JavaScript

File [../demo/static/js/main.js](../demo/static/js/main.js) xử lý một số logic chung:

- kiểm tra `/api/status/`;
- set ngày mặc định;
- gọi realtime prediction;
- hiển thị signal badge;
- hiển thị phân tích ngắn theo model;
- helper loading/error.

Một phần logic chart/backtest nằm trực tiếp trong từng template HTML, đặc biệt là `compare.html`.

## Tại Sao Demo Tính Lại Indicator?

Model cần state cùng schema như train. Khi người dùng nhập symbol/date mới, backend không thể dùng sẵn file CSV trong `data/`. Nó phải:

1. tải OHLCV mới;
2. tính lại indicator;
3. drop NaN warm-up;
4. chuẩn hóa bằng train stats;
5. đưa vào model.

Điều này cho phép demo chạy với ticker/khoảng thời gian khác, miễn là Yahoo Finance có dữ liệu đủ dài và schema indicator đầy đủ.

## Điểm Khác Giữa Notebook Và Django

| Nội dung | Notebook | Django demo |
|---|---|---|
| Mục tiêu | Train, phân tích, lưu artifact | Load artifact, inference/backtest |
| Device | Có thể dùng CUDA | CPU |
| Action trong training | Sample từ Normal rồi `tanh` | Deterministic mean rồi `tanh` |
| Data | CSV/split cố định trong notebook | Tải lại theo request |
| Normalization | Tính từ train env | Đọc từ `norm_params.json` |
| Output | Bảng, hình, file artifact | JSON API và chart frontend |

## Các Lỗi Dễ Gặp

### Thiếu package

`/api/status/` trả `packages_available = false` nếu thiếu `torch`, `yfinance`, `ta`, `pandas`, ...

Cách xử lý:

```powershell
pip install -r demo\requirements.txt
```

### Không tải được dữ liệu Yahoo Finance

Realtime/backtest có thể lỗi nếu:

- mất mạng;
- ticker sai;
- Yahoo Finance không có dữ liệu;
- khoảng ngày quá ngắn, không đủ warm-up indicator;
- proxy/environment làm yfinance trả empty data.

Trong `views.py` có `yfinance_network_context()` để tránh một số proxy discard làm yfinance trả dữ liệu rỗng.

### Thiếu hoặc sai `norm_params.json`

Backend có thể raise `NormalizationConfigError` nếu:

- không có file;
- thiếu `feat_cols`;
- mean/std không khớp số feature;
- `state_dim` không bằng feature count + 3;
- std không dương.

### Shape mismatch khi load model

Nếu weights được train với state/action dim khác, `load_state_dict` sẽ lỗi. Khi đổi feature set, phải retrain hoặc lưu đúng weights/norm params tương ứng.

### Số liệu demo khác notebook

Do active folder `demo/weights` có thể khác `model_v2`, metric trong web demo có thể lệch nhẹ so với bảng trong notebook/report. Khi viết báo cáo hoặc thuyết trình, nên nói rõ đang dùng artifact nào.

## Nếu Muốn Thay Artifact Active

Hiện demo load từ:

```text
demo/weights
```

Nếu muốn dùng `model_v2` làm active artifact, cần thay các file trong `demo/weights` bằng bộ tương ứng từ `demo/model_v2`:

```text
ppo_model.pth
cvar_ppo_model.pth
sortino_ppo_model.pth
norm_params.json
results.json
test_results.csv
```

Khi thay artifact, phải thay cả weights và `norm_params.json` cùng một bộ. Không nên trộn weights từ version này với norm params từ version khác.

## Checklist Khi Review Demo

- `python demo\manage.py check` chạy không lỗi.
- `/api/status/` báo packages available và models loaded.
- Realtime signal trả action/signal hợp lệ.
- Backtest với SPY 2023-2024 trả đủ results cho 4 strategy.
- Compare page vẽ equity curve và drawdown.
- Metric trong UI khớp với artifact đang load.
- `norm_params.json` có 16 feature columns và `state_dim = 19`.

## Tóm Tắt

Django demo là lớp triển khai inference/backtest cho project. Nó không thay thế notebook training, mà dùng artifact từ notebook để tạo tín hiệu và đánh giá trên dữ liệu người dùng chọn. Phần quan trọng nhất của demo là giữ nhất quán feature engineering và normalization; nếu input state khác lúc train, mọi signal/model comparison đều mất ý nghĩa.
