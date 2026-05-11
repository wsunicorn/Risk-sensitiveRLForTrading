# Risk-Sensitive RL for Trading

## 1. Giới thiệu

Đây là web demo học thuật mô phỏng policy trading bằng Reinforcement Learning trên Django. Hệ thống không được thiết kế để dự đoán giá trực tiếp; trong demo này, model học policy/action cho giao dịch dựa trên market state và trạng thái danh mục, sau đó action được diễn giải thành tín hiệu BUY / SELL / HOLD.

Mục tiêu của project là so sánh các policy PPO, CVaR-PPO, Sortino-PPO với benchmark Buy & Hold bằng backtest và risk metrics. Kết quả chỉ phục vụ mục đích thực nghiệm, báo cáo học thuật và demo local, không phải khuyến nghị đầu tư.

## 2. Ý tưởng chính

Trading được mô hình hóa như một bài toán ra quyết định tuần tự:

- Market state gồm OHLCV, technical indicators và trạng thái danh mục.
- Policy sinh action liên tục trong khoảng đầu ra của actor network.
- Tín hiệu hiển thị được map từ action thành BUY / SELL / HOLD. Trong realtime signal: `action > 0.1` là BUY, `action < -0.1` là SELL, còn lại là HOLD.
- Trong backtest, môi trường trading thực thi mua/bán theo action với transaction cost `0.001` và slippage `0.0005`.
- Kết quả được đánh giá bằng Equity Curve, Total Return, Sharpe Ratio, Max Drawdown, Volatility, Win Rate và Number of Trades.

Risk-sensitive RL tập trung vào trade-off giữa return và risk, không chỉ tối đa hóa lợi nhuận kỳ vọng. Vì vậy, kết quả cần đọc cùng drawdown, volatility và các metric điều chỉnh theo rủi ro.

## 3. Các mô hình được so sánh

| Mô hình | Vai trò | Trực giác |
|---|---|---|
| PPO | Baseline risk-neutral | Tối ưu lợi nhuận kỳ vọng, không có thành phần riêng để phạt tail risk. |
| CVaR-PPO | Risk-sensitive policy | Chú trọng tail risk và các kịch bản xấu. |
| Sortino-PPO | Downside-risk policy | Chú trọng downside volatility thay vì toàn bộ volatility. |
| Buy & Hold | Benchmark thụ động | Mua và nắm giữ tài sản để đối chiếu với policy RL. |

## 4. Luồng xử lý dữ liệu và thực nghiệm

Pipeline backend hiện tại nằm chủ yếu trong `demo/trading/views.py`:

1. Tải dữ liệu OHLCV / market data bằng `yfinance`.
2. Chuẩn hóa schema về các cột `date`, `open`, `high`, `low`, `close`, `volume`.
3. Làm sạch dữ liệu bằng cách parse date, sort date, drop duplicate dates, drop missing và inf trên các cột OHLCV.
4. Tạo technical indicators: SMA 10/20/50, RSI, MACD, Bollinger Bands, ATR, returns, log returns và volume SMA.
5. Drop warmup NaN sau khi tính indicators.
6. Chuẩn hóa feature bằng train normalization params trong `demo/norm_params.json`.
7. Tạo state vector gồm 16 market features đã normalize và 3 portfolio features: normalized balance, shares value và portfolio value.
8. Load model artifacts `.pth` từ `demo/` và cho policy sinh action deterministic.
9. Map action sang BUY / SELL / HOLD cho realtime signal, hoặc thực thi mua/bán trong backtest.
10. Backtest danh mục và tính return + risk metrics.

`norm_params.json` lưu `feat_cols`, `feat_mean`, `feat_std`, `state_dim = 19` và `action_dim = 1`. Backend bắt buộc dùng các tham số normalization này khi inference/backtest, giúp tránh việc tính mean/std từ dữ liệu test hoặc realtime và giảm nguy cơ data leakage.

Backend cũng phân loại market regime bằng rolling return 60 ngày và rolling volatility 60 ngày, gồm `bull`, `sideways` và `bear_high_vol`, để hỗ trợ regime analysis trong backtest/compare.

## 5. Tính năng web demo

| Route | Trang | Mô tả |
|---|---|---|
| `/` | Overview | Giới thiệu project, ý tưởng policy trading và workflow. |
| `/dashboard/` | Dashboard | Tổng quan cấu hình thực nghiệm, data, model và metric. |
| `/ppo/` | PPO | Realtime Signal và Backtest cho PPO. |
| `/cvar/` | CVaR-PPO | Realtime Signal và Backtest cho CVaR-PPO. |
| `/sortino/` | Sortino-PPO | Realtime Signal và Backtest cho Sortino-PPO. |
| `/compare/` | Compare | So sánh PPO, CVaR-PPO, Sortino-PPO và Buy & Hold trên cùng symbol/giai đoạn, kèm bảng metric và biểu đồ. |

## 6. API endpoints

| Endpoint | Method | Mục đích | Input chính | Output chính |
|---|---|---|---|---|
| `/api/status/` | GET | Kiểm tra trạng thái hệ thống. | Không cần body. | `packages_available`, `models_loaded`, `config`. |
| `/api/realtime_predict/` | POST | Lấy realtime signal cho một model. | JSON gồm `symbol` và `model` (`ppo`, `cvar`, `sortino`). | `success`, `symbol`, `current_price`, `timestamp`, và block theo model gồm `action`, `signal`, `strength`. |
| `/api/backtest/` | POST | Chạy backtest cho PPO, CVaR-PPO, Sortino-PPO và Buy & Hold. | JSON gồm `symbol`, `start_date`, `end_date`, `initial_balance`. | `results`, `regimes`, `regime_labels`, `data_info`. Mỗi result có metrics, history, portfolio values, drawdowns và regime metrics. |
| `/api/compare_models/` | GET/POST | So sánh các strategy trên cùng dữ liệu/giai đoạn. | GET query hoặc POST JSON gồm `symbol`, `start_date`, `end_date`, `initial_balance`. | `results`, `regimes`, `ranking`, `data_info`. |

Ghi chú: code hiện có thêm `/api/test_models/` cho việc test model, nhưng luồng UI chính đang sử dụng các endpoint ở bảng trên.

## 7. Cấu trúc thư mục

```text
.
├── demo/
│   ├── manage.py
│   ├── requirements.txt
│   ├── trading/
│   │   ├── urls.py
│   │   └── views.py
│   ├── trading_platform/
│   │   ├── settings.py
│   │   └── urls.py
│   ├── templates/
│   │   ├── index.html
│   │   ├── dashboard.html
│   │   ├── ppo.html
│   │   ├── cvar.html
│   │   ├── sortino.html
│   │   └── compare.html
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/main.js
│   ├── ppo_model.pth
│   ├── cvar_ppo_model.pth
│   ├── sortino_ppo_model.pth
│   ├── norm_params.json
│   ├── results.json
│   └── test_results.csv
├── risk_sensitive_trading_v1.ipynb
├── risk_sensitive_trading_v2.ipynb
├── risk_sensitive_trading_v2_result.ipynb
└── README.md
```

Thư mục `demo/v1/` và `demo/v2/` lưu thêm artifacts/results theo phiên bản thực nghiệm. Backend Django hiện tại load artifacts mặc định từ `demo/`.

## 8. Cài đặt và chạy project

Lệnh bên dưới dành cho Windows PowerShell, chạy từ root repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r demo\requirements.txt
python demo\manage.py check
python demo\manage.py runserver
```

Sau khi server chạy, mở:

```text
http://127.0.0.1:8000/
```

Dependency chính nằm trong `demo/requirements.txt`: Django, Django REST Framework, PyTorch, NumPy, pandas, yfinance, ta, plotly và seaborn. Frontend template dùng Bootstrap, Font Awesome, Google Fonts và Plotly.js qua CDN.

## 9. Cách sử dụng demo

1. Mở `/` để xem overview và workflow.
2. Vào `/dashboard/` để xem tổng quan cấu hình thực nghiệm, model và metric.
3. Vào `/ppo/`, `/cvar/` hoặc `/sortino/` để lấy Realtime Signal hoặc chạy Backtest cho từng model.
4. Vào `/compare/` để so sánh các model với Buy & Hold trên cùng `symbol`, `initial_balance`, `start_date`, `end_date`.
5. Đọc kết quả qua Equity Curve, Drawdown, Total Return, Sharpe Ratio, Volatility, Win Rate, Number of Trades và regime analysis nếu dữ liệu đủ dài.

Mặc định UI ưu tiên `SPY`, vì đây là symbol phù hợp với artifacts/normalization hiện có. Có thể thử symbol khác, nhưng kết quả phụ thuộc vào dữ liệu `yfinance`, feature availability và độ phù hợp với distribution đã train.

## 10. Validation / kiểm thử

Nên chạy các lệnh sau trước khi nộp/demo:

```powershell
python -B demo\manage.py check
git diff --check
git status --short
```

Có thể xem nội dung thay đổi README bằng:

```powershell
git diff -- README.md
```

Nếu `README.md` mới được tạo và chưa được add vào Git index, `git diff -- README.md` có thể không hiện nội dung. Khi đó dùng:

```powershell
git diff --no-index NUL README.md
```

Trên Windows, nếu Git báo CRLF/LF warning thì đó là cảnh báo line ending, không phải lỗi logic của Django backend.

## 11. Lưu ý học thuật và giới hạn

- Project này là research prototype / demo học thuật, không phải hệ thống giao dịch thật.
- Realtime Signal là phản ứng của policy theo state hiện tại, không phải dự báo giá chắc chắn.
- Backtest không đảm bảo hiệu suất tương lai.
- Kết quả phụ thuộc vào symbol, giai đoạn backtest, model artifacts, normalization params và dữ liệu từ `yfinance`.
- Model artifacts hiện tại được load local từ repo; nếu thay đổi artifacts hoặc `norm_params.json`, kết quả inference/backtest có thể thay đổi.
- Các risk metrics cần được đọc cùng nhau. Total Return cao nhưng Max Drawdown/Volatility lớn vẫn có thể là chiến lược kém ổn định.
- Chưa nên sử dụng demo này cho quyết định đầu tư thật nếu không có validation độc lập, walk-forward testing, transaction-cost modeling đầy đủ và quản trị rủi ro riêng.

## 12. Trạng thái hiện tại

- UI/UX đã được polish cho demo local và báo cáo học thuật.
- Backend Django dùng model artifacts `.pth` và `norm_params.json` hiện có trong `demo/`.
- Các trang chính và API endpoint cho realtime signal, backtest và compare đã sẵn sàng để chạy local.
- Repo phù hợp để giáo viên, teammate hoặc reviewer GitHub đọc hiểu mục tiêu, cách chạy, kiến trúc và logic thực nghiệm của project.
