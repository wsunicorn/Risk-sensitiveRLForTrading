# Risk-Sensitive Reinforcement Learning for Trading

Dự án này xây dựng một hệ thống thử nghiệm giao dịch định lượng bằng học tăng cường sâu, tập trung vào câu hỏi: **một agent có thể học chính sách mua/bán/nắm giữ vừa tạo lợi nhuận, vừa kiểm soát rủi ro giảm sâu của danh mục hay không?**

Khác với mô hình dự báo giá trực tiếp, project mô hình hóa giao dịch như một bài toán ra quyết định tuần tự. Ở mỗi ngày giao dịch, agent nhìn thấy trạng thái thị trường, trạng thái danh mục, sinh ra một action liên tục trong `[-1, 1]`, môi trường thực thi mua/bán với chi phí và trượt giá, rồi trả reward dựa trên thay đổi giá trị danh mục.

Notebook chính: [risk_sensitive_trading.ipynb](risk_sensitive_trading.ipynb)

Báo cáo Word đã được chuyển sang Markdown để dễ đọc và tra cứu:

- [docs/Nhom15_Report.md](docs/Nhom15_Report.md)
- Ảnh trích từ report nằm trong [docs/Nhom15_Report_media](docs/Nhom15_Report_media)

> Dự án phục vụ mục đích nghiên cứu/học thuật. Đây không phải khuyến nghị đầu tư, không phải hệ thống giao dịch thật và không đảm bảo sinh lời ngoài thị trường.

## Thông Tin Nhóm Thực Hiện

Đề tài được thực hiện trong môn **Giao dịch định lượng**, thuộc Khoa Công nghệ Thông tin, Trường Đại học Công nghiệp TP. Hồ Chí Minh.

| Thông tin                | Nội dung                                         |
| ------------------------- | ------------------------------------------------- |
| Tên đề tài            | Risk-Sensitive Reinforcement Learning for Trading |
| Giảng viên hướng dẫn | TS. Bùi Thanh Hùng                              |
| Lớp                      | 420301411301                                      |
| Khóa                     | 19                                                |
| Năm thực hiện          | 2026                                              |

| Thành viên          |     MSSV | Đóng góp chính                                                                                                                                             |
| --------------------- | -------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Nguyễn Ngọc Lân    | 22635801 | Thu thập và tiền xử lý dữ liệu, trích xuất đặc trưng, chuẩn hóa, huấn luyện PPO/CVaR-PPO/Sortino-PPO, hỗ trợ Django demo và viết báo cáo |
| Lê Huỳnh Tấn Đạt | 23634031 | Xây dựng logic tín hiệu BUY/SELL/HOLD, chỉ số đánh giá tài chính, kiểm thử backtest, hỗ trợ Django demo và viết báo cáo                     |

## Tóm Tắt Ý Tưởng

Bài toán trading có ba đặc điểm khiến học tăng cường phù hợp:

- Quyết định hiện tại ảnh hưởng đến trạng thái tương lai của danh mục.
- Kết quả không chỉ là dự đoán đúng/sai, mà là chuỗi hành động tạo ra equity curve.
- Lợi nhuận cao chưa đủ tốt nếu đi kèm drawdown, volatility hoặc tail loss quá lớn.

Vì vậy project so sánh bốn chiến lược:

| Chiến lược | Vai trò              | Ý nghĩa                                                                              |
| ------------- | --------------------- | -------------------------------------------------------------------------------------- |
| Buy & Hold    | Benchmark thụ động | Mua từ đầu kỳ và giữ đến cuối kỳ, dùng làm chuẩn đối chiếu đơn giản |
| PPO           | Baseline RL           | Tối ưu reward kỳ vọng, chưa có cơ chế phạt rủi ro riêng                     |
| CVaR-PPO      | Risk-sensitive RL     | Thêm penalty cho nhóm return xấu nhất để chú ý tail risk                       |
| Sortino-PPO   | Risk-sensitive RL     | Điều chỉnh reward theo downside deviation để phạt biến động bất lợi         |

Điểm quan trọng không phải là “RL luôn thắng Buy & Hold”. Trên bull market 2023-2024, Buy & Hold thắng lợi nhuận tuyệt đối. Ý nghĩa của project là chỉ ra cách đọc **trade-off giữa lợi nhuận và rủi ro**, đặc biệt trong crisis period như COVID crash hoặc bear market 2022.

## Pipeline Tổng Quát

1. Tải dữ liệu OHLCV daily từ Yahoo Finance cho `SPY`, `QQQ`, `GLD`.
2. Tạo technical indicators: SMA, RSI, MACD, Bollinger Bands, ATR, returns, log returns, volume SMA.
3. Chia dữ liệu theo thời gian và market regime:
   - Train: `2018-03-14` đến `2021-12-31`
   - Validation: `2022-01-03` đến `2022-12-30`
   - Test: `2023-01-03` đến `2024-12-30`
4. Xây dựng môi trường giao dịch dạng `gymnasium.Env`.
5. Huấn luyện PPO, CVaR-PPO và Sortino-PPO trong cùng kiến trúc Actor-Critic.
6. Đánh giá bằng Return, Sharpe, Sortino, Calmar, Max Drawdown, VaR-95, CVaR-95, Volatility, Win Rate.
7. Lưu model weights và normalization params để phục vụ Django demo.

## Dữ Liệu Và State

SPY là tài sản huấn luyện chính. QQQ và GLD dùng để kiểm tra khả năng tổng quát hóa khi model đã train trên SPY được áp dụng sang tài sản khác mà không retrain.

Sau feature engineering, mỗi dòng dữ liệu có 20 cột:

```text
date, open, high, low, close, volume,
sma_10, sma_20, sma_50, rsi,
macd, macd_signal, macd_diff,
bb_high, bb_low, bb_mid,
atr, returns, log_returns, volume_sma
```

Observation mà agent nhận gồm **19 chiều**:

- 16 market features: `close`, `volume`, các indicator và return.
- 3 portfolio features:
  - `balance / initial_balance`
  - `shares * price / initial_balance`
  - `portfolio_value / initial_balance`

Các market features được chuẩn hóa bằng Z-score dựa trên thống kê của tập train:

```text
normalized_feature = (raw_feature - train_mean) / train_std
```

File [demo/weights/norm_params.json](demo/weights/norm_params.json) lưu `feat_cols`, `feat_mean`, `feat_std`, `state_dim` và `action_dim`. Đây là cầu nối rất quan trọng giữa notebook và Django demo: nếu realtime/backtest dùng normalization khác lúc train, state đưa vào model sẽ lệch phân phối.

## Môi Trường Giao Dịch

Môi trường mô phỏng một tài sản, vốn khởi tạo `$10,000`, transaction cost `0.1%` và slippage `0.05%`.

Action là một số thực `a` trong `[-1, 1]`:

|                 Action | Diễn giải                                                  |
| ---------------------: | ------------------------------------------------------------ |
|           `a > 0.01` | BUY, dùng tỷ lệ `a` của cash hiện có để mua        |
|          `a < -0.01` | SELL, bán tỷ lệ `abs(a)` của số cổ phiếu đang giữ |
| `-0.01 <= a <= 0.01` | HOLD                                                         |

Reward cơ sở:

```text
reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
```

PPO dùng reward này trực tiếp. CVaR-PPO giữ reward cơ sở nhưng thêm penalty vào loss. Sortino-PPO sửa reward bằng downside penalty.

## Phương Pháp

Ba agent dùng cùng mạng Actor-Critic:

```text
Input state_dim = 19
-> Linear(19, 256) + ReLU
-> Linear(256, 256) + ReLU
-> Actor head: Linear(256, 1)
-> Critic head: Linear(256, 1)
```

Notebook ghi nhận tổng số tham số: **71,427**.

### PPO

PPO là baseline trung lập rủi ro. Thuật toán dùng clipped surrogate objective để cập nhật policy vừa đủ lớn để học, vừa tránh làm policy mới lệch quá mạnh so với policy cũ.

PPO phù hợp làm baseline vì:

- ổn định hơn policy gradient thuần;
- hỗ trợ action liên tục;
- dễ mở rộng bằng risk penalty;
- đủ phổ biến để so sánh với các nghiên cứu DRL trading khác.

### CVaR-PPO

CVaR-PPO thêm penalty cho tail risk. Ý tưởng là không chỉ quan tâm trung bình return, mà quan tâm phần đuôi xấu của phân phối return.

Trong notebook:

- `CVAR_ALPHA = 0.15`: xét worst 15% returns.
- `CVAR_LAMBDA = 0.15`: trọng số phạt ban đầu.
- `CVAR_LAMBDA_DECAY = 0.999`: giảm dần penalty.
- `CVAR_LAMBDA_MIN = 0.08`: mức penalty tối thiểu.
- Return được chuẩn hóa trước khi tính CVaR để tránh loss bị chi phối bởi scale.

CVaR-PPO có ý nghĩa nhất khi thị trường có tail event, ví dụ COVID crash.

### Sortino-PPO

Sortino-PPO đưa downside risk trực tiếp vào reward:

```text
reward_sortino = portfolio_return - penalty
penalty = sortino_lambda * downside_std^2
```

Notebook dùng rolling window 60 bước, chỉ tính standard deviation trên các return âm, và cap penalty tối đa 50% độ lớn reward để tránh penalty áp đảo tín hiệu lợi nhuận.

Sortino-PPO hữu ích vì trong trading, upside volatility không xấu như downside volatility. Một chiến lược tăng mạnh nhưng biến động lên xuống đều lớn có thể bị Sharpe phạt, còn Sortino tập trung vào phần giảm giá bất lợi hơn.

## Kết Quả Nhanh

Nguồn bảng dưới đây là output trong [risk_sensitive_trading.ipynb](risk_sensitive_trading.ipynb), tương ứng với artifact [demo/model_v2](demo/model_v2). Thư mục [demo/weights](demo/weights) là artifact đang được Django demo load, có một vài con số lệch nhẹ do khác lần chạy/phiên bản weights; xem chi tiết ở [references/04-training-and-results.md](references/04-training-and-results.md).

Test set: SPY, `2023-01-03` đến `2024-12-30`, vốn khởi tạo `$10,000`.

| Method      |    Final Value |      Return |     Sharpe |    Sortino |     Calmar |     Max DD |      CVaR-95 |
| ----------- | -------------: | ----------: | ---------: | ---------: | ---------: | ---------: | -----------: |
| PPO         | `$11,371.80` | `+13.72%` | `0.9706` | `1.1964` | `1.1366` | `-5.90%` | `-1.1034%` |
| CVaR-PPO    | `$11,900.86` | `+19.01%` | `1.0729` | `1.4165` | `1.1057` | `-8.31%` | `-1.3178%` |
| Sortino-PPO | `$11,553.35` | `+15.53%` | `0.9589` | `1.1930` | `1.0465` | `-7.23%` | `-1.2552%` |
| Buy & Hold  | `$15,882.05` | `+58.82%` | `1.8844` | `2.9956` | `2.6325` | `-9.97%` | `-1.7422%` |

Diễn giải:

- Buy & Hold thắng mạnh trong test period vì 2023-2024 là giai đoạn phục hồi và bull run.
- CVaR-PPO có Return, Sharpe và Sortino tốt nhất trong ba model RL trên test set của notebook.
- PPO có Max Drawdown và CVaR-95 tốt nhất trong test set này, nhưng không phải lúc nào cũng phòng thủ tốt nhất trong crisis.
- RL models có volatility thấp hơn Buy & Hold, đổi lại thường bỏ lỡ một phần upside của bull market.

Stress test trong notebook:

| Period                |   PPO Return / MaxDD | CVaR-PPO Return / MaxDD | Sortino-PPO Return / MaxDD | Buy & Hold Return / MaxDD |
| --------------------- | -------------------: | ----------------------: | -------------------------: | ------------------------: |
| COVID crash 2020      | `-9.73% / -26.36%` |    `+1.44% / -14.97%` |       `-5.62% / -23.32%` |      `-5.56% / -33.72%` |
| Bear market 2022      | `-6.92% / -13.66%` |    `-6.43% / -13.98%` |       `-8.64% / -15.18%` |     `-18.65% / -24.50%` |
| Bull market 2023-2024 | `+20.39% / -5.78%` |    `+26.76% / -6.21%` |       `+14.33% / -6.56%` |      `+58.82% / -9.97%` |

Kết luận quan trọng: **CVaR-PPO thể hiện rõ giá trị trong COVID crash**, nơi tail-risk penalty giúp giảm drawdown sâu và thậm chí tạo return dương trong notebook run.

## Cấu Trúc Thư Mục

```text
.
├── README.md
├── risk_sensitive_trading.ipynb
├── data/
│   ├── raw_SPY.csv
│   ├── featured_SPY.csv
│   ├── split_SPY_train.csv
│   ├── split_SPY_val.csv
│   └── split_SPY_test.csv
├── demo/
│   ├── manage.py
│   ├── requirements.txt
│   ├── trading/
│   │   ├── urls.py
│   │   └── views.py
│   ├── templates/
│   ├── static/
│   ├── weights/
│   ├── model_v1/
│   └── model_v2/
├── docs/
│   ├── Nhom15_Report.docx
│   ├── Nhom15_Report.pdf
│   ├── Nhom15_Report.md
│   └── Nhom15_Report_media/
└── references/
    ├── README.md
    ├── 01-data-and-features.md
    ├── 02-trading-environment.md
    ├── 03-methods.md
    ├── 04-training-and-results.md
    └── 05-django-demo.md
```

## Chạy Web Demo

Yêu cầu khuyến nghị: Python 3.11+.

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

Các trang chính:

- `/`: landing page.
- `/dashboard/`: dashboard tổng quan.
- `/ppo/`: realtime signal và backtest PPO.
- `/cvar/`: realtime signal và backtest CVaR-PPO.
- `/sortino/`: realtime signal và backtest Sortino-PPO.
- `/compare/`: so sánh PPO, CVaR-PPO, Sortino-PPO và Buy & Hold.

Các API chính:

```text
/api/status/
/api/realtime_predict/
/api/backtest/
/api/compare_models/
```

## Chạy Notebook

Notebook được thiết kế để chạy trên môi trường có GPU như Kaggle, nhưng vẫn có thể chạy local nếu cài đủ thư viện:

```powershell
pip install yfinance ta gymnasium torch pandas numpy matplotlib plotly seaborn
```

Sau đó mở [risk_sensitive_trading.ipynb](risk_sensitive_trading.ipynb) bằng Jupyter, VS Code hoặc Kaggle Notebook và chạy tuần tự các cell.

## Tài Liệu Chi Tiết

Nếu muốn review sâu trong tương lai, nên đọc theo thứ tự:

1. [references/01-data-and-features.md](references/01-data-and-features.md): dữ liệu, feature engineering, normalization và data leakage.
2. [references/02-trading-environment.md](references/02-trading-environment.md): MDP, state, action, reward, chi phí, slippage.
3. [references/03-methods.md](references/03-methods.md): PPO, CVaR-PPO, Sortino-PPO, lý do chọn và trade-off.
4. [references/04-training-and-results.md](references/04-training-and-results.md): training config, metrics, kết quả, stress test, multi-asset evaluation.
5. [references/05-django-demo.md](references/05-django-demo.md): kiến trúc web app, artifact, API, luồng inference/backtest.

## Hạn Chế

- Backtest không đồng nghĩa với hiệu quả ngoài thị trường.
- Dữ liệu daily đơn giản, chưa có intraday, order book, spread thật hoặc liquidity constraint.
- Agent train chủ yếu trên một tài sản SPY; multi-asset trong notebook là kiểm tra generalization, chưa phải portfolio optimization đa tài sản thật.
- Mỗi model mới được đánh giá trên số lần chạy hạn chế, chưa có nhiều seed để đo độ ổn định thống kê.
- Market regime thay đổi theo thời gian; policy học trên lịch sử có thể không thích nghi nếu phân phối thị trường đổi mạnh.
- Buy & Hold thắng lợi nhuận trong bull market là điều hợp lý; không nên kết luận RL kém chỉ từ return cuối kỳ.

## Hướng Phát Triển

- Chạy nhiều seed và báo cáo mean/std của metric.
- Thêm walk-forward validation để giảm phụ thuộc vào một split cố định.
- Huấn luyện multi-asset portfolio thật sự với joint action space.
- Thêm VIX, yield curve, macro, sentiment và market regime features.
- Thử LSTM/Transformer actor để học phụ thuộc thời gian thay vì chỉ nhìn state một ngày.
- Thử reward có drawdown penalty, turnover penalty hoặc transaction-cost-aware objective rõ hơn.
- Triển khai paper trading để kiểm tra độ ổn định trước khi nghĩ đến giao dịch thật.
