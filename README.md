# Risk-Sensitive Reinforcement Learning for Trading

Dự án xây dựng và so sánh các chiến lược giao dịch tự động bằng học tăng cường, tập trung vào câu hỏi: một agent có thể vừa tìm kiếm lợi nhuận, vừa kiểm soát rủi ro sụt giảm mạnh của thị trường hay không?

Notebook chính là [risk_sensitive_trading.ipynb](risk_sensitive_trading.ipynb). Từ notebook này, dự án triển khai ba biến thể agent:

- **PPO**: baseline tối ưu lợi nhuận kỳ vọng, chưa có cơ chế phạt rủi ro riêng.
- **CVaR-PPO**: thêm ràng buộc rủi ro đuôi bằng Conditional Value-at-Risk.
- **Sortino-PPO**: điều chỉnh reward theo downside deviation để ưu tiên rủi ro giảm giá.

Dự án cũng có demo Web App bằng Django trong thư mục [demo](demo), dùng model đã train để chạy realtime signal, backtest và so sánh chiến lược.

> Lưu ý: Đây là dự án nghiên cứu/học thuật, không phải khuyến nghị đầu tư.

## Mục Lục

- [Điểm nổi bật](#điểm-nổi-bật)
- [Kết quả nhanh](#kết-quả-nhanh)
- [Cách hoạt động](#cách-hoạt-động)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt và chạy](#cài-đặt-và-chạy)
- [Tài liệu chi tiết](#tài-liệu-chi-tiết)
- [Hướng phát triển](#hướng-phát-triển)

## Điểm Nổi Bật

- Mô hình hóa giao dịch thành bài toán **Markov Decision Process (MDP)** thay vì dự đoán giá trực tiếp.
- Dùng dữ liệu OHLCV từ Yahoo Finance cho **SPY, QQQ, GLD**, trong đó SPY là tài sản huấn luyện chính.
- Chia dữ liệu theo regime thị trường: train 2018-2021, validation 2022, test 2023-2024.
- Không gian trạng thái gồm **19 chiều**: 16 đặc trưng thị trường và 3 đặc trưng danh mục.
- Action liên tục trong khoảng `[-1, 1]`, biểu diễn tỷ trọng bán/mua.
- Mô phỏng chi phí giao dịch `0.1%` và slippage `0.05%`.
- Mạng Actor-Critic dùng chung cho ba agent, gồm **71,427 tham số**.
- Có đánh giá theo test set, crisis stress test và multi-asset generalization.

## Kết Quả Nhanh

Kết quả dưới đây lấy từ output notebook trên test set SPY, giai đoạn **2023-01-03 đến 2024-12-30**, vốn khởi tạo `$10,000`.

| Method      |    Final Value |      Return |     Sharpe |    Sortino |     Calmar |     Max DD |      CVaR-95 |
| ----------- | -------------: | ----------: | ---------: | ---------: | ---------: | ---------: | -----------: |
| PPO         | `$11,371.80` | `+13.72%` | `0.9706` | `1.1964` | `1.1366` | `-5.90%` | `-1.1034%` |
| CVaR-PPO    | `$11,900.86` | `+19.01%` | `1.0729` | `1.4165` | `1.1057` | `-8.31%` | `-1.3178%` |
| Sortino-PPO | `$11,553.35` | `+15.53%` | `0.9589` | `1.1930` | `1.0465` | `-7.23%` | `-1.2552%` |
| Buy & Hold  | `$15,882.05` | `+58.82%` | `1.8844` | `2.9956` | `2.6325` | `-9.97%` | `-1.7422%` |

Trong bull market 2023-2024, Buy & Hold có lợi nhuận cao nhất. Tuy nhiên, stress test cho thấy CVaR-PPO phòng thủ tốt hơn trong giai đoạn COVID crash:

| Period                |   PPO Return / MaxDD | CVaR-PPO Return / MaxDD | Sortino-PPO Return / MaxDD | Buy & Hold Return / MaxDD |
| --------------------- | -------------------: | ----------------------: | -------------------------: | ------------------------: |
| COVID crash 2020      | `-9.73% / -26.36%` |    `+1.44% / -14.97%` |       `-5.62% / -23.32%` |      `-5.56% / -33.72%` |
| Bear market 2022      | `-6.92% / -13.66%` |    `-6.43% / -13.98%` |       `-8.64% / -15.18%` |     `-18.65% / -24.50%` |
| Bull market 2023-2024 | `+20.39% / -5.78%` |    `+26.76% / -6.21%` |       `+14.33% / -6.56%` |      `+58.82% / -9.97%` |

Diễn giải ngắn:

- **CVaR-PPO** là biến thể nổi bật nhất trong COVID crash, vừa có return dương vừa giảm drawdown sâu.
- **PPO** có Max Drawdown tốt nhất trên test set 2023-2024, nhưng kém hơn CVaR-PPO trong crisis.
- **Sortino-PPO** nằm giữa hai hướng: ổn định hơn PPO trong một số pha rủi ro, nhưng chưa vượt CVaR-PPO ở stress test.
- **Buy & Hold** thắng mạnh trong bull market, đổi lại chịu tail risk và drawdown lớn hơn ở crisis.

## Cách Hoạt Động

Pipeline trong notebook gồm các bước chính:

1. **Tải dữ liệu** bằng `yfinance` cho SPY, QQQ và GLD từ `2018-01-01` đến `2024-12-31`.
2. **Tạo đặc trưng kỹ thuật**: SMA, RSI, MACD, Bollinger Bands, ATR, returns, log returns và volume SMA.
3. **Chia dữ liệu theo regime**:
   - Train: `2018-03-14` đến `2021-12-31`, gồm COVID crash.
   - Validation: `2022-01-03` đến `2022-12-30`, thị trường gấu.
   - Test: `2023-01-03` đến `2024-12-30`, phục hồi và bull run.
4. **Huấn luyện agent** trong môi trường `gymnasium.Env` tự xây dựng.
5. **Đánh giá** bằng Return, Sharpe, Sortino, Calmar, Max Drawdown, VaR-95, CVaR-95, volatility và win rate.
6. **Lưu model** và tham số chuẩn hóa vào thư mục weights để phục vụ demo Django.

## Cấu Trúc Thư Mục

```text
.
├── README.md
├── risk_sensitive_trading.ipynb
├── demo/
│   ├── manage.py
│   ├── requirements.txt
│   ├── trading/
│   │   ├── urls.py
│   │   └── views.py
│   ├── templates/
│   ├── static/
│   └── weights/
│       ├── ppo_model.pth
│       ├── cvar_ppo_model.pth
│       ├── sortino_ppo_model.pth
│       ├── norm_params.json
│       ├── results.json
│       └── test_results.csv
└── references/
    ├── README.md
    ├── 01-data-and-features.md
    ├── 02-trading-environment.md
    ├── 03-methods.md
    ├── 04-training-and-results.md
    └── 05-django-demo.md
```

## Cài Đặt Và Chạy

Yêu cầu khuyến nghị: Python 3.11+.

### Chạy Web Demo

Từ thư mục gốc của project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r demo\requirements.txt
python demo\manage.py check
python demo\manage.py runserver
```

Mở trình duyệt tại:

```text
http://127.0.0.1:8000/
```

Các chức năng chính của demo:

- Dashboard tổng quan.
- Realtime signal cho model đã train.
- Backtest theo mã tài sản và khoảng thời gian.
- Compare models giữa PPO, CVaR-PPO, Sortino-PPO và Buy & Hold.

### Chạy Notebook

Notebook được thiết kế để chạy trên Kaggle GPU, nhưng vẫn có thể chạy local nếu cài đủ thư viện:

```powershell
pip install yfinance ta gymnasium torch pandas numpy matplotlib plotly
```

Sau đó mở [risk_sensitive_trading.ipynb](risk_sensitive_trading.ipynb) bằng Jupyter, VS Code hoặc Kaggle Notebook và chạy lần lượt các cell.

## Tài Liệu Chi Tiết

Các phần phân tích sâu đã được tách vào thư mục [references](references):

- [Tổng quan references](references/README.md)
- [Dữ liệu và feature engineering](references/01-data-and-features.md)
- [Môi trường giao dịch](references/02-trading-environment.md)
- [Kiến trúc và thuật toán](references/03-methods.md)
- [Huấn luyện, đánh giá và phân tích kết quả](references/04-training-and-results.md)
- [Django demo và luồng hệ thống](references/05-django-demo.md)

## Hướng Phát Triển

- Huấn luyện dài hơn và kiểm tra nhiều seed để giảm phụ thuộc vào một lần chạy.
- Thêm LSTM/Transformer để học phụ thuộc thời gian thay vì chỉ dùng state tại một ngày.
- Mở rộng sang multi-asset portfolio thật sự với joint action space.
- Bổ sung VIX, yield curve, macro và sentiment features.
- Thêm walk-forward validation để đánh giá ổn định hơn trên nhiều giai đoạn thị trường.
