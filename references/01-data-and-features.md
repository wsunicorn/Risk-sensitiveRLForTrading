# 01. Dữ Liệu Và Feature Engineering

## Nguồn Dữ Liệu

Notebook tải dữ liệu lịch sử OHLCV từ Yahoo Finance thông qua `yfinance`.

| Symbol | Vai trò | Số dòng raw | Khoảng thời gian | Close min / max / mean |
|---|---|---:|---|---:|
| SPY | Tài sản huấn luyện chính | 1,760 | 2018-01-02 đến 2024-12-30 | `$204.94 / $597.11 / $361.48` |
| QQQ | Kiểm tra generalization | 1,760 | 2018-01-02 đến 2024-12-30 | `$137.16 / $533.92 / $289.74` |
| GLD | Kiểm tra generalization | 1,760 | 2018-01-02 đến 2024-12-30 | `$111.10 / $257.50 / $165.12` |

SPY được dùng làm `PRIMARY_SYMBOL` để train ba agent. QQQ và GLD được dùng ở phần multi-asset evaluation, không retrain model.

## Làm Sạch Dữ Liệu

Hàm `download_data()` trong notebook thực hiện:

- Tải dữ liệu daily OHLCV.
- Reset index và chuẩn hóa tên cột thành chữ thường.
- Giữ các cột `date`, `open`, `high`, `low`, `close`, `volume`.
- Loại bỏ ngày trùng.
- Loại bỏ dòng có giá OHLC không hợp lệ hoặc thiếu dữ liệu.
- Sắp xếp lại theo thời gian.

## Technical Indicators

Sau khi gọi `add_features()`, mỗi symbol còn 1,711 dòng và 20 cột. Các feature kỹ thuật chính:

| Nhóm | Feature |
|---|---|
| Trend | `sma_10`, `sma_20`, `sma_50` |
| Momentum | `rsi`, `macd`, `macd_signal`, `macd_diff` |
| Volatility | `bb_high`, `bb_low`, `bb_mid`, `atr` |
| Return | `returns`, `log_returns` |
| Volume | `volume_sma` |

Khi đưa vào môi trường giao dịch, notebook dùng 16 market features:

```text
close, volume, sma_10, sma_20, sma_50, rsi, macd, macd_signal,
macd_diff, bb_high, bb_low, bb_mid, atr, returns, log_returns, volume_sma
```

Ba đặc trưng danh mục được nối thêm vào observation:

- `balance / initial_balance`
- `shares * price / initial_balance`
- `portfolio_value / initial_balance`

Vì vậy observation cuối cùng có kích thước **19 chiều**.

## Chia Dữ Liệu Theo Regime

Notebook không chia ngẫu nhiên vì dữ liệu tài chính có tính chuỗi thời gian. Thay vào đó, dữ liệu được chia theo giai đoạn thị trường:

| Split | Số dòng | Khoảng thời gian | Ý nghĩa |
|---|---:|---|---|
| Train | 959 | 2018-03-14 đến 2021-12-31 | Có uptrend và COVID crash |
| Validation | 251 | 2022-01-03 đến 2022-12-30 | Bear market 2022 |
| Test | 501 | 2023-01-03 đến 2024-12-30 | Recovery và bull run |

Cách chia này giúp kiểm tra agent có học được chính sách bền vững qua nhiều regime hay chỉ overfit vào một pha thị trường.

## Chuẩn Hóa Feature

Môi trường tính `mean` và `std` của feature trên tập train, sau đó chuẩn hóa theo công thức:

```text
normalized_feature = (raw_feature - train_mean) / train_std
```

Các tham số chuẩn hóa được lưu trong:

```text
demo/weights/norm_params.json
```

File này rất quan trọng cho demo inference, vì dữ liệu realtime/backtest trên web phải được chuẩn hóa giống đúng lúc train. Nếu chuẩn hóa lại bằng dữ liệu test hoặc dữ liệu realtime, mô hình có nguy cơ bị data leakage hoặc lệch phân phối so với quá trình học.
