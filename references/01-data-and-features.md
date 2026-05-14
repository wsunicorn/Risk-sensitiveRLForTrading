# 01. Dữ Liệu Và Feature Engineering

File này giải thích toàn bộ tầng dữ liệu của project: dữ liệu lấy từ đâu, vì sao chọn các tài sản này, làm sạch thế nào, từng đặc trưng kỹ thuật có ý nghĩa gì và tại sao normalization là phần bắt buộc khi đưa model vào Django demo.

## Vai Trò Của Dữ Liệu Trong Project

Trong project này, model không học trực tiếp từ ảnh chart hoặc chuỗi giá thô. Model nhận một vector trạng thái gồm:

- thông tin thị trường đã chuẩn hóa;
- thông tin danh mục hiện tại;
- toàn bộ được ghép thành observation 19 chiều.

Vì vậy feature engineering quyết định “model nhìn thấy thị trường như thế nào”. Nếu feature sai, scale sai hoặc bị data leakage, policy học được cũng khó đáng tin.

## Nguồn Dữ Liệu

Notebook dùng `yfinance` để tải dữ liệu daily OHLCV từ Yahoo Finance.

| Symbol | Vai trò | Dòng raw | Giai đoạn raw | Close min / max / mean |
|---|---:|---:|---|---:|
| SPY | Tài sản huấn luyện chính | 1,760 | `2018-01-02` đến `2024-12-30` | `$204.94 / $597.11 / $361.48` |
| QQQ | Kiểm tra generalization | 1,760 | `2018-01-02` đến `2024-12-30` | `$137.16 / $533.92 / $289.74` |
| GLD | Kiểm tra generalization | 1,760 | `2018-01-02` đến `2024-12-30` | `$111.10 / $257.50 / $165.12` |

Ý nghĩa từng tài sản:

| Symbol | Diễn giải | Lý do đưa vào project |
|---|---|---|
| `SPY` | ETF theo S&P 500 | Đại diện thị trường cổ phiếu Mỹ rộng; dùng làm primary asset để train |
| `QQQ` | ETF theo Nasdaq-100 | Nhiều cổ phiếu công nghệ, biến động và upside khác SPY; dùng để kiểm tra model có chuyển sang tài sản tăng trưởng mạnh không |
| `GLD` | ETF đại diện vàng | Khác nhóm tài sản với equity ETF; dùng để xem policy train trên SPY có còn tạo return dương trên tài sản phòng thủ hơn không |

SPY được chọn làm tài sản train chính vì nó thanh khoản, phổ biến, có dữ liệu sạch và đủ đại diện cho market regime lớn giai đoạn 2018-2024.

## Schema Dữ Liệu Raw

Sau khi tải về, notebook giữ các cột:

```text
date, open, high, low, close, volume
```

Ý nghĩa:

| Cột | Ý nghĩa | Vai trò |
|---|---|---|
| `date` | Ngày giao dịch | Trục thời gian để split, backtest và vẽ equity curve |
| `open` | Giá mở cửa | Dùng để mô tả OHLCV, nhưng không đưa trực tiếp vào state cuối cùng |
| `high` | Giá cao nhất trong ngày | Dùng để tính ATR |
| `low` | Giá thấp nhất trong ngày | Dùng để tính ATR |
| `close` | Giá đóng cửa | Giá chính dùng để định giá danh mục và là một feature |
| `volume` | Khối lượng giao dịch | Đại diện mức độ hoạt động/thanh khoản |

Trong môi trường hiện tại, giao dịch được mô phỏng theo giá `close`. Đây là giả định đơn giản và phổ biến trong backtest nghiên cứu, nhưng chưa phản ánh execution intraday thật.

## Làm Sạch Dữ Liệu

Hàm `download_data()` trong notebook và các hàm tương ứng trong Django thực hiện:

1. Reset index để `date` thành cột rõ ràng.
2. Chuẩn hóa tên cột thành lowercase.
3. Giữ đúng schema OHLCV.
4. Bỏ ngày trùng.
5. Bỏ dòng có OHLC không hợp lệ hoặc nhỏ hơn/bằng 0.
6. Bỏ missing values.
7. Sắp xếp theo thời gian tăng dần.

Lý do:

- Dữ liệu tài chính là chuỗi thời gian, nên thứ tự ngày là bắt buộc.
- Indicator rolling như SMA, RSI, Bollinger Bands sẽ tạo NaN ở các ngày đầu; các dòng này phải bị loại bỏ.
- Giá bằng 0 hoặc âm là bất thường với ETF trong dữ liệu này.
- Dòng duplicate có thể làm backtest đi qua cùng một ngày nhiều lần, làm sai reward và metric.

## Feature Engineering

Sau khi gọi `add_features()`, mỗi symbol còn 1,711 dòng và 20 cột.

Các feature thị trường cuối cùng được dùng trong state là:

```text
close, volume,
sma_10, sma_20, sma_50,
rsi,
macd, macd_signal, macd_diff,
bb_high, bb_low, bb_mid,
atr,
returns, log_returns,
volume_sma
```

Tổng cộng 16 market features.

## Nhóm 1: Price Và Return

### `close`

Giá đóng cửa của tài sản trong ngày.

Vai trò:

- dùng để định giá số cổ phiếu đang nắm giữ;
- dùng để tính returns, log returns và nhiều indicator;
- là feature giá trực tiếp cho policy.

Điểm cần nhớ: `close` có scale rất lớn so với `returns`, nên bắt buộc phải chuẩn hóa trước khi đưa vào neural network.

### `returns`

Tỷ suất sinh lời đơn giản:

```text
returns_t = close_t / close_{t-1} - 1
```

Ý nghĩa:

- cho biết giá tăng/giảm bao nhiêu phần trăm so với ngày trước;
- trực tiếp liên quan đến thay đổi giá trị danh mục;
- dùng để agent nhận tín hiệu momentum ngắn hạn.

### `log_returns`

Log return:

```text
log_returns_t = log(close_t / close_{t-1})
```

Ý nghĩa:

- gần bằng simple return khi biến động nhỏ;
- cộng dồn thuận tiện hơn theo thời gian;
- thường ổn định hơn trong phân tích chuỗi tài chính.

Trong project, cả `returns` và `log_returns` cùng được đưa vào state để model có cả tín hiệu phần trăm trực quan và tín hiệu log scale.

## Nhóm 2: Trend Indicators

### `sma_10`, `sma_20`, `sma_50`

Simple Moving Average:

```text
sma_n = mean(close trong n ngày gần nhất)
```

Ý nghĩa từng window:

| Feature | Ý nghĩa |
|---|---|
| `sma_10` | Xu hướng rất ngắn hạn |
| `sma_20` | Xu hướng ngắn/trung hạn, cũng là middle band của Bollinger |
| `sma_50` | Xu hướng trung hạn |

Tại sao dùng nhiều SMA:

- Nếu `close > sma_10 > sma_20 > sma_50`, thị trường thường đang có xu hướng tăng ngắn hạn mạnh.
- Nếu `close < sma_50`, asset có thể đang suy yếu trung hạn.
- Khoảng cách giữa giá và SMA giúp agent nhận diện mức độ overextended hoặc mean-reversion.

SMA không dự đoán tương lai. Nó chỉ tóm tắt cấu trúc quá khứ gần.

## Nhóm 3: Momentum Indicators

### `rsi`

RSI 14 ngày đo cường độ tăng/giảm gần đây.

Diễn giải thường gặp:

| RSI | Diễn giải phổ biến |
|---:|---|
| `< 30` | Có thể oversold |
| `30-70` | Vùng trung tính |
| `> 70` | Có thể overbought |

Trong RL, không nên hiểu RSI như rule cứng “RSI thấp thì mua”. Agent tự học cách sử dụng RSI cùng các feature khác và trạng thái danh mục.

### `macd`, `macd_signal`, `macd_diff`

MACD đo momentum bằng chênh lệch giữa EMA nhanh và EMA chậm.

| Feature | Ý nghĩa |
|---|---|
| `macd` | Chênh lệch xu hướng nhanh/chậm |
| `macd_signal` | Đường tín hiệu làm mượt của MACD |
| `macd_diff` | Histogram, thường là `macd - macd_signal` |

Ý nghĩa cho agent:

- `macd_diff > 0` có thể báo momentum tăng đang mạnh hơn.
- `macd_diff < 0` có thể báo momentum suy yếu.
- Mức độ và dấu của MACD giúp policy phân biệt trend-following và giảm vị thế.

## Nhóm 4: Volatility Indicators

### Bollinger Bands: `bb_high`, `bb_low`, `bb_mid`

Bollinger Bands thường gồm:

```text
bb_mid = SMA_20
bb_high = SMA_20 + k * rolling_std_20
bb_low = SMA_20 - k * rolling_std_20
```

Ý nghĩa:

- Khoảng cách giữa `bb_high` và `bb_low` đại diện volatility rolling.
- Giá gần `bb_high` có thể là momentum mạnh hoặc overbought.
- Giá gần `bb_low` có thể là panic/oversold hoặc downtrend tiếp diễn.

Vì RL không dùng rule cứng, Bollinger Bands đóng vai trò mô tả bối cảnh volatility và vị trí tương đối của giá.

### `atr`

Average True Range đo biên độ dao động thực tế, có xét cả gap.

Ý nghĩa:

- ATR cao: thị trường biến động mạnh, execution risk lớn hơn.
- ATR thấp: thị trường yên hơn, tín hiệu có thể ít nhiễu hơn.

ATR khác `returns`: returns đo hướng tăng/giảm, ATR đo độ rộng dao động. Một ngày tăng mạnh và một ngày giảm mạnh đều có thể làm ATR cao.

## Nhóm 5: Volume

### `volume`

Khối lượng giao dịch trong ngày.

Ý nghĩa:

- Volume cao có thể xác nhận trend hoặc panic.
- Volume thấp có thể làm tín hiệu giá kém đáng tin hơn.

### `volume_sma`

SMA 20 ngày của volume.

Ý nghĩa:

- So sánh volume hiện tại với volume trung bình.
- Giúp agent nhận ra phiên có hoạt động bất thường.

## Portfolio Features

Ngoài market features, state có 3 biến danh mục:

| Feature | Công thức | Ý nghĩa |
|---|---|---|
| Cash ratio | `balance / initial_balance` | Agent còn bao nhiêu tiền mặt so với vốn ban đầu |
| Position value ratio | `shares * price / initial_balance` | Giá trị cổ phiếu đang nắm giữ so với vốn ban đầu |
| Portfolio value ratio | `portfolio_value / initial_balance` | Tổng giá trị danh mục so với vốn ban đầu |

Ba biến này rất quan trọng vì cùng một thị trường nhưng quyết định đúng có thể khác nhau tùy agent đang full cash hay full position.

Ví dụ:

- Nếu market rất tích cực nhưng agent đã gần như full invested, action BUY mạnh không còn nhiều ý nghĩa.
- Nếu market xấu và agent đang nắm nhiều cổ phiếu, SELL có thể quan trọng hơn HOLD.
- Nếu portfolio đang drawdown, policy có thể học giảm risk exposure.

## Observation 19 Chiều

State cuối cùng:

```text
[
  close, volume,
  sma_10, sma_20, sma_50,
  rsi,
  macd, macd_signal, macd_diff,
  bb_high, bb_low, bb_mid,
  atr,
  returns, log_returns,
  volume_sma,
  balance_ratio,
  position_value_ratio,
  portfolio_value_ratio
]
```

Output sanity check trong notebook:

```text
Observation shape: (19,)
Action space: Box(-1.0, 1.0, (1,), float32)
```

## Chia Dữ Liệu Theo Market Regime

Notebook không chia random vì dữ liệu tài chính có thứ tự thời gian. Nếu random split, dữ liệu tương lai có thể rò vào train qua phân phối, làm kết quả quá lạc quan.

| Split | Dòng | Giai đoạn | Ý nghĩa |
|---|---:|---|---|
| Train | 959 | `2018-03-14` đến `2021-12-31` | Có uptrend và COVID crash |
| Validation | 251 | `2022-01-03` đến `2022-12-30` | Bear market 2022 |
| Test | 501 | `2023-01-03` đến `2024-12-30` | Recovery và bull run |

Lý do chia như vậy:

- Train có nhiều trạng thái khác nhau, gồm cú sốc COVID.
- Validation là một năm giảm mạnh, dùng để quan sát khả năng chịu rủi ro.
- Test là giai đoạn tăng, giúp thấy trade-off khi Buy & Hold có lợi thế.

## Normalization

Neural network nhạy với scale. Trong state, `volume` có thể hàng chục triệu, `returns` khoảng 0.001, `rsi` khoảng 0-100. Nếu đưa raw vào mạng, feature có scale lớn có thể áp đảo gradient.

Project dùng Z-score:

```text
z = (x - mean_train) / std_train
```

Điểm bắt buộc: mean/std phải lấy từ tập train, không lấy từ validation/test/realtime.

Nếu tính normalization trên test, ta vô tình dùng thông tin phân phối tương lai. Đó là data leakage.

Nếu tính normalization riêng trên realtime, input demo sẽ không cùng phân phối với lúc train. Model có thể nhận state sai ý nghĩa.

## `norm_params.json`

File [../demo/weights/norm_params.json](../demo/weights/norm_params.json) chứa:

```json
{
  "feat_cols": ["close", "volume", "..."],
  "feat_mean": [303.4205, 83621609.8019, "..."],
  "feat_std": [64.3276, 47184103.9856, "..."],
  "state_dim": 19,
  "action_dim": 1
}
```

Vai trò trong Django:

1. Backend tải dữ liệu mới bằng `yfinance`.
2. Backend tính lại các indicator theo đúng schema.
3. Backend đọc `feat_cols`, `feat_mean`, `feat_std`.
4. Backend chuẩn hóa features theo train stats.
5. Backend ghép portfolio features để tạo state 19 chiều.
6. Model `.pth` nhận state và sinh action.

Nếu thiếu `norm_params.json`, web demo không nên chạy inference vì không biết chuẩn hóa đúng như lúc train.

## Data Files Trong Repo

| File | Nội dung |
|---|---|
| [../data/raw_SPY.csv](../data/raw_SPY.csv) | OHLCV raw của SPY |
| [../data/raw_QQQ.csv](../data/raw_QQQ.csv) | OHLCV raw của QQQ |
| [../data/raw_GLD.csv](../data/raw_GLD.csv) | OHLCV raw của GLD |
| [../data/featured_SPY.csv](../data/featured_SPY.csv) | SPY sau khi thêm indicator |
| [../data/featured_QQQ.csv](../data/featured_QQQ.csv) | QQQ sau khi thêm indicator |
| [../data/featured_GLD.csv](../data/featured_GLD.csv) | GLD sau khi thêm indicator |
| [../data/split_SPY_train.csv](../data/split_SPY_train.csv) | Tập train SPY |
| [../data/split_SPY_val.csv](../data/split_SPY_val.csv) | Tập validation SPY |
| [../data/split_SPY_test.csv](../data/split_SPY_test.csv) | Tập test SPY |

## Những Điểm Cần Cẩn Thận

- Technical indicators không phải “tín hiệu chắc thắng”; chúng chỉ là cách mã hóa lịch sử giá/volume.
- Feature càng nhiều không luôn càng tốt; nếu thêm feature nhiễu, agent có thể overfit.
- Split theo thời gian quan trọng hơn random split trong trading.
- Normalization phải dùng train stats.
- Khi test trên QQQ/GLD, model vẫn dùng normalization train từ SPY. Đây là cách kiểm tra generalization nghiêm ngặt hơn, nhưng cũng có thể làm distribution shift mạnh.
- Dữ liệu daily bỏ qua nhiều yếu tố thực tế như bid/ask spread, khớp lệnh intraday, thanh khoản theo thời điểm và market impact.

## Tóm Tắt

Tầng dữ liệu của project biến OHLCV thành state 19 chiều. 16 chiều đầu mô tả thị trường: giá, trend, momentum, volatility, return, volume. 3 chiều cuối mô tả trạng thái danh mục. Việc chuẩn hóa bằng train stats giúp model học ổn định và đảm bảo inference trong Django nhất quán với training trong notebook.
