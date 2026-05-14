# 02. Môi Trường Giao Dịch

File này giải thích cách project biến trading thành một bài toán Markov Decision Process (MDP): state là gì, action là gì, reward là gì, môi trường thực thi lệnh thế nào và các giả định backtest có ý nghĩa ra sao.

## Vì Sao Cần Môi Trường RL?

Nếu dùng supervised learning, ta thường dự đoán giá hoặc nhãn tăng/giảm. Nhưng trading không chỉ là dự báo đúng hướng giá. Một chiến lược còn phải quyết định:

- đang giữ bao nhiêu cổ phiếu;
- còn bao nhiêu tiền mặt;
- có nên tăng/giảm vị thế không;
- giao dịch nhiều có bị phí ăn mất lợi nhuận không;
- drawdown có chấp nhận được không.

Học tăng cường phù hợp vì agent tương tác với môi trường theo chuỗi:

```text
state_t -> action_t -> execution -> portfolio_{t+1} -> reward_t -> state_{t+1}
```

Mỗi action ảnh hưởng đến danh mục tương lai, nên bài toán đúng bản chất là sequential decision-making.

## MDP Formulation

| Thành phần | Thiết kế trong project |
|---|---|
| State `s_t` | 16 market features đã chuẩn hóa + 3 portfolio features |
| Action `a_t` | Một giá trị liên tục trong `[-1, 1]` |
| Reward `r_t` | Tỷ suất thay đổi giá trị danh mục sau một bước |
| Transition | Sau khi thực thi lệnh, môi trường chuyển sang ngày giao dịch kế tiếp |
| Episode | Một lượt agent đi qua toàn bộ dữ liệu của split tương ứng |
| Policy | Mạng Actor-Critic sinh action từ state |

Điểm khác với trading rule truyền thống: agent không được lập trình sẵn “RSI < 30 thì mua”. Nó tự học action thông qua reward sau nhiều episode.

## State Space

Observation có shape `(19,)`.

```text
state = concat(normalized_market_features, portfolio_features)
```

### Market Features

16 feature thị trường:

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

Các feature này được chuẩn hóa theo mean/std của tập train.

### Portfolio Features

3 feature danh mục:

```text
balance / initial_balance
shares * current_price / initial_balance
portfolio_value / initial_balance
```

Tại sao cần portfolio features?

Vì action tốt phụ thuộc vào vị thế hiện tại. Cùng một thị trường, nếu agent đang full cash thì BUY có ý nghĩa khác với khi agent đã full invested. Không có portfolio state, policy sẽ không biết mình còn bao nhiêu vốn để mua hoặc có bao nhiêu cổ phiếu để bán.

## Action Space

Action là một số thực `a` trong `[-1, 1]`.

| Vùng action | Tín hiệu | Cách thực thi |
|---:|---|---|
| `a > 0.01` | BUY | Dùng tỷ lệ `a` của cash hiện có để mua |
| `a < -0.01` | SELL | Bán tỷ lệ `abs(a)` của số cổ phiếu đang giữ |
| `-0.01 <= a <= 0.01` | HOLD | Không giao dịch |

Ví dụ:

- `a = 0.50`: dùng 50% cash hiện tại để mua.
- `a = -0.30`: bán 30% số shares đang nắm giữ.
- `a = 0.005`: HOLD vì nằm trong vùng nhiễu.

### Vì sao dùng action liên tục?

Trading thực tế thường là bài toán sizing, không chỉ chọn `BUY/SELL/HOLD`. Action liên tục giúp agent học mức độ giao dịch:

- mua ít khi tín hiệu yếu;
- mua mạnh khi tín hiệu rõ;
- bán một phần thay vì thoát hết;
- giữ vị thế khi không có edge rõ ràng.

Nếu dùng action rời rạc, policy sẽ thô hơn và khó biểu diễn tỷ trọng.

## Execution Logic

Môi trường dùng giá đóng cửa hiện tại làm giá tham chiếu.

### Khi BUY

```text
invest = balance * action
exec_price = price * (1 + slippage)
bought = invest * (1 - transaction_cost) / exec_price
shares += bought
balance -= invest
```

Ý nghĩa:

- Agent chỉ được dùng tiền mặt hiện có, không margin.
- Giá mua bị cộng slippage vì khớp lệnh thường tệ hơn giá quan sát.
- Transaction cost làm số cổ phiếu mua được giảm xuống.

### Khi SELL

```text
sell = shares * abs(action)
exec_price = price * (1 - slippage)
proceeds = sell * exec_price * (1 - transaction_cost)
shares -= sell
balance += proceeds
```

Ý nghĩa:

- Agent chỉ bán số shares đang nắm giữ, không short.
- Giá bán bị trừ slippage.
- Transaction cost làm tiền thu về giảm.

## Chi Phí Giao Dịch Và Slippage

Notebook dùng:

| Tham số | Giá trị |
|---|---:|
| Initial balance | `$10,000` |
| Transaction cost | `0.001` = `0.1%` |
| Slippage | `0.0005` = `0.05%` |

Tại sao cần cost và slippage?

- Nếu không có phí, agent có thể học giao dịch quá nhiều vì mỗi thay đổi nhỏ đều “miễn phí”.
- Trong thực tế, phí, spread và trượt giá làm turnover cao trở nên đắt đỏ.
- Cost/slippage tạo penalty gián tiếp cho hành vi overtrading.

Trong project này, cost/slippage vẫn là giả định đơn giản. Chúng chưa mô phỏng spread động, market impact hoặc thanh khoản theo volume.

## Reward Cơ Sở

Reward của `TradingEnv`:

```text
reward_t = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
```

Ý nghĩa:

- Nếu portfolio tăng, reward dương.
- Nếu portfolio giảm, reward âm.
- Reward được tính theo tỷ lệ, nên không phụ thuộc tuyệt đối vào vốn ban đầu.

PPO dùng reward này trực tiếp.

## Reward Trong Sortino-PPO

Sortino-PPO dùng môi trường `SortinoTradingEnv`, override `_reward()`.

Logic:

```text
r = portfolio_return
window = last_60_returns
negative_returns = returns trong window nhỏ hơn 0
downside_std = std(negative_returns)
penalty = sortino_lambda * downside_std^2
reward = r - min(penalty, abs(r) * 0.5)
```

Trong notebook:

- `SORTINO_LAMBDA = 0.2`
- rolling window = 60 bước
- penalty cap = 50% độ lớn reward

Ý nghĩa:

- Chỉ các return âm mới tạo downside deviation.
- Policy được khuyến khích tránh chuỗi lỗ hoặc biến động giảm mạnh.
- Cap penalty giúp reward không bị penalty áp đảo hoàn toàn.

## CVaR-PPO Nằm Ở Đâu?

CVaR-PPO không sửa môi trường. Nó vẫn dùng reward cơ sở của `TradingEnv`, nhưng thêm risk penalty vào loss trong quá trình update policy.

Điều này tạo khác biệt thiết kế:

| Method | Risk xử lý ở đâu? |
|---|---|
| PPO | Không có risk penalty riêng |
| CVaR-PPO | Thêm penalty trong loss |
| Sortino-PPO | Sửa reward của environment |

CVaR-PPO tập trung vào distribution của return trong batch/update. Sortino-PPO đưa tín hiệu downside vào từng bước reward.

## Episode

Một episode là một lần agent chạy từ đầu đến gần cuối dataset split.

Với train set SPY:

- reset vốn về `$10,000`;
- `shares = 0`;
- `balance = 10000`;
- đi qua từng ngày giao dịch;
- mỗi bước chọn action, môi trường thực thi, cập nhật portfolio;
- kết thúc khi hết dữ liệu.

Training chạy 300 episodes. Điều này nghĩa là agent nhiều lần đi qua cùng tập train để cải thiện policy.

## History Được Lưu

Notebook lưu mỗi bước:

```text
step, action, price, portfolio_value, reward
```

Django backend lưu thêm:

```text
date, balance, shares
```

History dùng để tính:

- equity curve;
- daily returns;
- drawdown series;
- Sharpe;
- Sortino;
- Calmar;
- VaR-95;
- CVaR-95;
- volatility;
- win rate;
- action distribution.

## Buy & Hold Baseline

Buy & Hold không dùng policy network. Logic:

```text
shares = initial_balance / first_close
portfolio_value_t = shares * close_t
```

Tại sao cần Buy & Hold?

- Nó là benchmark đơn giản nhất trong một thị trường tăng.
- Nếu model phức tạp không vượt hoặc không kiểm soát rủi ro tốt hơn benchmark, cần xem lại giá trị của model.
- Nó giúp diễn giải trade-off: RL có thể bỏ lỡ upside nhưng giảm drawdown/tail risk.

## Giả Định Của Môi Trường

Môi trường hiện tại có các giả định:

- Giao dịch một tài sản tại một thời điểm.
- Không dùng margin.
- Không short.
- Không có cash yield.
- Không có thuế.
- Giá khớp dựa trên close cộng/trừ slippage cố định.
- Cost cố định theo tỷ lệ giao dịch.
- Không giới hạn thanh khoản.
- Không có partial fill hoặc order book.
- Mỗi step là một ngày giao dịch.

Các giả định này hợp lý cho đồ án nghiên cứu, nhưng chưa đủ để triển khai giao dịch thật.

## Vì Sao Không Dự Đoán Giá?

Dự đoán giá và ra quyết định giao dịch là hai bài toán khác nhau.

Một mô hình dự đoán đúng hướng giá vẫn có thể thua nếu:

- sizing sai;
- giao dịch quá nhiều;
- phí lớn;
- drawdown quá sâu;
- không biết đang nắm bao nhiêu vị thế;
- tín hiệu đúng nhưng reward/risk trade-off xấu.

RL giải quyết trực tiếp câu hỏi: “Trong trạng thái này, nên hành động thế nào để danh mục tốt hơn theo thời gian?”

## Vấn Đề Cần Cẩn Thận

### Overtrading

Nếu reward chỉ là return và cost quá thấp, agent có thể giao dịch liên tục. Cost/slippage giúp giảm nhưng chưa hoàn toàn thay thế turnover penalty.

### Look-ahead Bias

Môi trường phải đảm bảo action tại ngày `t` không dùng thông tin từ ngày tương lai. Feature rolling chỉ dùng dữ liệu quá khứ/tới hiện tại, split theo thời gian và normalization theo train giúp giảm nguy cơ này.

### Data Leakage Qua Normalization

Nếu dùng mean/std của toàn bộ dataset, model gián tiếp biết phân phối test. Vì vậy `norm_params.json` phải lấy từ train.

### Bull Market Bias

Test set 2023-2024 là bull market. Buy & Hold có lợi thế tự nhiên. Vì vậy cần đọc thêm crisis stress test để đánh giá khả năng phòng thủ.

### Deterministic vs Stochastic Action

Trong training, actor sample từ phân phối Normal để exploration. Trong Django demo, backend dùng deterministic mean qua `act_deterministic()` để inference/backtest ổn định và tái lập hơn.

## Tóm Tắt

Môi trường giao dịch là lõi của project. Nó định nghĩa agent được nhìn thấy gì, được làm gì và được thưởng/phạt ra sao. State 19 chiều cho agent biết cả thị trường lẫn danh mục. Action liên tục cho phép sizing vị thế. Reward dựa trên portfolio return, còn CVaR-PPO và Sortino-PPO thêm góc nhìn rủi ro để tránh chỉ tối ưu lợi nhuận ngắn hạn.
