# 02. Môi Trường Giao Dịch

Notebook xây dựng `TradingEnv` kế thừa từ `gymnasium.Env`. Mục tiêu là biến bài toán giao dịch thành MDP để agent học policy trực tiếp.

## MDP Formulation

| Thành phần | Thiết kế trong notebook |
|---|---|
| State | 16 market features đã chuẩn hóa + 3 portfolio features |
| Action | Một giá trị liên tục trong `[-1, 1]` |
| Reward | Tỷ suất thay đổi portfolio value sau mỗi bước |
| Transition | Dịch sang ngày giao dịch kế tiếp sau khi thực hiện lệnh |
| Episode | Một lượt đi qua toàn bộ tập dữ liệu của split tương ứng |

## Observation Space

Observation có shape `(19,)`:

- 16 feature thị trường: giá, volume, indicators, returns.
- 3 feature danh mục: tỷ lệ tiền mặt, tỷ lệ giá trị cổ phiếu đang giữ, tỷ lệ tổng portfolio value.

Output kiểm tra nhanh trong notebook:

```text
Observation shape: (19,)
Action space: Box(-1.0, 1.0, (1,), float32)
```

## Action Space

Action là một số thực `a` trong `[-1, 1]`:

| Điều kiện | Ý nghĩa |
|---|---|
| `a > 0.01` | BUY, dùng tỷ lệ `a` của cash hiện có để mua |
| `a < -0.01` | SELL, bán tỷ lệ `abs(a)` của lượng cổ phiếu đang nắm giữ |
| `-0.01 <= a <= 0.01` | HOLD |

Thiết kế liên tục này giúp agent học mức độ mua/bán, thay vì chỉ chọn discrete action như `buy`, `sell`, `hold`.

## Transaction Cost Và Slippage

Notebook mô phỏng chi phí thị trường:

| Tham số | Giá trị |
|---|---:|
| Initial balance | `$10,000` |
| Transaction cost | `0.001` tương đương `0.1%` |
| Slippage | `0.0005` tương đương `0.05%` |

Khi mua:

```text
exec_price = price * (1 + slippage)
bought = invest * (1 - transaction_cost) / exec_price
```

Khi bán:

```text
exec_price = price * (1 - slippage)
proceeds = shares_to_sell * exec_price * (1 - transaction_cost)
```

Việc thêm cost và slippage làm môi trường thực tế hơn, đồng thời phạt implicit các policy giao dịch quá thường xuyên.

## Reward Gốc

Reward mặc định là step return của portfolio:

```text
reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
```

PPO dùng reward này trực tiếp. Sortino-PPO override `_reward()` để cộng thêm downside penalty, còn CVaR-PPO giữ reward gốc nhưng thêm penalty vào loss.

## History Được Lưu

Mỗi bước giao dịch lưu vào `history`:

```text
step, action, price, portfolio_value, reward
```

`history` này được dùng để tính Return, Sharpe, Sortino, Calmar, Max Drawdown, VaR-95, CVaR-95, volatility và win rate.
