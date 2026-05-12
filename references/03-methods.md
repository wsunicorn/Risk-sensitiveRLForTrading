# 03. Kiến Trúc Và Thuật Toán

Notebook so sánh ba hướng tối ưu policy: PPO, CVaR-PPO và Sortino-PPO. Cả ba dùng chung kiến trúc Actor-Critic.

## Actor-Critic Network

Kiến trúc mạng:

```text
Input state_dim=19
→ Linear(19, 256) + ReLU
→ Linear(256, 256) + ReLU
→ Actor head: Linear(256, 1)
→ Critic head: Linear(256, 1)
```

Actor sinh tham số mean của phân phối Normal. `actor_log_std` là tham số học được. Trong training, action được sample để exploration; trong evaluation/demo, có thể dùng policy ổn định hơn qua mean.

Thông tin từ notebook:

```text
ActorCritic: state_dim=19, action_dim=1
Total parameters: 71,427
```

## PPO Baseline

PPO là baseline risk-neutral. Agent tối ưu clipped surrogate objective:

```text
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio * advantage
surr2 = clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
policy_loss = -mean(min(surr1, surr2))
```

Loss tổng:

```text
loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
```

Các điểm đáng chú ý:

- `gamma = 0.99`
- `epsilon = 0.2`
- `ppo_epochs = 8`
- Gradient clipping ở `0.5`
- Advantage được chuẩn hóa để ổn định training

PPO không có penalty rủi ro riêng, nên thường học policy tương đối aggressive khi thị trường tăng.

## CVaR-PPO

CVaR-PPO thêm penalty cho tail risk vào loss của PPO. Ý tưởng là phạt phần lợi nhuận ở đuôi xấu của phân phối return.

Tham số chính:

| Tham số | Giá trị |
|---|---:|
| `CVAR_ALPHA` | `0.15` |
| `CVAR_LAMBDA` | `0.15` |
| `CVAR_LAMBDA_DECAY` | `0.999` |
| `CVAR_LAMBDA_MIN` | `0.08` |
| `CVAR_L2` | `1e-5` |

Notebook chuẩn hóa returns trước khi tính CVaR:

```text
r_norm = (returns - returns.mean()) / returns.std()
CVaR = mean(worst alpha portion of r_norm)
CVaR loss = relu(-CVaR)
```

Loss bổ sung:

```text
extra_loss = lambda * CVaR_loss
```

Điểm cải thiện quan trọng so với phiên bản cũ là chuẩn hóa returns trước khi tính CVaR, giúp loss không tăng vô hạn do scale của return.

## Sortino-PPO

Sortino-PPO không sửa loss trực tiếp. Thay vào đó, nó sửa reward của environment theo downside deviation:

```text
reward_sortino = portfolio_return - lambda * downside_std^2
```

Notebook dùng:

| Tham số | Giá trị |
|---|---:|
| `SORTINO_LAMBDA` | `0.2` |
| Rolling window | 60 bước |
| Penalty cap | tối đa 50% độ lớn reward |

Pseudo-logic:

```text
r = portfolio_return
window = last_60_returns
negative_returns = returns trong window nhỏ hơn 0
downside_std = std(negative_returns)
penalty = sortino_lambda * downside_std^2
reward = r - min(penalty, abs(r) * 0.5)
```

Cách này thường ổn định hơn CVaR loss vì agent nhận tín hiệu rủi ro ngay từ reward từng bước, nhưng mức hiệu quả phụ thuộc nhiều vào lựa chọn rolling window và hệ số phạt.

## So Sánh Cơ Chế Rủi Ro

| Method | Tối ưu chính | Cách xử lý rủi ro | Trade-off |
|---|---|---|---|
| PPO | Expected return | Không có penalty riêng | Dễ aggressive, tốt khi uptrend |
| CVaR-PPO | PPO loss + tail penalty | Phạt worst 15% normalized returns | Phòng thủ tốt trong crisis, có thể bỏ lỡ một phần upside |
| Sortino-PPO | Reward đã chỉnh downside | Phạt downside volatility rolling 60 ngày | Cân bằng hơn, nhạy với thiết kế reward |
