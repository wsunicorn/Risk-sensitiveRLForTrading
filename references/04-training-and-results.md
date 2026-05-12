# 04. Huấn Luyện Và Kết Quả

## Cấu Hình Huấn Luyện

| Nhóm | Tham số | Giá trị |
|---|---|---:|
| Data | Primary symbol | `SPY` |
| Data | Start / End | `2018-01-01` đến `2024-12-31` |
| Environment | Initial balance | `$10,000` |
| Environment | Transaction cost | `0.001` |
| Environment | Slippage | `0.0005` |
| Network | Hidden dim | `256` |
| PPO | Gamma | `0.99` |
| PPO | Learning rate | `2e-4` |
| PPO | Epsilon | `0.2` |
| PPO | Epochs | `8` |
| CVaR-PPO | Alpha | `0.15` |
| CVaR-PPO | Lambda | `0.15` decays to min `0.08` |
| Sortino-PPO | Lambda | `0.2` |
| Training | Episodes | `300` |
| Training | Update interval | `256` steps |

Notebook được chạy trên Kaggle GPU, output ghi nhận GPU Tesla T4.

## Training Progress

Kết quả cuối quá trình training trong notebook:

| Method | Final train portfolio | Train return |
|---|---:|---:|
| PPO | `$12,434.05` | `+24.34%` |
| CVaR-PPO | `$14,089.07` | `+40.89%` |
| Sortino-PPO | `$13,060.67` | `+30.61%` |

CVaR-PPO có train portfolio cao nhất trong lần chạy notebook này, dù mục tiêu của nó không chỉ là tăng return mà còn giảm tail loss.

## Test Set Results

Test set: SPY từ `2023-01-03` đến `2024-12-30`.

| Method | Final Value | Return | Sharpe | Sortino | Calmar | Max DD | CVaR-95 | Volatility | Win Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | `$11,371.80` | `+13.72%` | `0.9706` | `1.1964` | `1.1366` | `-5.90%` | `-1.1034%` | `0.0694` | `0.5451` |
| CVaR-PPO | `$11,900.86` | `+19.01%` | `1.0729` | `1.4165` | `1.1057` | `-8.31%` | `-1.3178%` | `0.0851` | `0.5611` |
| Sortino-PPO | `$11,553.35` | `+15.53%` | `0.9589` | `1.1930` | `1.0465` | `-7.23%` | `-1.2552%` | `0.0783` | `0.5511` |
| Buy & Hold | `$15,882.05` | `+58.82%` | `1.8844` | `2.9956` | `2.6325` | `-9.97%` | `-1.7422%` | `0.1281` | `0.5760` |

Nhận xét:

- Buy & Hold thắng về return trong bull market 2023-2024.
- PPO có Max Drawdown và CVaR-95 tốt nhất trên riêng test set.
- CVaR-PPO có Return, Sharpe và Sortino tốt nhất trong ba agent RL.
- Các agent RL giữ volatility thấp hơn Buy & Hold.

## Crisis Period Stress Test

Stress test chạy trên các giai đoạn đặc biệt trong toàn bộ dữ liệu.

| Period | Method | Return | Max DD | Sortino |
|---|---|---:|---:|---:|
| COVID crash 2020 | PPO | `-9.73%` | `-26.36%` | `-0.831` |
| COVID crash 2020 | CVaR-PPO | `+1.44%` | `-14.97%` | `+0.259` |
| COVID crash 2020 | Sortino-PPO | `-5.62%` | `-23.32%` | `-0.557` |
| COVID crash 2020 | Buy & Hold | `-5.56%` | `-33.72%` | `-0.409` |
| Bear 2022 | PPO | `-6.92%` | `-13.66%` | `-0.681` |
| Bear 2022 | CVaR-PPO | `-6.43%` | `-13.98%` | `-0.726` |
| Bear 2022 | Sortino-PPO | `-8.64%` | `-15.18%` | `-1.046` |
| Bear 2022 | Buy & Hold | `-18.65%` | `-24.50%` | `-1.249` |
| Bull 2023-2024 | PPO | `+20.39%` | `-5.78%` | `+2.150` |
| Bull 2023-2024 | CVaR-PPO | `+26.76%` | `-6.21%` | `+2.321` |
| Bull 2023-2024 | Sortino-PPO | `+14.33%` | `-6.56%` | `+1.193` |
| Bull 2023-2024 | Buy & Hold | `+58.82%` | `-9.97%` | `+2.996` |

Kết luận từ stress test:

- COVID crash là nơi CVaR-PPO thể hiện rõ nhất: return dương và drawdown thấp hơn đáng kể.
- Trong Bear 2022, cả ba agent RL đều giảm lỗ mạnh so với Buy & Hold.
- Trong Bull 2023-2024, Buy & Hold thắng return nhưng chịu drawdown lớn hơn các agent RL.

## Multi-Asset Evaluation

Các model train trên SPY được test trực tiếp trên SPY, QQQ và GLD mà không retrain.

| Symbol | Method | Return | Sharpe | Max DD | CVaR-95 |
|---|---|---:|---:|---:|---:|
| SPY | PPO | `+12.32%` | `+0.920` | `-5.02%` | `-1.005%` |
| SPY | CVaR-PPO | `+21.80%` | `+1.303` | `-6.55%` | `-1.169%` |
| SPY | Sortino-PPO | `+17.20%` | `+1.110` | `-6.12%` | `-1.078%` |
| SPY | Buy & Hold | `+58.82%` | `+1.884` | `-9.97%` | `-1.742%` |
| QQQ | PPO | `+39.64%` | `+1.835` | `-8.18%` | `-1.285%` |
| QQQ | CVaR-PPO | `+46.14%` | `+1.662` | `-8.11%` | `-1.637%` |
| QQQ | Sortino-PPO | `+41.24%` | `+1.595` | `-7.57%` | `-1.572%` |
| QQQ | Buy & Hold | `+97.47%` | `+2.004` | `-13.56%` | `-2.376%` |
| GLD | PPO | `+16.03%` | `+1.002` | `-9.38%` | `-1.079%` |
| GLD | CVaR-PPO | `+13.04%` | `+0.687` | `-9.26%` | `-1.491%` |
| GLD | Sortino-PPO | `+12.28%` | `+0.746` | `-9.01%` | `-1.231%` |
| GLD | Buy & Hold | `+40.67%` | `+1.281` | `-11.35%` | `-1.980%` |

Điểm đáng chú ý: policy học trên SPY vẫn tạo return dương khi chuyển sang QQQ và GLD, nhưng chưa vượt Buy & Hold trong các bull test period. Đây là tín hiệu tốt cho generalization ban đầu, nhưng chưa đủ để kết luận robust.

## File Đầu Ra

Notebook lưu các artifact sau:

```text
ppo_model.pth
cvar_ppo_model.pth
sortino_ppo_model.pth
norm_params.json
results.json
test_results.csv
```

Trong repo demo, các file tương ứng nằm ở [../demo/weights](../demo/weights).
