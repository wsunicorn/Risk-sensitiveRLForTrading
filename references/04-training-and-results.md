# 04. Huấn Luyện Và Kết Quả

File này tổng hợp cấu hình huấn luyện, cách tính metric, kết quả test/stress test/multi-asset evaluation và cách diễn giải kết quả cho đúng. Mục tiêu là giúp bạn review lại project mà không chỉ nhìn bảng số, mà hiểu vì sao từng kết quả có ý nghĩa.

## Nguồn Số Liệu

Trong repo có nhiều nguồn kết quả:

| Nguồn | Vai trò |
|---|---|
| [../risk_sensitive_trading.ipynb](../risk_sensitive_trading.ipynb) | Notebook nghiên cứu chính, có output training/test/stress/multi-asset |
| [../docs/Nhom15_Report.md](../docs/Nhom15_Report.md) | Report đã chuyển Markdown, có diễn giải kết quả và hình ảnh |
| [../demo/model_v2](../demo/model_v2) | Artifact khớp với output notebook/report ở nhiều bảng |
| [../demo/weights](../demo/weights) | Artifact active mà Django demo đang load |
| [../demo/model_v1](../demo/model_v1) | Bản artifact lưu riêng, hiện giống `demo/weights` |

Vì reinforcement learning có sampling và repo lưu nhiều phiên bản weights, các con số giữa `demo/weights` và notebook/model_v2 lệch nhẹ. Phần kết quả nghiên cứu dưới đây ưu tiên notebook/model_v2, đồng thời ghi riêng bảng active demo khi cần.

## Cấu Hình Huấn Luyện

| Nhóm | Tham số | Giá trị |
|---|---|---:|
| Data | Symbols | `SPY`, `QQQ`, `GLD` |
| Data | Primary symbol | `SPY` |
| Data | Start / End | `2018-01-01` đến `2024-12-31` |
| Split | Train end | `2021-12-31` |
| Split | Validation end | `2022-12-31` |
| Environment | Initial balance | `$10,000` |
| Environment | Transaction cost | `0.001` |
| Environment | Slippage | `0.0005` |
| Network | Hidden dim | `256` |
| PPO | Learning rate | `2e-4` |
| PPO | Gamma | `0.99` |
| PPO | Epsilon | `0.2` |
| PPO | Epochs | `8` |
| CVaR-PPO | Alpha | `0.15` |
| CVaR-PPO | Lambda | `0.15` giảm dần đến `0.08` |
| CVaR-PPO | L2 | `1e-5` |
| Sortino-PPO | Lambda | `0.2` |
| Training | Episodes | `300` |
| Training | Update interval | `256` steps |
| Device | Notebook output | `cuda` |

Notebook ghi nhận GPU Tesla T4 trên Kaggle trong quá trình chạy.

## Quy Trình Training

Với mỗi model:

1. Khởi tạo environment trên train set.
2. Reset vốn `$10,000`, shares `0`.
3. Agent đi qua từng ngày giao dịch.
4. Tại mỗi bước:
   - nhận state 19 chiều;
   - sample action từ actor;
   - môi trường thực thi BUY/SELL/HOLD;
   - nhận reward;
   - lưu state/action/reward/log_prob/value/done vào memory.
5. Sau mỗi `256` steps, PPO update policy.
6. Kết thúc episode, lưu total reward và portfolio value.
7. Lặp 300 episodes.

Khác biệt chính:

- PPO dùng reward cơ sở.
- CVaR-PPO thêm tail-risk penalty trong update loss.
- Sortino-PPO dùng environment có reward downside-adjusted.

## Kết Quả Training

Notebook output cuối training:

| Method | Avg Reward ở episode 300 | Avg Portfolio ở episode 300 | Final train portfolio |
|---|---:|---:|---:|
| PPO | `+0.3971` | `$14,535.70` | `$12,434.05` |
| CVaR-PPO | `+0.3410` | `$13,652.02` | `$14,089.07` |
| Sortino-PPO | `+0.3488` | `$13,894.47` | `$13,060.67` |

Cách đọc:

- Cả ba model đều học được policy làm portfolio vượt vốn ban đầu trên train.
- PPO có avg reward/avg portfolio cao ở episode 300.
- CVaR-PPO có final train portfolio cao nhất trong lần chạy notebook.
- Kết quả train không phải kết luận cuối cùng vì model có thể overfit train regime.

## Metrics Đánh Giá

Từ history portfolio values, notebook tính:

### Total Return

```text
total_return = (final_value - initial_balance) / initial_balance
```

Cho biết lợi nhuận toàn kỳ.

### Annual Return

```text
annual_return = (final_value / initial_balance)^(252 / N) - 1
```

Quy đổi return về năm giao dịch.

### Volatility

```text
volatility = std(daily_returns) * sqrt(252)
```

Độ biến động annualized.

### Sharpe

```text
sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
```

Lợi nhuận trên mỗi đơn vị volatility tổng. Notebook giả định risk-free rate bằng 0.

### Sortino

```text
sortino = annual_return / downside_std
```

Chỉ dùng standard deviation của returns âm, nên tập trung vào downside risk.

### Max Drawdown

```text
peak_t = max(portfolio_value_0...t)
drawdown_t = (portfolio_value_t - peak_t) / peak_t
max_drawdown = min(drawdown_t)
```

Cho biết danh mục từng giảm sâu nhất bao nhiêu từ đỉnh.

### Calmar

```text
calmar = annual_return / abs(max_drawdown)
```

Đo return so với mức drawdown nặng nhất.

### VaR-95 Và CVaR-95

```text
var_95 = percentile(daily_returns, 5)
cvar_95 = mean(daily_returns <= var_95)
```

VaR-95 là ngưỡng lỗ trong nhóm 5% ngày xấu nhất. CVaR-95 là lỗ trung bình trong nhóm đó.

### Win Rate

```text
win_rate = number_of_positive_return_days / total_days
```

Không nên đọc win rate một mình. Một chiến lược có win rate cao nhưng lỗ lớn ở vài ngày vẫn có thể rất xấu.

## Test Set Results Theo Notebook / Model V2

Test set: SPY, `2023-01-03` đến `2024-12-30`, vốn `$10,000`.

| Method | Final Value | Return | Sharpe | Sortino | Calmar | Max DD | CVaR-95 | Volatility | Win Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | `$11,371.80` | `+13.72%` | `0.9706` | `1.1964` | `1.1366` | `-5.90%` | `-1.1034%` | `0.0694` | `0.5451` |
| CVaR-PPO | `$11,900.86` | `+19.01%` | `1.0729` | `1.4165` | `1.1057` | `-8.31%` | `-1.3178%` | `0.0851` | `0.5611` |
| Sortino-PPO | `$11,553.35` | `+15.53%` | `0.9589` | `1.1930` | `1.0465` | `-7.23%` | `-1.2552%` | `0.0783` | `0.5511` |
| Buy & Hold | `$15,882.05` | `+58.82%` | `1.8844` | `2.9956` | `2.6325` | `-9.97%` | `-1.7422%` | `0.1281` | `0.5760` |

### Diễn Giải Test Set

Buy & Hold thắng tất cả metric return/risk-adjusted chính trên test set vì giai đoạn 2023-2024 là bull market mạnh. Khi tài sản tăng nhiều, chiến lược nắm giữ 100% từ đầu kỳ thường rất khó bị đánh bại về return.

Trong ba RL model:

- CVaR-PPO có Return, Sharpe và Sortino tốt nhất.
- PPO có Max Drawdown và CVaR-95 tốt nhất trong notebook test set.
- Sortino-PPO nằm giữa PPO và CVaR-PPO về return, nhưng không vượt CVaR-PPO ở lần chạy này.

Điểm quan trọng: RL models có volatility thấp hơn Buy & Hold. Điều đó cho thấy agent thường không full exposure toàn thời gian, nên equity curve ít nhạy với thị trường hơn nhưng cũng bỏ lỡ upside.

## Test Results Của Artifact Active `demo/weights`

Django demo hiện load [../demo/weights](../demo/weights). File [../demo/weights/test_results.csv](../demo/weights/test_results.csv) có bảng:

| Method | Final Value | Return | Sharpe | Sortino | Calmar | Max DD | CVaR-95 | Volatility | Win Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | `$11,483.01` | `+14.83%` | `1.0476` | `1.2739` | `1.3306` | `-5.44%` | `-1.1110%` | `0.0689` | `0.5451` |
| CVaR-PPO | `$11,953.64` | `+19.54%` | `1.0746` | `1.4155` | `1.1271` | `-8.37%` | `-1.3407%` | `0.0872` | `0.5571` |
| Sortino-PPO | `$11,414.24` | `+14.14%` | `0.9124` | `1.1318` | `0.9869` | `-7.00%` | `-1.2077%` | `0.0753` | `0.5471` |
| Buy & Hold | `$15,882.05` | `+58.82%` | `1.8844` | `2.9956` | `2.6325` | `-9.97%` | `-1.7422%` | `0.1281` | `0.5760` |

Diễn giải không thay đổi về chất:

- Buy & Hold vẫn thắng return.
- CVaR-PPO vẫn là RL model có return cao nhất.
- PPO vẫn có drawdown thấp nhất trong nhóm RL.
- Sortino-PPO vẫn là phương án cân bằng nhưng chưa thắng CVaR-PPO trong lần chạy này.

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

### Diễn Giải Stress Test

COVID crash là điểm sáng của CVaR-PPO:

- Return dương `+1.44%`.
- MaxDD `-14.97%`, thấp hơn nhiều so với Buy & Hold `-33.72%`.
- Sortino dương, trong khi các chiến lược khác âm.

Bear 2022:

- Cả ba RL model đều giảm ít hơn Buy & Hold.
- CVaR-PPO có return tốt nhất trong bảng, nhưng PPO có MaxDD tốt nhất.
- Điều này cho thấy không có model nào thống trị tuyệt đối mọi risk metric.

Bull 2023-2024:

- Buy & Hold thắng mạnh return.
- CVaR-PPO thắng trong nhóm RL về return.
- PPO có drawdown tốt nhất trong nhóm RL.

Kết luận: risk-sensitive RL có ý nghĩa nhất khi thị trường bất lợi hoặc có tail event. Trong bull market, benchmark thụ động thường thắng return.

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

### Diễn Giải Multi-Asset

Điểm tích cực:

- Policy train trên SPY vẫn tạo return dương trên QQQ và GLD.
- RL models thường có drawdown thấp hơn Buy & Hold.
- Điều này cho thấy một mức generalization ban đầu.

Điểm cần thận trọng:

- Buy & Hold vẫn thắng return trên cả ba tài sản trong giai đoạn test.
- QQQ tăng rất mạnh giai đoạn 2023-2024, nên chiến lược nắm giữ có lợi thế lớn.
- GLD khác phân phối với SPY; dùng normalization từ SPY có thể gây distribution shift.
- Đây chưa phải multi-asset portfolio optimization. Model vẫn giao dịch từng asset độc lập, không học phân bổ vốn đồng thời giữa nhiều tài sản.

## Return Distribution

Notebook ghi nhận daily return distribution trên test set:

| Method | Mean daily % | Std daily % | VaR-95 | CVaR-95 |
|---|---:|---:|---:|---:|
| PPO | `+0.0267` | `0.4371` | `-0.7325` | `-1.1034` |
| CVaR-PPO | `+0.0362` | `0.5362` | `-0.9162` | `-1.3178` |
| Sortino-PPO | `+0.0298` | `0.4934` | `-0.8230` | `-1.2552` |
| Buy & Hold | `+0.0958` | `0.8072` | `-1.3775` | `-1.7422` |

Buy & Hold có mean daily return cao nhất nhưng cũng có std và tail loss lớn hơn. Đây là hình ảnh rõ của trade-off: nhiều upside hơn nhưng chịu biến động và tail risk lớn hơn.

## Vì Sao Buy & Hold Thắng Test Set?

Test set là 2023-2024, giai đoạn thị trường phục hồi và tăng trưởng mạnh. Buy & Hold giữ toàn bộ exposure từ đầu đến cuối, nên hưởng trọn đà tăng.

RL models thường điều chỉnh vị thế, có lúc giữ cash hoặc giảm exposure để tránh risk. Trong bull market mạnh, hành vi phòng thủ này làm mất upside.

Do đó:

- Nếu mục tiêu là return tuyệt đối trong bull market, Buy & Hold có lợi thế.
- Nếu mục tiêu là kiểm soát drawdown/tail event, cần đọc stress test và risk metrics.

## Vì Sao CVaR-PPO Tốt Trong COVID Crash?

CVaR-PPO được thiết kế để phạt nhóm return xấu nhất. Khi gặp tail event như COVID crash:

- policy có xu hướng giảm exposure nhanh hơn hoặc tránh vị thế quá rủi ro;
- tail-risk penalty trở nên có ích;
- drawdown giảm đáng kể so với Buy & Hold và PPO trong notebook stress test.

Đây là kết quả phù hợp với mục tiêu thiết kế, không chỉ là “may mắn metric”.

## Vì Sao PPO Có MaxDD Tốt Trên Test Set?

PPO không có risk penalty riêng, nhưng trong lần chạy này policy của PPO có thể học exposure thận trọng hơn trên test set, dẫn đến MaxDD thấp.

Điều này nhắc rằng:

- risk-sensitive method không đảm bảo luôn thắng mọi risk metric;
- policy behavior phụ thuộc training dynamics;
- cần nhiều seed để khẳng định ổn định thống kê.

## Artifact Đầu Ra

Notebook lưu:

```text
ppo_model.pth
cvar_ppo_model.pth
sortino_ppo_model.pth
norm_params.json
results.json
test_results.csv
```

Trong demo:

| File | Vai trò |
|---|---|
| `ppo_model.pth` | Trọng số Actor-Critic của PPO |
| `cvar_ppo_model.pth` | Trọng số Actor-Critic của CVaR-PPO |
| `sortino_ppo_model.pth` | Trọng số Actor-Critic của Sortino-PPO |
| `norm_params.json` | Thống kê chuẩn hóa và shape state/action |
| `results.json` | Config, test metrics, crisis summary |
| `test_results.csv` | Bảng metric test |

## Hạn Chế Của Kết Quả

- Chỉ một hoặc ít seed, chưa có confidence interval.
- Test period là bull market, thiên vị cho Buy & Hold.
- Crisis test dùng sub-period trong toàn bộ dữ liệu; một phần crisis có thể nằm trong train, nên nên đọc như stress replay, không phải unseen out-of-sample tuyệt đối.
- Không có walk-forward validation nhiều giai đoạn.
- Không tính thuế, spread động, liquidity, market impact.
- Model train chính trên SPY, chưa phải portfolio đa tài sản thật.
- Các bảng so sánh với nghiên cứu khác trong report nên đọc như tham khảo bối cảnh, vì tài sản/giai đoạn/giả định khác nhau không thể so sánh tuyệt đối.

## Kết Luận Thực Nghiệm

Kết luận cân bằng nhất:

- Buy & Hold thắng return trong giai đoạn test tăng mạnh.
- RL models có ý nghĩa khi phân tích risk-return trade-off.
- CVaR-PPO nổi bật nhất trong COVID crash theo notebook stress test.
- PPO kiểm soát MaxDD tốt trong test set của lần chạy này.
- Sortino-PPO thể hiện hướng tiếp cận downside-risk hợp lý nhưng cần tune thêm để vượt CVaR-PPO/PPO ổn định hơn.
- Project chứng minh pipeline xây dựng, đánh giá và demo risk-sensitive RL trading, nhưng chưa đủ để kết luận khả năng giao dịch thật.
