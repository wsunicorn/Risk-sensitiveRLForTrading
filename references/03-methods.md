# 03. Kiến Trúc Và Thuật Toán

File này giải thích các phương pháp trong project: Buy & Hold, PPO, CVaR-PPO, Sortino-PPO và mạng Actor-Critic dùng chung. Trọng tâm là hiểu **vì sao chọn các phương pháp này**, chúng tối ưu cái gì, khác nhau ở đâu và trade-off rủi ro/lợi nhuận nên được đọc như thế nào.

## Bài Toán Cần Giải

Project không cố trả lời “ngày mai giá SPY tăng hay giảm?”. Project trả lời:

```text
Với trạng thái thị trường và trạng thái danh mục hiện tại,
agent nên mua, bán hay giữ bao nhiêu để danh mục tốt hơn theo thời gian?
```

Vì vậy model phải học một policy:

```text
policy(state) -> action
```

Trong đó action là số liên tục trong `[-1, 1]`.

## Vì Sao Cần So Sánh Nhiều Phương Pháp?

Một chiến lược trading có thể tốt theo metric này nhưng xấu theo metric khác.

Ví dụ:

- Buy & Hold có thể lợi nhuận cao trong bull market nhưng drawdown lớn trong crisis.
- PPO có thể ổn định nhưng chưa nhạy với tail risk.
- CVaR-PPO có thể phòng thủ tốt hơn khi có cú sụt giảm cực đoan.
- Sortino-PPO có thể cân bằng hơn bằng cách phạt downside volatility.

Do đó project không chỉ tìm “model có return cao nhất”, mà phân tích **cơ chế tạo ra return và rủi ro**.

## Buy & Hold Benchmark

Buy & Hold là benchmark thụ động:

```text
shares = initial_balance / first_price
portfolio_value_t = shares * price_t
```

Ưu điểm:

- đơn giản;
- dễ giải thích;
- rất mạnh trong bull market;
- là baseline bắt buộc để biết model phức tạp có tạo thêm giá trị không.

Nhược điểm:

- không giảm vị thế khi rủi ro tăng;
- chịu toàn bộ drawdown của thị trường;
- không kiểm soát tail risk;
- không dùng thông tin technical indicators hoặc trạng thái rủi ro.

Trong project, Buy & Hold thắng return trên test set 2023-2024 vì đó là giai đoạn thị trường tăng mạnh. Điều này không làm project thất bại; nó chỉ cho thấy khi thị trường tăng đều, chiến lược nắm giữ toàn bộ tài sản thường có lợi thế.

## Actor-Critic Network

Ba agent PPO, CVaR-PPO và Sortino-PPO dùng chung kiến trúc Actor-Critic.

```text
Input state_dim = 19
-> Linear(19, 256) + ReLU
-> Linear(256, 256) + ReLU
-> shared representation
-> Actor head: Linear(256, 1)
-> Critic head: Linear(256, 1)
```

Notebook ghi nhận:

```text
ActorCritic: state_dim=19, action_dim=1
Total parameters: 71,427
```

## Actor Là Gì?

Actor là phần sinh action.

Trong notebook:

1. State đi qua shared MLP.
2. Actor head sinh `mean`.
3. `actor_log_std` là tham số học được.
4. Tạo phân phối Normal.
5. Sample action thô trong training.
6. Dùng `tanh` để ép action về `[-1, 1]`.

Pseudo-flow:

```text
h = shared(state)
mean = actor_mean(h)
std = exp(actor_log_std)
raw_action ~ Normal(mean, std)
action = tanh(raw_action)
```

Vì action được sample trong training, agent có exploration. Trong Django demo, backend dùng deterministic mean để kết quả ổn định:

```text
action = tanh(actor_mean(shared(state)))
```

## Critic Là Gì?

Critic ước lượng value của state:

```text
V(s_t) = expected future return from state s_t
```

Critic giúp tính advantage:

```text
advantage = return - value_estimate
```

Nếu advantage dương, action đã chọn tốt hơn kỳ vọng của critic. Nếu advantage âm, action đó kém hơn kỳ vọng. PPO dùng advantage để cập nhật policy.

## PPO Baseline

PPO là Proximal Policy Optimization. Đây là baseline risk-neutral trong project.

### Vì Sao Chọn PPO?

PPO phù hợp vì:

- hoạt động tốt với action liên tục;
- dễ triển khai hơn TRPO;
- ổn định hơn vanilla policy gradient;
- dùng clipped update để tránh policy thay đổi quá mạnh;
- là thuật toán phổ biến trong DRL trading, dễ so sánh với nghiên cứu khác;
- có thể mở rộng bằng risk penalty như CVaR hoặc reward shaping như Sortino.

### Objective Cốt Lõi

PPO so sánh policy mới và policy cũ qua ratio:

```text
ratio = exp(new_log_prob - old_log_prob)
```

Surrogate objective:

```text
surr1 = ratio * advantage
surr2 = clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
policy_loss = -mean(min(surr1, surr2))
```

Ý nghĩa clipping:

- Nếu policy mới khác policy cũ quá nhiều, ratio bị clip.
- Điều này làm update “vừa phải”, giúp training ổn định hơn.

Loss tổng:

```text
loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
```

Trong đó:

- `policy_loss`: cải thiện policy.
- `value_loss`: làm critic dự đoán return tốt hơn.
- `entropy`: khuyến khích exploration, tránh policy quá chắc quá sớm.

### Tham Số PPO

| Tham số | Giá trị |
|---|---:|
| `GAMMA` | `0.99` |
| `PPO_LR` | `2e-4` |
| `PPO_EPSILON` | `0.2` |
| `PPO_EPOCHS` | `8` |
| Gradient clipping | `0.5` |
| Update interval | `256` steps |

### PPO Tối Ưu Điều Gì?

PPO tối ưu reward kỳ vọng. Trong project, reward cơ sở là step return của portfolio:

```text
reward = portfolio_return
```

Điều này làm PPO trung lập rủi ro theo nghĩa: nó không có term riêng để nói “đừng để tail loss quá nặng” hoặc “đừng có downside volatility cao”. Nếu reward trong train cho thấy mua nhiều tạo return tốt, PPO có thể học policy tương đối aggressive.

## CVaR-PPO

CVaR-PPO mở rộng PPO bằng cách thêm penalty cho rủi ro đuôi.

### CVaR Là Gì?

VaR-95 trả lời:

```text
Trong 5% ngày xấu nhất, ngưỡng lỗ là bao nhiêu?
```

CVaR-95 trả lời sâu hơn:

```text
Nếu đã rơi vào 5% ngày xấu nhất, mức lỗ trung bình là bao nhiêu?
```

CVaR hữu ích vì VaR chỉ là một ngưỡng, còn CVaR nhìn vào độ nặng của phần đuôi.

Ví dụ:

- Chiến lược A và B cùng VaR-95 = `-2%`.
- A có worst losses quanh `-2.1%`.
- B có vài ngày `-8%`, `-10%`.
- VaR có thể giống nhau, nhưng CVaR của B sẽ xấu hơn nhiều.

### CVaR-PPO Trong Notebook

CVaR-PPO giữ nguyên môi trường và reward cơ sở, nhưng override `_extra_loss()`.

Logic:

```text
r_norm = (returns - returns.mean()) / returns.std()
sorted_returns = sort(r_norm)
worst = first alpha portion
cvar = mean(worst)
cvar_loss = relu(-cvar)
extra_loss = lambda * cvar_loss
```

Tham số:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `CVAR_ALPHA` | `0.15` | Xét worst 15% returns |
| `CVAR_LAMBDA` | `0.15` | Trọng số risk penalty ban đầu |
| `CVAR_LAMBDA_DECAY` | `0.999` | Giảm penalty dần qua training |
| `CVAR_LAMBDA_MIN` | `0.08` | Penalty tối thiểu |
| `CVAR_L2` | `1e-5` | Weight decay cho optimizer |

### Vì Sao Chuẩn Hóa Returns Trước Khi Tính CVaR?

Nếu returns có scale thay đổi mạnh, CVaR loss có thể phình lên hoặc nhỏ đi chỉ vì scale, không phải vì policy thật sự rủi ro hơn/kém hơn.

Notebook chuẩn hóa:

```text
r_norm = (returns - mean) / std
```

Điều này làm CVaR penalty tập trung vào tương quan/tail shape thay vì đơn vị đo tuyệt đối.

### Vì Sao Lambda Decay?

Penalty cao từ đầu giúp agent học phòng thủ. Nhưng nếu penalty quá cao mãi, agent có thể quá thận trọng và bỏ lỡ upside. Decay cho phép training ban đầu chú ý risk, sau đó giảm dần để policy vẫn tối ưu return.

### Trade-off Của CVaR-PPO

Ưu điểm:

- chú ý tail risk;
- có giá trị trong crisis;
- trong notebook, COVID crash là nơi CVaR-PPO nổi bật nhất.

Nhược điểm:

- cần chọn `alpha` và `lambda`;
- nếu penalty yếu, không khác PPO nhiều;
- nếu penalty mạnh, agent có thể quá phòng thủ;
- CVaR ước lượng từ batch nhỏ có thể nhiễu.

## Sortino-PPO

Sortino-PPO không thêm penalty vào PPO loss. Thay vào đó, nó sửa reward của environment.

### Sortino Ratio Là Gì?

Sharpe Ratio dùng độ lệch chuẩn của toàn bộ return, cả tăng lẫn giảm. Nhưng trong đầu tư, volatility tăng không xấu như volatility giảm.

Sortino tập trung vào downside deviation:

```text
Sortino = excess_return / downside_deviation
```

Ý tưởng: chỉ phạt các biến động bất lợi.

### Sortino-PPO Trong Notebook

SortinoTradingEnv override reward:

```text
r = portfolio_return
window = last_60_returns
negative_returns = returns < 0
downside_std = std(negative_returns)
penalty = sortino_lambda * downside_std^2
reward = r - min(penalty, abs(r) * 0.5)
```

Tham số:

| Tham số | Giá trị |
|---|---:|
| `SORTINO_LAMBDA` | `0.2` |
| Rolling window | `60` steps |
| Penalty cap | `50%` độ lớn reward |

### Vì Sao Dùng Reward Shaping?

Reward shaping đưa risk signal vào từng bước tương tác. Thay vì chỉ đến lúc update mới tính tail distribution, agent nhận reward đã phản ánh downside risk ngay trong environment.

Ưu điểm:

- trực quan;
- ổn định hơn một số risk loss nếu batch nhỏ;
- phù hợp với ý tưởng giảm downside volatility.

Nhược điểm:

- phụ thuộc rolling window;
- penalty cap là heuristic;
- có thể làm agent thận trọng và bỏ lỡ bull market upside.

## So Sánh PPO, CVaR-PPO, Sortino-PPO

| Method | Tối ưu chính | Cách đưa rủi ro vào | Khi nào có lợi | Rủi ro/nhược điểm |
|---|---|---|---|---|
| PPO | Expected portfolio return | Không có risk penalty riêng | Baseline ổn định, thị trường tăng | Có thể không phản ứng tốt với tail event |
| CVaR-PPO | PPO objective + tail penalty | Thêm loss dựa trên worst 15% normalized returns | Crisis, tail-risk control | Nhạy với `alpha`, `lambda`, batch distribution |
| Sortino-PPO | Reward đã chỉnh downside | Phạt downside deviation trong rolling window | Khi cần giảm biến động âm | Có thể quá thận trọng, phụ thuộc window |

## Vì Sao Cần Cùng Kiến Trúc?

Nếu mỗi model dùng architecture khác nhau, khó biết kết quả khác nhau là do:

- thuật toán risk-sensitive tốt hơn;
- hay mạng lớn hơn;
- hay hyperparameter khác;
- hay dữ liệu khác.

Project giữ cùng Actor-Critic để so sánh tập trung vào **cơ chế rủi ro**.

## Quan Hệ Giữa Training Và Evaluation

Trong training:

- actor sample action để exploration;
- policy update theo PPO;
- CVaR/Sortino thêm risk mechanism;
- agent đi qua train set nhiều episodes.

Trong evaluation/demo:

- dùng model đã train;
- thường dùng deterministic action;
- không update weights;
- chạy môi trường trên test/realtime data;
- tính metric từ history.

Điều này giống khác biệt giữa học chiến lược và kiểm tra chiến lược.

## Các Metric Liên Quan Đến Phương Pháp

| Metric | Liên quan nhất đến | Ý nghĩa |
|---|---|---|
| Total Return | Tất cả | Lợi nhuận cuối kỳ |
| Sharpe | PPO/overall | Return trên mỗi đơn vị volatility tổng |
| Sortino | Sortino-PPO | Return trên mỗi đơn vị downside risk |
| Max Drawdown | Risk management | Mức sụt giảm sâu nhất từ đỉnh |
| CVaR-95 | CVaR-PPO | Lỗ trung bình trong nhóm ngày xấu nhất |
| Volatility | Overall risk | Độ biến động annualized |
| Win Rate | Hành vi giao dịch | Tỷ lệ ngày return dương |

Không nên chọn model chỉ bằng một metric. Ví dụ, Buy & Hold thắng Total Return nhưng có Max DD/CVaR xấu hơn trong nhiều giai đoạn. PPO có thể Max DD thấp nhưng return thấp. CVaR-PPO có thể phòng thủ tốt hơn crisis nhưng không luôn thắng bull market.

## Hạn Chế Phương Pháp

- PPO, CVaR-PPO và Sortino-PPO đều học từ dữ liệu lịch sử; nếu regime tương lai khác mạnh, policy có thể suy giảm.
- Action space một tài sản không phải portfolio allocation đa tài sản thật.
- CVaR penalty trong notebook là một approximation theo batch returns.
- Sortino reward shaping dùng rolling window và penalty cap thủ công.
- Chưa có nhiều seed để chứng minh độ ổn định thống kê.
- Chưa có transaction-cost-aware objective riêng ngoài cost/slippage trong môi trường.
- Chưa có temporal model như LSTM/Transformer; state hiện tại chứa indicator rolling nhưng mạng không trực tiếp nhớ chuỗi dài.

## Tóm Tắt

PPO là baseline để học policy giao dịch ổn định. CVaR-PPO thêm góc nhìn tail risk bằng penalty trong loss. Sortino-PPO thêm góc nhìn downside risk bằng reward shaping. Ba phương pháp dùng chung Actor-Critic để khác biệt kết quả phản ánh chủ yếu cách xử lý rủi ro, không phải khác biệt kiến trúc.
