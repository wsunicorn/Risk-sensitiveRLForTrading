# Risk-Sensitive Reinforcement Learning for Trading

## 1. Tổng quan Dự án (Project Overview)

Đây là hệ thống giao dịch tự động bằng **Học tăng cường (Reinforcement Learning - RL)**, được xây dựng để giải quyết bài toán: *Làm thế nào để một agent giao dịch không chỉ tối đa hóa lợi nhuận mà còn tự động nhận biết và phòng vệ trước các rủi ro sụt giảm mạnh (tail-risk/downside risk) của thị trường?*

Thay vì dự đoán giá (Price Prediction) như các mô hình Supervised Learning truyền thống (LSTM, ARIMA), dự án này mô hình hóa việc giao dịch thành bài toán **Markov Decision Process (MDP)**. Agent sẽ trực tiếp học ra **Chính sách (Policy)** giao dịch: Ở trạng thái thị trường hiện tại, nên đưa ra quyết định (Action) Mua/Bán/Giữ với tỷ trọng bao nhiêu.

Hệ thống cung cấp một pipeline hoàn chỉnh từ thu thập dữ liệu, huấn luyện mô hình, backtest đến một Web App (Django) trực quan để demo và so sánh các chiến lược.

## 2. Xử lý Dữ liệu & Kỹ nghệ Đặc trưng (Data Pipeline & Feature Engineering)

Dự án sử dụng dữ liệu lịch sử OHLCV (Open, High, Low, Close, Volume) tải từ Yahoo Finance qua thư viện `yfinance`. 

### 2.1. Phân chia dữ liệu (Data Split - Regime Based)
Để chứng minh Agent không bị "học vẹt" (overfitting) và có thể sống sót qua các pha thị trường khác nhau, dữ liệu được chia có chủ đích:
- **Tập Train (2018 - 2021):** Bao gồm cả uptrend và đợt sụp đổ kinh hoàng do COVID-19 (tháng 2-3/2020) để Agent học cách đối phó với khủng hoảng.
- **Tập Validation (2022):** Giai đoạn thị trường gấu (Bear Market), lạm phát cao, chứng khoán giảm điểm liên tục (-19%).
- **Tập Test (2023 - 2024):** Giai đoạn thị trường phục hồi và tăng trưởng (Bull run). Mô hình hoàn toàn chưa từng nhìn thấy dữ liệu này khi huấn luyện.

### 2.2. Không gian Trạng thái (State Space / Features)
Agent quan sát thị trường qua một vector trạng thái gồm **19 chiều (19 dimensions)**:
- **16 Technical Indicators (Market Features):**
  - *Xu hướng (Trend):* SMA 10, SMA 20, SMA 50.
  - *Động lượng (Momentum):* RSI 14, MACD, MACD Signal, MACD Diff.
  - *Biến động (Volatility):* Bollinger Bands (High, Mid, Low), ATR (Average True Range).
  - *Khác:* Daily Returns, Log Returns, Volume SMA 20.
- **3 Portfolio Features:** 
  - `balance / initial_balance` (Tỷ lệ tiền mặt hiện có)
  - `(shares * price) / initial_balance` (Giá trị cổ phiếu đang nắm giữ)
  - `portfolio_value / initial_balance` (Tổng giá trị danh mục)

**Chuẩn hóa (Normalization):** Tất cả 16 Market Features đều được chuẩn hóa (Z-score normalization) dựa trên `mean` và `std` của *riêng tập Train*. Các tham số này được lưu vào `weights/norm_params.json` để dùng cho lúc Test và trên Web App. Việc này cực kỳ quan trọng để chống rò rỉ dữ liệu (Data Leakage) và giúp Neural Network hội tụ nhanh hơn.

## 3. Môi trường Giao dịch (Trading Environment)

Được kế thừa từ `gymnasium.Env` của OpenAI, môi trường mô phỏng chân thực thị trường tài chính:
- **Action Space:** Không gian hành động liên tục `[-1.0, 1.0]`. 
  - `action > 0.01`: **BUY**. Agent sẽ dùng tỷ lệ tiền tương ứng với `action` để mua cổ phiếu.
  - `action < -0.01`: **SELL**. Agent sẽ bán ra tỷ lệ cổ phiếu đang nắm giữ tương ứng với `abs(action)`.
  - Còn lại `[-0.01, 0.01]`: **HOLD** (Đứng ngoài hoặc giữ nguyên).
- **Frictions (Chi phí ma sát):** Mô phỏng chi phí giao dịch (Transaction Cost) `0.1%` và trượt giá (Slippage) `0.05%`. Điều này ép Agent không được giao dịch "quá độ" (overtrading) vì mỗi lần trade đều mất phí.
- **Reward (Phần thưởng):** Mặc định ở mỗi bước thời gian $t$, phần thưởng là tỷ suất lợi nhuận của danh mục: 

$$
r_t = \frac{\text{Portfolio Value}_{t} - \text{Portfolio Value}_{t-1}}{\text{Portfolio Value}_{t-1}}
$$

## 4. Kiến trúc Mạng Neural (Actor-Critic Network)

Các Agent chia sẻ chung một kiến trúc mạng Neural (Actor-Critic):
- **Shared Backbone:** Dữ liệu State (19 chiều) đi qua 2 lớp ẩn (Hidden Layers) Dense với 256 nốt, dùng hàm kích hoạt ReLU.
- **Critic Head:** Nhận output từ backbone, đi qua 1 lớp Linear(256, 1) để dự đoán Value function (giá trị kỳ vọng của state hiện tại).
- **Actor Head:** Dự đoán tham số của phân phối chuẩn (Normal Distribution):
  - `actor_mean`: Lớp Linear(256, 1).
  - `actor_log_std`: Tham số học được độc lập (Parameter).
  - Lúc Train: Lấy mẫu (sample) hành động từ phân phối này để tăng cường Exploration. Lúc Test/Demo: Lấy trực tiếp `mean` (Deterministic) để đảm bảo kết quả ổn định. Sau cùng, áp dụng hàm `tanh` để ép giá trị hành động về khoảng `[-1, 1]`.

## 5. Chi tiết các Thuật toán & Phương pháp (Methodology)

Đây là trọng tâm khoa học của dự án. So sánh 3 phương pháp tối ưu chính sách:

### 5.1. PPO (Proximal Policy Optimization) - Risk Neutral Baseline
PPO là thuật toán tiêu chuẩn, tối đa hóa trực tiếp hàm lợi nhuận kỳ vọng, sử dụng kỹ thuật "Clipping" để chính sách không cập nhật quá mạnh trong 1 bước.
- **Điểm mạnh:** Lợi nhuận rất cao trong thị trường uptrend, vì mô hình có xu hướng "All-in" (aggressive).
- **Điểm yếu:** Nó bị mù rủi ro. Khi thị trường sập (như Bear Market 2022), nó chịu Max Drawdown rất nặng nề vì không có hàm loss nào ép nó phòng thủ.

### 5.2. CVaR-PPO (Tail Risk Constraint)
Thêm thành phần rủi ro đuôi (Tail Risk) trực tiếp vào Hàm Loss của PPO. 
- **Định nghĩa CVaR:** Conditional Value-at-Risk ở mức $\alpha=0.15$ tính toán trung bình của 15% lợi nhuận *tệ nhất* trong batch. 
- **Toán học:** Hàm Loss của policy giờ đây được cộng thêm một khoản penalty: 

$$
\text{Loss} = \text{Loss}_{PPO} + \lambda \times \max(0, -CVaR_{\alpha})
$$
  *(Lưu ý thuật toán thực tế đã chuẩn hóa return trong batch trước khi tính CVaR để tránh hàm loss phát triển đến vô hạn).*
- **Cách thức hoạt động:** Trọng số $\lambda$ bắt đầu ở mức 0.15 và phân rã (decay) dần với hệ số 0.999. Hàm penalty này buộc Agent phải thay đổi trọng số neural network sao cho các tình huống "thua lỗ nặng" (nằm trong 15% đuôi) ít xảy ra hơn.
- **Kết quả:** Max Drawdown được hạn chế tối đa trong giai đoạn khủng hoảng. Tuy nhiên, trong Bull market, do bị "sợ hãi" rủi ro đuôi, mô hình có thể e dè và sinh lời ít hơn PPO cơ bản.

### 5.3. Sortino-PPO (Downside Deviation Penalty)
Thay vì phạt vào Hàm Loss như CVaR-PPO, Sortino-PPO phạt trực tiếp vào **Hàm Reward** trong Environment.
- **Cơ chế:** Tính toán độ lệch chuẩn của các khoản lợi nhuận âm (downside volatility) trong khung thời gian 60 ngày gần nhất (rolling window).
- **Toán học:** 

$$
\text{Reward}_{Sortino} = r_t - \lambda \times (\sigma_{downside})^2
$$
  (Trong code, $\lambda = 0.2$ và mức phạt được cap (giới hạn) không quá 50% mức lợi nhuận tuyệt đối để không làm sụp đổ quá trình hội tụ học).
- **Kết quả:** Thuật toán mang tính "cân bằng" hơn. Nó tối ưu hóa tỷ lệ Sortino (Lợi nhuận / Rủi ro sụt giảm). Nó tốt hơn PPO trong Down-trend và không quá "nhát gan" như CVaR-PPO trong Up-trend.

## 6. Quá trình Huấn luyện (Training Execution)

- **Tham số Train:** Mô hình train trong 300 Episodes. Thu thập trải nghiệm và update network mỗi `256` steps (Update Interval). Learning rate của Actor-Critic là `2e-4`.
- **CVaR-L2 Regularization:** Riêng CVaR-PPO dùng L2 regularization (`weight_decay=1e-5`) để tăng thêm độ ổn định cho trọng số khi đối mặt với tính chất phi tuyến của hàm loss CVaR.

## 7. Các Chỉ số Đo lường & Đánh giá (Evaluation Metrics)

Để đánh giá một hệ thống giao dịch, chỉ xem Lợi nhuận (Return) là hoàn toàn thiếu sót. Dự án sử dụng hệ metric toàn diện:
- **Total Return & Annualized Return:** Tỷ suất sinh lời tổng và tỷ suất sinh lời quy năm.
- **Sharpe Ratio:** Lợi nhuận quy năm sinh ra trên mỗi đơn vị rủi ro tổng thể (Volatility).
- **Sortino Ratio:** Lợi nhuận sinh ra trên mỗi đơn vị *rủi ro sụt giảm (Downside Volatility)*. Đây là thước đo tốt hơn cho nhà đầu tư vì Volatility dương (tăng giá) là điều tốt và không nên bị phạt.
- **Max Drawdown (Max DD):** Mức sụt giảm sâu nhất từ đỉnh (Peak-to-Trough). Chỉ số sinh tử của các quỹ đầu tư.
- **Calmar Ratio:** Tỷ lệ giữa Lợi nhuận quy năm / |Max Drawdown|.
- **VaR-95% & CVaR-95%:** Đo lường tính toán rủi ro đuôi (Tail-Risk). Định lượng mức thua lỗ bình quân ở 5% số ngày tệ nhất.

## 8. Cấu trúc Web App (Django) và System Flow

Giao diện Web được xây dựng không chỉ để hiển thị tĩnh mà có backend tính toán trực tiếp:
1. Khi user bấm Backtest/Signal trên Web, Django (ở `demo/trading/views.py`) lập tức dùng `yfinance` kéo dữ liệu thực từ thị trường.
2. Tính toán lại 16 Technical Indicators theo logic khớp 100% với file Jupyter Notebook.
3. Chuẩn hóa State array bằng `norm_params.json` đã lưu.
4. Load trọng số `.pth` (từ `demo/weights/`) vào Actor-Critic model và thực thi bước nhảy (forward pass).
5. Nếu là Backtest, hệ thống tạo Environment giả lập việc mua bán, trừ phí trượt giá (slippage) và gửi History array về cho Frontend vẽ biểu đồ bằng Javascript (Plotly).
6. API phân tách kiến trúc rành mạch: `/api/realtime_predict/`, `/api/backtest/`, `/api/compare_models/`.

## 9. Hướng dẫn Cài đặt & Chạy Code

Project yêu cầu Python 3.8+ (khuyên dùng 3.11+). Các lệnh bên dưới dùng cho PowerShell / Terminal:

```powershell
# 1. Khởi tạo và kích hoạt môi trường ảo (Virtual Environment)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt các thư viện lõi (Django, PyTorch, Pandas, yfinance, ta...)
pip install -r demo\requirements.txt

# 3. Chạy System Check của Django để đảm bảo model (.pth) được load đầy đủ
python demo\manage.py check

# 4. Chạy Server
python demo\manage.py runserver
```

Truy cập `http://127.0.0.1:8000/` trên trình duyệt để trải nghiệm Dashboard, xem Realtime Signal và chạy Backtest Comparison.
