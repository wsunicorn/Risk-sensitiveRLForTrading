# References

Thư mục này là “sổ tay kỹ thuật” cho project [Risk-Sensitive Reinforcement Learning for Trading](../README.md). README chính giúp nắm bức tranh tổng quan; các file trong thư mục này đi sâu vào dữ liệu, môi trường, thuật toán, kết quả và web demo.

Nguồn nội dung được tổng hợp từ:

- [../risk_sensitive_trading.ipynb](../risk_sensitive_trading.ipynb)
- [../docs/Nhom15_Report.md](../docs/Nhom15_Report.md)
- [../demo/trading/views.py](../demo/trading/views.py)
- Các artifact trong [../data](../data), [../demo/weights](../demo/weights), [../demo/model_v1](../demo/model_v1), [../demo/model_v2](../demo/model_v2)

## Cách Đọc Nhanh

Nếu bạn chỉ có 10 phút:

1. Đọc [README chính](../README.md) để hiểu mục tiêu, pipeline và kết quả nhanh.
2. Đọc phần “So sánh cơ chế rủi ro” trong [03-methods.md](03-methods.md).
3. Đọc “Diễn giải kết quả” trong [04-training-and-results.md](04-training-and-results.md).

Nếu bạn cần review sâu để bảo vệ đồ án hoặc phát triển tiếp:

1. [01-data-and-features.md](01-data-and-features.md)
2. [02-trading-environment.md](02-trading-environment.md)
3. [03-methods.md](03-methods.md)
4. [04-training-and-results.md](04-training-and-results.md)
5. [05-django-demo.md](05-django-demo.md)

## Bản Đồ Tài Liệu

| File | Nên đọc khi muốn hiểu |
|---|---|
| [01-data-and-features.md](01-data-and-features.md) | Vì sao dùng SPY/QQQ/GLD, dữ liệu được làm sạch thế nào, từng technical indicator có ý nghĩa gì, tại sao phải chuẩn hóa theo train |
| [02-trading-environment.md](02-trading-environment.md) | Cách biến trading thành MDP, state/action/reward, transaction cost, slippage và các giả định của môi trường |
| [03-methods.md](03-methods.md) | Lý do chọn Buy & Hold, PPO, CVaR-PPO, Sortino-PPO; Actor-Critic hoạt động ra sao; risk penalty khác nhau thế nào |
| [04-training-and-results.md](04-training-and-results.md) | Config huấn luyện, metric tài chính, kết quả test, stress test, multi-asset generalization, cách diễn giải đúng |
| [05-django-demo.md](05-django-demo.md) | Luồng web app, API, model artifacts, inference/backtest trong Django và các lỗi dễ gặp |

## Mental Model Của Project

Một cách ngắn gọn:

```text
OHLCV data
-> technical indicators
-> normalized market features
-> portfolio features
-> 19-dimensional state
-> Actor-Critic policy
-> continuous action [-1, 1]
-> buy/sell/hold execution
-> portfolio value
-> reward and risk metrics
```

Phần cần nhớ nhất là project không dự báo giá ngày mai. Model học một **policy giao dịch**: ở trạng thái này nên tăng vị thế, giảm vị thế hay đứng ngoài bao nhiêu phần trăm.

## Ghi Chú Về Số Liệu

Trong repo hiện có nhiều bộ artifact:

| Thư mục | Vai trò |
|---|---|
| [../demo/weights](../demo/weights) | Artifact active mà Django demo đang load |
| [../demo/model_v1](../demo/model_v1) | Bản artifact lưu riêng, metric giống `demo/weights` trong repo hiện tại |
| [../demo/model_v2](../demo/model_v2) | Bản artifact khớp với nhiều output trong notebook/report |

Vì học tăng cường có sampling, mỗi lần train hoặc mỗi phiên bản weights có thể lệch nhẹ. Các tài liệu này ưu tiên giải thích bằng số liệu từ notebook/report khi nói về kết quả nghiên cứu, đồng thời ghi chú khi số liệu active demo khác một chút.

## Tài Liệu Gốc

Report Word đã được chuyển thành Markdown kèm ảnh:

- [../docs/Nhom15_Report.md](../docs/Nhom15_Report.md)
- [../docs/Nhom15_Report_media](../docs/Nhom15_Report_media)

Bản Markdown report giữ nội dung chính và hình ảnh để tra cứu nhanh. Một số layout Word như merge cell, công thức tự động hoặc caption có thể đã được đơn giản hóa khi chuyển đổi.
