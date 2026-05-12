# References

Thư mục này chứa phần trình bày chi tiết được tách ra từ [risk_sensitive_trading.ipynb](../risk_sensitive_trading.ipynb). README gốc giữ vai trò tổng quan; các file trong thư mục này dùng như phụ lục kỹ thuật cho báo cáo, slide hoặc phần thuyết minh.

## Danh Sách Tài Liệu

| File | Nội dung |
|---|---|
| [01-data-and-features.md](01-data-and-features.md) | Dữ liệu, symbol, split theo regime, feature engineering và chuẩn hóa |
| [02-trading-environment.md](02-trading-environment.md) | Mô hình MDP, state, action, reward, transaction cost và slippage |
| [03-methods.md](03-methods.md) | Actor-Critic, PPO, CVaR-PPO và Sortino-PPO |
| [04-training-and-results.md](04-training-and-results.md) | Tham số huấn luyện, kết quả test, stress test và multi-asset evaluation |
| [05-django-demo.md](05-django-demo.md) | Cấu trúc demo Django, model weights, API và cách chạy web app |

## Cách Đọc Nhanh

Nếu cần hiểu dự án để trình bày trong vài phút, đọc [README chính](../README.md) trước. Nếu cần giải thích cơ chế học và rủi ro, đọc tiếp [03-methods.md](03-methods.md). Nếu cần bảo vệ kết quả thực nghiệm, ưu tiên [04-training-and-results.md](04-training-and-results.md).
