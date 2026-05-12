# 05. Django Demo Và Luồng Hệ Thống

Thư mục [../demo](../demo) chứa Web App Django để demo model đã train. App cho phép người dùng xem dashboard, lấy realtime signal, chạy backtest và so sánh nhiều chiến lược.

## Cấu Trúc Chính

```text
demo/
├── manage.py
├── requirements.txt
├── trading/
│   ├── urls.py
│   └── views.py
├── trading_platform/
│   └── settings.py
├── templates/
│   ├── dashboard.html
│   ├── index.html
│   ├── compare.html
│   ├── ppo.html
│   ├── cvar.html
│   └── sortino.html
├── static/
│   ├── css/style.css
│   └── js/main.js
└── weights/
    ├── ppo_model.pth
    ├── cvar_ppo_model.pth
    ├── sortino_ppo_model.pth
    ├── norm_params.json
    ├── results.json
    └── test_results.csv
```

## Model Artifacts

Các file trong `demo/weights` phục vụ inference:

| File | Vai trò |
|---|---|
| `ppo_model.pth` | Trọng số Actor-Critic của PPO |
| `cvar_ppo_model.pth` | Trọng số Actor-Critic của CVaR-PPO |
| `sortino_ppo_model.pth` | Trọng số Actor-Critic của Sortino-PPO |
| `norm_params.json` | Feature columns, mean/std train, state/action dim |
| `results.json` | Config, test metrics và crisis summary |
| `test_results.csv` | Bảng metric test set |

`norm_params.json` là cầu nối quan trọng giữa notebook và web app. Khi người dùng chọn mã/ticker trên web, backend phải tính feature và chuẩn hóa đúng bằng mean/std từ training.

## Luồng Backtest

Luồng xử lý backtest trong Django:

1. Frontend gửi request với ticker, ngày bắt đầu, ngày kết thúc và model cần dùng.
2. Backend tải dữ liệu bằng `yfinance`.
3. Backend tính lại technical indicators giống notebook.
4. State được chuẩn hóa bằng `norm_params.json`.
5. Actor-Critic load trọng số `.pth` tương ứng.
6. Backend mô phỏng mua/bán với transaction cost và slippage.
7. Kết quả history và metrics được trả về frontend để vẽ biểu đồ bằng Plotly.

## API Chính

README cũ và cấu trúc app cho thấy các API chính:

```text
/api/realtime_predict/
/api/backtest/
/api/compare_models/
```

Ý nghĩa:

- `realtime_predict`: sinh tín hiệu hiện tại từ model đã chọn.
- `backtest`: chạy lại chiến lược trên khoảng thời gian người dùng nhập.
- `compare_models`: so sánh PPO, CVaR-PPO, Sortino-PPO và Buy & Hold.

## Chạy Demo

Từ thư mục gốc project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r demo\requirements.txt
python demo\manage.py check
python demo\manage.py runserver
```

Truy cập:

```text
http://127.0.0.1:8000/
```

## Gợi Ý Khi Trình Bày

Khi demo, nên đi theo thứ tự:

1. Mở dashboard để giới thiệu mục tiêu dự án.
2. Chạy realtime signal cho một ticker quen thuộc như SPY.
3. Chạy backtest riêng từng model để giải thích action policy.
4. Mở compare page để cho thấy trade-off giữa return và drawdown.
5. Kết nối lại với kết quả stress test trong [04-training-and-results.md](04-training-and-results.md).
