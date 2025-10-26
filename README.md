# CONQ028-AIO2025-Project-5.1

## Installation

Sử dụng thư viện cookiecutter-data-science template để tạo cấu trúc cơ bản cho dự án:

```bash
# tạo môi trường ảo độc lập
python -m venv venv
source venv/bin/activate

pip install cookiecutter-data-science
```

Khởi tạo project:
```bash
ccds
```
Cây thư viện của project:

```
project_51/
├── .env
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── data/
│   ├── external/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
├── docs/
│   └── .gitkeep
├── inventory_optimization/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── plots.py
│   └── modeling/
│       ├── __init__.py
│       ├── predict.py
│       └── train.py
├── models/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── references/
│   └── .gitkeep
├── reports/
│   ├── .gitkeep
│   └── figures/
│       └── .gitkeep
└── tests/
    └── test_data.py
```



