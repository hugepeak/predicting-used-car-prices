# Used Car Price Project

当前目录使用 `.venv` + `data/` 做统一管理，训练数据位于：
- `data/used_car_train_20200313.csv`
- `data/used_car_testB_20200421.csv`
- `data/used_car_sample_submit.csv`

## 已建文件
- `inspect_csv.py`：查看字段与前 5 行
- `train_steps.py`：按轻量 -> 中量 -> 全量的分步验证脚本（推荐）
- `train_car_price.py`：完整基线脚本
- `.gitignore`：忽略虚拟环境、模型和临时文件
- `requirements.txt`：依赖清单

## 第一步：安装与环境
```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 第二步：数据检查
```bash
.venv\Scripts\python.exe inspect_csv.py
```

## 第三步：按你电脑资源分步跑（推荐）
```bash
.venv\Scripts\python.exe train_steps.py --mode quick
.venv\Scripts\python.exe train_steps.py --mode step2
# quick + step2 表现稳定后再跑：
.venv\Scripts\python.exe train_steps.py --mode full
```

## 这次按 PDF 方案“第十三名总结”提取的关键点
- 目标值 `price` 做偏态处理（本文用 `Box-Cox`）
- 去掉 `SaleID`、`name`；剔除部分异常点（例如 `seller == 1`）
- `power` 缺失与异常处理，并截断到 0~600
- `regDate`、`creatDate` 做日期特征（年/月/日、使用时长）
- 对 `power`、`model`、`kilometer` 做分桶
- 加入分组目标统计特征（如品牌/型号的 price 均值/最大值/中位数）
- 以 CatBoost + MAE 为主，轻量步骤逐步加量

默认不强制你一次性全量训练；每个 mode 可单独运行，适合配置较低的机器。
