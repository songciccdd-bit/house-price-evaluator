# 北京二手房价格评估服务

这是一个基于 Flask 的北京二手房价格评估服务，使用真实挂牌数据训练的机器学习与深度学习模型进行房价预测。

## 功能

- 选择区域、商圈、小区和房源属性
- 根据面积、户型、楼层、朝向、装修等特征评估总价与单价
- XGBoost、LightGBM、GradientBoosting、ExtraTrees、DNN 五模型堆叠

## 本地运行

建议使用 Python 3.12。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

打开 <http://localhost:7860>。

## Docker 运行

```bash
docker build -t house-price-evaluator .
docker run --rm -p 7860:7860 house-price-evaluator
```

打开 <http://localhost:7860>。

## 在线运行

点击下面的按钮，在 GitHub Codespaces 中打开项目：

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/songciccdd-bit/house-price-evaluator)

在 Codespaces 中运行 `python app.py` 后，将端口 `7860` 设置为公开即可分享访问链接。

## 模型文件

`custom_hybrid_model.pkl` 和 `custom_hybrid_keras.keras` 通过 Git LFS 管理。克隆仓库前请安装 [Git LFS](https://git-lfs.com/)。

## 免责声明

估价结果仅供研究参考，不构成任何交易或投资依据。
