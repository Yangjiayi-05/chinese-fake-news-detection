# Chinese Fake News Detection (MCFEND Dataset)

本项目基于 BiLSTM 与 Self-Attention 机制，实现中文社交媒体虚假新闻检测。

## 运行环境
- Python 3.8+
- PyTorch 1.10+
- 依赖安装: `pip install -r requirements.txt`

## 目录说明
- `main.py`: 主训练脚本
- `models/`: 包含核心模型与消融对比模型
- `utils/`: 数据加载、训练辅助与可视化工具
- `app.py`: Streamlit 交互式检测界面
