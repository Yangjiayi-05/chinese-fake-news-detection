import os

# ===== 自动适配本地与 Kaggle 环境 =====
if os.path.exists('/kaggle/input'):
    # 在 Kaggle 环境运行时
    # 注意：'mcfend-processed-data' 是你稍后要在 Kaggle 上传的 Dataset 名字，保持一致
    PROCESSED_DIR = '/kaggle/input/mcfend-processed-data'
    CHECKPOINT_DIR = '/kaggle/working/checkpoints'
    FIGURES_DIR = '/kaggle/working/figures'
    LOGS_DIR = '/kaggle/working/logs'
else:
    # 在本地 PyCharm 环境运行时
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# 自动创建目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ===== 下面保留你原本的超参数 =====
# 随机种子（跑多种子实验时改这里）
SEED = 42    # 可选：42 / 123 / 2024
...
