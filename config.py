import os

# ===== 自动适配本地与 Kaggle 环境 =====
if os.path.exists('/kaggle/input'):
    # 在 Kaggle 环境运行时
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

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ===== 超参数 =====
SEED = 42    # 可选：42 / 123 / 2024
BATCH_SIZE = 64
EMBEDDING_DIM = 200
HIDDEN_DIM = 128
N_LAYERS = 1
DROPOUT = 0.4
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 5e-5
LABEL_SMOOTHING = 0.05
EPOCHS = 30
PATIENCE = 5
MAX_LEN = 200
MIN_DELTA = 0.001

# 模型选择：'fasttext' / 'bilstm' / 'bilstm_attention'
MODEL_NAME = 'fasttext'
