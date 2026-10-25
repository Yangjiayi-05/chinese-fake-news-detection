import os

# ===== Environment Setup =====
if os.path.exists('/kaggle/input'):
    # Kaggle 绝对路径
    PROCESSED_DIR = '/kaggle/input/datasets/yanghuanhuan05/mcfend-processed-data/processed'
    CHECKPOINT_DIR = '/kaggle/working/checkpoints'
    FIGURES_DIR = '/kaggle/working/figures'
    LOGS_DIR = '/kaggle/working/logs'
else:
    # 本地 PyCharm 路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ===== Optimized Hyperparameters =====
SEED = 2024
BATCH_SIZE = 64
EMBEDDING_DIM = 300      # Increased from 200
HIDDEN_DIM = 256         # Increased from 128
N_LAYERS = 1
DROPOUT = 0.5            # Increased to prevent overfitting
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 5e-5
LABEL_SMOOTHING = 0.05
EPOCHS = 30
PATIENCE = 5
MAX_LEN = 200
MIN_DELTA = 0.001

# Choose: 'fasttext', 'bilstm', 'bilstm_avgpool', 'bilstm_attention'
MODEL_NAME = 'bilstm_attention' 
