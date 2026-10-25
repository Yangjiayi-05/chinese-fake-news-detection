import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# 随机种子（跑多种子实验时改这里）
SEED = 42    # 可选：42 / 123 / 2024

# 超参数
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