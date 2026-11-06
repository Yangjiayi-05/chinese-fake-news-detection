# -*- coding: utf-8 -*-
"""全局配置：路径、序列长度、超参、阈值、随机种子，全项目只在这里定义一次。"""
import glob
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IS_KAGGLE = os.path.exists('/kaggle/input')


def _first_existing(paths):
    """按顺序返回第一个存在的路径；都不存在时返回第一个，便于报错时看到期望位置。"""
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


# 原始数据：本地统一放在 data/raw 下，其余是 Kaggle 挂载点候选
RAW_CSV = _first_existing([
    os.path.join(BASE_DIR, 'data', 'raw', 'mcfend', 'news.csv'),
    '/kaggle/input/mcfend-original-dataset/mcfend/news.csv',
    '/kaggle/input/mcfend/mcfend/news.csv',
])

# ==================== 划分方式 ====================
# 'random'：随机分层划分，快；但同源新闻会跨训练/测试，存在数据泄漏
# 'group' ：按重复簇分组划分，近似重复的样本绑在同一侧，泄漏为 0
# 切换方式：Windows  set CFND_SPLIT=random  /  Linux  export CFND_SPLIT=random
SPLIT_MODE = os.environ.get('CFND_SPLIT', 'group')
SPLIT_SEED = 42
PROCESSED_SUBDIR = 'processed' if SPLIT_MODE == 'random' else 'processed_group'

# 分组划分的两个超参
GROUP_CONT_RATIO = 0.8     # 两条样本的内容包含度超过它，判为同一簇
GROUP_TEMPLATE_DF = 0.02   # 出现在 2% 以上文档里的 n-gram 视为模板，聚类前剔除


def _resolve_kaggle_processed(input_root='/kaggle/input'):
    """在 Kaggle 的只读挂载区里找到预处理数据。

    先按约定的数据集名尝试，找不到再扫一层通配兜底；
    这样即使数据集改了名字，只要里面有 vocab.pkl 就能跑起来。
    """
    candidates = [
        os.path.join(input_root, 'mcfend-processed-data', PROCESSED_SUBDIR),
        os.path.join(input_root, 'datasets', 'yanghuanhuan05',
                     'mcfend-processed-data', PROCESSED_SUBDIR),
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, 'vocab.pkl')):
            return path
    for path in (glob.glob(os.path.join(input_root, '*', PROCESSED_SUBDIR))
                 + glob.glob(os.path.join(input_root, '*', '*', PROCESSED_SUBDIR))):
        if os.path.exists(os.path.join(path, 'vocab.pkl')):
            return path
    return candidates[0]


if IS_KAGGLE:
    # Kaggle 上数据以 Dataset 形式只读挂载，产出统一写到 working 目录
    PROCESSED_DIR = _resolve_kaggle_processed()
    WORK_DIR = '/kaggle/working'
else:
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', PROCESSED_SUBDIR)
    WORK_DIR = BASE_DIR

CHECKPOINT_DIR = os.path.join(WORK_DIR, 'checkpoints')
LOGS_DIR = os.path.join(WORK_DIR, 'logs')
FIGURES_DIR = os.path.join(WORK_DIR, 'figures')
RESULTS_DIR = os.path.join(WORK_DIR, 'results')


def ensure_dir(path):
    """写文件之前再创建目录，避免 import config 时就在项目里堆出几个空文件夹。"""
    os.makedirs(path, exist_ok=True)

# 序列长度：预处理、训练、推理三处统一从这里取，避免训练 256 推理 200 的错位
MAX_LEN = 256
NUM_CLASSES = 2
LABEL_MAP = {'谣言': 0, '事实': 1}

# 实验协议：4 个模型 × 3 个种子 = 12 次
SEEDS = [42, 123, 2024]
MODEL_LIST = ['fasttext', 'bilstm', 'bilstm_avgpool', 'bilstm_attention']
DEFAULT_MODEL = 'bilstm_attention'
DEPLOY_SEED = 42

# 超参档位：paper 是论文正式口径，tuned 是 Kaggle 上试过的强正则那套
PROFILE = os.environ.get('CFND_PROFILE', 'paper')
PROFILES = {
    'paper': dict(EMBEDDING_DIM=200, HIDDEN_DIM=128, DROPOUT=0.4,
                  LEARNING_RATE=5e-4, WEIGHT_DECAY=5e-5),
    'tuned': dict(EMBEDDING_DIM=300, HIDDEN_DIM=256, DROPOUT=0.6,
                  LEARNING_RATE=3e-4, WEIGHT_DECAY=1e-4),
}
_cfg = PROFILES[PROFILE]
EMBEDDING_DIM = _cfg['EMBEDDING_DIM']
HIDDEN_DIM = _cfg['HIDDEN_DIM']
DROPOUT = _cfg['DROPOUT']
LEARNING_RATE = _cfg['LEARNING_RATE']
WEIGHT_DECAY = _cfg['WEIGHT_DECAY']

BATCH_SIZE = 64
N_LAYERS = 1
LABEL_SMOOTHING = 0.05
EPOCHS = 30
PATIENCE = 5
MIN_DELTA = 0.001

# 决策阈值：判定"事实/谣言"的分界，由 search_threshold.py 在验证集上搜出来后写回这里
DECISION_THRESHOLD = 0.5
# 模糊带半宽：系统层面的人工复核成本参数，与上面的分界线不是一回事
BAND_HALF_WIDTH = 0.07
