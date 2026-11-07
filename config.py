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
# group  ：按重复簇分组，近似重复的样本绑在同一侧（主实验）
# random ：随机分层，存在跨集合重复（对照，量化泄漏影响）
# time   ：按发布时间切分（对照，检验时间漂移）
# 切换：Windows  set CFND_SPLIT=random   /  Linux  export CFND_SPLIT=random
SPLIT_MODE = os.environ.get('CFND_SPLIT', 'group')
SPLIT_SEED = 42
_SUBDIR = {'random': 'processed', 'group': 'processed_group', 'time': 'processed_time'}
PROCESSED_SUBDIR = _SUBDIR.get(SPLIT_MODE, 'processed_group')

# 分组划分参数
GROUP_CONT_RATIO = 0.8     # 内容包含度超过它，判为同一簇
GROUP_TEMPLATE_DF = 0.02   # 出现在 2% 以上文档里的 n-gram 视为模板，聚类前剔除


def _resolve_kaggle_processed(input_root='/kaggle/input'):
    """在 Kaggle 的只读挂载区里找到预处理数据（按当前划分方式找对应子目录）。"""
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
    PROCESSED_DIR = _resolve_kaggle_processed()
    WORK_DIR = '/kaggle/working'
else:
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', PROCESSED_SUBDIR)
    WORK_DIR = BASE_DIR

# 输出目录按划分方式加后缀：跑对照组时不会覆盖主实验的权重与结果
_RUN_SUFFIX = '' if SPLIT_MODE == 'group' else '_' + SPLIT_MODE
CHECKPOINT_DIR = os.path.join(WORK_DIR, 'checkpoints' + _RUN_SUFFIX)
LOGS_DIR = os.path.join(WORK_DIR, 'logs' + _RUN_SUFFIX)
FIGURES_DIR = os.path.join(WORK_DIR, 'figures' + _RUN_SUFFIX)
RESULTS_DIR = os.path.join(WORK_DIR, 'results' + _RUN_SUFFIX)


def ensure_dir(path):
    """写文件之前再创建目录，避免 import config 时就在项目里堆出几个空文件夹。"""
    os.makedirs(path, exist_ok=True)

# 序列长度：预处理、训练、推理三处统一从这里取
MAX_LEN = 256
NUM_CLASSES = 2
LABEL_MAP = {'谣言': 0, '事实': 1}

# ==================== 实验协议 ====================
SEEDS = [42, 123, 2024]
MAIN_MODEL_LIST = ['fasttext', 'bilstm', 'bilstm_attention']
MODEL_LIST = ['fasttext', 'bilstm', 'bilstm_attention', 'bilstm_attention_nomask']
DEFAULT_MODEL = 'bilstm_attention'
DEPLOY_SEED = 42

# 正则档位：阶段 0 只跑核心模型，依据验证集 Macro-F1 选优，之后固定
REG_PROFILES = {
    'A': dict(DROPOUT=0.50, WEIGHT_DECAY=1e-4, WORD_DROPOUT=0.10),
    'B': dict(DROPOUT=0.60, WEIGHT_DECAY=2e-4, WORD_DROPOUT=0.15),
    'C': dict(DROPOUT=0.40, WEIGHT_DECAY=5e-5, WORD_DROPOUT=0.05),
}
# ==================== 最终配置 ====================
# 由全因子实验（24 配置）按验证集综合确定，改动只在这里：
#   reg=C（A/B/C 三档主效应差异仅 0.09，C 与 A 并列最优）
#   不使用类别权重（主效应 −0.15，不加权略优）
#   保留 Padding Mask（主效应 +0.19，验证集 +0.36）
#   保留词向量初始化（主效应 +1.02，C 档消融达 −2.86）
FINAL_REG = 'C'
FINAL_USE_CW = False
FINAL_USE_MASK = True
FINAL_USE_W2V = True

# 正则档位：阶段 0 / 全因子实验按验证集选定，作为默认档位
REG = os.environ.get('CFND_REG', FINAL_REG)
_reg = REG_PROFILES[REG]
DROPOUT = _reg['DROPOUT']
WEIGHT_DECAY = _reg['WEIGHT_DECAY']
WORD_DROPOUT = _reg['WORD_DROPOUT']

# 模型与训练超参（固定，不参与阶段 0）
EMBEDDING_DIM = 200
HIDDEN_DIM = 128
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
N_LAYERS = 1
LABEL_SMOOTHING = 0.05
EPOCHS = 100          # 上限；各模型由早停决定实际轮数（FastText 收敛慢，40 轮不够）
PATIENCE = 8
MIN_DELTA = 0.001

# 开关（默认值即最终配置）
USE_PRETRAINED_EMB = FINAL_USE_W2V   # Word2Vec 初始化嵌入层
USE_CLASS_WEIGHT = FINAL_USE_CW      # 类别加权交叉熵
W2V_WINDOW = 5
W2V_EPOCHS = 5

# 决策阈值：验证集搜索后写回这里
DECISION_THRESHOLD = 0.5
BAND_HALF_WIDTH = 0.07
