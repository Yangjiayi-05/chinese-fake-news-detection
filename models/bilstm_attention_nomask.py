from models.bilstm_attention import BiLSTM_Attention


class BiLSTM_Attention_NoMask(BiLSTM_Attention):
    """消融模型：结构与主模型完全一致，只把 Padding Mask 关掉。

    实现复用父类，靠 use_mask 开关区分，避免两份几乎相同的代码各自演化。
    保留这个类名是为了让既有的结果文件、脚本引用继续有效。
    """

    def __init__(self, *args, **kwargs):
        kwargs['use_mask'] = False
        super().__init__(*args, **kwargs)
