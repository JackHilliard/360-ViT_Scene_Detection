#from .swin_transformer4 import SwinTransformer4, refine_net
from models.simple_panoswin_transformer import SimplePanoSwinTransformer
#from .trans_ablation import Ablation
import torch
import random

def build_model(config):
    model_type = config['TYPE']
    if model_type == 'panoswin':
        model = SimplePanoSwinTransformer(patch_size=config['PATCH_SIZE'],
                                     in_chans=config['IN_CHANS'],
                                     n_classes=1,
                                     embed_dim=config['EMBED_DIM'],
                                     depths=config['DEPTHS'],
                                     num_heads=config['NUM_HEADS'],
                                     window_size=config['WINDOW_SIZE'],
                                     mlp_ratio=config['MLP_RATIO'],
                                     qkv_bias=config['QKV_BIAS'],
                                     qk_scale=config['QK_SCALE'],
                                     drop_rate=config['DROP_RATE'],
                                     attn_drop_rate=0.,
                                     drop_path_rate=config['DROP_PATH_RATE'],
                                     ape=True,
                                     patch_norm=config['PATCH_NORM'],
                                     out_indices=(0, 1, 2, 3),
                                     frozen_stages=-1,
                                     use_checkpoint=config['TRAIN.USE_CHECKPOINT'],
                                     pano_mode=True)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model