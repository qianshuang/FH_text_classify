# -*- coding: utf-8 -*-

from modelscope import snapshot_download

model_dir = snapshot_download("damo/nlp_roberta_backbone_base_std", revision='v1.0.0', cache_dir='')
