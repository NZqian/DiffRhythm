import json

from .factory import create_model_from_config
from .utils import load_ckpt_state_dict

from huggingface_hub import hf_hub_download

def get_pretrained_model(config_path, ckpt_path):

    with open(config_path) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)

    model.load_state_dict(load_ckpt_state_dict(ckpt_path))

    return model, model_config