#!/usr/bin/env python


import os
import shutil
import argparse 
import logging 
import sys 
import yaml


from distutils.dir_util import copy_tree
from pathlib import Path
from tempfile import TemporaryDirectory
from huggingface_hub import snapshot_download
from tensorizer import TensorSerializer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from tensorize_model import tensorize_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


def download_model_from_hf_hub(
        model_id: str, 
        hf_model_path: str,
        rm_existing_model: bool = True,
    ) -> dict:
    """
    This function downloads a model from the Hugging Face Hub and saves it locally.
    It also saves the tokenizer in a separate location so that it can be easely included in a docker Image
    without including the model weights.

    Args:
        model_id (str): Name of model on hugging face hub
        hf_model_path (str): Local path where Hugging Face model is saved
        rm_existing_model (bool, optional): Whether to remove the existing model or not. Defaults to False.

    Returns:
        dict: Dictionary containing the model name and path
    """

    # model_weights_path = os.path.join(os.getcwd(), "model_weights/torch_weights")
    # hf_model_path = os.path.join(model_weights_path, model_id)


    if rm_existing_model:
        logger.info(f"Removing existing model at {hf_model_path}")
        if os.path.exists(hf_model_path):
            shutil.rmtree(hf_model_path)

    # setup temporary directory
    with TemporaryDirectory() as tmpdir:
        logger.info(f"Downloading {model_id} weights to temp...")

        snapshot_dir = snapshot_download(
            repo_id=model_id, 
            cache_dir=tmpdir,
            allow_patterns=["*.bin", "*.json", "*.md", "tokenizer.model"],
        )
        # copy snapshot to model dir
        logger.info(f"Copying weights to {hf_model_path}...")
        copy_tree(snapshot_dir, str(hf_model_path))
    
    return {"model_id": model_id, "hf_model_path": hf_model_path}


def download_hf_model_and_copy_tokenizer(
        model_id: str,
        hf_model_path: str,
        tokenizer_path: str,
        rm_existing_model: bool = True,
):
    
    # if not hf_model_path:
    #     hf_model_path = os.path.join(os.getcwd(), "models", model_id, "hf")
    
    # if not tokenizer_path:
    #     # Write config to tensorized model weights directory
    #     tokenizer_path = os.path.dirname(hf_model_path)
    

    model_info = download_model_from_hf_hub(model_id, hf_model_path)

    # Move tokenizer to separate location
    logging.info(f"Copying tokenizer and model config to {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, padding_side="left")
    tokenizer.save_pretrained(tokenizer_path)

    # Set the source and destination file paths
    config_path = os.path.join(hf_model_path, "config.json")

    # Use the shutil.copy() function to copy the file to the destination directory
    shutil.copy(config_path, tokenizer_path)

    return model_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_id", type=str)
    # parser.add_argument("--hf_model_path", type=str)
    # parser.add_argument("--tokenizer_path", type=str)
    # parser.add_argument("--tensorize", action="store_true", default=False)
    # parser.add_argument("--dtype", type=str, default="fp32")

    parser.add_argument("--config", type=str)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        args = yaml.safe_load(f)
        args = Config(args)

    download_hf_model_and_copy_tokenizer(args.model_id, hf_model_path=args.hf_model_path, tokenizer_path=args.tokenizer_path)

    if args.tensorize:
        model = tensorize_model(args.model_id, dtype=args.dtype)