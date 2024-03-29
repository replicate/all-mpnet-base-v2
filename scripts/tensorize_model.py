#!/usr/bin/env python
import torch
import os
import argparse
import logging 
import sys

from tensorizer import TensorSerializer
from transformers import AutoModelForCausalLM, AutoConfig


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def tensorize_model(
        model_id: str,
        model_path: str,
        tensorizer_path: str,
        dtype: str = "fp32",
) -> dict:
    """
    Create a tensorized version of model weights. If fp16 or bf16 is True, 
    the model will be converted to fp16 or bf16. 

    If `model_path` is None weights will be saved in `./model_weights/torch_weights/model_id`.
    If `tensorizer_path` is None weights will be saved in `./model_weights/tensorizer_weights/model_id/dtype_str`.

    Args:
        model_id (str): Name of model on hugging face hub
        model_path (str, optional): Local path where model weights are saved.
        tensorizer_path (str, optional): Local path where tensorizer weights are saved.
        path (str): Local path where tensorized model weights are saved
        dtype (str): One of `"fp32"`, `"fp16"`, and `"bf16"`. Defaults to `"fp32"`.
    
    Returns:
        dict: Dictionary containing the tensorized model path and dtype.
    """


    if dtype == 'fp32' or dtype is None:
        torch_dtype = torch.float32
    
    elif dtype == 'bf16':
        torch_dtype = torch.bfloat16
    
    elif dtype == 'fp16':
        torch_dtype = torch.float16

    logger.info(f"Loading {model_id} in {dtype} from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to('cuda:0')

    logger.info(f"Tensorizing model {model_id} in {dtype} and writing tensors to {tensorizer_path}...")

    serializer = TensorSerializer(tensorizer_path)
    serializer.write_module(model)
    serializer.close()

    # Write config to tensorized model weights directory
    dir_path = os.path.dirname(tensorizer_path)
    config_path = os.path.join(dir_path, 'config.json')
    model_config.save_pretrained(config_path)

    logger.info(f"Tensorized model {model_id} in {dtype} and wrote tensors to {tensorizer_path} and config to {config_path}...")

    return {"tensorized_weights_path": tensorizer_path, "dtype": dtype}

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description=(
        "A simple script for tensorizing a torch model."
        )
    )
    
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tensorizer_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="fp32")

    args = parser.parse_args()

    model_info = tensorize_model(
        args.model_id,
        model_path=args.model_path,
        tensorizer_path=args.tensorizer_path,
        dtype=args.dtype
    )