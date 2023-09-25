# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, BaseModel

import torch
import os
import time
import json
import subprocess

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
from collections import OrderedDict 

from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor

from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn.functional as F

import yaml


from utils.memory_utils import MemoryTrace

from contextlib import ExitStack, contextmanager

@contextmanager
def dummy_context():
    yield None

MODEL_SETUP_CONFIG = "/src/model_setup.yaml"

class Embedding(BaseModel):
    embedding: List[float]

def embeddings_to_json(embeddings_list: List[Embedding]) -> str:
    """Serializes a list of Embedding objects to a JSON string."""
    return json.dumps([embedding.dict() for embedding in embeddings_list])

def write_embeddings_to_file(embeddings_list: List[Embedding], filename: str) -> None:
    """Writes the serialized list of Embedding objects to a file."""
    with open(filename, 'w') as f:
        json_data = embeddings_to_json(embeddings_list)
        f.write(json_data)

def maybe_download(path):
    if path.startswith("gs://"):
        output_path = "/tmp/weights.tensors"
        subprocess.check_call(["gcloud", "storage", "cp", path, output_path])
        return output_path
    return path

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            

            with open(MODEL_SETUP_CONFIG, 'r') as f:
                model_args = yaml.safe_load(f)
            
            config_path = os.path.join(model_args["tokenizer_path"], "config.json")
            self.tokenizer = self.load_tokenizer(path=model_args["tokenizer_path"])

            if weights is not None and weights.name == "weights":
                # bugfix
                weights = None
            
            if hasattr(weights, "filename") and "tensors" in weights.filename:
                self.model = self.load_tensorizer(
                    weights=weights, plaid_mode=True, cls=AutoModel, config_path=config_path,
                )

            elif hasattr(weights, "suffix") and "tensors" in weights.suffix:
                self.model = self.load_tensorizer(
                    weights=weights, plaid_mode=True, cls=AutoModel, config_path=config_path,
                )

            else:

                target_model = model_args["predict_model"]

                if target_model.endswith(".tensors"):
                    self.model = self.load_tensorizer(
                        weights=maybe_download(target_model), plaid_mode=True, cls=AutoModel, config_path=config_path,
                    )
        
                else:
                    self.model = self.load_huggingface_model(weights=target_model)
            
    def load_tokenizer(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path)
        return tokenizer

    def load_huggingface_model(self, weights=None):
        st = time.time()
        print(f"loading weights from {weights} w/o tensorizer")
        # model = SentenceTransformer(weights)
        model = AutoModel.from_pretrained(weights)
        model.to(self.device)
        print(f"weights loaded in {time.time() - st}")
        return model
    
    def load_tensorizer(self, weights, plaid_mode, cls, config_path):
        # raise NotImplementedError("Loading tensorizer not implemented yet.")
    
        st = time.time()
        print(f"deserializing weights from {weights}")
        config = AutoConfig.from_pretrained(config_path)

        model = no_init_or_tensor(
            lambda: cls.from_pretrained(
                None, config=config, state_dict=OrderedDict()
            )
        )

        des = TensorDeserializer(weights, plaid_mode=True)
        des.load_into_module(model)
        print(f"weights loaded in {time.time() - st}")
        return model
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def predict(
        self,
        text: str = Input(default=None, description="A single string to encode."),
        text_batch: str = Input(default=None, description="A JSON-formatted list of strings to encode."),
        text_file: Path = Input(default=None, description="A file containing a JSON array of texts to encode."),
        profile_memory: bool = Input(default=False, description="Whether to profile memory usage. If 'True', memory usage will be printed to logs."),

    ) -> Path: # List[Embedding]:
        """
        Encode a single `text` or a `text_batch` into embeddings.

        Parameters:
        ----------
        text : str, optional
            A single string to encode.
        text_batch : str, optional
            A JSON-formatted list of strings to encode.
        text_file : Path, optional
            A file containing a JSON array of texts to encode.

        Returns:
        -------
        List[List[float]]
            A list of embeddings ordered the same as the inputs.
        """

        predict_context = dummy_context() if not profile_memory else MemoryTrace() 
        self._validate_inputs(text, text_batch, text_file)

        if text:
            docs = [text]

        elif text_batch:
            docs = json.loads(text_batch)
        
        elif text_file:
            with open(text_file, 'r') as f:
                docs = json.load(f)
        
        with predict_context as context:

            # Tokenize sentences
            encoded_input = self.tokenizer(docs, padding=True, truncation=True, return_tensors='pt').to(self.device)
                
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform pooling
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # embeddings = self.model.encode(docs)

            outputs = []
            for embedding in embeddings:
                outputs.append(Embedding(embedding=[float(x) for x in embedding.tolist()]))
            
        if profile_memory:
            self._print_memory_profile(context)
        
        write_embeddings_to_file(outputs, "./predict_output/embeddings.json")
        return Path("./predict_output/embeddings.json")
        
        # return outputs

    def _validate_inputs(self, text, text_batch, text_file):
        if text and text_batch:
            raise ValueError("Only one of 'text' or 'text_batch' can be specified.")

        if text and text_file:
            raise ValueError("Only one of 'text' or 'file' can be specified.")

        if text_batch and text_file:
            raise ValueError("Only one of 'text_batch' or 'file' can be specified.")

        if not text and not text_batch and not text_file:
            raise ValueError("One of 'text', 'text_batch', or 'file' must be specified.")

    
    def _print_memory_profile(self, memtrace):
        print("Memory Profile:","-"*20)
        print(f"Max CUDA memory allocated: {memtrace.peak} GB")
        print(f"Max CUDA memory reserved: {memtrace.max_reserved} GB")
        print(f"CPU Total Peak Memory consumed during: {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        print("-"*40)