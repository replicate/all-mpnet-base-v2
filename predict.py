# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, BaseModel

import torch
import os
import time
import json

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional

import yaml

MODEL_SETUP_CONFIG = "/src/model_setup.yaml"


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

                target_model = model_args["target_model"]

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
        model = SentenceTransformer(weights)
        model.to(self.device)
        print(f"weights loaded in {time.time() - st}")
        return model
    
    def load_tensorizer(self, weights, plaid_mode, cls, config_path):
        raise NotImplementedError("Loading tensorizer not implemented yet.")
    
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

    def predict(
        self,
        text: str = Input(default=None, description="A single string to encode."),
        text_batch: str = Input(default=None, description="A JSON-formatted list of strings to encode."),

    ) -> List[List[float]]:
        """
        Encode a single `text` or a `text_batch` into embeddings.

        Parameters:
        ----------
        text : str, optional
            A single string to encode.
        text_batch : str, optional
            A JSON-formatted list of strings to encode.

        Returns:
        -------
        List[List[float]]
            A list of embeddings ordered the same as the inputs.
        """

        # TODO: 
        # * Push model
        # * Update README


        if text:
            docs = [text]

        elif text_batch:
            docs = json.loads(text_batch)
            print(docs)

        embeddings = self.model.encode(docs)

        outputs = []
        for embedding in embeddings:
            outputs.append([float(x) for x in embedding.tolist()])

        return outputs



