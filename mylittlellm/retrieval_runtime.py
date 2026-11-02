"""Load the same retrieval weights with less transient CPU memory for the CLI.

The frozen evaluation constructor remains unchanged. In-place dynamic INT8
conversion avoids deep-copying a full FP32 model alongside the live generator.
Encoding and reranking are inherited without changes.
"""
import ctypes
import gc

from .hierarchical import QwenRetrievalModels, load_config
from .retrieval import ROOT


def release_unused_cpu_memory():
    gc.collect()
    trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
    if trim is not None:
        trim(0)


class LowMemoryQwenRetrievalModels(QwenRetrievalModels):
    def __init__(self, config=None, reranker=True):
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        self.config = config or load_config()
        self.torch = torch
        torch.set_num_threads(self.config['threads'])

        def load(spec, model_class):
            tokenizer = AutoTokenizer.from_pretrained(ROOT/spec['path'],local_files_only=True,padding_side='left')
            model = model_class.from_pretrained(ROOT/spec['path'],local_files_only=True,
                                                torch_dtype=torch.float32,trust_remote_code=False).eval()
            if self.config.get('runtime_quantization')=='torch_dynamic_int8_linear':
                model = torch.ao.quantization.quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8,inplace=True)
            release_unused_cpu_memory()
            return tokenizer,model

        self.tokenizer,self.encoder = load(self.config['embedding'],AutoModel)
        self.reranker = None
        if reranker:
            self.rerank_tokenizer,self.reranker = load(self.config['reranker'],AutoModelForCausalLM)
            self.yes_id = self.rerank_tokenizer.convert_tokens_to_ids('yes')
            self.no_id = self.rerank_tokenizer.convert_tokens_to_ids('no')
