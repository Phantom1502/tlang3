from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import LlamaForCausalLM

from app.tokenizer.hub import load_tokenizer

class ModelInference:
    def __init__(
        self,
        model_repo: str,
        revision: Optional[str] = None,
        subfolder: Optional[str] = None,
        tokenizer_repo: Optional[str] = None,
        max_new_tokens: int = 24,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- Tokenizer: pin cùng revision với model nếu có, giữ quirk
        # add_eos_token=False/add_bos_token=True/padding_side=left (batch generate). ---
        self.tok = load_tokenizer(repo_id=tokenizer_repo or model_repo, revision=revision, allow_local_fallback=False)
        self.tok.add_eos_token = False
        self.tok.add_bos_token = True
        self.tok.padding_side = "left"
        
        model_kwargs: Dict[str, Any] = {}
        if revision is not None:
            model_kwargs["revision"] = revision
        if subfolder is not None:
            model_kwargs["subfolder"] = subfolder
        self.model = LlamaForCausalLM.from_pretrained(model_repo, **model_kwargs).to(self.device)
        self.model.eval()
        if self.model.config.vocab_size != self.tok.vocab_size:
            raise ValueError(
                f"model.vocab_size ({self.model.config.vocab_size}) != tokenizer.vocab_size "
                f"({self.tok.vocab_size}) — checkpoint và tokenizer không khớp "
                f"(model_repo={model_repo!r}, revision={revision!r}, subfolder={subfolder!r})."
            )
            
    def generate_batch(self, rows: Sequence[Dict[str, Any]]) -> List[str]:
        prompts = [r["prompt"] for r in rows]
        enc = self.tok(prompts, add_special_tokens=True, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        if self.do_sample:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

        gen_ids = out_ids[:, input_ids.shape[1]:]
        return self.tok.batch_decode(gen_ids, skip_special_tokens=True)