from pathlib import Path
from typing import Any

from app.core.config import (
    LOCAL_LLM_MAX_NEW_TOKENS,
    LOCAL_LLM_MODEL_ID,
    LOCAL_LLM_TEMPERATURE,
)


class LocalLLMService:
    """Hugging Face 로컬 LLM (기본: EXAONE Instruct) 응답 생성."""

    def __init__(
        self,
        model_id: str = LOCAL_LLM_MODEL_ID,
        max_new_tokens: int = LOCAL_LLM_MAX_NEW_TOKENS,
        temperature: float = LOCAL_LLM_TEMPERATURE,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer: Any = None
        self.model: Any = None
        self._device: str = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self.tokenizer is not None and self.model is not None

    def load(self) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Config 로드 시 remote modeling 파일이 내려받아짐 → transformers 5.9 호환 패치
        AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        _patch_exaone_remote_modeling(self.model_id)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": dtype,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        else:
            self._device = self._resolve_device()
            load_kwargs["dtype"] = (
                torch.float16 if self._device == "mps" else torch.float32
            )

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        if not torch.cuda.is_available():
            self.model = self.model.to(self._device)
        self.model.eval()

    def generate(self, messages: list[dict[str, str]]) -> str:
        import torch

        if not self.is_loaded:
            raise RuntimeError("Local LLM not loaded")

        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("Tokenizer does not support chat template")

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _resolve_device(self) -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def _patch_exaone_remote_modeling(model_id: str) -> None:
    """HF remote modeling_exaone.py 와 transformers 5.9 API 불일치 보정."""
    if "exaone" not in model_id.lower():
        return

    cache_root = Path.home() / ".cache/huggingface/modules/transformers_modules"
    if not cache_root.exists():
        return

    for modeling_file in cache_root.rglob("modeling_exaone.py"):
        text = modeling_file.read_text(encoding="utf-8")
        patched = text.replace("input_embeds=inputs_embeds", "inputs_embeds=inputs_embeds")
        patched = patched.replace("            cache_position=cache_position,\n", "")
        if patched != text:
            modeling_file.write_text(patched, encoding="utf-8")


local_llm_service = LocalLLMService()
