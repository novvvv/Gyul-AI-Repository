"""Kanana(GQA) 모델 — transformers 5.x LlamaConfig 검증 우회.

hidden_size(1792)가 num_attention_heads(24)로 나누어떨어지지 않아
head_dim이 명시된 GQA 구조에서 오탐 검증이 발생한다.
"""


def patch_kanana_llama_config() -> None:
    from transformers.models.llama.configuration_llama import LlamaConfig

    def validate_architecture_gqa(self) -> None:
        if getattr(self, "head_dim", None):
            return
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the "
                f"number of attention heads ({self.num_attention_heads})."
            )

    LlamaConfig.validate_architecture = validate_architecture_gqa
    LlamaConfig.__class_validators__ = [validate_architecture_gqa]
