"""Patch Fish Speech's tokenizer for OpenAudio S1-mini fine-tuning.

The current Fish Speech main branch points text2semantic_finetune at
``openaudio-s1-mini/tokenizer.tiktoken`` while its FishTokenizer wrapper tries
to load the path with Transformers AutoTokenizer. OpenAudio S1-mini ships a
tiktoken BPE file plus special_tokens.json, so this patch restores a tokenizer
implementation that reads those files directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TOKENIZER_SOURCE = r'''from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import List, Union

import tiktoken
import torch

logger = logging.getLogger(__name__)

FISH_TIKTOKEN_PATTERN = "|".join(
    [
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
        r"\p{P}",
        r"[^\r\n\p{L}\p{N}]?\p{L}+",
        r"\p{N}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(\?!\S)",
        r"\s+",
    ]
)
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"
LEGACY_EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
PHONEME_START_TOKEN = "<|phoneme_start|>"
PHONEME_END_TOKEN = "<|phoneme_end|>"

MODALITY_TEXT_TOKEN = "<|text|>"
MODALITY_VOICE_TOKEN = "<|voice|>"
MODALITY_INTERLEAVE_TOKEN = "<|interleave|>"
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"
AUDIO_EMBED_TOKEN = "<|audio_pad|>"

MODALITY_TOKENS = {
    "text": MODALITY_TEXT_TOKEN,
    "voice": MODALITY_VOICE_TOKEN,
    "interleave": MODALITY_INTERLEAVE_TOKEN,
}

SEMANTIC_TOKEN_TEMPLATE = "<|semantic:{i}|>"
SEMANTIC_TOKENS = [SEMANTIC_TOKEN_TEMPLATE.format(i=i) for i in range(4096)]

ALL_SPECIAL_TOKENS = [
    BOS_TOKEN,
    EOS_TOKEN,
    LEGACY_EOS_TOKEN,
    PAD_TOKEN,
    IM_START_TOKEN,
    IM_END_TOKEN,
    PHONEME_START_TOKEN,
    PHONEME_END_TOKEN,
    MODALITY_TEXT_TOKEN,
    MODALITY_VOICE_TOKEN,
    MODALITY_INTERLEAVE_TOKEN,
    AUDIO_START_TOKEN,
    AUDIO_END_TOKEN,
    AUDIO_EMBED_TOKEN,
    *SEMANTIC_TOKENS,
]


class FishTokenizer:
    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        if model_path.is_dir():
            token_path = model_path / "tokenizer.tiktoken"
            special_tokens_path = model_path / "special_tokens.json"
        else:
            token_path = model_path
            special_tokens_path = model_path.parent / "special_tokens.json"

        if not token_path.exists():
            raise FileNotFoundError(f"Tokenizer BPE file not found: {token_path}")

        mergeable_ranks = self.load_tiktoken_bpe(token_path)
        self.all_special_tokens_with_ids = self._load_special_tokens(
            special_tokens_path, len(mergeable_ranks)
        )

        self.semantic_id_to_token_id: dict[int, int] = {}
        for token, token_id in self.all_special_tokens_with_ids.items():
            match = re.fullmatch(r"<\|semantic:(\d+)\|>", token)
            if match:
                self.semantic_id_to_token_id[int(match.group(1))] = int(token_id)

        if not self.semantic_id_to_token_id:
            raise ValueError(
                f"No semantic tokens found in {special_tokens_path}; cannot train S1-mini"
            )

        max_semantic_id = max(self.semantic_id_to_token_id)
        self.semantic_begin_id = self.semantic_id_to_token_id[0]
        self.semantic_end_id = self.semantic_id_to_token_id[max_semantic_id]
        self.semantic_map_tensor = torch.zeros(max_semantic_id + 1, dtype=torch.long)
        for semantic_id, token_id in self.semantic_id_to_token_id.items():
            self.semantic_map_tensor[semantic_id] = token_id

        self.tkt_model = tiktoken.core.Encoding(
            name=token_path.stem,
            pat_str=FISH_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.all_special_tokens_with_ids,
        )

        max_rank = max(mergeable_ranks.values(), default=-1)
        max_special = max(self.all_special_tokens_with_ids.values(), default=-1)
        self._vocab_size = max(max_rank, max_special) + 1

        logger.info(
            "Loaded tiktoken FishTokenizer. Semantic range: %s -> %s",
            self.semantic_begin_id,
            self.semantic_end_id,
        )

    @staticmethod
    def _load_special_tokens(path: Path, special_token_begin: int) -> dict[str, int]:
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)

            if isinstance(loaded, dict):
                return {str(token): int(token_id) for token, token_id in loaded.items()}

            if isinstance(loaded, list):
                return {
                    str(token): special_token_begin + idx
                    for idx, token in enumerate(loaded)
                }

            raise ValueError(f"Unsupported special token format in {path}")

        return {
            token: special_token_begin + idx
            for idx, token in enumerate(ALL_SPECIAL_TOKENS)
        }

    @staticmethod
    def load_tiktoken_bpe(tiktoken_bpe_file: str | Path) -> dict[bytes, int]:
        data: dict[bytes, int] = {}
        with Path(tiktoken_bpe_file).open("r", encoding="utf-8") as file:
            for line in file.read().splitlines():
                if not line:
                    continue
                token, rank = line.split()
                if token == "=":
                    token = ""
                data[base64.b64decode(token)] = int(rank)
        return data

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def num_special_tokens(self) -> int:
        return len(self.all_special_tokens_with_ids)

    @property
    def pad_token_id(self) -> int:
        return self.get_token_id(PAD_TOKEN)

    @property
    def eos_token_id(self) -> int:
        return self.get_token_id(EOS_TOKEN)

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self.all_special_tokens_with_ids)

    def get_token_id(self, token: str) -> int:
        if token in self.all_special_tokens_with_ids:
            return self.all_special_tokens_with_ids[token]

        aliases = {
            EOS_TOKEN: LEGACY_EOS_TOKEN,
            LEGACY_EOS_TOKEN: EOS_TOKEN,
        }
        alias = aliases.get(token)
        if alias and alias in self.all_special_tokens_with_ids:
            return self.all_special_tokens_with_ids[alias]

        encoded = self.encode(token, allowed_special=False)
        if len(encoded) != 1:
            raise KeyError(f"Token is not a single token or special token: {token}")
        return encoded[0]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.get_token_id(token)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        allowed_special: Union[bool, set[str], str] = True,
        **_: object,
    ) -> List[int]:
        del add_special_tokens

        if allowed_special is True:
            allowed = self.tkt_model.special_tokens_set
        elif allowed_special is False:
            allowed = set()
        else:
            allowed = allowed_special

        chunks = [
            text[i : i + TIKTOKEN_MAX_ENCODE_CHARS]
            for i in range(0, len(text), TIKTOKEN_MAX_ENCODE_CHARS)
        ]
        return sum(
            self.tkt_model.encode_batch(
                chunks,
                allowed_special=allowed,
                disallowed_special=set(),
            ),
            start=[],
        )

    def decode(self, tokens: Union[List[int], int], **_: object) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.tkt_model.decode(tokens)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with (path / "tokenizer.tiktoken").open("w", encoding="utf-8") as file:
            for token, rank in self.tkt_model._mergeable_ranks.items():
                encoded = base64.b64encode(token).decode()
                if encoded == "":
                    encoded = "="
                file.write(f"{encoded} {rank}\n")

        with (path / "special_tokens.json").open("w", encoding="utf-8") as file:
            json.dump(
                self.all_special_tokens_with_ids,
                file,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "FishTokenizer":
        return cls(path)
'''


def patch_tokenizer(fish_speech_dir: Path) -> Path:
    tokenizer_path = fish_speech_dir / "fish_speech" / "tokenizer.py"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Cannot find Fish Speech tokenizer: {tokenizer_path}")

    backup_path = tokenizer_path.with_suffix(".py.autotokenizer.bak")
    if not backup_path.exists():
        backup_path.write_text(tokenizer_path.read_text(encoding="utf-8"), encoding="utf-8")

    tokenizer_path.write_text(TOKENIZER_SOURCE, encoding="utf-8")
    return tokenizer_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fish-speech-dir",
        default="/content/fish-speech",
        type=Path,
        help="Path to the cloned fish-speech repository.",
    )
    args = parser.parse_args()

    tokenizer_path = patch_tokenizer(args.fish_speech_dir)
    print(f"Patched FishTokenizer: {tokenizer_path}")


if __name__ == "__main__":
    main()
