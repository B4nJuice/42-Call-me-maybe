import json
from typing import Any
import numpy as np
from pydantic import BaseModel, PrivateAttr


class Tokenizer(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model: Any
    _vocab: dict[str, int] = PrivateAttr()
    _token_vocab: dict[int, str] = PrivateAttr()
    _token_cache: dict[str, int | None] = PrivateAttr()
    _string_cache: dict[int, str | None] = PrivateAttr()

    def model_post_init(self, _: Any):
        with open(self.model.get_path_to_vocab_file(), 'r') as f:
            self._vocab = json.load(f)
        self._token_vocab = {v: k for k, v in self._vocab.items()}
        self._token_cache = {}
        self._string_cache = {}

    @property
    def token_cache(self) -> dict[str, int | None]:
        return self._token_cache

    @property
    def string_cache(self) -> dict[int, str | None]:
        return self._string_cache

    @property
    def vocab(self) -> dict[str, int]:
        return self._vocab

    @property
    def token_vocab(self) -> dict[int, str]:
        return self._token_vocab

    def encode(self, string: str) -> list[int]:
        str_len: int = len(string)
        i: int = 0
        start: int = 0
        result: list[int] = []

        while i < str_len:
            token_slice: str = string[start:str_len-i]
            if (sub_string := self.get_token(token_slice)) is not None:
                start = str_len - i
                result.append(sub_string)
                i = -1
            i += 1

        return result

    def decode(self, token_list: list[int] | int) -> str:
        result: str = ""

        if isinstance(token_list, int):
            token_list = [token_list]

        for token in token_list:
            string: str = self.get_string(token)
            if string:
                result += string

        return result

    def get_constrained_token(
                self,
                logits: list[float],
                mask: str
            ) -> int:
        if not mask:
            return int(np.argmax(np.asarray(logits)))

        allowed_chars: set[str] = set(mask)
        sorted_ids: np.ndarray[Any, Any] = np.argsort(np.asarray(logits))[::-1]

        for token_id in sorted_ids:
            token: int = int(token_id)
            decoded: str | None = self.get_string(token)
            if not decoded:
                continue
            if all(char in allowed_chars for char in decoded):
                return token

        return int(np.argmax(np.asarray(logits)))

    def get_next_token_from_possible_outputs(
                self,
                logits: list[float],
                current_tokens: list[int],
                possible_outputs_tokens: list[list[int]]
            ) -> int:
        possible_next_tokens: set[int] = set()
        current_len: int = len(current_tokens)

        matching_outputs: list[list[int]] = []
        for output_tokens in possible_outputs_tokens:
            if output_tokens[:current_len] != current_tokens:
                continue
            matching_outputs.append(output_tokens)

        if not matching_outputs:
            raise ValueError(
                "No possible output starts with current_tokens."
            )

        for output_tokens in matching_outputs:
            if len(output_tokens) <= current_len:
                continue
            possible_next_tokens.add(output_tokens[current_len])

        if not possible_next_tokens:
            raise ValueError(
                "No next token available for current_tokens prefix."
            )

        return max(possible_next_tokens, key=lambda token_id: logits[token_id])

    def get_token(self, string: str) -> int | None:
        normalized: str = string.replace("\n", "ĠĊ").replace(" ", "Ġ")
        if normalized not in self.token_cache:
            self.token_cache[normalized] = self.vocab.get(normalized)
        return self.token_cache[normalized]

    def get_string(self, token: int) -> str | None:
        if token not in self.string_cache:
            self.string_cache[token] = self.token_vocab.get(token).replace(
                "ĠĊ", "\n").replace("Ġ", " ").replace("Ċ", "\"")
        return self.string_cache[token]
