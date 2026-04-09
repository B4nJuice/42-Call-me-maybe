import json
from typing import Any
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
                result += string.replace("ĠĊ", "\n").replace("Ġ", " ")

        return result

    def get_token(self, string: str) -> int | None:
        normalized: str = string.replace("\n", "ĠĊ").replace(" ", "Ġ")
        if normalized not in self.token_cache:
            self.token_cache[normalized] = self.vocab.get(normalized)
        return self.token_cache[normalized]

    def get_string(self, token: int) -> str | None:
        if token not in self.string_cache:
            self.string_cache[token] = self.token_vocab.get(token)
        return self.string_cache[token]
