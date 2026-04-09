import json
from typing import Any
import numpy as np
from pydantic import BaseModel, PrivateAttr


class Tokenizer(BaseModel):
    """Provide lightweight token/string conversion and constrained selection.

    The tokenizer loads a vocabulary JSON exposed by the backend model and
    offers helper methods used during constrained decoding.
    """

    model_config = {"arbitrary_types_allowed": True}

    model: Any
    _vocab: dict[str, int] = PrivateAttr()
    _token_vocab: dict[int, str] = PrivateAttr()
    _token_cache: dict[str, int | None] = PrivateAttr()
    _string_cache: dict[int, str | None] = PrivateAttr()

    def model_post_init(self, _: Any) -> None:
        """Initialize vocabularies and internal caches after construction.

        Parameters
        ----------
        _ : Any
            Pydantic post-init context.

        Returns
        -------
        None
            Populates vocab mappings and cache dictionaries.
        """
        with open(self.model.get_path_to_vocab_file(), 'r') as f:
            self._vocab = json.load(f)
        self._token_vocab = {v: k for k, v in self._vocab.items()}
        self._token_cache = {}
        self._string_cache = {}

    @property
    def token_cache(self) -> dict[str, int | None]:
        """Return memoized string-to-token lookups.

        Returns
        -------
        dict[str, int | None]
            Cache keyed by normalized string values.
        """
        return self._token_cache

    @property
    def string_cache(self) -> dict[int, str | None]:
        """Return memoized token-to-string lookups.

        Returns
        -------
        dict[int, str | None]
            Cache keyed by token identifier.
        """
        return self._string_cache

    @property
    def vocab(self) -> dict[str, int]:
        """Return raw normalized string-to-token vocabulary.

        Returns
        -------
        dict[str, int]
            Vocabulary loaded from model files.
        """
        return self._vocab

    @property
    def token_vocab(self) -> dict[int, str]:
        """Return reverse token-to-string vocabulary.

        Returns
        -------
        dict[int, str]
            Reverse mapping built from ``vocab``.
        """
        return self._token_vocab

    def encode(self, string: str) -> list[int]:
        """Encode text using greedy longest-substring vocabulary matching.

        Parameters
        ----------
        string : str
            Input text to encode.

        Returns
        -------
        list[int]
            Sequence of token identifiers found in the vocabulary.
        """
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
        """Decode a token or token list into plain text.

        Parameters
        ----------
        token_list : list[int] | int
            Token identifier(s) to decode.

        Returns
        -------
        str
            Decoded text with normalized control markers restored.
        """
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
        """Select the best token whose decoded chars are allowed by a mask.

        Parameters
        ----------
        logits : list[float]
            Next-token logits over the full vocabulary.
        mask : str
            Set of allowed output characters.

        Returns
        -------
        int
            Selected token identifier.
        """
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
        """Pick the max-logit next token constrained by valid output prefixes.

        Parameters
        ----------
        logits : list[float]
            Next-token logits over the full vocabulary.
        current_tokens : list[int]
            Tokens already generated for the current field.
        possible_outputs_tokens : list[list[int]]
            Candidate full outputs represented as token sequences.

        Returns
        -------
        int
            Highest-logit next token among candidates sharing the prefix.

        Raises
        ------
        ValueError
            If no candidate output starts with ``current_tokens`` or if no
            continuation token exists for the matching candidates.
        """
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
        """Return token ID for a string after internal normalization.

        Parameters
        ----------
        string : str
            Raw string fragment to map into a token identifier.

        Returns
        -------
        int | None
            Matching token identifier, if available.
        """
        normalized: str = string.replace("\n", "ĠĊ").replace(" ", "Ġ")
        if normalized not in self.token_cache:
            self.token_cache[normalized] = self.vocab.get(normalized)
        return self.token_cache[normalized]

    def get_string(self, token: int) -> str | None:
        """Return decoded string for a token with cache reuse.

        Parameters
        ----------
        token : int
            Token identifier to decode.

        Returns
        -------
        str | None
            Decoded string fragment, if available in the vocabulary.
        """
        if token not in self.string_cache:
            self.string_cache[token] = self.token_vocab.get(token).replace(
                "ĠĊ", "\n").replace("Ġ", " ").replace("Ċ", "\"")
        return self.string_cache[token]
