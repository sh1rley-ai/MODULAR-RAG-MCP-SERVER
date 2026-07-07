"""LLMReranker — reranks candidates by calling an LLM with a scoring prompt (B7.7).

The reranker reads a prompt template from `config/prompts/rerank.txt` (or an
injected path/text), sends the query + candidates to the LLM, parses the JSON
response, and returns the candidates reordered by descending relevance score.

Output schema (one object per candidate):
    [{"passage_id": "<id>", "score": 0-3, "reasoning": "<text>"}, ...]

Any schema violation (missing passage_id, non-numeric score, unknown id,
missing candidates) raises `RerankerError` with a readable message rather than
silently degrading — the Core layer (D6) is responsible for fallback.

Constructor parameters:
    rerank_settings  RerankSettings from global config (model / top_k).
    llm              Optional BaseLLM instance. Must be provided before
                     `rerank()` is called; typically injected in tests and
                     wired by the Core layer (D6) in production.
    prompt_path      Optional path to an alternative prompt file. When None,
                     defaults to config/prompts/rerank.txt relative to the
                     project root.
    prompt_text      Optional prompt text that overrides prompt_path. Used
                     to inject a compact prompt in unit tests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from libs.llm.base_llm import BaseLLM
from libs.reranker.base_reranker import BaseReranker, RerankerError
from libs.reranker.reranker_factory import RerankerFactory

# config/prompts/rerank.txt is at <project_root>/config/prompts/rerank.txt
# __file__ lives at <project_root>/src/libs/reranker/llm_reranker.py
_DEFAULT_PROMPT_PATH: Path = (
    Path(__file__).parents[3] / "config" / "prompts" / "rerank.txt"
)

_RESPONSE_INSTRUCTION = (
    "\nReturn ONLY a JSON array with one object per passage:\n"
    '[{"passage_id": "<id>", "score": <0-3>, "reasoning": "<text>"}]'
)


class LLMReranker(BaseReranker):
    """Reranker that uses an LLM to score and reorder candidates."""

    provider_name = "llm"

    def __init__(
        self,
        rerank_settings: Any,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ) -> None:
        self.settings = rerank_settings
        self._llm = llm

        if prompt_text is not None:
            self._prompt_template = prompt_text
        else:
            path = Path(prompt_path) if prompt_path is not None else _DEFAULT_PROMPT_PATH
            try:
                self._prompt_template = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise RerankerError(
                    f"[{self.provider_name}] failed to load rerank prompt from {path}: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Any | None = None,
    ) -> List[Dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)

        if not candidates:
            return []

        if self._llm is None:
            raise RerankerError(
                f"[{self.provider_name}] reranker requires an injected BaseLLM instance"
            )

        user_msg = self._build_user_message(query, candidates)
        try:
            response = self._llm.chat(
                [
                    {"role": "system", "content": self._prompt_template},
                    {"role": "user", "content": user_msg},
                ]
            )
        except Exception as exc:
            raise RerankerError(
                f"[{self.provider_name}] LLM call failed: {exc}"
            ) from exc

        return self._parse_and_reorder(response.content, candidates)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        lines: List[str] = [f"Query: {query}\n\nPassages:"]
        for i, candidate in enumerate(candidates, start=1):
            text = (candidate.get("text") or "").strip()
            lines.append(f'[{i}] id: "{candidate["id"]}"\ntext: "{text}"')
        lines.append(_RESPONSE_INSTRUCTION)
        return "\n".join(lines)

    def _parse_and_reorder(
        self, content: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        candidate_ids = {c["id"] for c in candidates}
        candidate_map: Dict[str, Dict[str, Any]] = {c["id"]: c for c in candidates}

        # Extract the first JSON array from the response (LLMs sometimes add prose)
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if not match:
            raise RerankerError(
                f"[{self.provider_name}] response contains no JSON array.\n"
                f"Response: {content!r}"
            )
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise RerankerError(
                f"[{self.provider_name}] response is not valid JSON: {exc}\n"
                f"Response: {content!r}"
            ) from exc

        if not isinstance(parsed, list):
            raise RerankerError(
                f"[{self.provider_name}] expected a JSON array, got {type(parsed).__name__}"
            )

        scored: List[tuple[str, float]] = []
        seen_ids: set[str] = set()

        for item in parsed:
            if not isinstance(item, dict):
                raise RerankerError(
                    f"[{self.provider_name}] each item must be a dict, "
                    f"got {type(item).__name__}"
                )
            passage_id = item.get("passage_id")
            if not isinstance(passage_id, str) or not passage_id:
                raise RerankerError(
                    f"[{self.provider_name}] each item must have a non-empty string "
                    f"'passage_id', got {passage_id!r}"
                )
            if passage_id not in candidate_ids:
                raise RerankerError(
                    f"[{self.provider_name}] passage_id {passage_id!r} not in candidates"
                )
            score = item.get("score")
            if not isinstance(score, (int, float)):
                raise RerankerError(
                    f"[{self.provider_name}] score for '{passage_id}' must be numeric, "
                    f"got {type(score).__name__}"
                )
            seen_ids.add(passage_id)
            scored.append((passage_id, float(score)))

        missing = candidate_ids - seen_ids
        if missing:
            raise RerankerError(
                f"[{self.provider_name}] LLM response is missing passage_ids: "
                f"{sorted(missing)}"
            )

        scored.sort(key=lambda x: x[1], reverse=True)
        return [candidate_map[pid] for pid, _ in scored]


RerankerFactory.register("llm", LLMReranker)
