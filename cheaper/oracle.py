"""
Oracle interface and implementations for Battleship-style initial labeling.
Oracle: (tuple1, tuple2) -> 0 | 1 (no-match | match).
"""
import logging
import re
from typing import Callable, List, Tuple, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _pair_to_display_text(tuple1: List[Any], tuple2: List[Any]) -> str:
    """Format a pair for display or LLM prompt."""
    left = " | ".join(str(a) for a in tuple1)
    right = " | ".join(str(a) for a in tuple2)
    return f"Record A: {left}\nRecord B: {right}"


class SimilarityOracle:
    """
    Oracle that uses a similarity function and a threshold.
    sim_function(tuple1, tuple2) should return a list of scores (e.g. [0.85]);
    the first value is compared to threshold.
    """
    def __init__(self, sim_function: Callable, threshold: float = 0.5):
        self.sim_function = sim_function
        self.threshold = threshold

    def __call__(self, tuple1: List[Any], tuple2: List[Any]) -> int:
        scores = self.sim_function(tuple1, tuple2)
        score = float(scores[0]) if scores else 0.0
        return 1 if score >= self.threshold else 0


class SBERTOracle:
    """
    Oracle using sentence-transformers: embed both records, compute cosine similarity, threshold.
    Requires sentence_transformers. Falls back to SimilarityOracle with a simple sim if not available.
    """
    def __init__(self, threshold: float = 0.5, model_name: str = "all-MiniLM-L6-v2", fallback_sim=None):
        self.threshold = threshold
        self.model_name = model_name
        self._model = None
        self._fallback = fallback_sim

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning("sentence_transformers not installed; SBERTOracle requires it or a fallback_sim")
                raise
        return self._model

    def __call__(self, tuple1: List[Any], tuple2: List[Any]) -> int:
        try:
            model = self._get_model()
            text1 = " ".join(str(a) for a in tuple1)
            text2 = " ".join(str(a) for a in tuple2)
            emb = model.encode([text1, text2])
            sim = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-9))
            return 1 if float(sim) >= self.threshold else 0
        except Exception as e:
            if self._fallback is not None:
                return self._fallback(tuple1, tuple2)
            raise e


class LLMOracle:
    """
    Oracle that asks an LLM whether two records refer to the same entity.
    client: callable(messages: list) -> str, or None to use default OpenAI-style client.
    """
    def __init__(
        self,
        client: Optional[Callable] = None,
        prompt_template: Optional[str] = None,
        yes_pattern: Optional[str] = None,
    ):
        self.client = client
        self.prompt_template = prompt_template or (
            "Do the following two records refer to the same real-world entity? "
            "Answer only with Yes or No.\n\n{pair_text}"
        )
        self.yes_pattern = yes_pattern or r"\b(?:yes|true|1|match)\b"

    def __call__(self, tuple1: List[Any], tuple2: List[Any]) -> int:
        pair_text = _pair_to_display_text(tuple1, tuple2)
        prompt = self.prompt_template.format(pair_text=pair_text)
        if self.client is not None:
            response = self.client(prompt)
        else:
            response = self._default_client(prompt)
        return 1 if re.search(self.yes_pattern, response.strip().lower()) else 0

    def _default_client(self, prompt: str) -> str:
        """Default: try OpenAI API via openai package. Set OPENAI_API_KEY."""
        try:
            import openai
            client = openai.OpenAI()
            r = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("LLMOracle default client failed: %s", e)
            return "No"


def get_oracle(
    oracle_type: str,
    threshold: float = 0.5,
    sim_function: Optional[Callable] = None,
    sbert_model: str = "all-MiniLM-L6-v2",
    llm_client: Optional[Callable] = None,
) -> Callable[[List[Any], List[Any]], int]:
    """
    Factory: returns an oracle callable (tuple1, tuple2) -> 0|1.
    oracle_type: 'similarity' | 'sbert' | 'llm'
    """
    if oracle_type == "similarity":
        if sim_function is None:
            from cheaper.similarity import sim_function as sf
            sim_function = sf.jaro
        return SimilarityOracle(sim_function, threshold)
    if oracle_type == "sbert":
        fallback = SimilarityOracle(sim_function or (lambda t1, t2: [0.5]), threshold) if sim_function else None
        return SBERTOracle(threshold=threshold, model_name=sbert_model, fallback_sim=fallback)
    if oracle_type == "llm":
        return LLMOracle(client=llm_client)
    raise ValueError("oracle_type must be 'similarity', 'sbert', or 'llm'")
