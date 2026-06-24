"""
Oracle interface and implementations for Battleship-style initial labeling.
Oracle: (tuple1, tuple2) -> 0 | 1 (no-match | match).
"""
import logging
import re
from typing import Callable, List, Tuple, Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def _pair_to_display_text(tuple1: List[Any], tuple2: List[Any]) -> str:
    """Format a pair for display or LLM prompt."""
    left = " | ".join(str(a) for a in tuple1)
    right = " | ".join(str(a) for a in tuple2)
    return f"Record A: {left}\nRecord B: {right}"


def adaptive_threshold(scores: List[float], method: str = "median") -> float:
    """
    Compute a data-driven threshold from a list of similarity scores.
    method: 'median' (balanced split), 'otsu' (maximize between-class variance),
    or 'p75' (75th percentile, conservative).
    """
    scores_arr = np.asarray(scores, dtype=float)
    if len(scores_arr) == 0:
        return 0.5
    if len(scores_arr) == 1:
        return float(scores_arr[0])
    if method == "median":
        return float(np.median(scores_arr))
    if method == "p75":
        return float(np.percentile(scores_arr, 75))
    if method == "otsu":
        return _otsu_threshold_1d(scores_arr)
    return float(np.median(scores_arr))


def _otsu_threshold_1d(scores: np.ndarray) -> float:
    """1D Otsu: threshold that maximizes between-class variance."""
    lo, hi = float(scores.min()), float(scores.max())
    if lo >= hi:
        return (lo + hi) / 2.0
    n_bins = min(256, max(2, len(scores) // 5))
    hist, bin_edges = np.histogram(scores, bins=n_bins, range=(lo, hi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 0.5
    total_mean = (hist * bin_centers).sum() / total
    best_var = 0.0
    best_t = bin_centers[0]
    w_below = 0.0
    sum_below = 0.0
    for i in range(len(hist)):
        w_below += hist[i]
        if w_below == 0:
            continue
        w_above = total - w_below
        if w_above == 0:
            break
        sum_below += hist[i] * bin_centers[i]
        mean_below = sum_below / w_below
        sum_above = (hist * bin_centers).sum() - sum_below
        mean_above = sum_above / w_above
        var_between = w_below * w_above * (mean_below - mean_above) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = bin_centers[i]
    return float(best_t)


class SimilarityOracle:
    """
    Oracle that uses a similarity function and a threshold.
    sim_function(tuple1, tuple2) should return a list of scores (e.g. [0.85]);
    the first value is compared to threshold.
    """
    def __init__(self, sim_function: Callable, threshold: float = 0.5):
        self.sim_function = sim_function
        self.threshold = threshold

    def score(self, tuple1: List[Any], tuple2: List[Any]) -> float:
        """Return raw similarity score (for adaptive threshold)."""
        scores = self.sim_function(tuple1, tuple2)
        return float(scores[0]) if scores else 0.0

    def __call__(self, tuple1: List[Any], tuple2: List[Any]) -> int:
        s = self.score(tuple1, tuple2)
        return 1 if s >= self.threshold else 0


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

    def score(self, tuple1: List[Any], tuple2: List[Any]) -> float:
        """Return raw cosine similarity (for adaptive threshold)."""
        try:
            model = self._get_model()
            text1 = " ".join(str(a) for a in tuple1)
            text2 = " ".join(str(a) for a in tuple2)
            emb = model.encode([text1, text2])
            return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-9))
        except Exception as e:
            if self._fallback is not None and hasattr(self._fallback, "score"):
                return self._fallback.score(tuple1, tuple2)
            if self._fallback is not None:
                return 0.5
            raise e

    def __call__(self, tuple1: List[Any], tuple2: List[Any]) -> int:
        try:
            sim = self.score(tuple1, tuple2)
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
                model="gpt-5-nano-2025-08-07",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1000,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("LLMOracle default client failed: %s", e)
            return "No"


ADAPTIVE_THRESHOLD_METHODS = ("median", "otsu", "p75")


def get_oracle(
    oracle_type: str,
    threshold: Union[float, str] = 0.5,
    sim_function: Optional[Callable] = None,
    sbert_model: str = "all-MiniLM-L6-v2",
    llm_client: Optional[Callable] = None,
) -> Callable[[List[Any], List[Any]], int]:
    """
    Factory: returns an oracle callable (tuple1, tuple2) -> 0|1.
    oracle_type: 'similarity' | 'sbert' | 'llm'
    threshold: float (e.g. 0.5) or 'median' | 'otsu' | 'p75' for data-driven threshold (used in pipeline with .score()).
    """
    th = 0.5 if (isinstance(threshold, str) and threshold in ADAPTIVE_THRESHOLD_METHODS) else float(threshold)
    if oracle_type == "similarity":
        if sim_function is None:
            from cheaper.similarity import sim_function as sf
            sim_function = sf.jaro
        return SimilarityOracle(sim_function, th)
    if oracle_type == "sbert":
        fallback = SimilarityOracle(sim_function or (lambda t1, t2: [0.5]), th) if sim_function else None
        return SBERTOracle(threshold=th, model_name=sbert_model, fallback_sim=fallback)
    if oracle_type == "llm":
        return LLMOracle(client=llm_client)
    raise ValueError("oracle_type must be 'similarity', 'sbert', or 'llm'")
