"""
Conformal Language Modelling (CLM) for reliable agentic outputs.

Based on: Quach et al., "Conformal Language Modeling", ICLR 2024.
    https://arxiv.org/abs/2306.10193

Implements:
    1. **ConformalSampler** – Algorithm 1 from the paper:
       Sample → Accept/Reject → Stop/Repeat, producing a prediction set
       with statistical coverage guarantees.
    2. **ConformalCalibrator** – Learn-then-Test (LTT) calibration of the
       threshold triple λ = (λ_sim, λ_qual, λ_conf) on a held-out
       calibration set.
    3. **Component-level selection** – §4.4 of the paper: identifies
       sentence-level subsets that are independently reliable.

Adapts the framework for API-based LLMs (OpenAI) where token logprobs
are available via the `logprobs` parameter.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring functions (§5.2 of the paper)
# ---------------------------------------------------------------------------

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def similarity_rouge_l(a: str, b: str) -> float:
    """Text similarity 𝒮 via ROUGE-L F-measure (used for duplicate rejection)."""
    if not a.strip() or not b.strip():
        return 0.0
    return _rouge.score(a, b)["rougeL"].fmeasure


def quality_logprob(logprob: float, length: int, alpha: float = 0.6) -> float:
    """Quality function 𝒬 based on length-normalised log-probability.

    Q(x, y) = exp( logprob(y|x) / |y|^α )   (Wu et al., 2016)
    """
    if length <= 0:
        return 0.0
    normalised = logprob / (length ** alpha)
    return math.exp(normalised)


def quality_self_consistency(candidate: str, other_candidates: list[str]) -> float:
    """Quality function 𝒬 based on self-consistency (voting).

    When logprobs are unavailable, measure how much a candidate agrees with
    the rest of the prediction set via average ROUGE-L.
    """
    if not other_candidates:
        return 1.0
    scores = [similarity_rouge_l(candidate, o) for o in other_candidates]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Set-confidence functions ℱ (§5.2)
# ---------------------------------------------------------------------------

def set_confidence_max(qualities: list[float]) -> float:
    """ℱ_Max: set is as good as its best element."""
    return max(qualities) if qualities else 0.0


def set_confidence_sum(qualities: list[float]) -> float:
    """ℱ_Sum: sum of element-level quality scores."""
    return sum(qualities)


def set_confidence_first_k(k: int, _qualities: list[float] | None = None) -> float:
    """ℱ_First-K: just the count of samples taken."""
    return float(k)


# ---------------------------------------------------------------------------
# Conformal Sampler – Algorithm 1
# ---------------------------------------------------------------------------

@dataclass
class SamplerConfig:
    """Configurable thresholds for conformal sampling."""
    # λ₁ – max ROUGE-L overlap with existing set members (diversity)
    lambda_sim: float = 0.85
    # λ₂ – min quality score to accept a sample
    lambda_qual: float = 0.0
    # λ₃ – min set confidence to stop sampling
    #   For first_k: number of total samples drawn before stopping.
    #   For max/sum: quality threshold (scale depends on logprobs).
    lambda_conf: float = 3.0
    # k_max – sampling budget
    k_max: int = 5
    # Temperature for diverse sampling (higher → more diverse)
    sampling_temperature: float = 0.7
    # Set-confidence function: "max", "sum", or "first_k"
    #   first_k is recommended for API-based LLMs (scale-invariant).
    confidence_fn: str = "first_k"
    # γ – ROUGE-L threshold for component-level sentence reliability
    component_threshold: float = 0.5


@dataclass
class ConformalPredictionSet:
    """Output of the conformal sampler."""
    candidates: list[str] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)
    logprobs: list[float | None] = field(default_factory=list)
    set_confidence: float = 0.0
    total_samples_drawn: int = 0
    total_rejected: int = 0
    best_index: int = 0
    # Component-level reliability (sentence → confidence)
    reliable_components: list[dict[str, Any]] = field(default_factory=list)

    @property
    def best_candidate(self) -> str:
        if not self.candidates:
            return ""
        return self.candidates[self.best_index]

    @property
    def set_size(self) -> int:
        return len(self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "set_size": self.set_size,
            "total_samples_drawn": self.total_samples_drawn,
            "total_rejected": self.total_rejected,
            "set_confidence": round(self.set_confidence, 4),
            "best_index": self.best_index,
            "qualities": [round(q, 4) for q in self.qualities],
            "reliable_components": self.reliable_components,
        }


class ConformalSampler:
    """Implements CLM Algorithm 1: conformal sampling with rejection.

    Works with any LLM callable that returns (text, logprob, token_count).
    For OpenAI API, we request `logprobs=True` and `top_logprobs=1`.
    """

    def __init__(self, config: SamplerConfig | None = None):
        self.config = config or SamplerConfig()

    def _get_confidence_fn(self) -> Callable:
        name = self.config.confidence_fn
        if name == "max":
            return set_confidence_max
        elif name == "sum":
            return set_confidence_sum
        elif name == "first_k":
            return set_confidence_first_k
        else:
            raise ValueError(f"Unknown confidence function: {name}")

    def sample(
        self,
        generate_fn: Callable[[], tuple[str, float | None, int]],
    ) -> ConformalPredictionSet:
        """Run the conformal sampling loop.

        Args:
            generate_fn: Callable that returns (response_text, total_logprob, n_tokens).
                         total_logprob may be None if unavailable.

        Returns:
            ConformalPredictionSet with the accepted candidates.
        """
        cfg = self.config
        conf_fn = self._get_confidence_fn()

        candidates: list[str] = []
        qualities: list[float] = []
        logprobs_list: list[float | None] = []
        total_drawn = 0
        total_rejected = 0

        for k in range(1, cfg.k_max + 1):
            # Step 1: Sample
            text, logprob, n_tokens = generate_fn()
            total_drawn += 1

            # Step 2a: Reject duplicates (similarity check)
            is_duplicate = False
            for existing in candidates:
                sim = similarity_rouge_l(text, existing)
                if sim > cfg.lambda_sim:
                    is_duplicate = True
                    break

            if is_duplicate:
                total_rejected += 1
                # Still check stopping condition with current set
                if cfg.confidence_fn == "first_k":
                    current_conf = set_confidence_first_k(len(candidates))
                else:
                    current_conf = conf_fn(qualities) if qualities else 0.0
                if current_conf >= cfg.lambda_conf:
                    break
                continue

            # Step 2b: Reject low quality
            if logprob is not None:
                q = quality_logprob(logprob, n_tokens)
            else:
                q = quality_self_consistency(text, candidates)

            if q < cfg.lambda_qual:
                total_rejected += 1
                continue

            # Accept the sample
            candidates.append(text)
            qualities.append(q)
            logprobs_list.append(logprob)

            # Step 3: Check stopping condition
            if cfg.confidence_fn == "first_k":
                current_conf = set_confidence_first_k(len(candidates))
            else:
                current_conf = conf_fn(qualities)

            if current_conf >= cfg.lambda_conf:
                break

        # Select best candidate by quality
        best_idx = int(np.argmax(qualities)) if qualities else 0

        # Final set confidence
        if cfg.confidence_fn == "first_k":
            final_conf = set_confidence_first_k(len(candidates))
        else:
            final_conf = conf_fn(qualities) if qualities else 0.0

        result = ConformalPredictionSet(
            candidates=candidates,
            qualities=qualities,
            logprobs=logprobs_list,
            set_confidence=final_conf,
            total_samples_drawn=total_drawn,
            total_rejected=total_rejected,
            best_index=best_idx,
        )

        # Component-level reliability analysis
        if candidates:
            result.reliable_components = self._component_selection(
                candidates, qualities
            )

        return result

    def _component_selection(
        self,
        candidates: list[str],
        qualities: list[float],
    ) -> list[dict[str, Any]]:
        """§4.4 (heuristic approximation): Identify reliable sentence-level components.

        This is a self-consistency heuristic, NOT a calibrated conformal procedure.
        The paper's C_inner requires reference texts for calibration, which are
        unavailable at inference time.  Instead, we check whether each sentence
        in the best candidate is corroborated by other candidates in the set.
        A sentence is "reliable" if ≥50% of other candidates contain a similar
        statement (ROUGE-L > component_threshold).
        """
        if len(candidates) < 2:
            # Cannot do component selection with < 2 candidates
            return []

        # Decompose best candidate into sentences
        best_idx = int(np.argmax(qualities))
        best_text = candidates[best_idx]
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', best_text) if s.strip()]

        if not sentences:
            return []

        # For each sentence, check how many other candidates contain a
        # similar statement (via ROUGE-L > threshold)
        threshold = self.config.component_threshold
        components = []
        for sent in sentences:
            support_count = 0
            for j, cand in enumerate(candidates):
                if j == best_idx:
                    continue
                # Check if any sentence in this candidate is similar
                cand_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cand) if s.strip()]
                max_sim = max(
                    (similarity_rouge_l(sent, cs) for cs in cand_sents),
                    default=0.0,
                )
                if max_sim > threshold:
                    support_count += 1

            # Fraction of other candidates that support this sentence
            support_frac = support_count / max(len(candidates) - 1, 1)
            components.append({
                "sentence": sent,
                "support_fraction": round(support_frac, 3),
                "reliable": support_frac >= 0.5,
            })

        return components


# ---------------------------------------------------------------------------
# Conformal Calibrator – Learn-then-Test (§4.3)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Output of the LTT calibration procedure."""
    lambda_sim: float
    lambda_qual: float
    lambda_conf: float
    empirical_coverage: float
    avg_set_size: float
    avg_samples_drawn: float
    n_calibration: int
    is_valid: bool  # Whether a valid λ was found
    epsilon: float  # Target miscoverage rate
    delta: float    # Confidence level

    def to_config(self) -> SamplerConfig:
        """Convert calibration result to a SamplerConfig."""
        return SamplerConfig(
            lambda_sim=self.lambda_sim,
            lambda_qual=self.lambda_qual,
            lambda_conf=self.lambda_conf,
        )


class ConformalCalibrator:
    """Calibrate λ = (λ_sim, λ_qual, λ_conf) using LTT on a calibration set.

    The calibration set consists of (input, reference) pairs where we know
    the ground truth.  For each configuration λ we measure the empirical
    risk (fraction of calibration examples where the prediction set does NOT
    contain an admissible answer) and apply Fixed Sequence Testing (FST)
    along configurations sorted by ascending cost.  This avoids the
    conservativeness of Bonferroni while maintaining the coverage guarantee.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        delta: float = 0.05,
        admission_threshold: float = 0.3,
    ):
        """
        Args:
            epsilon: Target miscoverage rate (1 - coverage).
            delta: Confidence level for the guarantee.
            admission_threshold: ROUGE-L threshold for admission function A.
                A candidate y is "admissible" if ROUGE-L(y, y_ref) ≥ threshold.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.admission_threshold = admission_threshold

    def _admission_fn(self, candidate: str, reference: str) -> bool:
        """Admission function A(y): is the candidate acceptable?"""
        return similarity_rouge_l(candidate, reference) >= self.admission_threshold

    def _set_admits(self, candidates: list[str], reference: str) -> bool:
        """Check if at least one candidate in the set is admissible."""
        return any(self._admission_fn(c, reference) for c in candidates)

    def calibrate(
        self,
        generate_fn: Callable[[str], Callable[[], tuple[str, float | None, int]]],
        calibration_data: list[dict[str, str]],
        cal_k_max: int = 3,
    ) -> CalibrationResult:
        """Run LTT calibration over a grid of λ configurations.

        Args:
            generate_fn: Factory that takes an input prompt and returns a
                         sampling callable (as expected by ConformalSampler.sample).
            calibration_data: List of dicts with 'input' and 'reference' keys.
            cal_k_max: Sampling budget during calibration (lower than inference
                       k_max to reduce LLM call count).

        Returns:
            CalibrationResult with the best valid configuration.
        """
        n = len(calibration_data)
        if n == 0:
            logger.warning("Empty calibration set; returning defaults.")
            return CalibrationResult(
                lambda_sim=0.85, lambda_qual=0.0, lambda_conf=0.5,
                empirical_coverage=0.0, avg_set_size=0.0,
                avg_samples_drawn=0.0, n_calibration=0,
                is_valid=False, epsilon=self.epsilon, delta=self.delta,
            )

        # Reduced 3×3×3 grid (27 configs) to limit calibration cost
        sim_grid = [0.7, 0.85, 0.95]
        qual_grid = [0.0, 0.1, 0.3]
        conf_grid = [0.3, 0.5, 0.9]
        configs = [
            (s, q, c) for s in sim_grid for q in qual_grid for c in conf_grid
        ]

        logger.info(
            "CLM calibration: %d configs × %d samples (ε=%.2f, δ=%.2f, "
            "cal_k_max=%d)",
            len(configs), n, self.epsilon, self.delta, cal_k_max,
        )

        # For each config, compute empirical risk on calibration set
        results: list[dict[str, Any]] = []
        for lam_sim, lam_qual, lam_conf in configs:
            cfg = SamplerConfig(
                lambda_sim=lam_sim,
                lambda_qual=lam_qual,
                lambda_conf=lam_conf,
                k_max=cal_k_max,
            )
            sampler = ConformalSampler(cfg)

            n_failures = 0
            set_sizes: list[int] = []
            samples_drawn: list[int] = []
            for item in calibration_data:
                gen_fn = generate_fn(item["input"])
                pred_set = sampler.sample(gen_fn)
                admitted = self._set_admits(pred_set.candidates, item["reference"])
                if not admitted:
                    n_failures += 1
                set_sizes.append(pred_set.set_size)
                samples_drawn.append(pred_set.total_samples_drawn)

            results.append({
                "lambda": (lam_sim, lam_qual, lam_conf),
                "n_failures": n_failures,
                "empirical_risk": n_failures / n,
                "avg_set_size": float(np.mean(set_sizes)),
                "avg_samples_drawn": float(np.mean(samples_drawn)),
                "coverage": 1.0 - n_failures / n,
            })

        # Fixed Sequence Testing (FST): sort by ascending cost, test at
        # level δ (no Bonferroni division needed for ordered tests).
        # This mirrors the official repo's walk along the Pareto frontier.
        def _cost(res: dict) -> float:
            return 0.5 * res["avg_set_size"] + 0.5 * res["avg_samples_drawn"]

        results.sort(key=_cost)

        from scipy.stats import binom

        best_valid: dict[str, Any] | None = None
        for res in results:
            p_value = binom.cdf(res["n_failures"], n, self.epsilon)
            res["p_value"] = p_value
            # Reject H_λ (risk > ε) if p_value ≤ δ
            if p_value <= self.delta:
                res["valid"] = True
                best_valid = res  # Keep the lowest-cost valid config
                break  # FST: first valid in cost-sorted order is optimal
            else:
                res["valid"] = False

        if best_valid is None:
            logger.warning(
                "No valid CLM configuration found (n=%d, ε=%.2f, δ=%.2f). "
                "Using lowest-risk config with abstention flag.",
                n, self.epsilon, self.delta,
            )
            # Fall back to config with lowest empirical risk
            best_fallback = min(results, key=lambda r: r["empirical_risk"])
            lam = best_fallback["lambda"]
            return CalibrationResult(
                lambda_sim=lam[0], lambda_qual=lam[1], lambda_conf=lam[2],
                empirical_coverage=best_fallback["coverage"],
                avg_set_size=best_fallback["avg_set_size"],
                avg_samples_drawn=best_fallback["avg_samples_drawn"],
                n_calibration=n, is_valid=False,
                epsilon=self.epsilon, delta=self.delta,
            )

        lam = best_valid["lambda"]

        logger.info(
            "CLM calibrated: λ=(%.2f, %.2f, %.2f), coverage=%.2f, "
            "avg_size=%.1f, avg_drawn=%.1f, p=%.4f",
            *lam, best_valid["coverage"], best_valid["avg_set_size"],
            best_valid["avg_samples_drawn"], best_valid["p_value"],
        )

        return CalibrationResult(
            lambda_sim=lam[0], lambda_qual=lam[1], lambda_conf=lam[2],
            empirical_coverage=best_valid["coverage"],
            avg_set_size=best_valid["avg_set_size"],
            avg_samples_drawn=best_valid["avg_samples_drawn"],
            n_calibration=n, is_valid=True,
            epsilon=self.epsilon, delta=self.delta,
        )


# ---------------------------------------------------------------------------
# OpenAI-specific generation wrapper
# ---------------------------------------------------------------------------

def make_openai_generate_fn(
    llm: Any,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.7,
) -> Callable[[], tuple[str, float | None, int]]:
    """Create a generate_fn compatible with ConformalSampler.

    Uses the OpenAI API with logprobs enabled for quality estimation.

    Args:
        llm: LangChain ChatOpenAI model.
        system_prompt: System prompt for the agent.
        user_message: The user query.
        temperature: Sampling temperature (>0 for diversity).

    Returns:
        Callable that returns (response_text, total_logprob, n_tokens).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    def _generate() -> tuple[str, float | None, int]:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        # Use higher temperature for diverse sampling
        response = llm.invoke(
            messages,
            temperature=temperature,
            logprobs=True,
            top_logprobs=1,
        )

        text = response.content
        n_tokens = len(text.split())  # Approximate token count

        # Extract logprobs from response metadata
        total_logprob = None
        if hasattr(response, "response_metadata"):
            meta = response.response_metadata
            # OpenAI returns logprobs in the response metadata
            if "logprobs" in meta and meta["logprobs"]:
                content_logprobs = meta["logprobs"].get("content", [])
                if content_logprobs:
                    total_logprob = sum(
                        t.get("logprob", 0.0) for t in content_logprobs
                    )
                    n_tokens = len(content_logprobs)

        return text, total_logprob, n_tokens

    return _generate


def make_calibration_generate_factory(
    llm: Any,
    system_prompt: str,
    temperature: float = 0.7,
) -> Callable[[str], Callable[[], tuple[str, float | None, int]]]:
    """Create a generate_fn *factory* compatible with ConformalCalibrator.

    The calibrator needs a factory: input_text → sampling_callable.
    This wraps make_openai_generate_fn so that each calibration sample
    gets its own closure with the correct user message.

    Args:
        llm: LangChain ChatOpenAI model.
        system_prompt: System prompt for the synthesis step.
        temperature: Sampling temperature for diverse generation.

    Returns:
        Factory callable: (user_message: str) → generate_fn.
    """
    def factory(user_message: str) -> Callable[[], tuple[str, float | None, int]]:
        return make_openai_generate_fn(llm, system_prompt, user_message, temperature)
    return factory
