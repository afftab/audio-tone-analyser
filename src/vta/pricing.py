"""Inference cost model for the one paid component (the tone/intensity LLM call).

Every other stage -- ffmpeg, DSP, Silero VAD, PANNs, pyannote, parakeet,
emotion2vec+ -- runs on local CPU at $0 marginal cost, so the whole per-clip
cost is this single API call.

Rates are the published gpt-5.6-luna short-context prices. Cached input is
billed at a tenth of fresh input, which matters here because the ~1,500-token
rubric is identical on every call in a batch: the first clip pays full price
and the rest hit the cache.
"""

from dataclasses import dataclass

from vta.tone_llm import TokenUsage

# USD per 1M tokens, gpt-5.6-luna (short context).
INPUT_PER_MTOK = 0.20
CACHED_INPUT_PER_MTOK = 0.02
OUTPUT_PER_MTOK = 1.20

# Brief §5: the final production approach must cost no more than this.
COST_CEILING_PER_AUDIO_MIN = 0.003


@dataclass
class ClipCost:
    usd: float
    audio_s: float

    @property
    def usd_per_audio_min(self) -> float:
        if self.audio_s <= 0:
            return 0.0
        return self.usd / (self.audio_s / 60.0)


@dataclass
class CostBreakdown:
    """Dollar cost split by rate class, not just the total.

    Reasoning and visible-completion tokens bill at the same OUTPUT_PER_MTOK
    rate, so splitting completion_tokens between them is exact -- unlike an
    activity-based allocation, there is no ambiguity in mapping the dollars
    back to token counts here.
    """

    fresh_input_usd: float
    cached_input_usd: float
    reasoning_usd: float
    visible_output_usd: float
    fresh_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    visible_tokens: int

    @property
    def total_usd(self) -> float:
        return (
            self.fresh_input_usd + self.cached_input_usd
            + self.reasoning_usd + self.visible_output_usd
        )


def clip_cost_breakdown(usage: TokenUsage) -> CostBreakdown:
    fresh = max(usage.prompt_tokens - usage.cached_tokens, 0)
    visible = max(usage.completion_tokens - usage.reasoning_tokens, 0)
    return CostBreakdown(
        fresh_input_usd=fresh * INPUT_PER_MTOK / 1_000_000,
        cached_input_usd=usage.cached_tokens * CACHED_INPUT_PER_MTOK / 1_000_000,
        reasoning_usd=usage.reasoning_tokens * OUTPUT_PER_MTOK / 1_000_000,
        visible_output_usd=visible * OUTPUT_PER_MTOK / 1_000_000,
        fresh_tokens=fresh,
        cached_tokens=usage.cached_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        visible_tokens=visible,
    )


def clip_cost(usage: TokenUsage, audio_s: float) -> ClipCost:
    """Cost of one clip's tone call. `cached_tokens` is a subset of `prompt_tokens`."""
    return ClipCost(usd=clip_cost_breakdown(usage).total_usd, audio_s=audio_s)


def batch_totals(costs: list[ClipCost]) -> dict:
    """Aggregate for the dashboard. $/audio-min is computed on batch totals, not
    averaged per clip -- short clips carry fixed rubric overhead and would skew a
    naive mean upward."""
    total_usd = sum(c.usd for c in costs)
    total_audio_s = sum(c.audio_s for c in costs)
    per_min = total_usd / (total_audio_s / 60.0) if total_audio_s > 0 else 0.0
    return {
        "total_usd": total_usd,
        "total_audio_s": total_audio_s,
        "usd_per_audio_min": per_min,
        "ceiling_per_audio_min": COST_CEILING_PER_AUDIO_MIN,
        "pct_of_ceiling": (
            100.0 * per_min / COST_CEILING_PER_AUDIO_MIN
            if COST_CEILING_PER_AUDIO_MIN
            else 0.0
        ),
        "within_ceiling": per_min <= COST_CEILING_PER_AUDIO_MIN,
    }
