"""Cost model: the total must equal what a reader gets by summing the
dashboard's own cost-breakdown panel, not a separately-maintained figure."""

import pytest

from vta.pricing import (
    CACHED_INPUT_PER_MTOK,
    INPUT_PER_MTOK,
    OUTPUT_PER_MTOK,
    clip_cost,
    clip_cost_breakdown,
)
from vta.tone_llm import TokenUsage


def test_breakdown_components_sum_to_the_same_total_clip_cost_reports():
    usage = TokenUsage(prompt_tokens=3087, cached_tokens=1383, completion_tokens=908, reasoning_tokens=512)
    breakdown = clip_cost_breakdown(usage)
    total_cost = clip_cost(usage, audio_s=171.9)
    assert breakdown.total_usd == pytest.approx(total_cost.usd)


def test_reasoning_and_visible_tokens_partition_completion_exactly():
    usage = TokenUsage(prompt_tokens=100, cached_tokens=0, completion_tokens=908, reasoning_tokens=512)
    b = clip_cost_breakdown(usage)
    assert b.reasoning_tokens + b.visible_tokens == usage.completion_tokens
    assert b.reasoning_tokens == usage.reasoning_tokens


def test_fresh_tokens_exclude_the_cached_subset():
    usage = TokenUsage(prompt_tokens=1848, cached_tokens=1383, completion_tokens=100, reasoning_tokens=0)
    b = clip_cost_breakdown(usage)
    assert b.fresh_tokens == 1848 - 1383
    assert b.cached_tokens == 1383


def test_rates_applied_per_component():
    usage = TokenUsage(prompt_tokens=1_000_000, cached_tokens=0, completion_tokens=0, reasoning_tokens=0)
    assert clip_cost_breakdown(usage).fresh_input_usd == pytest.approx(INPUT_PER_MTOK)

    usage = TokenUsage(prompt_tokens=1_000_000, cached_tokens=1_000_000, completion_tokens=0, reasoning_tokens=0)
    assert clip_cost_breakdown(usage).cached_input_usd == pytest.approx(CACHED_INPUT_PER_MTOK)

    usage = TokenUsage(prompt_tokens=0, cached_tokens=0, completion_tokens=1_000_000, reasoning_tokens=1_000_000)
    b = clip_cost_breakdown(usage)
    assert b.reasoning_usd == pytest.approx(OUTPUT_PER_MTOK)
    assert b.visible_output_usd == 0.0


def test_zero_usage_costs_nothing():
    usage = TokenUsage(prompt_tokens=0, cached_tokens=0, completion_tokens=0, reasoning_tokens=0)
    b = clip_cost_breakdown(usage)
    assert b.total_usd == 0.0
