"""The tone call's response handling and schema contract.

The API is never touched: a stub client stands in for the three failure modes
that previously reached json.loads() as a bare TypeError or JSONDecodeError.
"""

import json
import types

import pytest

from vta import tone_llm
from vta.tone_llm import (
    TONE_SCHEMA,
    ProsodySummary,
    ToneClassificationError,
    ToneJudgment,
    classify_tone,
)

PROSODY = ProsodySummary(
    pitch_mean_hz=200.0, pitch_std_hz=30.0, energy_mean_db=-30.0,
    energy_std_db=5.0, energy_dynamic_range_db=40.0, voiced_ratio=0.5,
    speaking_rate_wpm=180.0, energy_contour_db=[-30.0, -28.0],
)

VALID = {
    "step1_lexical_evidence": "customer asks for a refund, third call",
    "step2_emotional_tone": "frustrated",
    "step3_acoustic_evidence": "arousal 0.45, moderate activation",
    "step4_intensity_rationale": "clear and sustained",
    "step5_emotional_intensity": "medium",
}


def _response(content, *, finish_reason="stop", refusal=None):
    usage = types.SimpleNamespace(
        prompt_tokens=1000, completion_tokens=200,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=800),
        completion_tokens_details=types.SimpleNamespace(reasoning_tokens=150),
    )
    message = types.SimpleNamespace(content=content, refusal=refusal)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    return types.SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def stub_client(monkeypatch):
    holder = {}

    class _Completions:
        def create(self, **kwargs):
            holder["kwargs"] = kwargs
            return holder["response"]

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))
    monkeypatch.setattr(tone_llm, "_get_client", lambda: client)
    return holder


def test_happy_path_returns_judgment_and_usage(stub_client):
    stub_client["response"] = _response(json.dumps(VALID))
    judgment, usage = classify_tone("I want a refund.", PROSODY)

    assert judgment.step2_emotional_tone == "frustrated"
    assert judgment.step5_emotional_intensity == "medium"
    assert usage.prompt_tokens == 1000
    assert usage.cached_tokens == 800
    assert usage.reasoning_tokens == 150


def test_truncated_response_raises_actionable_error(stub_client):
    """Reasoning can eat max_completion_tokens before the JSON object closes."""
    stub_client["response"] = _response('{"step1_lexical_evidence": "cus', finish_reason="length")
    with pytest.raises(ToneClassificationError) as exc:
        classify_tone("hello", PROSODY)
    msg = str(exc.value)
    assert "truncated" in msg
    assert "VTA_TONE_MAX_COMPLETION_TOKENS" in msg, "the error should say what to change"


def test_refusal_raises(stub_client):
    stub_client["response"] = _response(None, refusal="I can't help with that")
    with pytest.raises(ToneClassificationError, match="refused"):
        classify_tone("hello", PROSODY)


def test_empty_content_raises(stub_client):
    stub_client["response"] = _response("")
    with pytest.raises(ToneClassificationError, match="empty response"):
        classify_tone("hello", PROSODY)


def test_malformed_json_raises(stub_client):
    stub_client["response"] = _response("this is not json")
    with pytest.raises(ToneClassificationError, match="malformed JSON"):
        classify_tone("hello", PROSODY)


def test_out_of_enum_label_is_rejected(stub_client):
    """The model is the contract; a label outside the brief's five must fail."""
    bad = dict(VALID, step2_emotional_tone="furious")
    stub_client["response"] = _response(json.dumps(bad))
    with pytest.raises(Exception):  # pydantic ValidationError
        classify_tone("hello", PROSODY)


# --- schema contract ---

def test_schema_is_generated_from_the_model():
    """One definition, not two hand-synced ones."""
    assert set(TONE_SCHEMA["properties"]) == set(ToneJudgment.model_fields)
    assert TONE_SCHEMA["required"] == list(TONE_SCHEMA["properties"])


def test_schema_is_strict_mode_compatible():
    assert TONE_SCHEMA["additionalProperties"] is False
    assert TONE_SCHEMA["type"] == "object"
    # Strict mode rejects $ref/$defs, which pydantic emits for Literal fields.
    assert "$defs" not in TONE_SCHEMA
    assert "$ref" not in json.dumps(TONE_SCHEMA)


def test_schema_carries_the_brief_enums():
    props = TONE_SCHEMA["properties"]
    assert props["step2_emotional_tone"]["enum"] == [
        "neutral", "satisfied", "frustrated", "upset", "distressed",
    ]
    assert props["step5_emotional_intensity"]["enum"] == ["low", "medium", "high"]


def test_field_prefixes_force_evidence_before_label():
    """Strict outputs emit keys lexicographically; evidence must sort first."""
    keys = sorted(TONE_SCHEMA["properties"])
    assert keys.index("step1_lexical_evidence") < keys.index("step2_emotional_tone")
    assert keys.index("step3_acoustic_evidence") < keys.index("step5_emotional_intensity")
    assert keys.index("step4_intensity_rationale") < keys.index("step5_emotional_intensity")


def test_audio_is_never_sent_to_the_api(stub_client):
    """Brief §5: only derived text and numbers leave local infrastructure."""
    stub_client["response"] = _response(json.dumps(VALID))
    classify_tone("I want a refund.", PROSODY)

    payload = json.dumps(stub_client["kwargs"]["messages"])
    assert "I want a refund." in payload
    for banned in ("samples", "waveform", "base64", "audio_url", "input_audio"):
        assert banned not in payload
