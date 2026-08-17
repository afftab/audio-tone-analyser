"""GPT-5.6 Luna: transcript + prosody/emotion features -> emotional_tone, emotional_intensity.

Only derived text and numeric features are sent to the API -- never raw
audio. The frustrated/upset boundary is semantic ("annoyed, impatient,
dissatisfied *without* strong anger" vs "clearly angry, agitated"), which
prosody alone cannot resolve (two callers can be acoustically identical
while one grumbles about a delay and the other demands a refund) -- see
PLAN.md §3 for why this field is split from the deterministic 7.
"""

import json
import os
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from vta.config import OPENAI_API_KEY
from vta.schema import EmotionalIntensity, EmotionalTone

MODEL = os.environ.get("VTA_TONE_MODEL", "gpt-5.6-luna")

# Empty string omits the parameter. Reasoning tokens bill at the OUTPUT rate.
# "high" is worth +0.03 tone macro F1 on length-matched IEMOCAP; "low"
# collapses emotional_intensity to "low" on every clip.
TONE_EFFORT = os.environ.get("VTA_TONE_EFFORT", "high") or None

# Caps reasoning + visible output together, so a tight value can truncate the
# JSON mid-object. Generous, since billing is per token produced. This model
# rejects `max_tokens` in favour of `max_completion_tokens`.
TONE_MAX_COMPLETION_TOKENS = int(
    os.environ.get("VTA_TONE_MAX_COMPLETION_TOKENS", "16000")
)

# Verbatim from PROJECT_BRIEF.md §2, so the model applies the same
# definitions the hidden test set will be scored against.
RUBRIC = """\
emotional_tone (Enum)
Allowed values: neutral | satisfied | frustrated | upset | distressed
Definition: The primary emotional tone expressed by the customer. Neutral means no clear
positive or negative emotion. Satisfied means pleased, relieved, appreciative, or clearly positive.
Frustrated means annoyed, impatient, or dissatisfied without strong anger or distress. Upset
means clearly angry, agitated, or strongly dissatisfied. Distressed means highly emotional,
overwhelmed, panicked, crying, or otherwise emotionally escalated.

emotional_intensity (Enum)
Allowed values: low | medium | high
Definition: The strength of the detected emotional tone. Low is subtle or mild. Medium is clear
and sustained. High is strong, escalated, or likely to require attention.

Evaluation note: Do not infer frustration or distress solely from loudness or acoustic energy --
weigh the words and their context. A quiet caller can be furious; a loud caller can be merely
enthusiastic.
"""

SYSTEM_PROMPT = f"""You are an expert call-center QA analyst classifying customer emotional tone.

{RUBRIC}

You will be given the call transcript and acoustic-prosody summary statistics (pitch, energy,
speaking rate, arousal, emotion-class posteriors) already extracted from the audio.

WHOSE TONE TO CLASSIFY. When the transcript is speaker-labelled (lines like "SPEAKER_00: ..."),
the call has two parties: an agent (often a scripted or automated assistant) and the customer.
Identify which label is the CUSTOMER -- the agent is the one who opens with a company greeting,
offers help, quotes policy, and offers to transfer; the customer is the one with a request or a
problem. Classify the CUSTOMER's tone only. Do not let the agent's words or composure influence
the label: agents are trained or scripted to stay neutral, and roughly half the speech in a
typical call is theirs. Name the speaker you identified as the customer in step1.

If the transcript is not speaker-labelled, treat it as the customer's speech.

CONTEXT VS TARGET. If a "preceding_conversation_context" field is present, it contains earlier
turns of the same conversation, provided so you can tell whether the customer is escalating,
de-escalating, or responding to something specific. Use it to interpret the target -- but
classify ONLY the tone of the utterance in "transcript". Do not classify the context, and do not
let an emotional earlier turn override what the target utterance actually expresses. When no
context field is present, the transcript is all there is.

Complete the output fields in order. Each label is evidence-gated by what precedes it in the
output: state the lexical evidence and commit to emotional_tone, then read out the acoustic
evidence and grade emotional_intensity from it.

- emotional_tone is primarily semantic: read it from what the customer says -- what they
  ask for, what has gone wrong, how many times, whether they thank or blame, escalate or
  de-escalate across the call.
- emotional_intensity is primarily prosodic: grade it from the acoustic evidence, with the
  words as context. arousal is a 0-1 acoustic-emotion score: ~0.2-0.3 is typical calm
  neutral speech, 0.35-0.6 is moderately activated, >=0.6 is strongly activated, ~1.0 is
  near the model's maximum. Energy/pitch/rate, the per-second energy contour, and the
  emotion model's class posteriors corroborate it.

If the transcript is very short or affectively ambiguous, weight the acoustic evidence more
heavily -- do not default to neutral tone or low intensity for lack of words. Conversely, do
not infer frustration or distress solely from loudness or acoustic energy: check what the
words actually describe.

These are summary statistics over the whole clip, so they are coarse: a caller who starts
calm and escalates averages out to unremarkable numbers. The energy contour shows the arc.

Choosing between the classes:
- "neutral" means the customer genuinely expresses no clear positive or negative emotion --
  a routine, transactional exchange. It is NOT an "uncertain" or "can't tell" option. If
  there is any directional signal in the words, choose the closest non-neutral class instead.
- "frustrated" vs "upset" is a matter of degree of anger: annoyance and impatience with a
  situation is frustrated; anger directed with force, or strong dissatisfaction, is upset.
- "distressed" is about being overwhelmed rather than angry: panic, crying, fear, pleading,
  helplessness, or a personal stake beyond the transaction. A distressed caller is escalated
  but not necessarily hostile. Do not read distress as satisfaction because the voice sounds
  animated -- check what the words are actually describing.
- "satisfied" requires positive content: thanks, relief, praise, or a resolved problem.
  Animated or high-energy speech alone is not satisfaction.

Output only the five required fields, in field order."""


class ToneClassificationError(RuntimeError):
    """The tone call returned nothing usable (refusal, truncation, bad JSON)."""


# Strict structured outputs emit JSON keys in lexicographic order, so the
# step1_..step5_ prefixes force evidence before each label. This model is
# the response-shape definition; TONE_SCHEMA is generated from it.
class ToneJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")  # -> additionalProperties: false

    step1_lexical_evidence: str = Field(
        description=(
            "What the customer's words indicate: what they ask for, what has gone "
            "wrong, how many times, whether they thank or blame, escalate or "
            "de-escalate. Quote short spans. If the transcript is speaker-labelled, "
            "FIRST state which speaker label you identified as the customer "
            "and why, then cite only that speaker's words. If the transcript "
            "is short, thin, or affectively ambiguous, say so explicitly."
        )
    )
    step2_emotional_tone: EmotionalTone
    step3_acoustic_evidence: str = Field(
        description=(
            "Read out the acoustic evidence explicitly: arousal (0-1 acoustic-emotion "
            "score; ~0.2-0.3 typical calm neutral speech; 0.35-0.6 moderately "
            "activated; >=0.6 strongly activated; ~1.0 near maximum), energy "
            "mean/std/dynamic range in dB, pitch mean/std, speaking rate, energy "
            "contour trend, and the emotion model's class posteriors. State what "
            "they imply about activation level."
        )
    )
    step4_intensity_rationale: str = Field(
        description=(
            "Why this evidence maps to the chosen intensity. Low = subtle or mild. "
            "Medium = clear and sustained. High = strong, escalated, likely to "
            "require attention."
        )
    )
    step5_emotional_intensity: EmotionalIntensity


def _strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON Schema, inlined for OpenAI strict structured outputs.

    Literal fields become $defs with a $ref, which strict mode rejects, so
    splice them back in. `title` keys are dropped -- they ship on every
    request and cached input is still billed.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return inline({k: v for k, v in defs[name].items() if k != "title"})
            return {k: inline(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    schema = inline(schema)
    # Strict mode requires every property listed in `required`.
    schema["required"] = list(schema["properties"])
    schema["additionalProperties"] = False
    return schema


TONE_SCHEMA = _strict_schema(ToneJudgment)


@dataclass
class TokenUsage:
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    # Subset of completion_tokens spent on internal reasoning (0 when effort
    # is unset or the model is non-reasoning). Billed at the output rate.
    reasoning_tokens: int = 0


@dataclass
class ProsodySummary:
    pitch_mean_hz: float
    pitch_std_hz: float
    energy_mean_db: float
    energy_std_db: float
    energy_dynamic_range_db: float
    voiced_ratio: float
    speaking_rate_wpm: float
    energy_contour_db: list[float]
    arousal: float | None = None  # from emotion2vec, if available
    emotion_posteriors: dict[str, float] | None = None  # from emotion2vec, if available


# The contour is one float per second, serialized into the prompt. Longer
# contours are block-averaged down to this many points to bound the payload.
MAX_CONTOUR_POINTS = int(os.environ.get("VTA_MAX_CONTOUR_POINTS", "180"))


def _downsample_contour(contour: list[float], max_points: int = MAX_CONTOUR_POINTS) -> list[float]:
    n = len(contour)
    if n <= max_points or max_points < 1:
        return contour
    # Block-average, not slice: peaks survive as raised blocks.
    edges = [round(i * n / max_points) for i in range(max_points + 1)]
    return [
        sum(contour[a:b]) / (b - a)
        for a, b in zip(edges, edges[1:])
        if b > a
    ]


def _prosody_block(prosody: ProsodySummary) -> dict:
    contour = _downsample_contour(prosody.energy_contour_db)
    block = {
        "pitch_mean_hz": round(prosody.pitch_mean_hz, 1),
        "pitch_std_hz": round(prosody.pitch_std_hz, 1),
        "energy_mean_db": round(prosody.energy_mean_db, 1),
        "energy_std_db": round(prosody.energy_std_db, 1),
        "energy_dynamic_range_db": round(prosody.energy_dynamic_range_db, 1),
        "voiced_ratio": round(prosody.voiced_ratio, 3),
        "speaking_rate_wpm": round(prosody.speaking_rate_wpm, 1),
        "energy_contour_db_per_second": [round(x, 1) for x in contour],
    }
    if len(contour) != len(prosody.energy_contour_db):
        # Otherwise the model reads a resampled series as one point/second.
        block["energy_contour_note"] = (
            f"{len(prosody.energy_contour_db)} one-second samples averaged down to "
            f"{len(contour)} points; each point covers about "
            f"{len(prosody.energy_contour_db) / len(contour):.1f}s."
        )
    if prosody.arousal is not None:
        block["arousal"] = round(prosody.arousal, 3)
    if prosody.emotion_posteriors is not None:
        block["emotion_model_posteriors"] = {
            k: round(v, 3) for k, v in prosody.emotion_posteriors.items()
        }
    return block


def _user_content(
    transcript: str,
    prosody: ProsodySummary,
    per_speaker_prosody: dict[str, ProsodySummary] | None = None,
    context: str | None = None,
) -> str:
    """Build the request payload.

    `context` carries preceding conversational turns -- the largest reported
    gain for transcript-based LLM emotion recognition (arXiv:2602.06270).
    Per-speaker prosody sends both speakers: picking one would duplicate the
    customer identification with a heuristic here.
    """
    payload: dict = {}
    if context:
        payload["preceding_conversation_context"] = context
    payload["transcript"] = transcript

    if per_speaker_prosody:
        payload["prosody_features_by_speaker"] = {
            spk: _prosody_block(p) for spk, p in sorted(per_speaker_prosody.items())
        }
        payload["prosody_features_whole_clip"] = _prosody_block(prosody)
        payload["note"] = (
            "prosody_features_by_speaker is measured over each speaker's own "
            "audio only. Grade emotional_intensity from the block belonging to "
            "the speaker you identified as the customer."
        )
    else:
        payload["prosody_features"] = _prosody_block(prosody)

    return json.dumps(payload, indent=2)


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def classify_tone(
    transcript: str,
    prosody: ProsodySummary,
    per_speaker_prosody: dict[str, ProsodySummary] | None = None,
    context: str | None = None,
) -> tuple[ToneJudgment, TokenUsage]:
    client = _get_client()
    kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": _user_content(transcript, prosody, per_speaker_prosody, context)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "tone_judgment", "strict": True, "schema": TONE_SCHEMA},
        },
    }
    if TONE_EFFORT:
        kwargs["reasoning_effort"] = TONE_EFFORT
    if TONE_MAX_COMPLETION_TOKENS:
        kwargs["max_completion_tokens"] = TONE_MAX_COMPLETION_TOKENS
    response = client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    # The ways a structured-output call returns no usable JSON.
    if getattr(choice.message, "refusal", None):
        raise ToneClassificationError(f"model refused the request: {choice.message.refusal}")
    if choice.finish_reason == "length":
        # Reasoning and output share the budget, so at high effort it can
        # run out before the JSON object closes.
        raise ToneClassificationError(
            f"response truncated at max_completion_tokens={TONE_MAX_COMPLETION_TOKENS} "
            f"(reasoning tokens alone: "
            f"{getattr(getattr(response.usage, 'completion_tokens_details', None), 'reasoning_tokens', '?')}). "
            "Raise VTA_TONE_MAX_COMPLETION_TOKENS or lower VTA_TONE_EFFORT."
        )
    if not choice.message.content:
        raise ToneClassificationError(
            f"empty response content (finish_reason={choice.finish_reason!r})"
        )

    try:
        data = json.loads(choice.message.content)
    except json.JSONDecodeError as e:
        raise ToneClassificationError(
            f"malformed JSON (finish_reason={choice.finish_reason!r}): {e}"
        ) from e

    usage = response.usage
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    reasoning = getattr(
        getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0
    ) or 0
    token_usage = TokenUsage(
        prompt_tokens=usage.prompt_tokens,
        cached_tokens=cached,
        completion_tokens=usage.completion_tokens,
        reasoning_tokens=reasoning,
    )
    return ToneJudgment(**data), token_usage
