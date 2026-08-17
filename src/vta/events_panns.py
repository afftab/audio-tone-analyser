"""PANNs (CNN14, AudioSet-527) acoustic event tagging -> background noise fields.

Resolved independently of the LLM tone head per PLAN.md's architecture split:
background noise must never be inferred from poor audio quality or from the
LLM's read of the transcript, per the brief's explicit warning against
conflating these signals.
"""

import re
import warnings
from dataclasses import dataclass
from functools import lru_cache

import librosa
import numpy as np

from vta.audio_io import NormalizedAudio

PANNS_SR = 32_000

PRESENCE_THRESHOLD = 0.08  # min matched-category probability mass to call noise "present"
BAND_DEVIATION_GATE = 0.08  # min fraction of energy outside 300-3400Hz to corroborate presence
STRONG_MASS_OVERRIDE = 0.3  # PANNs mass strong enough to flag presence even in-band

# First substring match wins; phrasing mirrors the brief's own examples.
# Only labels matching a category count toward noise presence/severity.
CANONICAL_NOISE_CATEGORIES: list[tuple[str, list[str]]] = [
    ("keyboard typing", ["typing", "typewriter", "computer keyboard"]),
    ("television", ["television", "radio"]),
    (
        "static or interference",
        ["static", "white noise", "pink noise", "distortion", "mains hum", "cacophony"],
    ),
    ("music", ["music", "singing", "musical instrument"]),
    (
        "road noise",
        [
            "vehicle",
            "car",
            "traffic",
            "truck",
            "motorcycle",
            "train",
            "railroad",
            "bus",
            "aircraft",
        ],
    ),
    ("wind", ["wind"]),
    (
        "mechanical noise",
        [
            "mechanical fan",
            "air conditioning",
            "machine",
            "hum",
            "engine",
            "motor",
            "vibration",
            "drill",
            "vacuum cleaner",
            "sewing machine",
            "printer",
            "cash register",
            "gears",
            "pulleys",
        ],
    ),
    (
        "office chatter",
        ["chatter", "crowd", "hubbub, speech noise, speech babble", "children playing"],
    ),
    ("dog barking", ["dog", "bark"]),
    ("door or knocking", ["door", "knock", "doorbell"]),
    ("alarm or siren", ["siren", "alarm", "buzzer"]),
]

# PANNs fires on the phone line itself ("Dial tone"), not the environment --
# kept out of the whitelist above, as is speech.

# Keyword collisions ("Machine gun" contains "machine") that aren't plausible
# noise in a support call.
EXCLUDED_LABELS = {
    "Machine gun",
    "Gunshot, gunfire",
    "Explosion",
    "Fusillade",
    "Artillery fire",
    "Cap gun",
    "Fireworks",
    "Firecracker",
}

_CATEGORY_PATTERNS = [
    (canonical, [re.compile(r"(?<!\w)" + re.escape(kw) + r"(?!\w)") for kw in keywords])
    for canonical, keywords in CANONICAL_NOISE_CATEGORIES
]


@dataclass
class NoiseResult:
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: str  # none | low | medium | high
    top_labels: list[tuple[str, float]]  # (label, prob) for the top matched-category tags
    noise_prob_mass: float


@lru_cache(maxsize=1)
def _load_model():
    """PANNs CNN14 audio event tagger. (A CED-mini backend was evaluated and
    removed: its raw logits needed every threshold here rescaled. See
    TECHNICAL_MEMO.md.)"""
    from panns_inference import AudioTagging

    return AudioTagging(checkpoint_path=None, device="cpu")


def _matched_category(label: str) -> str | None:
    if label in EXCLUDED_LABELS:
        return None
    lower = label.lower()
    for canonical, patterns in _CATEGORY_PATTERNS:
        if any(p.search(lower) for p in patterns):
            return canonical
    return None


def detect_background_noise(
    audio: NormalizedAudio,
    telephony_band_ratio: float,
    occupied_bandwidth_hz: float = 0.0,
) -> NoiseResult:
    """telephony_band_ratio: fraction of spectral energy inside 300-3400 Hz
    (from DSPFeatures). PANNs alone can't separate telephony clips (no
    channel prior), so out-of-band energy corroborates presence. An snr_db
    parameter was removed: it has no dynamic range on this audio, and
    severity built on it validated below chance.
    """
    model = _load_model()

    y32 = librosa.resample(audio.samples, orig_sr=audio.sr, target_sr=PANNS_SR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clipwise_output, _ = model.inference(y32[None, :])
    probs = clipwise_output[0]  # (527,)

    from panns_inference.config import labels as tagger_labels

    categories = [_matched_category(lbl) for lbl in tagger_labels]
    noise_mask = np.array([c is not None for c in categories])
    noise_prob_mass = float(probs[noise_mask].sum())

    ranked_idx = np.argsort(-probs)
    top_matched = [
        (tagger_labels[i], float(probs[i])) for i in ranked_idx if noise_mask[i]
    ][:5]

    band_deviation = 1.0 - telephony_band_ratio  # energy fraction outside the voice band
    present = noise_prob_mass > PRESENCE_THRESHOLD and (
        band_deviation > BAND_DEVIATION_GATE or noise_prob_mass > STRONG_MASS_OVERRIDE
    )
    if not present:
        return NoiseResult(
            background_noise_present=False,
            background_noise_type="",
            background_noise_severity="none",
            top_labels=top_matched,
            noise_prob_mass=noise_prob_mass,
        )

    # Summed across a category's labels, not the single highest: a TV splits
    # across "Radio" and "Television", each below "Music", but their sum wins.
    category_mass: dict[str, float] = {}
    for i, cat in enumerate(categories):
        if cat is not None:
            category_mass[cat] = category_mass.get(cat, 0.0) + float(probs[i])
    noise_type = max(category_mass, key=category_mass.get) if category_mass else ""

    # Banded on band_deviation, with occupied_bandwidth_hz as a second
    # high-band trigger: band_deviation saturates near 0.52 even at SNR 0dB,
    # while real telephony never exceeds 3.4kHz. snr_db was tried, worse.
    if band_deviation > 0.55 or occupied_bandwidth_hz > 4000.0:
        severity = "high"
    elif band_deviation > 0.10:
        severity = "medium"
    else:
        severity = "low"

    return NoiseResult(
        background_noise_present=True,
        background_noise_type=noise_type,
        background_noise_severity=severity,
        top_labels=top_matched,
        noise_prob_mass=noise_prob_mass,
    )
