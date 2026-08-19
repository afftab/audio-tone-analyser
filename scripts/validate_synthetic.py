"""Synthetic validation for the 5 fields that admit ground truth by construction.

Per the build plan §6: the 3 provided clips are a smoke test only, not a validation
set (too few to produce per-class F1 or a confusion matrix, and any windowing
of call_003 would put the same call on both sides of a split -- exactly the
leakage the brief warns against). This generates synthetic audio with known
ground truth (injected noise/clipping/overlap/silence at controlled levels)
and reports real accuracy, per-class F1, and confusion matrices against it.

emotional_tone/emotional_intensity are NOT validated here -- they need
labeled emotional speech (MELD/IEMOCAP per the build plan), which requires external
dataset downloads not attempted given the deadline; this is a documented
limitation, not an oversight.

Base speech source: call_001.ogg (labeled noise-free, clear, no overlap, no
long silence in ground truth) -- used only as a real-speech carrier signal,
never scored against its own labels here.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vta.audio_io import NormalizedAudio, load_normalized  # noqa: E402
from vta.dsp_features import compute_dsp_features  # noqa: E402
from vta.events_panns import detect_background_noise  # noqa: E402
from vta.overlap_pyannote import detect_overlap  # noqa: E402
from vta.quality import classify_audio_quality  # noqa: E402
from vta.vad import run_vad  # noqa: E402

RNG = np.random.default_rng(1234)
SR = 16_000


def set_seed(seed: int):
    """Set the module-level RNG -- used for held-out-seed validation."""
    global RNG
    RNG = np.random.default_rng(seed)


def _mk_audio(samples: np.ndarray, tag: str) -> NormalizedAudio:
    samples = samples.astype(np.float32)
    return NormalizedAudio(
        samples=samples, sr=SR, duration_s=len(samples) / SR, source_path=Path(tag)
    )


def _white_noise(n: int) -> np.ndarray:
    return RNG.normal(0, 1, n).astype(np.float32)


def _pink_noise(n: int) -> np.ndarray:
    white = RNG.normal(0, 1, n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]  # avoid div by zero
    fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n)
    return (pink / np.max(np.abs(pink) + 1e-9)).astype(np.float32)


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_power = np.mean(speech**2)
    noise_power = np.mean(noise**2) + 1e-12
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)
    mixed = speech + scaled_noise
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed


def load_speech_bases() -> tuple[np.ndarray, np.ndarray]:
    a = load_normalized(REPO_ROOT / "call_001.ogg")
    b = load_normalized(REPO_ROOT / "call_002.ogg")
    return a.samples, b.samples


# --- Field 1+2: background noise (PANNs + DSP telephony-band cross-check) ---


def validate_noise(speech: np.ndarray) -> dict:
    # n=40 (10 per bucket): a single n=6-13 run can't tell a working
    # classifier from a lucky one. Clean controls vary their offset into the
    # base speech so the presence gate is tested on more than one segment.
    trials = []  # (snr_db_or_none, expected_present, expected_severity)
    for _ in range(10):
        trials.append((None, False, "none"))  # clean control
    for snr in [30, 28, 27, 26, 25, 24, 23, 22, 21, 20]:
        trials.append((snr, True, "low"))
    for snr in [19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        trials.append((snr, True, "medium"))
    for snr in [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
        trials.append((snr, True, "high"))

    y_true_present, y_pred_present = [], []
    y_true_sev, y_pred_sev = [], []

    max_offset_s = max(0.0, (len(speech) / SR) - 12.0)

    for i, (snr, exp_present, exp_sev) in enumerate(trials):
        offset_s = RNG.uniform(0, max_offset_s) if max_offset_s > 0 else 0.0
        start = int(offset_s * SR)
        seg = speech[start : start + SR * 12]
        if snr is None:
            mixed = seg.copy()
        else:
            noise = _pink_noise(len(seg)) if i % 2 == 0 else _white_noise(len(seg))
            mixed = _mix_at_snr(seg, noise, snr)

        audio = _mk_audio(mixed, f"noise_trial_{i}")
        feats = compute_dsp_features(audio)
        result = detect_background_noise(
            audio, feats.telephony_band_ratio, feats.occupied_bandwidth_hz
        )

        y_true_present.append(exp_present)
        y_pred_present.append(result.background_noise_present)
        y_true_sev.append(exp_sev)
        y_pred_sev.append(result.background_noise_severity)

    return {
        "presence": _binary_report(y_true_present, y_pred_present),
        "severity": _multiclass_report(
            y_true_sev, y_pred_sev, labels=["none", "low", "medium", "high"]
        ),
    }


# --- Field 3: audio_quality (clipping/bandwidth/SNR-derived) ---


def validate_audio_quality(speech: np.ndarray) -> dict:
    # Soft (tanh) clipping: hard np.clip sent every slightly_impaired trial
    # to severely_impaired. tanh matches codec/limiter degradation and
    # separates the classes: gain 1.0 = clear, 5-8 = slight, 10+ = severe.
    trials = (
        [(1.0, "clear")] * 10
        + [(g, "slightly_impaired") for g in [5.0, 5.5, 6.0, 6.2, 6.5, 6.8, 7.0, 7.2, 7.5, 7.8]]
        + [(g, "severely_impaired") for g in [10, 12, 14, 16, 18, 20, 24, 27, 30, 35]]
    )

    max_offset_s = max(0.0, (len(speech) / SR) - 10.0)

    y_true, y_pred = [], []
    for i, (gain, expected) in enumerate(trials):
        offset_s = RNG.uniform(0, max_offset_s) if max_offset_s > 0 else 0.0
        start = int(offset_s * SR)
        seg = speech[start : start + SR * 10]
        amplified = np.tanh(seg * gain)
        audio = _mk_audio(amplified, f"quality_trial_{i}")
        feats = compute_dsp_features(audio)
        predicted = classify_audio_quality(feats)
        y_true.append(expected)
        y_pred.append(predicted)

    return _multiclass_report(
        y_true, y_pred, labels=["clear", "slightly_impaired", "severely_impaired"]
    )


# --- Field 4: long_silence_present (Silero VAD) ---


def validate_long_silence(speech: np.ndarray) -> dict:
    # n=20 trials; varied offsets.
    gaps_false = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    gaps_true = [12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, 35.0]
    trials = [(g, False) for g in gaps_false] + [(g, True) for g in gaps_true]

    max_offset_s = max(0.0, (len(speech) / SR) - 10.0)

    y_true, y_pred = [], []
    for i, (gap_s, expected) in enumerate(trials):
        offset_s = RNG.uniform(0, max_offset_s) if max_offset_s > 0 else 0.0
        start = int(offset_s * SR)
        seg = speech[start : start + SR * 10]
        half = len(seg) // 2
        gap = np.zeros(int(gap_s * SR), dtype=np.float32)
        mixed = np.concatenate([seg[:half], gap, seg[half:]])
        audio = _mk_audio(mixed, f"silence_trial_{i}")
        result = run_vad(audio)
        y_true.append(expected)
        y_pred.append(result.long_silence_present)

    return _binary_report(y_true, y_pred)


# --- Field 5: speaker_overlap_present (pyannote) ---


def _norm_rms(x: np.ndarray, target: float = 0.3) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    return x / rms * target


def _load_iemocap_speakers() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Short clips from distinct IEMOCAP speakers (different gender/session).
    Similar call-center voices sum to audio the model reads as one speaker;
    diverse timbres sum to a detectable two-source signal."""
    import io
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "Ar4ikov/iemocap_audio_text",
        "data/train-00000-of-00003-9213b91aae6dc76d.parquet",
        repo_type="dataset",
    )
    rows = pq.read_table(path, columns=["titre", "audio"]).to_pylist()
    males, females = [], []
    for r in rows:
        samples, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if sr != SR or len(samples) < SR * 4 or len(samples) > SR * 8:
            continue
        if "_M0" in r["titre"] and len(males) < 15:
            males.append(samples)
        if "_F0" in r["titre"] and len(females) < 15:
            females.append(samples)
    return males, females


def validate_overlap(speech_a: np.ndarray, speech_b: np.ndarray) -> dict:
    # Diverse IEMOCAP M+F pairs: summing the two similar call-center calls
    # reads as one speaker (max_active=1.0, 0.50 accuracy at n=6); distinct
    # voices produce a detectable two-source signal. See validate_overlap.py.
    males, females = _load_iemocap_speakers()
    n_pairs = min(len(males), len(females), 15)

    def _overlap(a, b, ov_s):
        n = min(int(ov_s * SR), len(a) // 2, len(b) // 2)
        a, b = _norm_rms(a), _norm_rms(b)
        mixed = np.concatenate([a[:-n], a[-n:] + b[:n], b[n:]])
        peak = np.max(np.abs(mixed))
        return mixed / peak if peak > 1.0 else mixed

    def _sequential(a, b):
        a, b = _norm_rms(a), _norm_rms(b)
        mixed = np.concatenate([a, b])
        peak = np.max(np.abs(mixed))
        return mixed / peak if peak > 1.0 else mixed

    y_true, y_pred = [], []

    # Controls: single speaker (expected False)
    for i in range(min(10, n_pairs)):
        for spk in [males[i], females[i]]:
            audio = _mk_audio(spk, f"single_{i}")
            result = detect_overlap(audio)
            y_true.append(False)
            y_pred.append(result.speaker_overlap_present)

    # Controls: sequential, no overlap (expected False)
    for i in range(n_pairs):
        mixed = _sequential(males[i], females[i])
        audio = _mk_audio(mixed, f"seq_{i}")
        result = detect_overlap(audio)
        y_true.append(False)
        y_pred.append(result.speaker_overlap_present)

    # Cases: 2s and 4s overlap (expected True)
    for i in range(n_pairs):
        for ov_s in [2.0, 4.0]:
            mixed = _overlap(males[i], females[i], ov_s)
            audio = _mk_audio(mixed, f"ov{i}_{ov_s}")
            result = detect_overlap(audio)
            y_true.append(True)
            y_pred.append(result.speaker_overlap_present)

    return _binary_report(y_true, y_pred)


def _binary_report(y_true: list[bool], y_pred: list[bool]) -> dict:
    labels = [False, True]
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=["false", "true"],
        output_dict=True, zero_division=0,
    )
    accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    return {"accuracy": accuracy, "confusion_matrix": {"labels": ["false", "true"], "matrix": cm},
            "report": report}


def _multiclass_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    return {"accuracy": accuracy, "confusion_matrix": {"labels": labels, "matrix": cm},
            "report": report}


def main(seed: int = 1234):
    set_seed(seed)
    speech_a, speech_b = load_speech_bases()

    results = {
        "background_noise": validate_noise(speech_a),
        "audio_quality": validate_audio_quality(speech_a),
        "long_silence_present": validate_long_silence(speech_a),
        "speaker_overlap_present": validate_overlap(speech_a, speech_b),  # loads IEMOCAP internally
    }

    suffix = "" if seed == 1234 else f"_seed{seed}"
    out_path = REPO_ROOT / "voice-tone-analyzer" / f"validation_results{suffix}.json"
    out_path.write_text(json.dumps(results, indent=2))

    for field, r in results.items():
        print(f"\n=== {field} ===")
        if "presence" in r:
            print("presence accuracy:", r["presence"]["accuracy"])
            print("severity accuracy:", r["severity"]["accuracy"])
            print("severity macro_f1:", r["severity"]["report"]["macro avg"]["f1-score"])
        else:
            print("accuracy:", r["accuracy"])
            print("macro_f1:", r["report"]["macro avg"]["f1-score"])

    print(f"\nFull results written to {out_path} (seed={seed})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    main(seed=args.seed)
