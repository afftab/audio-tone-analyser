"""background_noise_present/type/severity validation using real recorded
noise (ESC-50), mixed into real clean speech at controlled SNR.

Why this exists: validate_synthetic.py's noise validation uses a synthetic
noise generator. VOiCES (SRI/Lab41) would give genuinely objective ground
truth for this -- known distractor type and mic distance per recording,
not a survey label -- but it's distributed only as monolithic tarballs
(29.5GB minimum for the devkit, 448GB for the full release, no per-file
listing, no HF Hub mirror), so a small low-cost subset isn't achievable the
way it was for HarperValleyBank. This script is the practical middle
ground: real recorded noise (not a programmatic noise generator), mixed at
a controlled level we choose -- an upgrade over pure synthetic generation,
but still a constructed mixture, not field-recorded audio. Said plainly on
/findings, not oversold as equivalent to HarperValleyBank's validation.

Deliberately NOT used to validate audio_quality: the brief explicitly warns
against inferring background noise from audio quality or vice versa
(§2's evaluation note). Injecting noise is a valid ground-truth
construction for the three background_noise_* fields; it is not a valid
one for audio_quality, which is about technical degradation (clipping,
static, echo, low volume, robotic audio, packet loss) independent of
whatever's happening in the background. Scoring audio_quality against a
noise-injection experiment would repeat exactly the conflation the brief
warns about.

Clean speech base: 6 HarperValleyBank calls already confirmed clean by the
diagnostic work in TECHNICAL_MEMO.md §2f (caller_mos=5.0, 0% clipping,
38-79dB SNR, no separation from "impaired" calls on any acoustic feature --
i.e. these are as clean as this corpus gets). Reusing them here means zero
new download for the speech side.

Noise source: ESC-50 (Piczak, CC BY-NC 3.0, github.com/karolpiczak/ESC-50),
2,000 individually-hosted 5s real environmental recordings -- ordinary
per-file GitHub access, no monolithic-archive problem. Six categories
chosen for a clean, unambiguous expected mapping onto the canonical noise
categories vta.events_panns already recognizes (AudioSet-527 vocabulary):
    keyboard_typing  -> "keyboard typing"
    vacuum_cleaner   -> "mechanical noise"
    wind             -> "wind"
    car_horn         -> "road noise"
    siren            -> "alarm or siren"
    dog              -> "dog barking"

This is entirely local computation (PANNs on CPU) -- zero OpenAI cost,
unlike the tone-LLM validations elsewhere in this repo.
"""

import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vta.audio_io import NormalizedAudio, load_normalized  # noqa: E402
from vta.dsp_features import compute_dsp_features  # noqa: E402
from vta.events_panns import detect_background_noise  # noqa: E402

ESC50_RAW_BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
ESC50_CACHE = REPO_ROOT / "data" / "cache" / "esc50_raw"
HVB_CACHE = REPO_ROOT / "data" / "cache" / "harpervalley_raw" / "audio" / "caller"

# Confirmed clean (mos=5.0) HarperValleyBank caller clips from the §2f
# diagnostic pass -- already cached locally from that work.
CLEAN_SIDS = [
    "a3980352013548b8", "1e47d21492244b04", "23897136ed8241e9",
    "063ea0f1abb143c1", "ebdc2700b6524a53", "e02b6dc778204aff",
]

# category -> (expected canonical noise type, one ESC-50 filename)
NOISE_CATEGORIES = {
    "keyboard_typing": "keyboard typing",
    "vacuum_cleaner": "mechanical noise",
    "wind": "wind",
    "car_horn": "road noise",
    "siren": "alarm or siren",
    "dog": "dog barking",
}

SEVERITY_TARGET_SNR_DB = {"low": 18.0, "medium": 8.0, "high": -2.0}
SEVERITY_LABELS = ["none", "low", "medium", "high"]


def _fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        dest.write_bytes(resp.read())


def esc50_index() -> dict[str, list[str]]:
    """category -> list of filenames, from ESC-50's own metadata CSV."""
    import csv
    path = ESC50_CACHE / "esc50.csv"
    _fetch(f"{ESC50_RAW_BASE}/meta/esc50.csv", path)
    by_cat: dict[str, list[str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_cat.setdefault(row["category"], []).append(row["filename"])
    return by_cat


def fetch_noise_clip(filename: str) -> Path:
    dest = ESC50_CACHE / "audio" / filename
    _fetch(f"{ESC50_RAW_BASE}/audio/{filename}", dest)
    return dest


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def mix_at_snr(clean: NormalizedAudio, noise_path: Path, target_snr_db: float) -> NormalizedAudio:
    noise = load_normalized(noise_path)
    n = len(clean.samples)
    reps = int(np.ceil(n / len(noise.samples)))
    noise_tiled = np.tile(noise.samples, reps)[:n]

    signal_rms = _rms(clean.samples)
    noise_rms = _rms(noise_tiled)
    desired_noise_rms = signal_rms / (10 ** (target_snr_db / 20))
    scale = desired_noise_rms / noise_rms
    mixed = clean.samples + noise_tiled * scale

    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99  # avoid clipping the mix itself

    return NormalizedAudio(
        samples=mixed.astype(np.float32), sr=clean.sr,
        duration_s=clean.duration_s, source_path=clean.source_path,
    )


def main():
    by_cat = esc50_index()
    noise_files = {cat: fetch_noise_clip(by_cat[cat][0]) for cat in NOISE_CATEGORIES}

    y_true_present, y_pred_present = [], []
    y_true_severity, y_pred_severity = [], []
    type_rows = []
    rows_out = []

    combos = [(sid, cat, sev) for sid in CLEAN_SIDS for cat in NOISE_CATEGORIES
              for sev in ["none", "low", "medium", "high"]]
    # "none" only needs to be tested once per clean clip, not once per category
    # (it's the same unmodified clip regardless of which category loop it's in).
    seen_none = set()
    filtered = []
    for sid, cat, sev in combos:
        if sev == "none":
            if sid in seen_none:
                continue
            seen_none.add(sid)
        filtered.append((sid, cat, sev))

    for i, (sid, cat, sev) in enumerate(filtered):
        clean_path = HVB_CACHE / f"{sid}.wav"
        clean = load_normalized(clean_path)

        if sev == "none":
            mixed = clean
        else:
            mixed = mix_at_snr(clean, noise_files[cat], SEVERITY_TARGET_SNR_DB[sev])

        feats = compute_dsp_features(mixed)
        noise_result = detect_background_noise(
            mixed, feats.telephony_band_ratio, feats.occupied_bandwidth_hz
        )

        exp_present = sev != "none"
        y_true_present.append(exp_present)
        y_pred_present.append(noise_result.background_noise_present)
        y_true_severity.append(sev)
        y_pred_severity.append(noise_result.background_noise_severity)
        if exp_present:
            type_rows.append({
                "expected": NOISE_CATEGORIES[cat], "predicted": noise_result.background_noise_type,
            })

        rows_out.append({
            "sid": sid, "category": cat, "severity": sev,
            "predicted_present": noise_result.background_noise_present,
            "predicted_severity": noise_result.background_noise_severity,
            "predicted_type": noise_result.background_noise_type,
        })
        print(f"[{i+1}/{len(filtered)}] {sid}+{cat}@{sev}: "
              f"present={noise_result.background_noise_present} (exp {exp_present}), "
              f"severity={noise_result.background_noise_severity} (exp {sev}), "
              f"type={noise_result.background_noise_type!r} (exp {NOISE_CATEGORIES.get(cat, '')!r})")

    def score(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        return {
            "n": len(y_true), "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "confusion_matrix": {"labels": labels, "matrix": cm}, "report": report,
        }

    type_matches = sum(1 for r in type_rows if r["predicted"] == r["expected"])

    results = {
        "dataset": "ESC-50 real noise mixed into confirmed-clean HarperValleyBank speech",
        "n_combos": len(filtered),
        "background_noise_present": score(y_true_present, y_pred_present, [False, True]),
        "background_noise_severity": score(y_true_severity, y_pred_severity, SEVERITY_LABELS),
        "background_noise_type": {
            "n": len(type_rows), "n_exact_match": type_matches,
            "exact_match_rate": type_matches / len(type_rows) if type_rows else None,
            "rows": type_rows,
        },
        "rows": rows_out,
    }

    out_path = REPO_ROOT / "validation_results_esc50_noise.json"
    out_path.write_text(json.dumps(results, indent=2))

    for field in ["background_noise_present", "background_noise_severity"]:
        r = results[field]
        print(f"\n=== {field} (n={r['n']}) ===")
        print("accuracy:", r["accuracy"], "macro_f1:", r["macro_f1"])
        print("confusion matrix:", r["confusion_matrix"])
    print(f"\n=== background_noise_type ===")
    print(f"exact match: {type_matches}/{len(type_rows)} = {results['background_noise_type']['exact_match_rate']:.2f}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
