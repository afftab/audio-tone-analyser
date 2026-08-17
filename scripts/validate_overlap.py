"""Speaker-overlap validation using diverse IEMOCAP speaker pairs.

Replaces the broken overlap generator in validate_synthetic.py, which summed
call_001 + call_002 (acoustically similar call-center voices from the same
recording environment). Direct inspection showed pyannote/segmentation-3.0
output max_active_speakers=1.0 on that construction -- the model never saw a
second speaker, because summing two similar voices doesn't produce the
spectrally-distinct two-source signal the segmentation head was trained on.

Using diverse speakers (different gender, different IEMOCAP sessions, hence
different recording chains and voice timbres) produces summed audio the model
DOES read as overlapping: max_active reaches 2.0, overlap is detected. This
script validates the detector with that fixed generator at meaningful scale.

Controls (expected False):
  - single-speaker IEMOCAP clips (never two sources)
  - sequential pairs (speaker A fully, then speaker B, zero co-temporal)
Cases (expected True):
  - 2s and 4s overlap (summed overlap region between two distinct speakers)
"""
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vta.audio_io import NormalizedAudio  # noqa: E402
from vta.overlap_pyannote import detect_overlap  # noqa: E402

SR = 16_000
HF_DATASET = "Ar4ikov/iemocap_audio_text"
SHARD = "data/train-00000-of-00003-9213b91aae6dc76d.parquet"


def _mk(samples, tag):
    return NormalizedAudio(samples.astype(np.float32), SR, len(samples) / SR, Path(tag))


def _norm_rms(x, target=0.3):
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    return x / rms * target


def _load_speaker_clips():
    """Collect short clips from distinct speakers across sessions."""
    path = hf_hub_download(HF_DATASET, SHARD, repo_type="dataset")
    rows = pq.read_table(path, columns=["titre", "audio"]).to_pylist()
    males, females = [], []
    for r in rows:
        samples, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if sr != SR or len(samples) < SR * 4 or len(samples) > SR * 8:
            continue
        if "_M0" in r["titre"] and len(males) < 20:
            males.append(samples)
        if "_F0" in r["titre"] and len(females) < 20:
            females.append(samples)
    return males, females


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


def main():
    # Clear any cached inference (STEP_S may differ from a previous run)
    from vta.overlap_pyannote import _load_inference
    _load_inference.cache_clear()

    males, females = _load_speaker_clips()
    print(f"Loaded {len(males)} male, {len(females)} female clips")
    n_pairs = min(len(males), len(females), 15)

    y_true, y_pred = [], []
    detail = []

    # Controls: single speaker (expected False)
    for i in range(min(10, n_pairs)):
        for spk in [males[i], females[i]]:
            audio = _mk(spk, f"single_{i}")
            res = detect_overlap(audio)
            y_true.append(False); y_pred.append(res.speaker_overlap_present)
            detail.append(("single", res.total_overlap_s, res.speaker_overlap_present))

    # Controls: sequential, no overlap (expected False)
    for i in range(n_pairs):
        mixed = _sequential(males[i], females[i])
        audio = _mk(mixed, f"seq_{i}")
        res = detect_overlap(audio)
        y_true.append(False); y_pred.append(res.speaker_overlap_present)
        detail.append(("sequential", res.total_overlap_s, res.speaker_overlap_present))

    # Cases: 2s overlap (expected True)
    for i in range(n_pairs):
        mixed = _overlap(males[i], females[i], 2.0)
        audio = _mk(mixed, f"ov2_{i}")
        res = detect_overlap(audio)
        y_true.append(True); y_pred.append(res.speaker_overlap_present)
        detail.append(("overlap_2s", res.total_overlap_s, res.speaker_overlap_present))

    # Cases: 4s overlap (expected True)
    for i in range(n_pairs):
        mixed = _overlap(males[i], females[i], 4.0)
        audio = _mk(mixed, f"ov4_{i}")
        res = detect_overlap(audio)
        y_true.append(True); y_pred.append(res.speaker_overlap_present)
        detail.append(("overlap_4s", res.total_overlap_s, res.speaker_overlap_present))

    labels = [False, True]
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=["false", "true"],
        output_dict=True, zero_division=0,
    )
    accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))

    print(f"\nn={len(y_true)} (controls={sum(1 for t in y_true if not t)}, overlap={sum(y_true)})")
    print(f"accuracy: {accuracy:.3f}  macro_f1: {report['macro avg']['f1-score']:.3f}")
    print(f"confusion (false,true): {cm}")
    print(f"  false precision={report['false']['precision']:.3f} recall={report['false']['recall']:.3f}")
    print(f"  true  precision={report['true']['precision']:.3f} recall={report['true']['recall']:.3f}")
    print("\nDetected overlap duration by condition:")
    from collections import defaultdict
    by_cond = defaultdict(list)
    for cond, dur, present in detail:
        by_cond[cond].append((dur, present))
    for cond in ["single", "sequential", "overlap_2s", "overlap_4s"]:
        durs = [d for d, _ in by_cond[cond]]
        hits = sum(1 for _, p in by_cond[cond] if p)
        print(f"  {cond:14s}: n={len(durs):2d} detected={hits:2d}/{len(durs):2d}  "
              f"mean_overlap_dur={np.mean(durs):.2f}s")

    results = {
        "method": "diverse IEMOCAP M+F speaker pairs (replaces similar call-center summation)",
        "n_samples": len(y_true),
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "confusion_matrix": {"labels": ["false", "true"], "matrix": cm},
        "report": report,
    }
    out = Path(__file__).resolve().parents[1] / "validation_results_overlap.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {out}")


import json  # noqa: E402

if __name__ == "__main__":
    main()
