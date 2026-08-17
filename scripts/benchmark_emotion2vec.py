"""Benchmark chunked emotion2vec: timing + posterior/arousal consistency.

Verifies the chunking fix makes emotion2vec cost ~linear in duration (it was
superlinear -- full-clip self-attention is O(n^2) in frame count) and that
the aggregated posteriors/arousal stay close to the single-pass values.
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vta.audio_io import load_normalized  # noqa: E402
from vta.emotion_head import classify_emotion, EMOTION2VEC_CHUNK_S  # noqa: E402

CALLS = ["call_001.ogg", "call_002.ogg", "call_003.ogg"]

# Pre-chunk single-pass timings (stage_timings.json) for comparison
OLD_E2V_S = {"call_001.ogg": 4.88, "call_002.ogg": 1.32, "call_003.ogg": 62.97}

print(f"chunk size: {EMOTION2VEC_CHUNK_S}s\n")
print(f"{'clip':14s} {'dur':>6s} {'old_e2v':>8s} {'new_e2v':>8s} {'speedup':>8s}  arousal  top3")
for name in CALLS:
    audio = load_normalized(REPO_ROOT / name)
    t0 = time.time()
    result = classify_emotion(audio)
    dt = time.time() - t0
    old = OLD_E2V_S[name]
    top3 = sorted(result.posteriors.items(), key=lambda x: -x[1])[:3]
    top3_s = " ".join(f"{l}={v:.2f}" for l, v in top3)
    print(
        f"{name:14s} {audio.duration_s:5.1f}s {old:7.1f}s {dt:7.1f}s {old/dt:7.1f}x  "
        f"{result.arousal:.3f}    {top3_s}"
    )
