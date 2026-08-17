"""Per-stage latency breakdown for the 3 provided calls."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vta.pipeline import analyze_clip  # noqa: E402

CALLS = ["call_001.ogg", "call_002.ogg", "call_003.ogg"]


def main():
    all_timings = {}
    for name in CALLS:
        analysis = analyze_clip(REPO_ROOT / name)
        all_timings[name] = {
            "duration_s": analysis.diagnostics.duration_s,
            "processing_s": analysis.diagnostics.processing_s,
            "stage_timings_s": analysis.diagnostics.stage_timings_s,
        }
        print(f"\n=== {name} (duration {analysis.diagnostics.duration_s:.1f}s) ===")
        for stage, t in sorted(
            analysis.diagnostics.stage_timings_s.items(), key=lambda x: -x[1]
        ):
            pct = 100 * t / analysis.diagnostics.processing_s
            print(f"  {stage:20s} {t:7.2f}s  ({pct:4.1f}%)")
        print(f"  {'TOTAL':20s} {analysis.diagnostics.processing_s:7.2f}s")

    out = Path(__file__).resolve().parents[1] / "stage_timings.json"
    out.write_text(json.dumps(all_timings, indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
