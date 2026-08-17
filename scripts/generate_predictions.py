"""Deliverable #4: predictions for the 3 provided calls in the required schema."""

import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vta.pipeline import analyze_clip  # noqa: E402

CALLS = ["call_001.ogg", "call_002.ogg", "call_003.ogg"]


def main():
    predictions = {}
    timings = {}

    for name in CALLS:
        path = REPO_ROOT / name
        t0 = time.time()
        analysis = analyze_clip(path)
        dt = time.time() - t0
        predictions[name] = analysis.result.model_dump()
        timings[name] = {
            "duration_s": analysis.diagnostics.duration_s,
            "processing_s": analysis.diagnostics.processing_s,
            "wall_time_s": dt,
        }
        print(f"{name}: {json.dumps(predictions[name])}")

    out_dir = Path(__file__).resolve().parents[1]

    (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "result_json"])
        for name, result in predictions.items():
            writer.writerow([name, json.dumps(result)])

    (out_dir / "predictions_timing.json").write_text(json.dumps(timings, indent=2))

    print("\nWritten: predictions.json, predictions.csv, predictions_timing.json")


if __name__ == "__main__":
    main()
