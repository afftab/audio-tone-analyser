"""Cross-field validation using HarperValleyBank -- real simulated bank
call-center calls, not acted emotion speech.

Why this dataset: every other validation in this repo is either synthetic
degradation (validate_synthetic.py) or an acted emotion corpus (IEMOCAP,
RAVDESS) -- disclosed limitations on the /findings page. HarperValleyBank
(Gridspace + Stanford, github.com/cricketclub/gridspace-stanford-harper-valley,
public domain) is real two-sided phone recordings of simulated customer
service calls to a bank. It's the closest available domain match to
AutoAce's actual production calls of anything used so far.

Each session has:
  - data/audio/caller/{sid}.wav, data/audio/agent/{sid}.wav -- separate
    single-channel recordings of each side of the call, time-aligned.
  - data/transcript/{sid}.json -- per-turn transcript with offset_ms,
    duration_ms, speaker_role, and a 3-way emotion softmax
    (positive/neutral/negative) per turn.
  - data/metadata/{sid}.json -- labels.caller_mos, an MOS intelligibility
    score for the caller's audio.

This system analyzes a single mixed recording (like the 3 provided calls),
not two separate channels, so caller.wav + agent.wav are summed into one
clip before running the real production pipeline (analyze_clip) end to end.

Ground truths, all derived from the dataset's own real annotations, not
injected:
  - emotional_tone: caller-only turns' emotion softmax, duration-weighted,
    argmax -> positive/neutral/negative. Coarser than the brief's 5-class
    schema, so both ground truth and prediction collapse to the same
    3-way bucket. Disclosed as approximate, same as RAVDESS/IEMOCAP
    elsewhere in this repo.
  - speaker_overlap_present: real timestamp overlap between a caller turn
    and an agent turn (both channels share a clock) -- ground truth from
    the actual call, not a synthetic speaker-pair mix.

Two fields were validated in an earlier version of this script and then
removed from scoring after investigation showed the ground truth itself,
not the production model, was the problem (see TECHNICAL_MEMO.md and
/findings for the full writeup):

  - audio_quality: labels.caller_mos looked like a natural fit at first
    (1-5 MOS, banded to the brief's 3-tier enum), but HarperValleyBank's
    own README defines it as "how well could the caller be understood" --
    a transcriptionist's intelligibility rating, not the brief's specific
    definition of technical degradation (clipping/static/echo/low
    volume/robotic/packet loss). Checked: the acoustic features this
    system actually uses (clipping, SNR, bandwidth, raw loudness) show no
    separation at all between mos=5 and mos=3 clips in the sampled set
    (SNR 38-79dB and 0.00% clipping in *both* groups). Scoring against
    caller_mos would be fitting to the wrong target, so it's reported
    descriptively only, not as an accuracy/confusion-matrix claim.
  - long_silence_present: the production threshold is 12.0s, checked
    against Silero VAD's acoustic silence gap, not turn timestamps. A
    smaller diagnostic pass that ran VAD directly found the longest
    acoustic gap across the sample topped out at 9.9s -- below the
    threshold by construction. The transcript-timestamp gap reported
    below (max 12.1s in this run) is a *weaker* proxy for the same thing:
    turn annotations pad some silence around actual speech onset/offset,
    so they overstate true acoustic silence relative to what Silero VAD
    (what production actually uses) detects. Either way, HarperValleyBank
    is a clean scripted simulation with no genuinely broken/dead-air
    calls, so it can't supply the positive-class example needed to
    calibrate this threshold. Reported descriptively only.

Also not validated: emotional_intensity (no arousal/activation label in
this dataset) and background_noise_type/severity (no noise label exists).

A separate free check (150 sessions, transcripts only, no audio/API cost)
found zero negative-valence caller turns in the entire sample -- confirmed
structural, not a sampling fluke: HarperValleyBank is simulated and
non-adversarial, so it cannot validate upset/frustrated/distressed
detection at all, regardless of subset size. IEMOCAP remains the source
for that part of the schema.

Subset, not the full 1,446 sessions: this system's only paid stage (the
tone LLM call) runs once per clip, so validating on all 1,446 would cost
roughly 1,446x a single call's ~$0.0005-0.0015 (see TECHNICAL_MEMO.md's
cost table) for very little additional statistical power. VTA_HVB_N
(default 30) takes a seeded random sample instead.
"""

import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vta.pipeline import analyze_clip  # noqa: E402

RAW_BASE = (
    "https://raw.githubusercontent.com/cricketclub/"
    "gridspace-stanford-harper-valley/master/data"
)
TREE_URL = (
    "https://api.github.com/repos/cricketclub/"
    "gridspace-stanford-harper-valley/git/trees/master?recursive=1"
)
CACHE_DIR = REPO_ROOT / "data" / "cache" / "harpervalley_raw"  # gitignored (data/cache/)

N = int(os.environ.get("VTA_HVB_N", "30"))
SEED = 1234

TONE_BUCKET = {
    "satisfied": "positive",
    "neutral": "neutral",
    "frustrated": "negative",
    "upset": "negative",
    "distressed": "negative",
}
# Only positive/neutral are ever scored (see module docstring: this corpus
# has zero real negative-valence examples). "negative" stays in the bucket
# map so a prediction of it is still visible in the raw log, just excluded
# from the scored labels below so it doesn't drag macro-F1 down for a class
# with no ground truth to test against.
SCORED_TONE_LABELS = ["positive", "neutral"]
LONG_SILENCE_THRESHOLD_S = 12.0  # matches vta.vad.LONG_SILENCE_THRESHOLD_S


def _fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        dest.write_bytes(resp.read())


def list_all_sids() -> list[str]:
    with urllib.request.urlopen(TREE_URL) as resp:
        tree = json.load(resp)
    paths = [t["path"] for t in tree["tree"]]
    return sorted(
        p.split("/")[-1].removesuffix(".json")
        for p in paths
        if p.startswith("data/transcript/") and p.endswith(".json")
    )


def caller_tone_bucket(transcript: list[dict]) -> str | None:
    weights = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    total_ms = 0
    for turn in transcript:
        if turn["speaker_role"] != "caller":
            continue
        dur = turn["duration_ms"]
        emo = turn["emotion"]
        for k in weights:
            weights[k] += emo[k] * dur
        total_ms += dur
    if total_ms == 0:
        return None
    return max(weights, key=weights.get)


def turn_intervals(transcript: list[dict], role: str) -> list[tuple[int, int]]:
    return sorted(
        (t["offset_ms"], t["offset_ms"] + t["duration_ms"])
        for t in transcript
        if t["speaker_role"] == role
    )


def intervals_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    i = j = 0
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]
        if a_start < b_end and b_start < a_end:
            return True
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return False


def longest_merged_gap_ms(transcript: list[dict]) -> int:
    """Longest gap in the union of active-turn intervals across both
    channels. Descriptive only -- see module docstring for why this isn't
    scored as a pass/fail ground truth."""
    spans = sorted(
        (t["offset_ms"], t["offset_ms"] + t["duration_ms"]) for t in transcript
    )
    if not spans:
        return 0
    merged = [spans[0]]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    gaps = [b[0] - a[1] for a, b in zip(merged, merged[1:])]
    return max(gaps) if gaps else 0


def mix_clip(sid: str) -> Path:
    caller_path = CACHE_DIR / "audio" / "caller" / f"{sid}.wav"
    agent_path = CACHE_DIR / "audio" / "agent" / f"{sid}.wav"
    caller, sr_c = sf.read(caller_path, dtype="float32")
    agent, sr_a = sf.read(agent_path, dtype="float32")
    assert sr_c == sr_a, f"{sid}: sample rate mismatch {sr_c} vs {sr_a}"
    n = max(len(caller), len(agent))
    caller = np.pad(caller, (0, n - len(caller)))
    agent = np.pad(agent, (0, n - len(agent)))
    mixed = (caller + agent) / 2.0
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, mixed, sr_c)
    return Path(tmp.name)


def main():
    all_sids = list_all_sids()
    print(f"{len(all_sids)} total sessions available; sampling {N} (seed={SEED})")
    rng = __import__("random").Random(SEED)
    sids = rng.sample(all_sids, N)

    y_true_tone, y_pred_tone = [], []
    y_true_overlap, y_pred_overlap = [], []
    quality_rows, silence_rows = [], []
    skipped, errors = [], []

    for i, sid in enumerate(sids):
        try:
            _fetch(f"{RAW_BASE}/transcript/{sid}.json", CACHE_DIR / "transcript" / f"{sid}.json")
            _fetch(f"{RAW_BASE}/metadata/{sid}.json", CACHE_DIR / "metadata" / f"{sid}.json")
            _fetch(f"{RAW_BASE}/audio/caller/{sid}.wav", CACHE_DIR / "audio" / "caller" / f"{sid}.wav")
            _fetch(f"{RAW_BASE}/audio/agent/{sid}.wav", CACHE_DIR / "audio" / "agent" / f"{sid}.wav")

            transcript = json.loads((CACHE_DIR / "transcript" / f"{sid}.json").read_text())
            metadata = json.loads((CACHE_DIR / "metadata" / f"{sid}.json").read_text())

            exp_tone = caller_tone_bucket(transcript)
            caller_mos = metadata.get("labels", {}).get("caller_mos")
            if exp_tone is None or caller_mos is None:
                skipped.append(f"{sid}: missing caller turns or MOS label")
                continue

            caller_iv = turn_intervals(transcript, "caller")
            agent_iv = turn_intervals(transcript, "agent")
            exp_overlap = intervals_overlap(caller_iv, agent_iv)
            longest_gap_s = longest_merged_gap_ms(transcript) / 1000

            clip_path = mix_clip(sid)
            try:
                analysis = analyze_clip(clip_path)
            finally:
                clip_path.unlink(missing_ok=True)

            result = analysis.result
            pred_tone = TONE_BUCKET[result.emotional_tone]

            y_true_tone.append(exp_tone)
            y_pred_tone.append(pred_tone)
            y_true_overlap.append(exp_overlap)
            y_pred_overlap.append(result.speaker_overlap_present)
            quality_rows.append({"sid": sid, "caller_mos": caller_mos, "predicted": result.audio_quality})
            silence_rows.append({
                "sid": sid, "longest_turn_gap_s": longest_gap_s, "predicted": result.long_silence_present,
            })

            print(
                f"[{i+1}/{len(sids)}] {sid}: tone {pred_tone} (exp {exp_tone}), "
                f"quality {result.audio_quality} (mos {caller_mos}, not scored), "
                f"overlap {result.speaker_overlap_present} (exp {exp_overlap}), "
                f"silence {result.long_silence_present} (longest real gap {longest_gap_s:.1f}s, not scored)"
            )
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sid}: {type(e).__name__}: {e}")
            print(f"[{i+1}/{len(sids)}] {sid}: ERROR {e}")

    def score(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        return {
            "n": len(y_true),
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "confusion_matrix": {"labels": labels, "matrix": cm},
            "report": report,
        }

    n_quality = len(quality_rows)
    n_predicted_not_clear = sum(1 for r in quality_rows if r["predicted"] != "clear")
    max_real_gap_s = max((r["longest_turn_gap_s"] for r in silence_rows), default=0.0)
    n_predicted_silence = sum(1 for r in silence_rows if r["predicted"])

    results = {
        "dataset": "HarperValleyBank (gridspace-stanford-harper-valley)",
        "n_total_available": len(all_sids),
        "n_sampled": N,
        "seed": SEED,
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_errors": len(errors),
        "errors": errors,
        "emotional_tone_coarse": score(y_true_tone, y_pred_tone, SCORED_TONE_LABELS),
        "speaker_overlap_present": score(y_true_overlap, y_pred_overlap, [False, True]),
        "audio_quality_descriptive": {
            "note": "Not scored -- caller_mos measures intelligibility, not this "
                    "system's technical-degradation definition. See module docstring.",
            "n": n_quality,
            "n_predicted_not_clear": n_predicted_not_clear,
            "rows": quality_rows,
        },
        "long_silence_descriptive": {
            "note": "Not scored -- production threshold (12.0s) exceeds the longest "
                    "real gap in this dataset. See module docstring.",
            "n": len(silence_rows),
            "threshold_s": LONG_SILENCE_THRESHOLD_S,
            "max_real_gap_s": max_real_gap_s,
            "n_predicted_true": n_predicted_silence,
            "rows": silence_rows,
        },
    }

    out_path = REPO_ROOT / "validation_results_harpervalley.json"
    out_path.write_text(json.dumps(results, indent=2))

    for field in ["emotional_tone_coarse", "speaker_overlap_present"]:
        r = results[field]
        print(f"\n=== {field} (n={r['n']}) ===")
        print("accuracy:", r["accuracy"], "macro_f1:", r["macro_f1"])
        print("confusion matrix:", r["confusion_matrix"])
    print(f"\n=== audio_quality (descriptive, not scored) ===")
    print(f"n={n_quality}, predicted non-clear: {n_predicted_not_clear}")
    print(f"\n=== long_silence (descriptive, not scored) ===")
    print(f"n={len(silence_rows)}, max real gap {max_real_gap_s:.1f}s vs {LONG_SILENCE_THRESHOLD_S}s threshold, "
          f"predicted True: {n_predicted_silence}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
