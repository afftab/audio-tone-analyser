"""Real emotional_tone/emotional_intensity validation using RAVDESS.

The brief's own 3 provided clips give zero quantitative signal for these two
fields (validate_synthetic.py's synthetic ground-truth-by-construction
approach doesn't work for emotion -- there's no way to "inject" a target
emotion the way noise/clipping/silence can be injected). RAVDESS (Ryerson
Audio-Visual Database of Emotional Speech and Song) is real labeled
emotional speech, permissively licensed, downloaded via the HF Hub mirror
birgermoell/ravdess.

Mapping RAVDESS's 8 acted emotions onto the brief's 5-class schema is
inherently approximate -- the two taxonomies were not designed to align
(RAVDESS is general affect; the brief's schema is customer-service-specific,
e.g. distinguishing "frustrated" from "upset" by degree of anger, which
RAVDESS's "angry" doesn't itself subdivide). This is disclosed, not hidden:
treat this as a real but imperfect-by-construction validation, better than
zero measurement but not a substitute for labeled call-center data.

Filename format: {modality}-{vocal_channel}-{emotion}-{intensity}-
{statement}-{repetition}-{actor}.wav
emotion: 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised
intensity: 01=normal 02=strong
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from vta.audio_io import load_normalized  # noqa: E402
from vta.dsp_features import compute_dsp_features  # noqa: E402
from vta.emotion_head import classify_emotion  # noqa: E402
from vta.asr import transcribe  # noqa: E402
from vta.tone_llm import ProsodySummary, classify_tone  # noqa: E402

RAVDESS_ROOT = Path(
    "/Users/aftab/.cache/huggingface/hub/datasets--birgermoell--ravdess/"
    "snapshots/801611132d134432b0344b04c1545da4fdc93e17"
)

# Approximate mapping, disclosed as such -- see module docstring.
EMOTION_TO_TONE = {
    "01": "neutral",  # neutral
    "02": "neutral",  # calm
    "03": "satisfied",  # happy
    "04": "frustrated",  # sad -- imperfect, no direct RAVDESS analogue
    "05": "upset",  # angry
    "06": "distressed",  # fearful
    "07": "frustrated",  # disgust -- imperfect, closest non-angry negative
    "08": None,  # surprised -- ambiguous valence, excluded from validation
}

TONE_LABELS = ["neutral", "satisfied", "frustrated", "upset", "distressed"]
INTENSITY_LABELS = ["low", "medium", "high"]


def expected_intensity(emotion_code: str, intensity_code: str) -> str:
    if emotion_code == "01":  # neutral has no "strong" variant in RAVDESS
        return "low"
    return "medium" if intensity_code == "01" else "high"


def iter_actors(actor_dirs: list[str]):
    for actor_dir in actor_dirs:
        d = RAVDESS_ROOT / actor_dir
        if not d.is_dir():
            continue
        for wav in sorted(d.glob("*.wav")):
            parts = wav.stem.split("-")
            if len(parts) != 7:
                continue
            _, vocal_channel, emotion, intensity, _, _, _ = parts
            if vocal_channel != "01":  # speech only, not song
                continue
            expected_tone = EMOTION_TO_TONE.get(emotion)
            if expected_tone is None:
                continue
            yield wav, expected_tone, expected_intensity(emotion, intensity)


def main(actor_dirs: list[str]):
    y_true_tone, y_pred_tone = [], []
    y_true_intensity, y_pred_intensity = [], []
    errors = []

    samples = list(iter_actors(actor_dirs))
    print(f"Running {len(samples)} RAVDESS samples from {actor_dirs}...")

    for i, (wav_path, exp_tone, exp_intensity) in enumerate(samples):
        try:
            audio = load_normalized(wav_path)
            feats = compute_dsp_features(audio)
            asr_result = transcribe(audio)
            emotion_result = classify_emotion(audio)

            prosody = ProsodySummary(
                pitch_mean_hz=feats.pitch_mean_hz,
                pitch_std_hz=feats.pitch_std_hz,
                energy_mean_db=feats.energy_mean_db,
                energy_std_db=feats.energy_std_db,
                energy_dynamic_range_db=feats.energy_dynamic_range_db,
                voiced_ratio=feats.voiced_ratio,
                speaking_rate_wpm=asr_result.speaking_rate_wpm,
                energy_contour_db=feats.energy_contour_db,
                arousal=emotion_result.arousal,
                emotion_posteriors=emotion_result.posteriors,
            )
            judgment, _usage = classify_tone(asr_result.transcript, prosody)

            intensity_rank = {"low": 0, "medium": 1, "high": 2}
            intensity = judgment.emotional_intensity
            if emotion_result.arousal > 0.6 and intensity_rank[intensity] < intensity_rank["high"]:
                intensity = "high"
            elif emotion_result.arousal > 0.35 and intensity_rank[intensity] < intensity_rank["medium"]:
                intensity = "medium"

            y_true_tone.append(exp_tone)
            y_pred_tone.append(judgment.emotional_tone)
            y_true_intensity.append(exp_intensity)
            y_pred_intensity.append(intensity)

            print(
                f"[{i+1}/{len(samples)}] {wav_path.name}: "
                f"tone {judgment.emotional_tone} (exp {exp_tone}), "
                f"intensity {intensity} (exp {exp_intensity})"
            )
        except Exception as e:  # noqa: BLE001
            errors.append(f"{wav_path.name}: {type(e).__name__}: {e}")
            print(f"[{i+1}/{len(samples)}] {wav_path.name}: ERROR {e}")

    tone_cm = confusion_matrix(y_true_tone, y_pred_tone, labels=TONE_LABELS).tolist()
    tone_report = classification_report(
        y_true_tone, y_pred_tone, labels=TONE_LABELS, output_dict=True, zero_division=0
    )
    intensity_cm = confusion_matrix(
        y_true_intensity, y_pred_intensity, labels=INTENSITY_LABELS
    ).tolist()
    intensity_report = classification_report(
        y_true_intensity, y_pred_intensity, labels=INTENSITY_LABELS,
        output_dict=True, zero_division=0,
    )

    results = {
        "n_samples": len(samples),
        "n_errors": len(errors),
        "errors": errors,
        "emotional_tone": {
            "accuracy": tone_report["accuracy"],
            "macro_f1": tone_report["macro avg"]["f1-score"],
            "confusion_matrix": {"labels": TONE_LABELS, "matrix": tone_cm},
            "report": tone_report,
        },
        "emotional_intensity": {
            "accuracy": intensity_report["accuracy"],
            "macro_f1": intensity_report["macro avg"]["f1-score"],
            "confusion_matrix": {"labels": INTENSITY_LABELS, "matrix": intensity_cm},
            "report": intensity_report,
        },
    }

    out_path = Path(__file__).resolve().parents[1] / "validation_results_tone.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\n=== emotional_tone ===")
    print("accuracy:", tone_report["accuracy"], "macro_f1:", tone_report["macro avg"]["f1-score"])
    print("confusion matrix:", results["emotional_tone"]["confusion_matrix"])
    print(f"\n=== emotional_intensity ===")
    print("accuracy:", intensity_report["accuracy"], "macro_f1:", intensity_report["macro avg"]["f1-score"])
    print("confusion matrix:", results["emotional_intensity"]["confusion_matrix"])
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    actors = sys.argv[1:] if len(sys.argv) > 1 else ["Actor_01", "Actor_02"]
    main(actors)
