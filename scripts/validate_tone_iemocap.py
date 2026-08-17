"""Real emotional_tone/emotional_intensity validation using IEMOCAP.

Replaces the RAVDESS-based validation (validate_tone_ravdess.py), which was
invalid for this system's architecture: RAVDESS uses exactly two fixed,
deliberately affect-neutral sentences ("Kids are talking by the door" /
"Dogs are sitting by the door"), so the LLM tone head -- which classifies
from the ASR transcript -- received an emotionally-neutral sentence on every
sample and collapsed to "neutral" on 103/104. RAVDESS isolates *prosodic*
affect by stripping lexical content; our design routes prosody to
emotion2vec and sends the *transcript* to the LLM, so it had nothing to read.

IEMOCAP (Interactive Emotional Dyadic Motion Capture) is improvised and
scripted *conversational* speech with varied lexical content -- real
dialogue between two actors, not repeated fixed sentences. Crucially, it
includes "frustrated" as its own emotion label, so the brief's dominant
error axis (frustrated vs upset, distinguished by degree of anger) is
validated directly rather than via an approximate valence mapping.

Dataset: Ar4ikov/iemocap_audio_text (HF Hub mirror of IEMOCAP Full Release,
audio + text + annotated emotion/activation/valence/dominance). Not gated.

Mapping (IEMOCAP 3-letter code -> brief 5-class schema), disclosed as
approximate where the taxonomies differ:
    neu  -> neutral      (valence 2.89, near-neutral)
    hap  -> satisfied    (valence 3.74, positive)
    exc  -> satisfied    (valence 3.73, high-arousal positive -- common
                          SER practice merges hap+exc; disclosed)
    fru  -> frustrated   (DIRECT label match)
    ang  -> upset        (valence 1.96 lowest, arousal 3.56 highest)
    fea  -> distressed   (fearful / panicked / overwhelmed)
    sad  -> excluded     (low-arousal negative; no clean brief analogue,
                          and would muddy the real frustrated class)
    sur  -> excluded     (ambiguous valence)
    oth  -> excluded     (uninformative)

emotional_intensity uses IEMOCAP's annotated continuous activation (1-5
arousal), banded the same way the production arousal cross-check bands:
    act < 2.5  -> low
    2.5-3.5    -> medium
    >= 3.5     -> high

The full production tone path is run per sample (ASR transcript + DSP
prosody + emotion2vec arousal/posteriors + LLM tone head + intensity
cross-check), so this measures the real system end-to-end, not the tone
head in isolation.
"""

import io
import json
import os
import re
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
from vta.dsp_features import compute_dsp_features  # noqa: E402
from vta.emotion_head import classify_emotion  # noqa: E402
from vta.asr import transcribe  # noqa: E402
from vta.tone_llm import ProsodySummary, classify_tone  # noqa: E402

SHARDS = [
    "data/train-00000-of-00003-9213b91aae6dc76d.parquet",
    "data/train-00001-of-00003-1d173227731b6f0e.parquet",
    "data/train-00002-of-00003-def37c465ef11c32.parquet",
]
HF_DATASET = "Ar4ikov/iemocap_audio_text"

IEMOCAP_TO_TONE = {
    "neu": "neutral",
    "hap": "satisfied",
    "exc": "satisfied",
    "fru": "frustrated",
    "ang": "upset",
    "fea": "distressed",
}
TONE_LABELS = ["neutral", "satisfied", "frustrated", "upset", "distressed"]
INTENSITY_LABELS = ["low", "medium", "high"]

# PER_CLASS_N x 5 classes; default 30 (=150), VTA_PER_CLASS_N for fast
# iteration. The seeded shuffle makes a smaller N a strict prefix of the
# full sample, so small runs stay comparable (but too noisy to report).
PER_CLASS_N = int(os.environ.get("VTA_PER_CLASS_N", "30"))
SEED = int(os.environ.get("VTA_SAMPLE_SEED", "1234"))

# Minimum reference-transcript word count for pool entry. Most IEMOCAP turns
# are a few words and carry affect in delivery, so the tone head defaults
# them to neutral; accuracy is monotone in length. Set VTA_MIN_REF_WORDS to
# match production's 70-450-word calls.
MIN_REF_WORDS = int(os.environ.get("VTA_MIN_REF_WORDS", "0"))


def _word_count(text: str) -> int:
    return len(re.sub(r"[^a-zA-Z' ]", " ", text or "").split())


# Buckets for the length-stratified breakdown, always reported so the
# length dependence is visible rather than hidden inside one aggregate.
LENGTH_BUCKETS = [(0, 3), (4, 8), (9, 20), (21, 10**6)]

# Preceding dialogue turns supplied as context (0 = off). The largest
# reported gain for transcript-based LLM emotion recognition (GPT-4o 43.4
# -> 55.5 UA, arXiv:2602.06270), and production already runs in this
# condition: a real call is one clip with the whole conversation.
CONTEXT_TURNS = int(os.environ.get("VTA_CONTEXT_TURNS", "0"))

# All pool rows before emotion filtering -- context needs every turn,
# including emotions outside the 5-class mapping.
_ALL_ROWS: list[dict] = []


def _parse_titre(titre: str):
    """'Ses01F_script02_2_F041' -> ('Ses01F_script02_2', 'F', 41)."""
    m = re.match(r"^(.*)_([MF])(\d+)$", titre)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def build_dialogue_index(rows: list[dict]) -> dict:
    """dialogue_id -> turn-ordered [(turn_no, speaker, text)]."""
    idx: dict[str, list] = {}
    for r in rows:
        parsed = _parse_titre(r["titre"])
        if not parsed:
            continue
        dlg, spk, turn = parsed
        text = (r.get("to_translate") or "").strip()
        if text:
            idx.setdefault(dlg, []).append((turn, spk, text))
    for dlg in idx:
        idx[dlg].sort(key=lambda x: x[0])
    return idx


def context_for(titre: str, idx: dict, k: int):
    """(context_string, target_speaker_label) or (None, None).

    Up to k preceding turns, speaker-labelled with the same SPEAKER_xx form
    the production diarized transcript uses. The target's speaker is always
    in the label map so the model can tell which party is being classified.
    """
    if k <= 0:
        return None, None
    parsed = _parse_titre(titre)
    if not parsed:
        return None, None
    dlg, target_spk, turn = parsed
    turns = idx.get(dlg, [])
    preceding = [t for t in turns if t[0] < turn][-k:]
    if not preceding:
        return None, None

    # Assign labels in first-appearance order, seeding with the target speaker
    # so its label is stable regardless of who speaks in the window.
    spk_map: dict[str, str] = {}

    def label(spk: str) -> str:
        if spk not in spk_map:
            spk_map[spk] = f"SPEAKER_{len(spk_map):02d}"
        return spk_map[spk]

    for _, spk, _text in preceding:
        label(spk)
    target_label = label(target_spk)
    lines = [f"{label(spk)}: {text}" for _, spk, text in preceding]
    return "\n".join(lines), target_label


def expected_intensity(activation: float) -> str:
    if activation < 2.5:
        return "low"
    if activation < 3.5:
        return "medium"
    return "high"


def load_pool():
    rows = []
    for shard in SHARDS:
        path = hf_hub_download(HF_DATASET, shard, repo_type="dataset")
        rows.extend(
            pq.read_table(
                path,
                columns=["emotion", "activation", "titre", "to_translate", "audio"],
            ).to_pylist()
        )
    global _ALL_ROWS
    _ALL_ROWS = rows
    pool = [r for r in rows if r["emotion"] in IEMOCAP_TO_TONE]
    if MIN_REF_WORDS > 0:
        before = len(pool)
        pool = [r for r in pool if _word_count(r["to_translate"]) >= MIN_REF_WORDS]
        print(
            f"Length filter: >={MIN_REF_WORDS} reference words -> "
            f"{len(pool)}/{before} utterances retained"
        )
    return pool


def balanced_sample(pool):
    """30 per tone class, stratified across sessions & speakers for
    representativeness (avoids over-sampling one actor/session)."""
    rng = np.random.default_rng(SEED)
    by_tone = {t: [] for t in TONE_LABELS}
    for r in pool:
        by_tone[IEMOCAP_TO_TONE[r["emotion"]]].append(r)

    sample = []
    for tone in TONE_LABELS:
        cands = by_tone[tone]
        rng.shuffle(cands)
        # round-robin across sessions so no single session dominates
        by_sess = {}
        for r in cands:
            by_sess.setdefault(r["titre"][:5], []).append(r)
        sess_order = sorted(by_sess)
        picked, si = [], 0
        while len(picked) < min(PER_CLASS_N, len(cands)):
            s = sess_order[si % len(sess_order)]
            if by_sess[s]:
                picked.append(by_sess[s].pop())
            si += 1
        sample.extend(picked)
        print(f"  {tone:12s}: {len(picked)} (pool {len(cands)})")
    return sample


def decode_audio(row):
    samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return NormalizedAudio(
        samples=samples, sr=sr, duration_s=len(samples) / sr, source_path=Path(row["titre"])
    )


def main():
    print("Loading IEMOCAP pool...")
    pool = load_pool()
    print(f"Pool: {len(pool)} mappable utterances. Sampling {PER_CLASS_N}/class:")
    sample = balanced_sample(pool)
    dlg_index = build_dialogue_index(_ALL_ROWS) if CONTEXT_TURNS > 0 else {}
    if CONTEXT_TURNS > 0:
        print(f"Context: up to {CONTEXT_TURNS} preceding turns from "
              f"{len(dlg_index)} indexed dialogues")
    print(f"Running {len(sample)} samples through the full tone pipeline...")

    y_true_tone, y_pred_tone = [], []
    y_true_int, y_pred_int = [], []
    records, errors = [], []

    for i, row in enumerate(sample):
        titre = row["titre"]
        exp_tone = IEMOCAP_TO_TONE[row["emotion"]]
        exp_int = expected_intensity(row["activation"])
        try:
            audio = decode_audio(row)
            feats = compute_dsp_features(audio)
            asr = transcribe(audio)
            emo = classify_emotion(audio)

            prosody = ProsodySummary(
                pitch_mean_hz=feats.pitch_mean_hz,
                pitch_std_hz=feats.pitch_std_hz,
                energy_mean_db=feats.energy_mean_db,
                energy_std_db=feats.energy_std_db,
                energy_dynamic_range_db=feats.energy_dynamic_range_db,
                voiced_ratio=feats.voiced_ratio,
                speaking_rate_wpm=asr.speaking_rate_wpm,
                energy_contour_db=feats.energy_contour_db,
                arousal=emo.arousal,
                emotion_posteriors=emo.posteriors,
            )
            ctx, ctx_target_spk = context_for(titre, dlg_index, CONTEXT_TURNS)
            target_transcript = asr.transcript
            if ctx and ctx_target_spk:
                # Label the target so the model can find its speaker in the context.
                target_transcript = f"{ctx_target_spk}: {asr.transcript}"
            judgment, usage = classify_tone(
                target_transcript, prosody, None, ctx
            )

            # The LLM's judgment is final -- no acoustic post-correction.
            # The override copies that used to live here are gone, so this
            # measures what actually ships.
            tone = judgment.step2_emotional_tone
            intensity = judgment.step5_emotional_intensity

            y_true_tone.append(exp_tone)
            y_pred_tone.append(tone)
            y_true_int.append(exp_int)
            y_pred_int.append(intensity)
            records.append({
                "titre": titre,
                "emotion": row["emotion"],
                "asr_transcript": asr.transcript[:200],
                "ref_transcript": (row["to_translate"] or "")[:200],
                "arousal_model": round(emo.arousal, 3),
                "true_tone": exp_tone,
                "pred_tone": tone,
                "true_intensity": exp_int,
                "pred_intensity": intensity,
                "context_turns_used": 0 if not ctx else len(ctx.splitlines()),
                "reasoning_tokens": usage.reasoning_tokens,
                "completion_tokens": usage.completion_tokens,
                "prompt_tokens": usage.prompt_tokens,
            })
            mark = "OK" if tone == exp_tone else "XX"
            override = " *" if tone != judgment.step2_emotional_tone else ""
            print(
                f"[{i+1}/{len(sample)}] {mark} {titre:28s} "
                f"pred={tone:11s} exp={exp_tone:11s} "
                f"int={intensity:6s} exp={exp_int} ar={emo.arousal:.2f}"
            )
        except Exception as e:  # noqa: BLE001
            errors.append(f"{titre}: {type(e).__name__}: {e}")
            print(f"[{i+1}/{len(sample)}] ERROR {titre}: {e}")

    # --- Length-stratified breakdown ---
    # Reported unconditionally: the aggregate hides a strong monotone
    # dependence on transcript length, and production sits far right of it.
    by_length = []
    for lo, hi in LENGTH_BUCKETS:
        sub = [r for r in records if lo <= _word_count(r["ref_transcript"]) <= hi]
        if not sub:
            continue
        n = len(sub)
        by_length.append({
            "bucket": f"{lo}-{hi}" if hi < 10**6 else f"{lo}+",
            "n": n,
            "tone_accuracy": round(
                sum(r["true_tone"] == r["pred_tone"] for r in sub) / n, 4
            ),
            "intensity_accuracy": round(
                sum(r["true_intensity"] == r["pred_intensity"] for r in sub) / n, 4
            ),
            "pred_neutral_rate": round(
                sum(r["pred_tone"] == "neutral" for r in sub) / n, 4
            ),
            "tone_macro_f1": round(
                classification_report(
                    [r["true_tone"] for r in sub], [r["pred_tone"] for r in sub],
                    labels=TONE_LABELS, output_dict=True, zero_division=0,
                )["macro avg"]["f1-score"], 4
            ),
        })

    tone_cm = confusion_matrix(y_true_tone, y_pred_tone, labels=TONE_LABELS).tolist()
    tone_rep = classification_report(
        y_true_tone, y_pred_tone, labels=TONE_LABELS, output_dict=True, zero_division=0
    )
    int_cm = confusion_matrix(y_true_int, y_pred_int, labels=INTENSITY_LABELS).tolist()
    int_rep = classification_report(
        y_true_int, y_pred_int, labels=INTENSITY_LABELS, output_dict=True, zero_division=0
    )

    results = {
        "dataset": "IEMOCAP (Ar4ikov/iemocap_audio_text)",
        "n_samples": len(sample),
        "n_errors": len(errors),
        "per_class_n": PER_CLASS_N,
        "mapping": IEMOCAP_TO_TONE,
        "excluded_emotions": ["sad", "sur", "oth"],
        "errors": errors,
        "emotional_tone": {
            "accuracy": tone_rep["accuracy"],
            "macro_f1": tone_rep["macro avg"]["f1-score"],
            "confusion_matrix": {"labels": TONE_LABELS, "matrix": tone_cm},
            "report": tone_rep,
        },
        "emotional_intensity": {
            "accuracy": int_rep["accuracy"],
            "macro_f1": int_rep["macro avg"]["f1-score"],
            "confusion_matrix": {"labels": INTENSITY_LABELS, "matrix": int_cm},
            "report": int_rep,
        },
        "per_sample": records,
        "min_ref_words_filter": MIN_REF_WORDS,
        "context_turns": CONTEXT_TURNS,
        "by_transcript_length": by_length,
    }

    out = Path(__file__).resolve().parents[1] / os.environ.get(
        "VTA_TONE_OUT", "validation_results_tone_iemocap.json"
    )
    out.write_text(json.dumps(results, indent=2))

    print("\n=== emotional_tone ===")
    print(f"accuracy: {tone_rep['accuracy']:.3f}  macro_f1: {tone_rep['macro avg']['f1-score']:.3f}")
    print("confusion:", TONE_LABELS)
    for lbl, row_cm in zip(TONE_LABELS, tone_cm):
        print(f"  {lbl:12s} {row_cm}")
    print("\n=== emotional_intensity ===")
    print(f"accuracy: {int_rep['accuracy']:.3f}  macro_f1: {int_rep['macro avg']['f1-score']:.3f}")
    print("confusion:", INTENSITY_LABELS)
    for lbl, row_cm in zip(INTENSITY_LABELS, int_cm):
        print(f"  {lbl:8s} {row_cm}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
