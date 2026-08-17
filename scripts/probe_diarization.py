"""STEP 1 feasibility probe: can we attribute transcript words to speakers?

The production transcript is currently undiarized, so the LLM tone head is
asked for "the primary emotional tone expressed by the customer" while
reading a flat string containing both the agent and the customer. On
call_002 that string is:

    "Hi, I'm Erica from Toyota of Braintree. How can I help?
     Spanish, please.
     Hola, soy Erica de Toyota de Braintree."

-- two of three turns are the agent's, and the customer's entire
contribution is two words. This probe checks whether pyannote can split the
speakers cleanly enough to (a) label the transcript per speaker and (b)
compute prosody over the customer's segments only.

Prints, per clip: speaker turns, a speaker-labelled transcript built by
assigning each parakeet word to the speaker whose turn covers its midpoint,
and the share of speech time per speaker.

Not wired into the pipeline. Feasibility only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vta.asr import transcribe  # noqa: E402
from vta.audio_io import load_normalized  # noqa: E402
from vta.config import HF_TOKEN  # noqa: E402

CLIPS = ["call_001.ogg", "call_002.ogg", "call_003.ogg"]
AUDIO_DIR = Path(__file__).resolve().parents[2]


_PIPE = None


def diarize(audio):
    """Full pyannote diarization: segmentation + embedding + clustering.

    Segmentation alone is permutation-invariant across its 10s windows and
    cannot give globally consistent speaker identity, which is what we need
    here. It is also 34-61x slower (TECHNICAL_MEMO §4) -- hence a probe.
    """
    global _PIPE
    import torch
    from pyannote.audio import Pipeline

    if _PIPE is None:
        # pyannote.audio 4.x renamed `use_auth_token` -> `token`.
        _PIPE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=HF_TOKEN
        )
    pipe = _PIPE
    waveform = torch.from_numpy(audio.samples).float().unsqueeze(0)
    out = pipe({"waveform": waveform, "sample_rate": audio.sr}, num_speakers=2)
    # 4.x returns a wrapper; the Annotation (.itertracks) is on
    # .speaker_diarization.
    ann = getattr(out, "speaker_diarization", out)
    return [
        (seg.start, seg.end, label)
        for seg, _, label in ann.itertracks(yield_label=True)
    ]


def label_transcript(words, turns):
    """Assign each word to the speaker whose turn contains its midpoint."""
    out, cur, buf = [], None, []
    for w in words:
        mid = (w.start_s + w.end_s) / 2
        spk = next((lb for s, e, lb in turns if s <= mid <= e), None)
        if spk != cur and buf:
            out.append((cur, " ".join(buf)))
            buf = []
        cur = spk
        buf.append(w.text)
    if buf:
        out.append((cur, " ".join(buf)))
    return out


def main():
    for name in CLIPS:
        path = AUDIO_DIR / name
        print("=" * 72)
        print(name)
        print("=" * 72)
        audio = load_normalized(path)
        asr = transcribe(audio)

        if not asr.words:
            print("  no word timestamps -- cannot attribute\n")
            continue

        turns = diarize(audio)
        print(f"  {len(turns)} speaker turns, duration {audio.duration_s:.1f}s")

        # Speech-time share per speaker; agent dominance is the suspected
        # dilution mechanism.
        share: dict[str, float] = {}
        for s, e, lb in turns:
            share[lb] = share.get(lb, 0.0) + (e - s)
        total = sum(share.values()) or 1.0
        for lb, secs in sorted(share.items()):
            print(f"    {lb}: {secs:5.1f}s ({100*secs/total:4.1f}% of speech)")

        print("\n  SPEAKER-LABELLED TRANSCRIPT:")
        for spk, text in label_transcript(asr.words, turns):
            print(f"    {spk or 'UNASSIGNED'}: {text}")
        print()


if __name__ == "__main__":
    main()
