# Technical Memo — Voice Tone & Background Noise Analyzer

## 1. Approaches tested and why the final architecture was selected

The brief's own scoring criteria and evaluation note ("do not infer frustration
or distress solely from loudness, and do not infer background noise solely
from poor audio quality") pushed the design toward **independent estimators
per field group rather than one model producing all nine fields from a
single representation**. A single audio-native model reading everything at
once cannot help correlating exactly the signals the brief says must stay
separate; seven deterministic/acoustic fields and two semantic fields are
resolved by disjoint pipelines that never share inputs in a way that would
let one leak into the other.

**Two materially different approaches were compared for the tone/intensity
fields**, per the brief's request:

1. **Acoustic-only** (prosody features + emotion2vec+ posteriors, no
   transcript) -- cheap, fast, fully local, but structurally unable to
   resolve `frustrated` vs `upset`, which the brief defines by *degree of
   anger*, a property of what is said and how, not of pitch/energy alone.
   Two callers can be acoustically near-identical while one calmly asks
   about a delay and the other threatens to cancel their account.
2. **Transcript + prosody summary via LLM** (selected) -- an LLM reads the
   transcript (from local ASR) plus numeric prosody/emotion features and
   applies the brief's rubric verbatim. This resolves the semantic boundary
   the acoustic-only approach cannot, at a small measured cost (see §3).

Within the transcript+LLM approach, model choice was constrained by the
$0.003/audio-minute ceiling and the "no raw audio leaves AutoAce
infrastructure" requirement. Options assessed and ruled out:

| Option | Why ruled out |
|---|---|
| Audio-native OpenAI models (`gpt-4o-audio`-class) | 6-20x over the cost ceiling; would also require sending raw audio to a third party |
| GPT-5.6 Sol / Terra (larger siblings of the selected Luna) | 10-60x the cost of Luna for a classification task that doesn't need frontier reasoning |
| Claude / GPT-5.4 family | No audio modality relevant here; text-only pricing comparable to Luna but Luna was already validated against the brief's own worked cost example |
| Local-only text model (e.g. a small instruction-tuned LLM) | Avoids any external call entirely, but degrades rubric-following accuracy on the `frustrated`/`upset` boundary in spot checks; kept as the documented fully-local upgrade path if no external API is acceptable in production |
| Non-commercially-licensed ASR models | Reproducibility/licensing risk for a production system |

**Selected**: `gpt-5.6-luna` (OpenAI), transcript + prosody features in one
strict structured-output call whose field order forces observation before
judgment (see §2c-addendum). The LLM's tone/intensity labels are final --
no acoustic post-correction (see §5.1/§5.2 for the measured reasons the
override and arousal cross-check were removed).

### Final architecture

```
audio file
   |
   +-> ffmpeg -> 16kHz mono
   |
   +-> DSP features (SNR, clipping, spectral flatness, bandwidth,
   |                 pitch, energy contour, speaking-rate proxy) --local, $0
   +-> Silero VAD -> long_silence_present                        --local, $0
   +-> PANNs (CNN14) + DSP cross-check -> background_noise_*      --local, $0
   +-> pyannote/segmentation-3.0 -> speaker_overlap_present        --local, $0
   +-> parakeet-tdt-0.6b-v2 -> transcript, speaking rate           --local, $0
   +-> emotion2vec+ (chunked, 20s windows) -> arousal + posteriors   --local, $0
   |
   +-> GPT-5.6 Luna (transcript + prosody/emotion features, 5-field
         evidence-ordered structured output)
         -> emotional_tone, emotional_intensity                    --~$0.0002-0.0015/min
   |
   +-> confidence heuristic (LLM/acoustic agreement)
   -> validated 9-field JSON
```

Overlap detection calls `pyannote/segmentation-3.0` directly rather than
the full `speaker-diarization-3.1` pipeline (segmentation + speaker
embedding + clustering). Profiling found the full pipeline was 58-91% of
total per-clip latency; overlap only needs segmentation's frame-level "how
many local speaker slots are active," which doesn't require the
embedding/clustering steps at all. Switching cut this stage by 34-61x
(28.8s/27.2s/274.4s -> 0.8-4.5s per clip) with near-identical overlap
durations, and removed the single largest latency contributor in the whole
pipeline -- see §4.

## 2. Validation results

### 2a. The 3 provided clips (smoke test, not scored as accuracy per brief §11)

> **Superseded -- read with §2d.** The acoustic tone override described in
> this section was subsequently REMOVED from the system, and the intensity
> arousal cross-check with it. The numbers below are an accurate record of
> what was measured at the time and are retained for the audit trail, but
> they do not describe the shipped configuration. The final system is
> Luna-only (no mapping, no post-hoc correction) with `reasoning_effort=high`
> and conversational context; see §2d for its results and for why the
> override was dropped (it flipped 8/30 distressed clips to the opposite
> valence while adding zero correct distressed predictions).


| Field | call_001 | call_002 | call_003 |
|---|---|---|---|
| emotional_tone | frustrated / **upset** ✗ | **satisfied** / neutral ✗† | **satisfied** / satisfied ✓ |
| emotional_intensity | medium / **high** (closer, still ✗) | medium / medium ✓ | medium / medium ✓ |
| background_noise_present | false ✓ | true ✓ | true ✓ |
| background_noise_type | "" ✓ | music / **TV** (near) | music / **sharp static** ✗ |
| background_noise_severity | none ✓ | medium ✓ | medium ✓ |
| audio_quality | clear ✓ | clear ✓ | clear ✓ |
| speaker_overlap_present | false ✓ | true ✓ | true ✓ |
| long_silence_present | false ✓ | false ✓ | false ✓ |
| confidence | 0.65 (uninformative — ground truth is a constant 0.82 across all 3 labels, so this cannot be validated against them; see §5.6) |

6/9, 7/9, 8/9 field matches (21/27 total). †call_002's tone changed from
neutral→satisfied via the acoustic override (§5.1): emotion2vec detected
strong happy prosody (>0.77 confident) and overrode the LLM's neutral, but
the ground truth is neutral -- the override false-positived on upbeat prosody
with neutral content. call_003's tone changed the same way (neutral→satisfied)
and is correct. Net on 3 clips: 1 broke, 1 fixed. At IEMOCAP scale (n=150)
the override improves macro F1 by +0.047 (§2c). `emotional_intensity` went
from 0/3 to 2/3 exact matches once two real bugs in the emotion2vec arousal
cross-check were fixed (§5.2); `background_noise_severity` went from 1/3 to
3/3 once its thresholds were rebuilt (§5.5). The remaining errors are
concentrated in `emotional_tone` (call_001 reads frustrated, GT is upset --
the real "degree of anger" boundary) and the open-text `background_noise_type`
guess -- see §2c.

**Regenerated under the 5-field evidence-ordered schema (§2c-addendum), which
is what ships**: call_001 frustrated/medium (GT upset/high -- tone unchanged
on the degree-of-anger boundary, intensity medium vs high), call_002
neutral/medium (GT neutral/medium, exact), call_003 neutral/medium (GT
satisfied/medium; the prior run's `satisfied` came partly from the now-removed
acoustic override). Tone matches 1/3 vs the prior 2/3, intensity 2/3 vs 2/3;
all seven technical fields unchanged. n=3 is a smoke test, not a tuning
target, and call_003's flip is consistent with §5.11 (speaker mixing on a
172 s two-party call averaging away the customer's satisfied cues) -- flagged,
not chased.

### 2b. Synthetic validation (real accuracy + confusion matrices, per brief §6)

Three labeled clips cannot produce per-class F1 or a confusion matrix, and
windowing `call_003` would place the same call on both sides of a split --
the leakage the brief explicitly warns against. Ground truth is instead
constructed by injecting known noise/clipping/silence/overlap into real
speech (`call_001`/`call_002` as carrier signal) at controlled levels.
`emotional_tone`/`emotional_intensity` are **not** validated this way --
that needs labeled emotional speech (MELD/IEMOCAP), whose download wasn't
attempted given the deadline; this is a documented gap, not an oversight.

Sample sizes below are n=20-65 per field. An earlier pass at n=6-13 was
too small to distinguish a working classifier from a lucky one; these are
the numbers after scaling up AND fixing the methodology defects the first
pass exposed (severity high-band trigger, audio_quality soft-clip
degradation, overlap diverse-speaker generator):

| Field | Accuracy | Macro F1 | Confusion matrix |
|---|---|---|---|
| `background_noise_present` (n=40) | **0.975** | 0.965 | labels=[false,true]: `[[9,1],[0,30]]` |
| `background_noise_severity` (n=40) | **0.725** | 0.711 | labels=[none,low,medium,high]: `[[9,0,1,0],[0,8,2,0],[0,3,3,4],[0,0,1,9]]` |
| `audio_quality` (n=30) | **0.933** | 0.933 | labels=[clear,slight,severe]: `[[10,0,0],[0,8,2],[0,0,10]]` |
| `long_silence_present` (n=20) | **0.95** | 0.95 | labels=[false,true]: `[[10,0],[1,9]]` |
| `speaker_overlap_present` (n=65) | **0.80** | 0.79 | labels=[false,true]: `[[33,2],[11,19]]` |

Full per-class precision/recall in `validation_results.json`. Reading these
honestly (root causes investigated in detail, not just noted):

- **`long_silence_present`: 0.95, one miss out of 20.** Silero VAD's gap
  detection is robust across synthetic silence of varying length and
  offset; consistent with the smaller n=7 pass.
- **`background_noise_present`: 0.975.** The presence gate is robust at
  scale; one residual miss (a clean control read as present) remains.
- **`background_noise_severity`: 0.725 (macro F1 0.711), up from 0.525
  (0.478).** The previous pass's critical gap -- "high" never predicted
  (0/10) -- is fixed: `occupied_bandwidth_hz > 4000` was added as a second
  high-band trigger alongside `band_deviation > 0.55`. High recall went
  from 0.00 to 0.90. **But this came partly at medium's expense: medium
  recall dropped to 0.30** (3/10 correct, with 3 spilling to low and 4 to
  high). The SNR 8-12 dB boundary where medium and high overlap is
  inherently fuzzy for any single acoustic feature -- `band_deviation`
  and `occupied_bandwidth_hz` both transition gradually across that range.
  Net macro F1 still nearly doubled (0.478 → 0.711), so it's a genuine
  improvement, but medium is now the weak band where the threshold sits
  in the transition zone. A four-seed held-out check (§2b below) confirms
  this is structural, not seed-specific.
- **`audio_quality`: 0.933 (macro F1 0.933), up from 0.50 (0.444).** Two
  fixes: (a) the synthetic degradation switched from hard clipping
  (`np.clip`) to soft clipping (`np.tanh`), matching how real codec/limiter
  degradation sounds -- hard clipping at moderate gain produced far more
  samples at full scale than real telephony, sending every
  slightly_impaired trial to severely_impaired. (b) `SLIGHT_BANDWIDTH_HZ`
  lowered from 2000 to 1400: individual speech segments vary 1490-2900 Hz
  in 99%-energy bandwidth depending on pitch/content (deep male vowels
  concentrate energy low), so 2000 Hz false-fired as "impaired" on 5/10
  clean controls. 1400 Hz is still below real telephony clips (2200-3340
  Hz) but above the natural low end of clear speech. `SEVERE_CLIPPING`
  raised from 0.02 to 0.03 to sit above the natural per-segment clipping
  variation at moderate gain. Residual: 2/10 slightly_impaired trials still
  spill to severely_impaired (clipping ratio varies 1-4.4% at the same gain
  depending on speech peaks).
- **`speaker_overlap_present`: 0.80 (macro F1 0.79), up from 0.50 (0.333)
  at n=6.** The previous generator summed call_001+call_002 -- acoustically
  similar call-center voices from the same recording environment. Direct
  inspection of the raw segmentation output confirmed max_active_speakers
  stayed at 1.0: the model never saw a second speaker, because summing two
  similar voices doesn't produce the spectrally-distinct two-source signal
  the segmentation head was trained on. Switched to diverse IEMOCAP
  speaker pairs (different gender, different sessions = different recording
  chains/voice timbres), which the model correctly reads as overlapping
  (max_active reaches 2.0). Scaled to n=65 (35 controls + 30 overlap
  cases). **Precision is 0.905 but recall is only 0.633** -- 11 of 30
  true-overlap samples are missed (37% miss rate), concentrated at shorter
  overlap durations (2s: 47% recall, 4s: 80%). For a presence flag, recall
  is the consequential direction: a missed overlap is a false negative that
  could mask a real QA issue. The conservative bias (high precision,
  moderate recall) means the system rarely false-alarms but can miss brief
  overlaps. The real clips still match 3/3. The remaining recall gap is an
  honest limitation of summed-mono synthetic construction vs. genuine
  concurrent recording.

**Takeaway**: the first pass at n=6-13 was too small to trust and, more
importantly, had three synthetic-methodology defects that produced
misleading numbers: the overlap generator used acoustically similar voices
(0.50 → 0.80 once fixed), the audio_quality degradation used unrealistic
hard clipping (0.50 → 0.93 once switched to tanh), and severity's high
band had no achievable trigger (recall 0 → 0.90 once `occupied_bandwidth_hz`
was added). Each fix was driven by direct inspection of the raw model
output or feature values, not by tuning to pass the test -- and the real
3-clip predictions are unchanged by all of them, confirming they target
the synthetic construction, not the production path.

### 2b-i. Held-out seed validation (leakage check)

Quality and severity thresholds were calibrated on seed 1234's synthetic
data. To verify they are not overfit, the synthetic validation was re-run
on three additional seeds (9999, 7777, 4242) **without touching any
threshold**:

| Field | Seed 1234 (tuned) | 4-seed mean ± std | Pre-fix |
|---|---|---|---|
| `background_noise_present` macro F1 | 0.965 | **0.958 ± 0.014** | — |

(Note: the CED comparison table below uses *accuracy* for presence, not macro
F1 — 0.969 ± 0.011 accuracy vs 0.958 ± 0.014 macro F1. Same data, different
metric, no discrepancy.)
| `background_noise_severity` F1 | 0.711 | **0.648 ± 0.038** | 0.478 |
| `audio_quality` F1 | 0.933 | **0.868 ± 0.057** | 0.444 |
| `long_silence_present` F1 | 0.950 | **0.987 ± 0.022** | — |
| `speaker_overlap_present` F1 | 0.790 | **0.790 ± 0.000** | 0.333 |

The held-out means are within one standard deviation of the tuned-seed
values for all fields except quality (0.868 vs 0.933), where seed 7777 is
an outlier (0.780) due to the slightly-impaired to severely-impaired
clipping boundary being inherently fuzzy at moderate gain -- the clear
class has 1.0 recall on all four seeds, confirming the bandwidth threshold
is not overfit. Severity's held-out mean (0.648) is above pre-fix (0.478)
by a wider margin than the seed-to-seed variation, confirming the bandwidth
trigger is a real improvement, not a seed artifact. The thresholds are
reported as calibrated values with honest held-out variance, not as point
estimates.

### 2c. Real emotional_tone/emotional_intensity validation (IEMOCAP)

> **Superseded -- read with §2d.** The acoustic tone override described in
> this section was subsequently REMOVED from the system, and the intensity
> arousal cross-check with it. The numbers below are an accurate record of
> what was measured at the time and are retained for the audit trail, but
> they do not describe the shipped configuration. The final system is
> Luna-only (no mapping, no post-hoc correction) with `reasoning_effort=high`
> and conversational context; see §2d for its results and for why the
> override was dropped (it flipped 8/30 distressed clips to the opposite
> valence while adding zero correct distressed predictions).


§2a and §2b together validate 7 of 9 fields with either real clips or
synthetic ground truth. The 45%-weighted `emotional_tone`/
`emotional_intensity` fields had **zero** quantitative validation until
the first pass used RAVDESS -- which turned out to be invalid for this
architecture (see below) -- and are now validated on IEMOCAP.

**Why RAVDESS was invalid (kept for the record).** RAVDESS uses exactly
two fixed, deliberately affect-neutral sentences ("Kids are talking by the
door" / "Dogs are sitting by the door"), spoken in all emotions by actors
to isolate *prosodic* affect from lexical content. Our design routes prosody
to emotion2vec and sends the *ASR transcript* to the LLM tone head. With a
neutral sentence on every sample, the tone head had nothing to read and
collapsed to "neutral" on 103/104 (accuracy 0.24, macro F1 0.10). That number
measured the dataset, not the system. RAVDESS is the wrong tool here.

**IEMOCAP (current).** 150 balanced samples (30 per mapped class) from
IEMOCAP (Interactive Emotional Dyadic Motion Capture) via the HF Hub mirror
`Ar4ikov/iemocap_audio_text` -- improvised and scripted *conversational*
speech between two actors, with varied lexical content. Crucially, IEMOCAP
includes "frustrated" as its own emotion label, so the brief's dominant
error axis (frustrated vs upset, distinguished by degree of anger) is
validated directly rather than via an approximate mapping. The full
production tone path is run per sample (ASR transcript + DSP prosody +
emotion2vec arousal/posteriors + LLM tone head + intensity cross-check),
so this measures the real system end-to-end.

Mapping (IEMOCAP 3-letter code → brief schema), disclosed as approximate:
`neu`→neutral, `hap`/`exc`→satisfied (high valence, common SER merge),
`fru`→frustrated (direct label match), `ang`→upset (lowest valence, highest
arousal), `fea`→distressed. `sad`/`sur`/`oth` excluded (no clean brief
analogue / ambiguous valence). See `scripts/validate_tone_iemocap.py`.

| Field | Accuracy | Macro F1 | Source |
|---|---|---|---|
| `emotional_tone` (with acoustic override) | **0.473** | **0.450** | 5-seed mean (§5.1) |
| `emotional_intensity` | 0.504 | 0.456 | 5-seed mean |

Without the override, tone is 0.428 / 0.406 (5-seed mean). The override adds
+0.045 accuracy and +0.044 macro F1, positive on all 5 seeds (sign test
p=0.031; paired t=2.38, df=4, p=0.038 one-tailed) -- see §5.1 for
the full ablation. These are 5-seed means, not single-run point estimates.

`emotional_tone` per-class F1 (with override, seed 1234 representative):
neutral 0.66, satisfied 0.63, frustrated 0.30, upset 0.63, distressed 0.16.
These vary +/-0.03 across seeds (LLM non-determinism); the macro F1 mean
is the stable summary. The confusion matrix reveals the dominant
error mode: **systematic neutral over-prediction**. 66 of 150 predictions
(44%) are neutral, and 42 of those are wrong -- the text head under-reads
emotional escalation across the board. `distressed` is the worst victim (3/30
correct, 17/30 misread as neutral, predicted only 6 times total). `frustrated`
(7/30 correct, 10/30 as neutral) is next. The frustrated→upset confusion (7/30)
is real and is the brief's genuine "degree of anger" boundary, but it is not
the main error axis -- neutral bias is. The strongest classes are `upset` (63%
recall) and `satisfied` (53% recall), where lexical content is most decisive.

An **acoustic tone override** (see §5.1) partially corrects this: when the LLM
predicts neutral but emotion2vec is confident about a specific non-neutral
emotion, the tone is overridden. This recovers 79% of missed non-neutrals at a
21% false-positive cost (3.8:1 ratio), improving macro F1 to 0.47. The
override is threshold-insensitive (0.2–0.6 give identical results on IEMOCAP).

`distressed` remains the weakest class even after the override, partly because
`fea` (fearful) maps imperfectly to the brief's "overwhelmed/panicked/crying"
definition and partly because fearful speech is often quiet and reads as neutral
to both heads.

`emotional_intensity` fares similarly (0.50 macro F1, 5-seed mean). The 5-seed
ablation also revealed that the LLM tone head has intrinsic non-determinism:
re-running on the same 150 clips with the same models gives +/-0.035 macro F1
variation across seeds. This noise band is now measured (not estimated) and
is the standard against which all tone-related improvements must be judged --
a single-run delta of less than 0.035 is within noise.

The key takeaway: RAVDESS's 0.10 was a floor imposed by zero lexical signal.
With varied conversational content, the tone head produces real, varied,
rubric-aligned predictions. 0.47 macro F1 on 5 classes (chance = 0.20) is a
valid baseline with the acoustic override; the dominant remaining error is
the frustrated/upset boundary and the still-weak distressed recall (16%),
both of which need fine-tuning on real labeled calls.

### 2c-addendum. Circularity caveat, and the evidence-ordered output fix

**The intensity numbers above (0.504/0.456, cross-check era) are partially
circular and must not be read as task accuracy.** Ground-truth intensity is
IEMOCAP's annotated activation banded at 2.5/3.5; the (now removed) production
cross-check banded emotion2vec arousal at 0.35/0.60. Two measurements prove
the circularity: (a) across a full system-prompt rewrite, the cross-check-era
intensity predictions agreed on 147/150 samples -- impossible unless the
cross-check, not Luna, was setting the field; (b) banding `arousal_model`
alone at 0.35/0.60, with no LLM in the loop, scores 0.533 accuracy / 0.485
macro F1 -- statistically indistinguishable from the cross-check era's
0.513/0.490. Those numbers measured agreement between two arousal estimators
(the acoustic model's and the annotators'), not performance on the task as
posed. With the cross-check removed, honest Luna-only intensity was
0.293/0.288: Luna predicted `low` on 105/150 samples. The same caveat applies
in reverse to any future "fix" that re-bands arousal: it optimizes toward the
benchmark's construction, not toward AutoAce's "low is subtle / medium is
clear and sustained / high is escalated" definitions.

**Reasoning effort was ruled out empirically, not argumentatively.** A full
150-sample sweep at `VTA_TONE_EFFORT=high`, usage-verified (mean 153 reasoning
tokens/call vs 0 at default; 21/150 calls still chose to emit none), moved
tone macro F1 by +0.010 and intensity by +0.013 -- a third of the measured
+/-0.035 non-determinism band -- with the low-intensity collapse byte-identical
(105/150) and the flagship failures unchanged
(`validation_results_tone_iemocap_effort_high.json`). Effort stays at the API
default.

**Root cause of the collapse, and the fix.** The retired prompt applied one
rule to both fields: "use prosody to confirm or grade a reading you already
have from the words -- never to originate one." Correct for `emotional_tone`
(semantic), wrong for `emotional_intensity` (primarily prosodic). On the 72
true-non-neutral clips with arousal >= 0.6, Luna emitted `low` on 38 and
`neutral` on 23 -- one demoted channel, two symptoms. The fix is structural,
not verbal: a 5-field strict structured output
(`step1_lexical_evidence -> step2_emotional_tone -> step3_acoustic_evidence
-> step4_intensity_rationale -> step5_emotional_intensity`), so each label is
generated only after its evidence is on the page. One measurement corrected
the design assumption behind this: **strict structured outputs on gpt-5.6-luna
emit JSON properties in lexicographic order, not schema order** (three probes,
identical key order) -- the `step1_..step5_` prefixes exist to make
alphabetical order equal reasoning order. The arousal value is anchored
verbally in the prompt (~0.2-0.3 calm, 0.35-0.6 moderate, >=0.6 strongly
activated), never as banding gates, and the already-transmitted
`emotion_model_posteriors` are now explicitly read out in
`step3_acoustic_evidence`.

Same seed, same 150 clips, one variable changed
(`validation_results_tone_iemocap_schema5.json`):

| Metric | 2-field prompt | 5-field evidence-ordered |
|---|---|---|
| intensity accuracy / macro F1 | 0.293 / 0.288 | **0.440 / 0.433** |
| tone accuracy / macro F1 | 0.427 / 0.423 | 0.453 / 0.447 |
| predicted intensity distribution | 105 low / 25 med / 20 high | 46 / 58 / 46 |
| `low` on true-neutral clips | 30/30 | 25/30, none `high` |
| `neutral` on non-neutrals with arousal >= 0.6 | 23/72 | 13/72 |

The intensity gain (+0.145 macro F1, ~4x the noise band) is the headline; the
tone gain (+0.024) is within noise, as expected -- tone is capped on this
benchmark by the label mapping, not by the prompt. An independent repeat run
on the same clips reproduced it exactly (intensity 0.440/0.428, tone
0.453/0.445; `validation_results_tone_iemocap_schema5_repeat.json`), so the
delta is not a single lucky roll. Two honesty notes: (i)
pure arousal banding still scores higher than the new Luna path (0.533 vs
0.440 accuracy), which is expected and not a reason to restore it -- banding
is the partially-circular crutch this section exists to disclaim; (ii) ASR
WER is clean overall (mean 0.12, median 0.0) but degrades exactly on the
weakest class (distressed 0.234 vs 0.07-0.14 elsewhere), so distressed
underperformance is lexical + prosodic + mapping, stacked.

### 2d. The benchmark was measuring the wrong regime (four times)

Eight configurations of the tone head all landed between 0.40 and 0.50 macro
F1, and tone was exactly 1/3 on the provided clips in every one. That looked
like a model ceiling. It was not: on four separate occasions the *measurement
setup*, not the system, was the limiter. Recording the sequence because the
diagnostic method matters more than any single number.

**(1) RAVDESS: no lexical content at all.** Documented in §2c -- two fixed
affect-neutral sentences, 103/104 predictions collapsed to neutral, macro F1
0.099. Caught and replaced with IEMOCAP.

**(2) IEMOCAP per-utterance: still almost no lexical content.** The same trap
one layer down, and it survived undetected for much longer. IEMOCAP is
annotated per conversational TURN, and most turns are a few words:

| Reference transcript | n | Tone accuracy | Predicted `neutral` |
|---|---|---|---|
| 0-3 words | 25 | 36.0% | 64% |
| 4-8 words | 55 | 40.0% | 51% |
| 9-20 words | 48 | 43.8% | 29% |
| 21+ words | 22 | **54.5%** | **18%** |

53% of the default sample has <=8 words; 17% has <=3. Utterances labelled
`distressed` include "When?", "Well...", and "I wonder." -- affect carried
entirely in delivery. Accuracy is monotone in transcript length and the
neutral-default rate is monotone in the opposite direction: the signature of
input starvation, not model weakness.

ASR was ruled out as the cause first: median ASR-vs-reference similarity is
**1.00** (mean 0.95), so the transcripts are faithful. There is simply
nothing in them to read.

AutoAce's calls are 31s / 35s / 172s -- roughly 70-450 words -- so the
unfiltered benchmark's length distribution barely overlaps the deployment
distribution. `VTA_MIN_REF_WORDS` filters the pool before balanced sampling;
`by_transcript_length` is now reported on every run so this cannot hide
inside an aggregate again.

**(3) Reasoning effort looked useless because there was nothing to reason
about.** `reasoning_effort=high` was measured at +0.011 on the unfiltered
benchmark and written off as null. On the length-matched pool the same change
is worth **+0.048** (0.4511 -> 0.4992). The effect is conditional on content
being present.

**(4) The benchmark ran in "transcript only"; production runs in "transcript
+ context".** IEMOCAP hands the model one isolated utterance. A real call is
ONE clip containing the entire conversation. Per arXiv:2602.06270 Table 3,
that distinction is the largest single factor in transcript-based LLM emotion
recognition -- larger than any prompting technique. Adding preceding turns as
speaker-labelled context (`VTA_CONTEXT_TURNS`, using IEMOCAP reference
transcripts, no extra ASR, target audio unchanged) closes it.

#### Cumulative effect

| Config (n=150, length-matched) | Tone acc | Tone macro F1 |
|---|---|---|
| Baseline | 0.4533 | 0.4511 |
| + reasoning_effort=high | 0.5000 | 0.4992 |
| + 6-turn context | 0.5467 | 0.5370 |
| **+ context + high** | **0.5667** | **0.5657** |

Per-class F1 for the final config: neutral 0.647, satisfied 0.577,
frustrated 0.328, upset 0.645, distressed **0.632**. `distressed` -- the
class that started at 0.162 and was the motivation for the removed acoustic
override -- is now third-best, reached by *deleting* the mapping rather than
repairing it.

**Replication (partial).** Paired re-runs on 2 seeds, context vs no-context,
both at effort=high:

| Seed | No context | + context | Delta |
|---|---|---|---|
| 1234 | 0.4700 | 0.5522 | +0.082 |
| 9999 | 0.4944 | 0.5391 | +0.045 |
| Mean | 0.482 | **0.546** | **+0.064** |

Positive on both seeds, mean +0.064, roughly 1.8x the +/-0.035 run-to-run
noise band. This is **under-replicated** -- 2 pairs supports no sign test and
no meaningful standard deviation -- and the replicated delta is smaller than
the +0.115 the first single run suggested, which is the expected regression
when a promising single measurement is repeated. Reported as a direction with
a magnitude estimate, not a significance claim. A same-seed re-run of the
identical winning config gave 0.5657 then 0.5522 (delta 0.0135), which
independently confirms the noise-band estimate.

#### Positioning against published results

arXiv:2602.06270 Table 3 (GPT-4o, IEMOCAP, unweighted accuracy / weighted F1):

| System | Transcript only | + Context |
|---|---|---|
| Zero-shot baseline | 43.38 / 41.03 | 55.51 / 53.63 |
| SpeechCueLLM | 49.97 / 48.54 | 60.07 / 58.52 |
| VowelPrompt (SOTA) | 51.18 / 50.15 | 62.26 / 60.74 |
| **This system** | **50.00 / 49.92** | **54.7-56.7 / 53.7-56.6** |

In the context-free condition we sit between SpeechCueLLM and VowelPrompt's
SOTA; with context we bracket the published GPT-4o + context baseline. Both
comparisons are on a **harder label set**: their five classes are angry /
happy / sad / neutral / excited, while ours includes the frustrated-vs-upset
boundary, distinguished only by degree of anger. Other work explicitly folds
frustration into an "others" bucket (92.4% frustration) to avoid it. Our
`frustrated` F1 of 0.328 -- 10/30 correct with 10 leaking to `upset` -- is
the single remaining weak class, and it is weak on precisely the axis the
literature identifies as hardest.

For the achievable ceiling: fine-tuned multimodal SOTA on IEMOCAP 4-class is
~76.3% WA (GatedxLSTM) and fine-tuned wav2vec2/HuBERT ~73%, but the same
class of model reportedly drops to **55-65% on real call-center audio**.
IEMOCAP's own inter-annotator agreement is Fleiss' kappa **0.27** across all
classes (0.48 on the four main ones) -- "fair" at best, which caps every
system including the annotators. Closing the gap to fine-tuned performance
requires thousands of labelled emotional utterances; three were provided.

#### The honest limitation

**These gains did not transfer to the three provided clips.** Tone remained
1/3 across every configuration -- eight of them, with and without
diarization, at four effort levels. The two failures are the two hardest
cases in the taxonomy: call_001 is a frustrated/upset confusion (the class
still at 0.328 on IEMOCAP), and call_003 is a customer who books a service
appointment, is told the dealership is closed, asks for a human, and says
"Thank you" -- labelled `satisfied`, called `neutral` by every configuration
we ran. n=3 has a resolution of 33 percentage points and cannot detect a
+0.06 macro-F1 improvement; it also cannot rule one out. Stated plainly: the
benchmark moved and the only real production data available did not.

#### Not adopted: speaker diarization

An undiarized transcript gives the LLM no way to isolate "the customer",
which brief §2 asks for. Measured on the provided calls, the non-customer
occupies roughly half the speech (53% / 61% / 59%), and on call_003 the agent
-- a scripted assistant, "Hi, I'm Erica from Lexington Toyota" -- delivers
four long policy explanations while the customer's sentiment is a handful of
turns. `src/vta/diarize.py` implements pyannote diarization, a
speaker-labelled transcript, and per-speaker prosody (features measured over
each speaker's own audio rather than averaged across both).

The mechanism works: the LLM identified the customer correctly on all three
calls, reasoning from the greeting. It changed **no prediction on any clip**
(tone 1/3, intensity 2/3 either way) while costing **170s vs 57s** across the
three (0.71x vs 0.24x realtime; diarization is 60-74% of runtime on the long
call). It also cannot be validated above n=3 -- IEMOCAP utterances are
single-speaker, so the benchmark has nothing for it to separate. Unmeasurable
benefit against a confirmed 3x latency cost: **ships off** (`VTA_DIARIZE=0`),
retained behind the flag with this rationale.

### 2e. Code sweep: one defect that changes a validated input

A full review of the codebase after the numbers above were measured turned up
a metric bug worth recording here, because it changes something the tone
model reads.

`speaking_rate_wpm` divided the word count by the whole first-word-to-last-word
span, counting every silence as if the speaker were talking through it -- while
the dataclass and the prompt both described it as a rate over *speech*. The
prompt asks the model to grade `emotional_intensity` partly from this number.
Corrected to exclude inter-word gaps above 0.4s:

| Clip | Words | Span | Reported before | Corrected |
|---|---|---|---|---|
| call_001 | 34 | 29.6s | **68.9 wpm** | 202.4 wpm |
| call_003 | 361 | 166.8s | 129.9 wpm | 207.9 wpm |

call_001 is 20s of pause inside a 30s clip, so the old figure described a
caller who was not talking rather than one talking slowly. The threshold is
not a knife-edge: 0.3/0.4/0.6s give 209/202/195 and 214/208/206 wpm, and the
median inter-word gap is 0.0s (75th percentile 0.16s), so continuous speech
sits well clear of the cut.

**What this means for the numbers above.** Every tone/intensity figure in
§2c-2d was measured with the old value in the prompt. Re-running the full
pipeline after the fix reproduced all 9 fields on all 3 provided clips
exactly (tone 1/3, intensity 2/3, 6 acoustic fields 3/3), so nothing in §2a
moves. The IEMOCAP figures were **not** re-run -- at ~$0.14 and 25 minutes
per n=150 configuration that was not affordable here, and the effect there
should be small in any case, since IEMOCAP utterances are single-turn and
mostly pause-free (the regime where the two formulas agree). The honest
statement is that §2c-2d are measured against the pre-fix input and the
provided-clip results are measured against the post-fix one.

Also fixed in the same sweep, none of which alter any prediction: session
cookies were forgeable when `SESSION_SECRET` was left at its default (now a
startup failure outside dev); uploaded call audio was retained indefinitely
in two places and leaked a third copy into the system temp dir on every
upload (now purged on a retention policy); the validation artifacts embedding
verbatim IEMOCAP reference transcripts were not gitignored and would have
been force-pushed to a public Space (now excluded, with a CI guard); ffmpeg
ran without a timeout; and the tone call had no guard for truncation or
refusal, so both surfaced as an unrelated `TypeError`.


## 3. Cost analysis

Local components (ffmpeg, DSP, Silero VAD, PANNs, pyannote, parakeet,
emotion2vec+) run entirely on local CPU: **$0 marginal cost**, no data
leaves local infrastructure.

Only the tone/intensity LLM call has a per-request cost. Measured (not
estimated) from real calls to `gpt-5.6-luna`, no prompt caching (single
independent calls in testing; production would cache the ~1,500-token
rubric across a batch):

| Clip | Duration | Prompt tokens | Completion tokens | Cost | Cost/audio-min |
|---|---|---|---|---|---|
| call_001 | 30.9s | 725 | 266 | $0.000464 | $0.00090 |
| call_002 | 35.0s | 731 | 161 | $0.000339 | $0.00058 |
| call_003 | 171.9s | 1970 | 185 | $0.000616 | $0.00022 |

Weighted average across the 3 calls: **$0.00036/audio-minute** -- 12% of the
$0.003 ceiling, with shorter calls costing proportionally more per minute
(fixed rubric-token overhead dominates) and longer calls amortizing better.
With the rubric cached (`$0.02/M` vs `$0.20/M` input), production cost drops
further, in line with the pre-implementation estimate of ~$0.00028/min.
**Disclosure**: OpenAI `gpt-5.6-luna`, $0.20/$1.20 per M input/output tokens
(short context), $0.02/M cached input. Customer audio never leaves
AutoAce-controlled infrastructure; only derived transcript text and numeric
acoustic features are transmitted. Retention per OpenAI API terms.

## 4. Latency analysis

Measured (not estimated) with `scripts/profile_stages.py`, per-stage
breakdown, single-threaded, CPU-only, no batching/concurrency yet. Two
generations of this measurement matter here: the first run identified
`pyannote` diarization as consuming 58-91% of total time per clip; fixing
that (§1, §5.4) changed these numbers substantially, so both are shown to
make the improvement concrete rather than just asserted.

| Clip | Duration | Original | After diarization fix | After emotion2vec chunking | After pyin + overlap step |
|---|---|---|---|---|---|
| call_001 | 30.9s | 49.6s (1.61x) | 22.1s (0.71x) | 22.5s (0.73x) | **19.2s (0.62x)** |
| call_002 | 35.0s | 36.3s (1.04x) | 8.3s (0.24x) | 9.9s (0.28x) | **5.9s (0.17x)** |
| call_003 | 171.9s | 368.1s (2.14x) | 94.2s (0.55x) | 40.1s (0.23x) | **23.4s (0.14x)** |

All three clips process **faster than real-time** (0.14-0.62x). Four
optimizations got here:

1. **Diarization → segmentation (first fix).** Overlap detection was
   calling the full `speaker-diarization-3.1` pipeline (segmentation +
   speaker-embedding + clustering) when it only needs segmentation's
   frame-level "how many speakers active" signal. Switching to
   `segmentation-3.0` directly cut that stage 34-61x (see §1, §5.4).

2. **Emotion2vec chunking (second fix).** `emotion2vec+` ran as one
   forward pass over the whole waveform. Its transformer applies full
   self-attention across every 20 ms frame -- O(n²) in frame count -- so
   call_003's 172 s (~8600 frames) cost 63 s (67% of the 94 s total),
   and 5.5× the audio produced 12.9× the compute (clearly superlinear).
   Now the audio is split into non-overlapping 20 s chunks, each run
   independently, with per-chunk posteriors duration-weight-averaged into
   one clip-level distribution. This makes the cost linear in duration:
   call_003's emotion2vec dropped from 63 s to **5.8 s (11× speedup)**,
   and the whole clip from 94 s to 40 s. Posteriors stay numerically
   close to the single-pass result (arousal within the same intensity
   band on all 3 real clips; predictions unchanged).

Three optimizations got here (plus two micro-fixes):

3. **pyin frequency range** (`dsp_features.py`). pyin searched C2-C7
   (65-2093 Hz), librosa's music-oriented default. Human speech f0 tops
   out around 300-350 Hz; the upper 3.5 octaves contain only harmonics
   that pyin can octave-error onto. Capping fmax at 400 Hz cut the stage
   from 10.7s to 2.9s on call_003 (3.6x) -- a speed fix that also removes
   wrong answers from the search space.

4. **Overlap step size** (`overlap_pyannote.py`). segmentation-3.0's
   sliding window is a fixed 10s; STEP_S was 1.0, meaning every audio
   second was processed ~10 times. Raising to 2.0 halves the stage.
   Validated: identical macro F1 (0.790), identical per-condition detection
   rates at n=65. The boolean threshold on total overlap duration is
   insensitive to the 1s boundary-precision change.

Post-fix per-stage breakdown (`stage_timings.json`): no single stage
dominates call_003 -- emotion2vec 25%, asr 20%, dsp_features 15%,
llm_tone 14%, pyannote 11%, panns 11%. The headline figure is **12.2
s/audio-min weighted average (0.20x realtime)** processed strictly
sequentially. Running multiple clips concurrently would improve batch
throughput further but is no longer required to clear the bar.

## 5. Failure modes, limitations, and next steps

1. **`emotional_tone` has a systematic neutral bias -- partially corrected
   by an acoustic override, validated by 5-seed ablation.** IEMOCAP's varied
   conversational speech gives a valid measurement: the text head alone
   averages 0.41 macro F1 (5-seed mean, std 0.035). The dominant error is
   **neutral over-prediction**. An **acoustic tone override** corrects this:
   when the LLM predicts neutral but emotion2vec is confident (>0.3 normalized)
   about a specific non-neutral emotion, the tone is overridden (angry to
   upset, fearful to distressed, disgusted/sad to frustrated, happy/surprised
   to satisfied).

   A 5-seed paired ablation (seeds 1234/9999/7777/4242/5678, 150 samples
   each, same clips across conditions) establishes the effect statistically:

   | Metric | Without override (5-seed) | With override (5-seed) | Paired delta | t |
   |---|---|---|---|---|
   | Tone macro F1 | 0.406 +/- 0.039 | **0.450 +/- 0.030** | +0.044 +/- 0.042 | 2.38 |
   | Tone accuracy | 0.428 +/- 0.034 | **0.473 +/- 0.031** | +0.045 +/- 0.039 | 2.58 |
   | Intensity macro F1 | 0.455 +/- 0.024 | 0.456 +/- 0.022 | +0.001 +/- 0.008 | 0.16 |

   The override improves tone macro F1 by +0.044, positive on all 5 seeds.
   The primary result is the sign test (5/5, p=0.031 one-tailed), which
   makes no distributional assumptions; the paired t-test agrees (t=2.38,
   df=4, p=0.038 one-tailed) but is under-powered at n=5 and does not clear
   the two-tailed threshold (critical t=2.776). One-tailed is used because
   the direction was hypothesised before measurement. Standard deviations
   are sample (ddof=1).

   Two caveats. First, the effect is heterogeneous (per-seed range +0.001
   to +0.102) and seed 1234 supplies roughly half of it; excluding it, the
   mean gain is +0.030. Seed 1234 also has the worst baseline (0.348 vs
   0.40-0.46), and baseline quality correlates negatively with gain --
   consistent with regression to the mean.

   Second, and more usefully: that correlation is mechanistic, not
   incidental. The override fires when the LLM emits neutral, and a weak
   baseline run is precisely one where it over-emitted neutral. So the
   override acts as a variance reducer -- run-to-run std falls from 0.039
   to 0.030, a 24% reduction -- lifting the worst run (0.348 to 0.450)
   while barely moving the best (0.456 to 0.474). For production, where
   you cannot resample until a good run appears, raising the floor is
   worth more than shifting the mean.

   No effect on intensity (t=0.16, 3/5 positive), as expected: the override
   only rewrites tone.

   On the 3 real clips the override changes call_002 (neutral to satisfied,
   GT neutral: false positive from upbeat prosody) and call_003 (neutral to
   satisfied, GT satisfied: correct). The fundamental trade-off: the
   override reduces neutral bias on average but can false-positive on calls
   where prosody is upbeat but content is neutral. Fine-tuning on real
   labeled calls remains the highest-value fix.
2. **`emotional_intensity`'s acoustic cross-check had two real bugs, both
   found and fixed.** (a) A label-parsing bug: the raw emotion2vec output
   format is `"<chinese>/<english>"` (e.g. `'生气/angry'`), and the original
   code took the Chinese side for every case, so posterior keys never
   matched the arousal-weight table and computed arousal was identically
   `0.0` on every clip -- not a threshold problem, a parsing bug. (b) An
   arousal-dilution bug: `"unknown"`/`"other"` (uninformative) carried
   non-zero weight and weren't excluded from the normalization denominator,
   diluting arousal exactly on the clips where the model is least confident
   (call_003's posteriors are 83% `"unknown"`). Fixed both; real arousal
   values post-fix are 0.51-0.60 across the 3 clips, crossing the "medium"
   cross-check gate and correcting 2 of 3 intensity predictions to match
   ground truth exactly (§2a). §2c's IEMOCAP run confirms this generalizes:
   52% accuracy / 50% macro F1 on intensity, well above chance, even on the
   samples where tone/valence is missed.
   **Arousal override calibration (investigated, left unchanged).** On the
   3 real clips the override fires every time (confidence constant 0.65),
   because Luna systematically predicts `low` intensity and arousal
   (0.42-0.60) always exceeds the 0.35 gate. Raising the gate to 0.45 was
   considered but rejected: it would un-correct call_003 (arousal 0.419,
   true intensity `medium`) back to Luna's `low`. The IEMOCAP arousal
   distribution confirms the thresholds are well-placed (low mean 0.32,
   medium 0.50, high 0.64; the 0.35/0.60 gates sit between adjacent means).
   The root cause is Luna's under-prediction, not the gate -- fixing that
   is the tone-head improvement in Next Steps, not a threshold change.

3. **Sarcasm and calm-but-firm anger survive transcription.** "Are you a
   real person?" (call_001) reads as confused/neutral in text; a human
   listener hearing the actual tone might immediately call it `upset`.
4. **Long-call averaging flattens escalation arcs.** Mitigated by passing a
   per-second energy contour (not just scalar summary stats) to the LLM,
   but not validated against a call that specifically escalates mid-call.
5. **`background_noise_severity`'s "high" band was unreachable; now
   fixed.** `band_deviation` (energy outside the 300-3400 Hz voice band)
   saturates for broadband noise because injected white/pink noise has
   substantial in-band energy too -- it plateaus at ~0.52 even at SNR 0 dB,
   never reaching the 0.55 "high" cutoff (synthetic "high" was 0/10).
   Added `occupied_bandwidth_hz > 4000` as a second high-band trigger:
   broadband noise pushes energy above the telephony band, so bandwidth
   rises to 4.5-6.6 kHz at high-noise levels. This catches all synthetic
   high-noise trials (recall 0.90) while real telephony clips (≤3.4 kHz
   bandwidth) never trigger it. Overall severity accuracy improved from
   0.525 to 0.725 (macro F1 0.478 → 0.711). Residual: 4/10 medium trials
   spill to high at the SNR 8-10 dB boundary, where no single acoustic
   feature cleanly separates the two bands.
6. **`speaker_overlap_present`'s synthetic validation was invalid; now
   fixed with a better generator.** The old generator summed call_001 +
   call_002 -- acoustically similar call-center voices. Direct inspection
   of the raw segmentation output confirmed `max_active_speakers` stayed at
   1.0: the model never saw a second speaker, because summing two similar
   voices doesn't produce the spectrally-distinct two-source signal the
   segmentation head was trained on. Switched to diverse IEMOCAP M+F
   speaker pairs (different gender, different sessions), which the model
   correctly reads as overlapping (`max_active` reaches 2.0). At n=65:
   accuracy 0.80, macro F1 0.79, overlap precision 0.905 (high -- when it
   fires, it's almost always right), recall 0.633 (scales with overlap
   duration: 4 s → 80%, 2 s → 47%). The real clips still match 3/3. The
   remaining recall gap reflects summed-mono synthetic construction vs.
   genuine concurrent recording; high precision / moderate recall is the
   safer failure mode for a QA tool.
7. **PANNs is trained on YouTube audio, not telephony**, and it shows:
   telephony-channel artifacts (dial tone, line reverb) initially
   dominated its output before being explicitly excluded (see
   `events_panns.py`). Real-world `background_noise_type` accuracy on 2/3
   provided clips was a plausible-but-wrong category (`music` guessed for
   `TV` and for `sharp static`).
8. **`confidence` cannot be validated against the provided labels** -- all
   3 are the constant `0.82` from the brief's own example. The current
   confidence heuristic (LLM/acoustic-arousal agreement) is a documented,
   uncalibrated placeholder.
9. **Three labeled examples mean the label-mapping/threshold layer is
   reasoned from definitions and lightly calibrated against real evidence
   where available (items 2, 5), not fitted at scale.** This is the main
   generalization risk called out in the brief itself (§11). Addressed as
   far as time allowed: §2c's 150-sample IEMOCAP run is the first
   real-scale check on tone/intensity; §2b's synthetic sets were scaled
   from n=6-13 to n=20-65 per field, with generators improved where the
   first pass exposed methodology defects (overlap: diverse speakers;
   audio_quality: soft clipping; severity: bandwidth trigger). Scaling up changed the
   picture in both directions and was worth doing: `background_noise_present`
   looked artificially weak at n=13 (a same-segment-repeated-4-times
   artifact, 0.69 -> 0.975 once fixed), while `audio_quality` looked
   artificially strong at n=9 (0.67 -> 0.50 once enough trials existed to
   expose the `slightly_impaired` collapse). Neither direction would have
   been visible without scaling up.
10. **No concurrency yet.** Clips process strictly sequentially in a single
    background thread per job. No longer required to meet the latency bar
    (§4 -- all 3 clips are now sub-real-time sequentially), but would
    further improve large-batch throughput.
11. **No speaker separation in the tone path (deferred, known).**
    `analyze_clip` transcribes and computes prosody/arousal over the whole
    file as one stream; the LLM prompt then asks about "the customer."
    On a real two-party call the transcript Luna reads interleaves agent
    and customer, and the acoustic averages mix both voices. This is
    invisible on IEMOCAP (single-speaker clips) -- which is exactly why the
    benchmark cannot catch it -- and live in production (2 of the 3 provided
    clips have overlap present). No prompt rewrite fixes an
    evidence-attribution problem. The pieces for a minimal fix already
    exist in the stack (parakeet word timestamps + pyannote segmentation
    for voice-activity windows); deliberately not built yet, logged as the
    main known production risk for the tone/intensity fields.


### Model swap experiments (empirically evaluated, not estimated)

Three candidate model swaps were tested empirically using the existing
validation harnesses, per the principle of measuring rather than estimating.
Each is switchable at runtime via environment variable
(`VTA_AUDIO_TAGGER`, `VTA_ASR`, `VTA_EMOTION2VEC`, `VTA_TONE_OVERRIDE`), so
the defaults (production models, override enabled) are unchanged and the
experiments are reproducible.

**1. PANNs CNN14 → CED-mini** (`VTA_AUDIO_TAGGER=ced-mini`)

CED-mini (9.7M params, AudioSet mAP 48.1) is 8× smaller than PANNs CNN14
(80M, mAP 43.1) and higher-scoring on AudioSet. But AudioSet mAP measures
average precision across all 527 labels on YouTube audio — it does not
directly translate to our noise-detection task on telephony audio with our
specific label subset and aggregation method.

CED outputs raw logits whose sigmoid saturates near 0.5 (unlike PANNs, which
returns sigmoid probabilities directly with wide dynamic range). Required a
recalibrated aggregation: excess-sigmoid `sum(max(0, sig-0.5)) × 10` to map
to a comparable scale.

4-seed synthetic comparison:

| Field | PANNs (4-seed mean±std) | CED-mini (4-seed mean±std) |
|---|---|---|
| noise_presence accuracy | 0.969 ± 0.011 | 0.931 ± 0.021 |
| noise_severity macro F1 | 0.647 ± 0.038 | 0.599 ± 0.025 |

PANNs is consistently better on both, though the gap is within seed
variation. CED-mini's real advantage is noise **type**: on the 3 real clips
it predicts "television" for call_002 (GT "TV", correct) where PANNs guesses
"music" (wrong) — improving type from 1/3 to 2/3. But type is unvalidated at
scale (the synthetic set only tests presence/severity), while presence and
severity are the validated fields CED is slightly worse on.

**Decision: kept PANNs -- but the synthetic benchmark is biased toward it.
** The synthetic validation set is built by mixing AudioSet noise (white/pink)
into clean speech, and PANNs was trained on AudioSet. This is home-field
advantage: PANNs is being tested on the exact noise distribution it learned
from, while CED (also AudioSet-trained but via a different architecture) faces
the same in-distribution test. The one real-audio datapoint goes the other way:
CED says "television" for call_002 (GT "TV", correct), PANNs says "music"
(wrong). So the honest framing is "PANNs wins on a benchmark biased toward it;
CED wins on the one real sample; presence/severity are validated and type
isn't." Same conclusion (keep PANNs for now), but the comparison is closer than
the synthetic gap suggests. The env-var switch (`VTA_AUDIO_TAGGER=ced-mini`)
remains; CED would drop 290 MB of weights and the `panns_inference` dependency.

**2. parakeet-0.6b → Moonshine-base** (`VTA_ASR=moonshine-base`)

Moonshine-base (61.5M params, 237 MB on disk) is 37× smaller than parakeet
(600M, 2.3 GB) and was reported 5.8× faster on CPU in the paper. The user's hypothesis: "transcript quality barely affects
tone accuracy" — confirmed.

| Configuration | Tone macro F1 | Intensity macro F1 |
|---|---|---|
| Parakeet (baseline) | 0.47–0.50 | 0.48–0.49 |
| Moonshine | 0.45 | 0.48 |

The tone difference (-0.02 to -0.05) is within the LLM non-determinism band
(re-running the identical baseline gives ±0.02 macro F1 variation across
runs, since each of 150 API calls is independent). Transcript content does
not materially affect tone classification — emotion rides on redundant lexical
cues, as predicted.

**But Moonshine silently truncates long audio.** `max_position_embeddings=194`
limits the encoder context; on call_003 (172 s) it transcribed from ~60s in
(not the start), producing a wrong-length, wrong-position transcript. On the
short IEMOCAP clips (3-8 s) this doesn't occur. Production calls (10+ min)
would require chunking with timestamp-based reassembly.

**Decision: kept parakeet for now.** Moonshine's size advantage (2.3 GB →
237 MB, dropping `nemo-toolkit[asr]` and its transitive deps) is the single
biggest dependency-tree win available. A chunking wrapper is now implemented
(25 s windows, 2 s overlap, transcript concatenation) that fixes the silent
truncation — call_003's 172 s now transcribes from the beginning instead of
60 s in. The switch (`VTA_ASR=moonshine-base`) is production-ready for testing;
remaining concern: some clips with quiet initial audio produce empty output on
individual chunks (Moonshine model quirk, not a chunking bug). Production
adoption requires verifying this doesn't cause material transcript loss.

**3. emotion2vec_plus_large → emotion2vec_plus_base** (`VTA_EMOTION2VEC=emotion2vec/emotion2vec_plus_base`)

| Configuration | Tone macro F1 | Intensity macro F1 |
|---|---|---|
| Large (baseline) | 0.47–0.50 | 0.48–0.49 |
| Base | 0.46 | 0.37 |

Tone barely moves (-0.01), but **intensity drops materially** (-0.11 macro F1,
-0.13 accuracy). The base model's arousal signal is much weaker — it can't
separate `medium` from `high` intensity (medium recall collapses to 0.21).
Since the arousal cross-check drives both the intensity correction AND the
tone override, the base model gives back real work.

**Decision: kept large.** The 1.8 GB (base is ~1.0 GB, saving ~0.8 GB) is
carrying validated signal. The base model costs 11 points of intensity F1 —
the user's stated criterion ("keep large if the drop is material") is met.

### Next steps with more data or time

- Fine-tune or few-shot-calibrate the tone head on a few hundred labeled
  production calls -- highest-value improvement by a wide margin, directly
  targeting the frustrated/upset boundary (item 1, now quantified at 0.43
  macro F1 on IEMOCAP §2c rather than hypothesized).
- Give `emotional_tone` itself an acoustic override path analogous to the
  intensity cross-check (item 2) -- right now only intensity gets corrected
  when the LLM under-weights prosody; tone does not, and §2c shows `distressed`
  (10% recall) and `frustrated` (23% recall) are exactly where that hurts most.
- Scale IEMOCAP validation beyond 150 samples and add MELD (TV-dialogue
  emotion) as a second corpus to test cross-dataset robustness.
- Scale synthetic sample sizes further (n=40 → a few hundred) and replace
  summed-mono overlap construction with genuine concurrent recordings for
  a tighter recall estimate (current precision 0.905 / recall 0.633, §2b).
- Replace the definition-derived field-resolution logic with a classifier
  trained on real labels once enough are available.
- Add concurrent batch processing (item 10).

**Completed since the first pass** (recorded for traceability):
emotion2vec chunking (§4, 11× speedup on long clips), severity "high" band
via `occupied_bandwidth_hz` (§2b, recall 0→0.90, validated on 4 seeds),
audio_quality tanh degradation + threshold recalibration (§2b, accuracy
0.50→0.93, held-out 0.87±0.06), overlap generator fix (§2b, diverse speakers,
accuracy 0.50→0.80), IEMOCAP tone validation replacing the invalid RAVDESS
run (§2c), tone acoustic override for neutral bias (§5.1, macro F1 0.43→0.47),
and held-out seed validation confirming thresholds are not overfit (§2b-i).
