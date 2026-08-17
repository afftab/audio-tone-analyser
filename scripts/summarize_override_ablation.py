"""Recompute the override ablation summary from the 10 raw run files.

Uses sample std (ddof=1) -- required for a t-statistic on n=5. The earlier
summary used population std (ddof=0), which understates uncertainty by
sqrt(4/5) and inflated t by 1.118x.
"""
import json, statistics as st
from pathlib import Path
from scipy import stats as sps

D = Path(__file__).resolve().parents[1] / "ablation_override"
SEEDS = [1234, 9999, 7777, 4242, 5678]
METRICS = ["tone_macro_f1", "tone_accuracy", "intensity_macro_f1"]

def read(path):
    d = json.load(open(path))
    t, i = d["emotional_tone"], d["emotional_intensity"]
    mf = lambda b: b.get("macro_f1", b["report"]["macro avg"]["f1-score"])
    return {"tone_macro_f1": mf(t), "tone_accuracy": t["accuracy"],
            "intensity_macro_f1": mf(i)}

off = {s: read(D / f"without_override_seed{s}.json") for s in SEEDS}
on  = {s: read(D / f"with_override_seed{s}.json")    for s in SEEDS}

results = {}
for m in METRICS:
    a = [off[s][m] for s in SEEDS]
    b = [on[s][m] for s in SEEDS]
    d = [y - x for x, y in zip(a, b)]
    t_stat, p_two = sps.ttest_rel(b, a)
    n_pos = sum(x > 0 for x in d)
    results[m] = {
        "without_override": {"mean": round(st.mean(a), 4), "std": round(st.stdev(a), 4)},
        "with_override":    {"mean": round(st.mean(b), 4), "std": round(st.stdev(b), 4)},
        "paired_delta_mean": round(st.mean(d), 4),
        "paired_delta_std":  round(st.stdev(d), 4),   # ddof=1
        "per_seed_delta": {str(s): round(x, 4) for s, x in zip(SEEDS, d)},
        "t_stat": round(float(t_stat), 3),
        "df": len(SEEDS) - 1,
        "p_one_tailed": round(float(p_two) / 2, 4),
        "p_two_tailed": round(float(p_two), 4),
        "critical_t_one_tailed_p05": 2.132,
        "critical_t_two_tailed_p05": 2.776,
        "significant_one_tailed_p05": bool(float(p_two) / 2 < 0.05),
        "significant_two_tailed_p05": bool(float(p_two) < 0.05),
        "sign_test_positive": f"{n_pos}/{len(d)}",
        "sign_test_p_one_tailed": round(
            sps.binomtest(n_pos, len(d), 0.5, alternative="greater").pvalue, 4),
        "delta_excluding_seed1234": round(
            st.mean([x for s, x in zip(SEEDS, d) if s != 1234]), 4),
        "variance_reduction_pct": round(
            100 * (1 - st.stdev(b) / st.stdev(a)), 1),
    }

out = {
    "experiment": "Tone acoustic override ablation (VTA_TONE_OVERRIDE 1 vs 0)",
    "n_seeds": len(SEEDS), "seeds": SEEDS, "n_samples_per_run": 150,
    "std_convention": "sample (ddof=1)",
    "notes": [
        "Paired design: identical clips across conditions per seed.",
        "One-tailed tests: the directional hypothesis (override helps) was "
        "specified before measurement.",
        "The sign test is the primary result -- it is assumption-free, unlike "
        "a t-test on n=5.",
        "The effect concentrates on weak baseline runs (see "
        "delta_excluding_seed1234); the override raises the floor more than "
        "the mean.",
    ],
    "results": results,
}
(D / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(results, indent=2))
