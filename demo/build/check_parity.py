"""Does the demo serve the study's B0? Run every dev question through the service's Retriever on the
built artifacts and compare with the study's saved R1+R4 RRF dev run (notebook 17). Dev questions only.

Pass: recall@10 matches the study's (0.766) and the top-10 lists are (nearly) all identical. Small
differences can come from the query encoder (ONNX fp32 on CPU here, sentence-transformers fp16 on GPU
in the study); anything larger means the demo isn't serving the study's setup.

It also reports memory, for choosing the Lightsail plan (target: the 4 GB plan). Linux counts touched pages of
memory-mapped files (the dense vectors, the Lucene index) in RSS, but the OS can drop those at will; "anon_mb"
is what the process really holds (Python, the JVM heap, ONNX), "file_mb" is reclaimable cache.

    python demo/build/check_parity.py --artifacts /content/b0_artifacts --study-run <runs/r1_r4_rrf_dev_...>
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo" / "backend"))

DEV_IDS = ROOT / "splits" / "dev_question_ids.txt"


_STATUS_FIELDS = {"VmHWM": "peak_rss_mb", "VmRSS": "rss_mb", "RssAnon": "anon_mb", "RssFile": "file_mb"}


def memory_report(status: str | None) -> dict:
    """MB figures from /proc/self/status text (Linux); {} where /proc isn't available."""
    if not status:
        return {}
    out = {}
    for line in status.splitlines():
        key, _, rest = line.partition(":")
        if key in _STATUS_FIELDS and rest.strip().endswith("kB"):
            out[_STATUS_FIELDS[key]] = int(rest.split()[0]) // 1024
    return out


def _proc_status() -> str | None:
    try:
        return Path("/proc/self/status").read_text()
    except OSError:
        return None


def compare_top_k(served: dict, study: dict, k: int) -> dict:
    """Per-question agreement of the top-k passage lists, over the questions both runs have."""
    shared = sorted(set(served) & set(study))
    identical = same_set = 0
    overlaps, differ = [], []
    for q in shared:
        a = [p for p, _ in served[q][:k]]
        b = [p for p, _ in study[q][:k]]
        identical += a == b
        same_set += set(a) == set(b)
        overlaps.append(len(set(a) & set(b)) / max(len(b), 1))
        if a != b:
            differ.append(q)
    return {"n": len(shared), "identical": identical, "same_set": same_set,
            "mean_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
            "missing": sorted(set(study) - set(served)), "differ": differ}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--study-run", type=Path, required=True, help="the study's r1_r4_rrf_dev_<date> folder")
    args = ap.parse_args()

    import time

    from lumbr import data, eval as ev
    from lumbr_demo.artifacts import load_retriever
    from lumbr_demo.service import FIRST_STAGE_K

    config = json.loads((args.artifacts / "config.json").read_text(encoding="utf-8"))
    if config.get("threshold") is None:  # parity ignores the filter; load_retriever needs a number
        config["threshold"] = 0.0
        (args.artifacts / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print("[parity] threshold was null; set to 0.0 temporarily. Run pick_threshold.py afterwards.")
    r = load_retriever(args.artifacts)

    dev = set(DEV_IDS.read_text(encoding="utf-8").split())
    qs = data.load_questions()
    qs = qs[qs.question_id.isin(dev)]
    served, times = {}, []
    for _, q in qs.iterrows():
        t = time.perf_counter()
        hits, _ = r._search(q.question, FIRST_STAGE_K)
        times.append(time.perf_counter() - t)
        served[str(q.question_id)] = hits
    raw = json.loads((args.study_run / "run.json").read_text(encoding="utf-8"))
    study = {qid: [tuple(h) for h in hits] for qid, hits in raw.items()}

    def recall(run):
        res = ev.evaluate_run(run, tag="parity", split="dev", k=10, questions_df=data.load_questions(),
                              split_ids=sorted(dev), write=False)
        return res["overall"]["recall@10"]

    times.sort()
    report = {"served_recall@10": round(recall(served), 4), "study_recall@10": round(recall(study), 4),
              "top10": compare_top_k(served, study, 10),
              "latency_s": {"median": round(times[len(times) // 2], 3), "p95": round(times[int(len(times) * 0.95)], 3)},
              "memory_after_all_dev_queries": memory_report(_proc_status()),
              "jvm_heap_cap": os.environ.get("LUMBR_JVM_HEAP", "1g")}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
