"""Time a cross-encoder reranker on CPU at several rerank depths and thread counts (#26).

The numbers in docs/superpowers/specs/2026-11-24-demo-setup-decision-rule.md come from this script.
Uses real dev questions and random real passages: latency depends on length, not relevance.

Usage:
    RR_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2 [RR_MAX_LENGTH=512] \
      python -u scripts/bench_reranker.py <passages.jsonl> <questions.parquet> <dev_ids.txt> [threads...]
"""
import json
import os
import random
import sys
import time

import pandas as pd
import psutil
import torch
from sentence_transformers import CrossEncoder

MODEL = os.environ["RR_MODEL"]
MAX_LENGTH = int(os.environ.get("RR_MAX_LENGTH", "512"))
DEPTHS = (10, 20, 50, 100)

passages_path, questions_path, dev_ids_path = sys.argv[1:4]
thread_counts = [int(t) for t in sys.argv[4:]] or [1, 2]

random.seed(0)
rows = [json.loads(line) for line in open(passages_path, encoding="utf-8")]
texts = [r["header"] + "\n" + r["body"] for r in rows]
dev_ids = set(open(dev_ids_path).read().split())
qs = pd.read_parquet(questions_path)
dev_q = qs[qs["question_id"].isin(dev_ids)]["question"].tolist()[:5]
print(f"{len(texts)} passages, mean tokens {sum(r['n_tokens'] for r in rows) / len(rows):.0f}; "
      f"{len(dev_q)} dev questions")

proc = psutil.Process(os.getpid())
rss0 = proc.memory_info().rss
t = time.perf_counter()
model = CrossEncoder(MODEL, max_length=MAX_LENGTH, device="cpu")
print(f"load {time.perf_counter() - t:.1f}s, model RSS +{(proc.memory_info().rss - rss0) / 2**20:.0f} MB")

for threads in thread_counts:
    torch.set_num_threads(threads)
    model.predict([(dev_q[0], texts[0])])  # warm-up
    for depth in DEPTHS:
        times = []
        for q in dev_q:
            cands = random.sample(texts, depth)
            t = time.perf_counter()
            model.predict([(q, p) for p in cands], batch_size=depth)
            times.append(time.perf_counter() - t)
        times.sort()
        print(f"threads={threads} depth={depth}: median {times[len(times) // 2]:.2f}s  "
              f"max {times[-1]:.2f}s  peak RSS {proc.memory_info().rss / 2**20:.0f} MB")
