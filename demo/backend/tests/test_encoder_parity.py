"""The ONNX query encoder matches the study's sentence-transformers encoding (needs built artifacts)."""
import json
from pathlib import Path

import numpy as np
import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
ROOT = Path(__file__).resolve().parents[3]

pytestmark = [
    pytest.mark.artifacts,
    pytest.mark.skipif(not (ARTIFACTS / "onnx" / "model.onnx").exists(),
                       reason="no demo/backend/artifacts: run demo/build/build_artifacts.py"),
]


@pytest.fixture(scope="module")
def queries() -> list[str]:
    from lumbr import data
    dev = (ROOT / "splits" / "dev_question_ids.txt").read_text(encoding="utf-8").split()[:20]
    qs = data.load_questions().set_index("question_id")
    return [qs.loc[q, "question"] for q in dev]


def test_onnx_matches_sentence_transformers(queries):
    from sentence_transformers import SentenceTransformer

    from lumbr_demo.encoder import OnnxEncoder
    from lumbr_demo.index import DenseIndex
    config = json.loads((ARTIFACTS / "config.json").read_text(encoding="utf-8"))
    onnx = OnnxEncoder(ARTIFACTS / "onnx", config["query_prefix"])
    st = SentenceTransformer(config["model"], revision=config["model_revision"], device="cpu")
    ref = st.encode([config["query_prefix"] + q for q in queries], normalize_embeddings=True)
    index = DenseIndex.load(ARTIFACTS)
    for q, r in zip(queries, ref):
        v = onnx.encode(q)
        assert float(v @ r) >= 0.999, q
        assert [i for i, _ in index.search(v, 5)] == [i for i, _ in index.search(r.astype(np.float32), 5)], q
