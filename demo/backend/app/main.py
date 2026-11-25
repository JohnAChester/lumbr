"""HTTP API for the demo search service (contract: docs/superpowers/specs/2026-11-23-interface-skeleton-design.md)."""
from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints

from lumbr_demo.service import MAX_HITS, Retriever

PassageId = Annotated[str, StringConstraints(pattern=r"^dsid_[0-9a-z]+#\d{3}$")]
Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
Turn = Annotated[str, StringConstraints(max_length=600)]


async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """422 without echoing the input: FastAPI's default echoes it, and text that isn't valid UTF-8
    (an emoji cut in half, a lone surrogate) then fails to encode and becomes a 500."""
    detail = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
              for e in exc.errors()]
    return JSONResponse({"detail": detail}, status_code=422)


class HistoryTurn(BaseModel):
    q: Turn
    a: Turn


class RetrieveBody(BaseModel):
    query: Query
    k: int = Field(5, ge=1, le=20)
    context_ids: list[PassageId] = Field(default_factory=list, max_length=MAX_HITS)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=6)


class PassagesBody(BaseModel):
    ids: list[str] = Field(max_length=16)


def create_app(retriever: Retriever | None = None, key: str | None = None) -> FastAPI:
    """With no arguments, the retriever and key load from the environment on first request."""
    app = FastAPI(title="Lumbr demo search")
    app.add_exception_handler(RequestValidationError, _validation_error)
    state: dict = {"retriever": retriever, "key": key}

    def get_retriever() -> Retriever:
        if state["retriever"] is None:
            from lumbr_demo.artifacts import load_retriever
            state["retriever"] = load_retriever(Path(os.environ.get("LUMBR_ARTIFACTS", "artifacts")))
        return state["retriever"]

    def authorised(authorization: str | None = Header(None)) -> None:
        want = state["key"] if state["key"] is not None else os.environ.get("LUMBR_BACKEND_KEY", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str (a 500 instead of a 401).
        given = (authorization or "").encode("utf-8", "replace")
        if not want or not given or not hmac.compare_digest(given, f"Bearer {want}".encode("utf-8")):
            raise HTTPException(401, "bad or missing key")

    @app.get("/health")
    def health() -> dict:
        r = get_retriever()
        return {"ok": True, "retriever": r.name, "passages": r.store.count(), "threshold": r.threshold,
                "prompt_sha256": r.prompt["sha256"]}

    @app.post("/retrieve", dependencies=[Depends(authorised)])
    def retrieve(body: RetrieveBody) -> dict:
        return get_retriever().retrieve(body.query, body.k, body.context_ids,
                                        [h.model_dump() for h in body.history])

    @app.post("/passages", dependencies=[Depends(authorised)])
    def passages(body: PassagesBody) -> dict:
        return {"passages": get_retriever().passages(body.ids)}

    return app


app = create_app()
