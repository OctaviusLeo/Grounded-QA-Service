"""Builds the pipeline's parts from the configuration (one place, shared by every entry point)."""

from __future__ import annotations

from gqa.config import Config
from gqa.ingest import read_passages
from gqa.model_client import ModelClient
from gqa.pipeline import AnsweringPipeline
from gqa.prompt import PromptBuilder
from gqa.retrieval import RagLiteBackend, RagLiteRetriever, RetrievalMode
from gqa.validate import CitationValidator


def load_backend(cfg: Config, dense: bool = True) -> RagLiteBackend:
    return RagLiteBackend(read_passages(cfg.corpus_path), use_embeddings=dense, use_reranker=dense,
                          weights=cfg.hybrid_weights, candidates=cfg.candidates)


def make_client(cfg: Config, model: str, **kwargs) -> ModelClient:
    ms = cfg.model_server
    return ModelClient(ms.base_url, model, ms.api_key_env, ms.timeout_s, cfg.seed,
                       api_key_required=ms.api_key_required, max_tokens=ms.max_tokens,
                       temperature=ms.temperature, **kwargs)


def make_pipeline(cfg: Config, backend: RagLiteBackend, mode: str, model: str, client=None,
                  threshold: float | None = None) -> AnsweringPipeline:
    return AnsweringPipeline(
        retriever=RagLiteRetriever(RetrievalMode(mode), backend, seed=cfg.seed),
        prompt_builder=PromptBuilder(),
        model_client=client or make_client(cfg, model),
        validator=CitationValidator(),
        threshold=cfg.threshold_for(mode) if threshold is None else threshold,
        k=cfg.k,
        seed=cfg.seed,
        candidates=cfg.candidates,
    )
