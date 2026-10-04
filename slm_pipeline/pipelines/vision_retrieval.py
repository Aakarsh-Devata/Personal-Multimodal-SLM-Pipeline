"""Caption-only local retrieval for model observations; no semantic or ASR stage."""

import json
import math
from pathlib import Path

import numpy as np

from slm_pipeline.config import config
from slm_pipeline.pipelines.phase_7_embed import EmbeddingGenerator, VectorStore


class CaptionOnlyRetriever:
    """Index model captions as evidence without turning them into verified facts."""

    def __init__(self, embedder=None, vector_store=None):
        self.embedder = embedder if embedder is not None else EmbeddingGenerator()
        self.vector_store = vector_store if vector_store is not None else VectorStore(
            dimension=self.embedder.dimension,
            memory_dir=Path(config["paths"]["memory_dir"]) / "vision_retrieval",
        )
        if self.embedder.dimension != self.vector_store.dimension:
            raise ValueError("Embedder and vector store dimensions must match")

    @staticmethod
    def observation_documents(caption_data):
        """Return timestamped, model-provenance documents from Phase 4 output."""
        if not isinstance(caption_data, dict) or not isinstance(caption_data.get("video_id"), str):
            raise ValueError("vision captions must contain a video_id")
        captions = caption_data.get("captions")
        if not isinstance(captions, list):
            raise ValueError("vision captions must contain a captions list")
        documents = []
        for item in captions:
            if not isinstance(item, dict) or item.get("source_type") != "model":
                raise ValueError("caption-only retrieval accepts only source_type=model observations")
            timestamp = item.get("source_timestamp_sec", item.get("timestamp_sec"))
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
                raise ValueError("model observation requires a finite source timestamp")
            caption = item.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError("model observation requires nonempty caption text")
            timestamp = float(timestamp)
            documents.append({
                "text": caption.strip(),
                "metadata": {
                    "video_id": caption_data["video_id"],
                    "time_range": [timestamp, timestamp],
                    "frame_timestamp_sec": timestamp,
                    "frame_id": item.get("frame_id"),
                    "source_path": item.get("source_path"),
                    "source_type": "model",
                    "verification_status": "unvalidated_model_hypothesis",
                    "accepted_as_fact": False,
                    "retrieval_mode": "caption_only",
                    "asr_status": "not_run",
                    "model": item.get("model"),
                    "backend": item.get("backend"),
                    "device": item.get("device"),
                    "observation": caption.strip(),
                },
            })
        return documents

    def index_caption_data(self, caption_data):
        documents = self.observation_documents(caption_data)
        video_id = caption_data["video_id"]
        texts = [document["text"] for document in documents]
        vectors = self.embedder.encode(texts) if texts else np.empty(
            (0, self.vector_store.dimension), dtype=np.float32
        )
        self.vector_store.replace_video(video_id, vectors, [document["metadata"] for document in documents])
        self.vector_store.save()
        return len(documents)

    def index_caption_file(self, path):
        with Path(path).open(encoding="utf-8") as stream:
            return self.index_caption_data(json.load(stream))

    def search(self, query, *, video_id=None, limit=5, time_range=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty text")
        scope = {} if time_range is None else {"time_range": time_range}
        return self.vector_store.search(
            self.embedder.encode_single(query.strip()), k=limit, video_id=video_id,
            **scope,
        )
