#!/usr/bin/env python3
"""Phase 7: generate embeddings and replace each video's searchable memory."""

import json
import logging
from numbers import Integral
from pathlib import Path
import sys
from typing import Dict, List, Optional

import faiss
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slm_pipeline.config import config
from slm_pipeline.runtime import phase_cli, selected_video_dirs

logging.basicConfig(
    level=config['logging']['level'],
    format=config['logging']['format'],
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Load sentence-transformers only when model inference is requested."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        model_config = config['models']['embeddings']
        model_name = model_config['name']
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(
            model_name, device=model_config.get('device', 'cpu'),
            local_files_only=not config['pipeline']['allow_model_downloads']
        )
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded (dimension: %s)", self.dimension)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class VectorStore:
    """Flat L2 index with exactly one metadata row per vector.

    ``replace_video`` is the indexing operation: it removes all prior rows for
    that video, including blocks that disappeared, while retaining other videos.
    ``memory_dir`` allows isolated stores in tests and other explicit workspaces.
    """

    def __init__(self, dimension: int = 384, memory_dir: Optional[Path] = None):
        if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension <= 0:
            raise ValueError("Embedding dimension must be a positive integer")
        self.dimension = int(dimension)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        root = Path(memory_dir if memory_dir is not None else config['paths']['memory_dir'])
        self.index_path = root / "vector_db" / "faiss_index.bin"
        self.metadata_path = root / "vector_db" / "metadata.json"
        if self.index_path.exists() or self.metadata_path.exists():
            self.load()

    def _validate_state(self):
        if self.index.d != self.dimension:
            raise ValueError(
                f"Index dimension {self.index.d} does not match embedding dimension {self.dimension}"
            )
        if not isinstance(self.metadata, list) or any(
            not isinstance(row, dict) for row in self.metadata
        ):
            raise ValueError("Vector metadata must be a list of objects")
        if self.index.ntotal != len(self.metadata):
            raise ValueError("Vector index and metadata cardinality do not match")

    def _validate_batch(self, embeddings, metadata):
        self._validate_state()
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Embeddings must have shape (n, {self.dimension})")
        if not isinstance(metadata, list) or any(not isinstance(row, dict) for row in metadata):
            raise ValueError("Metadata must be a list of objects")
        if vectors.shape[0] != len(metadata):
            raise ValueError("Number of embeddings must match metadata")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must contain only finite values")
        return np.ascontiguousarray(vectors)

    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Append a validated batch; pipeline callers should use replace_video."""
        vectors = self._validate_batch(embeddings, metadata)
        self.index.add(vectors)
        self.metadata.extend(row.copy() for row in metadata)

    def replace_video(self, video_id: str, embeddings: np.ndarray, metadata: List[Dict]):
        """Replace all rows for a video, even when the new batch is empty."""
        vectors = self._validate_batch(embeddings, metadata)
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("video_id must be a nonempty string")
        if any(row.get('video_id') != video_id for row in metadata):
            raise ValueError("Every replacement metadata row must belong to video_id")

        retained_ids = [
            idx for idx, row in enumerate(self.metadata)
            if row.get('video_id') != video_id
        ]
        replacement = faiss.IndexFlatL2(self.dimension)
        if retained_ids:
            retained = np.asarray(
                [self.index.reconstruct(idx) for idx in retained_ids], dtype=np.float32
            )
            replacement.add(retained)
        replacement.add(vectors)
        replacement_metadata = [self.metadata[idx] for idx in retained_ids]
        replacement_metadata.extend(row.copy() for row in metadata)
        # Do not alter the live store until validation and rebuilding succeed.
        self.index = replacement
        self.metadata = replacement_metadata
        logger.info("Replaced %s with %d embeddings", video_id, len(metadata))

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        video_id: Optional[str] = None,
        time_range: Optional[tuple] = None,
    ) -> List[Dict]:
        """Return up to k nearest eligible rows; time ranges overlap inclusively.

        Filtering happens before ranking, so a video's matches cannot be hidden
        by more than k (or an arbitrary oversampling factor) other-video results.
        """
        self._validate_state()
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.shape not in ((self.dimension,), (1, self.dimension)):
            raise ValueError(f"Query embedding must have dimension {self.dimension}")
        if not np.isfinite(query).all():
            raise ValueError("Query embedding must contain only finite values")
        query = np.ascontiguousarray(query.reshape(1, self.dimension))
        if time_range is not None:
            window = np.asarray(time_range, dtype=float)
            if window.shape != (2,) or not np.isfinite(window).all() or window[0] > window[1]:
                raise ValueError("time_range must be a finite (start, end) pair with start <= end")

        candidates = []
        for idx, row in enumerate(self.metadata):
            if video_id is not None and row.get('video_id') != video_id:
                continue
            if time_range is not None:
                row_range = row.get('time_range')
                if not isinstance(row_range, (list, tuple)) or len(row_range) != 2:
                    continue
                if row_range[1] < window[0] or row_range[0] > window[1]:
                    continue
            candidates.append(idx)
        if k == 0 or not candidates:
            return []

        search_index = self.index
        if len(candidates) != self.index.ntotal:
            search_index = faiss.IndexFlatL2(self.dimension)
            search_index.add(np.asarray(
                [self.index.reconstruct(idx) for idx in candidates], dtype=np.float32
            ))
        distances, indices = search_index.search(query, min(int(k), len(candidates)))
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            # FAISS uses -1 for a missing neighbor; never read metadata[-1].
            if not 0 <= idx < len(candidates) or not np.isfinite(distance):
                continue
            result = self.metadata[candidates[int(idx)]].copy()
            result['distance'] = float(distance)
            result['similarity'] = 1.0 / (1.0 + float(distance))
            results.append(result)
        return results

    def save(self):
        self._validate_state()
        # Serialize before writing so malformed metadata cannot destroy the index.
        metadata_json = json.dumps(self.metadata, indent=2, ensure_ascii=False, allow_nan=False)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(metadata_json, encoding='utf-8')
        logger.info("Saved vector store (%d vectors)", self.index.ntotal)

    def load(self):
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Vector store requires both faiss_index.bin and metadata.json")
        index = faiss.read_index(str(self.index_path))
        metadata = json.loads(self.metadata_path.read_text(encoding='utf-8'))
        if index.d != self.dimension:
            raise ValueError(
                f"Index dimension {index.d} does not match embedding dimension {self.dimension}"
            )
        if not isinstance(metadata, list) or any(not isinstance(row, dict) for row in metadata):
            raise ValueError("Vector metadata must be a list of objects")
        if index.ntotal != len(metadata):
            raise ValueError("Vector index and metadata cardinality do not match")
        self.index, self.metadata = index, metadata
        logger.info("Loaded vector store (%d vectors)", self.index.ntotal)


class MemoryIndexer:
    """Index semantic data, retaining transcript and source-frame evidence."""

    def __init__(self, embedder=None, vector_store=None):
        self.embedder = embedder if embedder is not None else EmbeddingGenerator()
        self.vector_store = vector_store if vector_store is not None else VectorStore(
            dimension=self.embedder.dimension
        )
        if self.embedder.dimension != self.vector_store.dimension:
            raise ValueError("Embedder and vector store dimensions must match")

    @staticmethod
    def is_accepted_fact(semantic_block: Dict) -> bool:
        return semantic_block.get('accepted_as_fact') is True and semantic_block.get(
            'verification_status'
        ) == 'accepted_independent_evidence'

    def create_searchable_text(self, semantic_block: Dict) -> str:
        accepted = self.is_accepted_fact(semantic_block)
        parts = []
        if semantic_block.get('summary'):
            parts.append(semantic_block['summary'])
        fields = [
            ('topics', 'Topics', ', '),
            ('entities', 'Entities', ', '),
        ]
        if accepted:
            fields.extend([('decisions', 'Decisions', ' | '), ('tasks', 'Tasks', ' | ')])
        for field, label, separator in fields:
            values = semantic_block.get(field, [])
            if values:
                parts.append(f"{label}: {separator.join(values)}")
        if not parts:
            return ""
        if not accepted:
            parts.insert(0, 'UNVALIDATED MODEL HYPOTHESIS')
        return " | ".join(parts)

    def process_video_folder(self, video_dir: Path):
        video_dir = Path(video_dir)
        logger.info("Indexing video: %s", video_dir.name)
        semantic_data = json.loads((video_dir / 'semantic_structure.json').read_text(encoding='utf-8'))
        context_data = json.loads((video_dir / 'context_blocks.json').read_text(encoding='utf-8'))
        context_blocks = {block['block_id']: block for block in context_data['context_blocks']}
        semantic_blocks = semantic_data['semantic_blocks']
        if not isinstance(semantic_blocks, list):
            raise ValueError("semantic_blocks must be a list")

        texts = []
        metadata_list = []
        for block in semantic_blocks:
            searchable_text = self.create_searchable_text(block)
            if not searchable_text.strip():
                continue
            block_id = block.get('context_block_id')
            if block_id not in context_blocks:
                raise ValueError(f"Missing context block {block_id!r} for {video_dir.name}")
            context = context_blocks[block_id]
            visual = context.get('visual', {})
            accepted_as_fact = self.is_accepted_fact(block)
            texts.append(searchable_text)
            metadata_list.append({
                'video_id': video_dir.name,
                'block_id': block_id,
                'time_range': context.get('time_range', block.get('time_range', [0, 0])),
                'speech_text': context.get('speech', {}).get('text', ''),
                'visual_caption': visual.get('caption', ''),
                'frame_ids': visual.get('frame_ids', []),
                'frame_timestamps_sec': visual.get('frame_timestamps_sec', []),
                'frames': visual.get('frames', []),
                'model_observations': visual.get('model_observations', []),
                'visual_evidence_status': visual.get('verification_status', 'no_visual_observation'),
                'raw_transcript_segment_id': context.get('raw_transcript_segment_id'),
                'source_type': context.get('source_type', 'unknown'),
                'verification_status': block.get('verification_status', 'unvalidated_derived_model_hypothesis'),
                'accepted_as_fact': accepted_as_fact,
                'topics': block.get('topics', []) if accepted_as_fact else [],
                'unvalidated_topics': block.get('topics', []) if not accepted_as_fact else [],
                'decisions': block.get('decisions', []) if accepted_as_fact else [],
                'unvalidated_decisions': block.get('decisions', []) if not accepted_as_fact else [],
                'tasks': block.get('tasks', []) if accepted_as_fact else [],
                'unvalidated_tasks': block.get('tasks', []) if not accepted_as_fact else [],
                'intent': block.get('intent', 'informational'),
                'summary': block.get('summary', ''),
                'model_confidence_unvalidated': block.get('confidence', 0.0),
                'confidence': 'not_calibrated',
                'searchable_text': searchable_text,
            })

        embeddings = self.embedder.encode(texts) if texts else np.empty(
            (0, self.vector_store.dimension), dtype=np.float32
        )
        self.vector_store.replace_video(video_dir.name, embeddings, metadata_list)
        logger.info("Indexed %d blocks from %s", len(texts), video_dir.name)
        return len(texts)

    def save(self):
        self.vector_store.save()


def main(video_id=None):
    """Index selected folders and propagate failures to the orchestrator."""
    video_folders = selected_video_dirs(config, video_id)
    indexer = MemoryIndexer()
    for video_dir in video_folders:
        indexer.process_video_folder(video_dir)
    indexer.save()
    logger.info("Phase 7 (Embedding & Memory Storage) completed")


if __name__ == '__main__':
    phase_cli(main)
