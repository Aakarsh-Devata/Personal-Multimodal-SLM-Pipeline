#!/usr/bin/env python3
"""
Phase 7: Embedding & Memory Storage
Generates embeddings and stores in FAISS vector database for semantic search.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

logging.basicConfig(
    level=config['logging']['level'],
    format=config['logging']['format']
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers."""
    
    def __init__(self):
        model_name = config['models']['embeddings']['name']
        logger.info(f"Loading embedding model: {model_name}")
        
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"✅ Embedding model loaded (dimension: {self.dimension})")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.model.encode([text], convert_to_numpy=True)[0]


class VectorStore:
    """FAISS-based vector store for semantic search."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        
        # Try to load existing index
        self.index_path = Path(config['paths']['memory_dir']) / "vector_db" / "faiss_index.bin"
        self.metadata_path = Path(config['paths']['memory_dir']) / "vector_db" / "metadata.json"
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.index_path.exists():
            self.load()
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add embeddings with metadata to the index."""
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Number of embeddings must match metadata")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Add metadata
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(metadata)} embeddings to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar embeddings."""
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['distance'] = float(dist)
                result['similarity'] = 1.0 / (1.0 + float(dist))
                results.append(result)
        
        return results
    
    def save(self):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved vector store ({self.index.ntotal} vectors)")
    
    def load(self):
        """Load index and metadata from disk."""
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            logger.info(f"✅ Loaded vector store ({self.index.ntotal} vectors)")


class MemoryIndexer:
    """Index semantic data into vector store."""
    
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore(dimension=self.embedder.dimension)
    
    def create_searchable_text(self, semantic_block: Dict) -> str:
        """Create searchable text from semantic block."""
        parts = []
        
        # Add summary
        if semantic_block.get('summary'):
            parts.append(semantic_block['summary'])
        
        # Add topics
        topics = semantic_block.get('topics', [])
        if topics:
            parts.append(f"Topics: {', '.join(topics)}")
        
        # Add decisions
        decisions = semantic_block.get('decisions', [])
        if decisions:
            parts.append(f"Decisions: {' | '.join(decisions)}")
        
        # Add tasks
        tasks = semantic_block.get('tasks', [])
        if tasks:
            parts.append(f"Tasks: {' | '.join(tasks)}")
        
        # Add entities
        entities = semantic_block.get('entities', [])
        if entities:
            parts.append(f"Entities: {', '.join(entities)}")
        
        return " | ".join(parts)
    
    def process_video_folder(self, video_dir: Path):
        """Index semantic data from a video folder."""
        logger.info(f"Indexing video: {video_dir.name}")
        
        # Load semantic structure
        semantic_file = video_dir / "semantic_structure.json"
        if not semantic_file.exists():
            logger.warning(f"No semantic structure found for {video_dir.name}")
            return
        
        # Load context blocks for reference
        context_file = video_dir / "context_blocks.json"
        context_data = {}
        if context_file.exists():
            with open(context_file, 'r', encoding='utf-8') as f:
                context_data = json.load(f)
        
        context_blocks = {
            block['block_id']: block 
            for block in context_data.get('context_blocks', [])
        }
        
        # Load semantic data
        with open(semantic_file, 'r', encoding='utf-8') as f:
            semantic_data = json.load(f)
        
        semantic_blocks = semantic_data.get('semantic_blocks', [])
        
        if not semantic_blocks:
            logger.warning(f"No semantic blocks to index for {video_dir.name}")
            return
        
        # Create searchable texts and metadata
        texts = []
        metadata_list = []
        
        for block in semantic_blocks:
            searchable_text = self.create_searchable_text(block)
            
            if not searchable_text.strip():
                continue
            
            texts.append(searchable_text)
            
            # Get original context
            block_id = block.get('context_block_id')
            context = context_blocks.get(block_id, {})
            
            metadata = {
                "video_id": video_dir.name,
                "block_id": block_id,
                "time_range": block.get('time_range', [0, 0]),
                "speech_text": context.get('speech', {}).get('text', ''),
                "visual_caption": context.get('visual', {}).get('caption', ''),
                "topics": block.get('topics', []),
                "decisions": block.get('decisions', []),
                "tasks": block.get('tasks', []),
                "intent": block.get('intent', 'informational'),
                "summary": block.get('summary', ''),
                "confidence": block.get('confidence', 0.0),
                "searchable_text": searchable_text
            }
            
            metadata_list.append(metadata)
        
        if not texts:
            logger.warning(f"No searchable content for {video_dir.name}")
            return
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} blocks")
        embeddings = self.embedder.encode(texts)
        
        # Add to vector store
        self.vector_store.add(embeddings, metadata_list)
        
        logger.info(f"✅ Indexed {len(texts)} blocks from {video_dir.name}")
    
    def save(self):
        """Save the vector store."""
        self.vector_store.save()


def main():
    """Main entry point."""
    processed_dir = Path(config['paths']['processed_dir'])
    
    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return
    
    # Initialize indexer
    indexer = MemoryIndexer()
    
    # Process all video folders
    video_folders = [d for d in processed_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(video_folders)} video folders to index")
    
    for video_dir in video_folders:
        try:
            indexer.process_video_folder(video_dir)
        except Exception as e:
            logger.error(f"Failed to index {video_dir.name}: {e}")
            continue
    
    # Save vector store
    indexer.save()
    
    logger.info("🎉 Phase 7 (Embedding & Memory Storage) completed!")


if __name__ == "__main__":
    main()
