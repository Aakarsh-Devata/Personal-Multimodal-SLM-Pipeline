# Personal Multimodal SLM System (Local Multimodal Memory Pipeline)

## Objective

Build a privacy-first, **fully local** multimodal Small Language Model (SLM) system that ingests first-person video and audio to create a searchable semantic memory. The system extracts speech, visual context, and structured meaning (topics, decisions, tasks) to enable intelligent recall, summarization, and querying — without sending data to the cloud.

**Design constraints**

* Fully offline execution (no external APIs during inference)
* Modular, model-agnostic pipeline design
* Deterministic, replayable processing for debugging and iteration

---

## Architecture

The core challenge is aligning asynchronous speech, visual context, and semantic meaning into a unified, queryable memory representation. The system is implemented as a deterministic, multi-stage pipeline where each phase produces structured artifacts consumed by downstream stages.

### Pipeline Phases

1. **Capture**

   - Validates raw video inputs and metadata

2. **Preprocessing**

   - Extracts audio (16 kHz mono)
   - Samples and deduplicates high-quality video frames

3. **Transcription**

   - Performs local speech-to-text using **OpenAI Whisper**

4. **Vision Understanding**

   - Generates detailed frame-level captions using **BLIP-2**

5. **Alignment**

   - Temporally aligns transcript segments with visual frames
   - Produces unified multimodal context windows (speech + visuals)
   - Handles mismatches between rapid visual changes and slower speech

6. **Semantic Extraction  
   - Uses a local instruction-tuned LLM (Mistral / Gemma via Ollama)  
   - Extracts structured meaning into normalized JSON schemas:
     - Topics
     - Decisions
     - Action items / tasks  
   - Generates confidence scores for extracted outputs to support filtering,
     ranking, and downstream decision-making
   - Designed to remain schema-stable across model swaps


7. **Memory Formation**

   - Generates semantic embeddings using **Sentence-Transformers**
   - Stores embeddings in a **FAISS** vector database for fast retrieval

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* [Ollama](https://ollama.com/) (for running local LLMs)
* FFmpeg

  ```bash
  brew install ffmpeg
  ```

### Installation

1. Clone the repository
2. Run the setup script:

   ```bash
   ./setup.sh
   ```

   This installs Python dependencies and pulls the required local models via Ollama.

---

## Basic Usage

### 1. Add Videos

Place video files (`.mp4`, `.mov`) into:

```
phone_videos/raw/
```

### 2. Run the Pipeline

Process all new videos and build the semantic memory:

```bash
python3 slm_pipeline/orchestrator.py
```

### 3. Query the Memory

```bash
# Natural language semantic search
python3 slm_pipeline/cli.py query "What did we discuss about the project?"

# Summarize a specific video
python3 slm_pipeline/cli.py summarize --meeting your_video_filename

# List extracted tasks
python3 slm_pipeline/cli.py tasks

# Filter queries by video
python3 slm_pipeline/cli.py query "design details" --video your_video_filename
```

---

## Project Structure

```
slm_pipeline/
├── pipelines/     # Vision, alignment, semantic extraction, embeddings
├── ingestion/     # Preprocessing, transcription, validation
├── agents/        # Retrieval and summarization agents
├── api/           # FastAPI backend
├── config/        # Centralized configuration
tests/             # Pipeline and system verification tests
phone_videos/
├── raw/            # Input videos
└── processed/      # Intermediate artifacts (audio, frames, JSON outputs)
```

---

## Configuration

Edit:

```
slm_pipeline/config/settings.yaml
```

to customize:

* Model selection (vision, LLM, embeddings)
* Input/output paths
* Processing parameters (frame rate, batch sizes, chunking)

---

## Why SLM?

The system prioritizes **small, local, instruction-tuned models** over cloud-hosted LLMs in order to:

* Preserve privacy and enable offline execution
* Reduce inference latency on consumer hardware
* Favor task-specialized reasoning over monolithic models

The architecture remains model-agnostic and can scale to larger models if needed.

---

## Evaluation & Limitations

* Transcription quality degrades with heavy overlap or background noise
* Visual captioning struggles with dense on-screen text or rapid motion
* Semantic extraction accuracy varies by meeting structure and speaking style
* Current pipeline is optimized for short length videos (< 30 minutes)

Planned improvements include adaptive chunking and lightweight domain adaptation.


## Contributing

This is a personal systems project focused on local execution and modular design. The pipeline is intentionally structured to support experimentation, extension, and adaptation to other multimodal memory use cases.


