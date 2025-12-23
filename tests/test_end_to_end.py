#!/usr/bin/env python3
"""
End-to-End Pipeline Test
Tests each phase of the pipeline on the existing video.
"""

import sys
from pathlib import Path
import json
import time

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from slm_pipeline.config import config

def check_phase_1():
    """Check Phase 1: Capture & Validation"""
    print("\n" + "="*80)
    print("PHASE 1: Capture & Validation")
    print("="*80)
    
    raw_dir = Path(config['paths']['raw_dir'])
    
    if not raw_dir.exists():
        print("❌ FAIL: Raw directory not found")
        return False
    
    videos = list(raw_dir.glob("*.mp4")) + list(raw_dir.glob("*.mov"))
    
    if not videos:
        print("❌ FAIL: No videos in raw directory")
        return False
    
    print(f"✅ PASS: Found {len(videos)} video(s)")
    for v in videos:
        print(f"   - {v.name}")
    
    return True


def check_phase_2():
    """Check Phase 2: Preprocessing"""
    print("\n" + "="*80)
    print("PHASE 2: Preprocessing (Audio + Frames)")
    print("="*80)
    
    processed_dir = Path(config['paths']['processed_dir'])
    
    if not processed_dir.exists():
        print("❌ FAIL: Processed directory not found")
        return False
    
    video_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    if not video_dirs:
        print("❌ FAIL: No processed videos")
        return False
    
    all_good = True
    for video_dir in video_dirs:
        print(f"\n📁 {video_dir.name}")
        
        # Check for audio
        audio_file = video_dir / "audio_clean.wav"
        if audio_file.exists():
            print(f"   ✅ Audio extracted")
        else:
            print(f"   ❌ Missing audio")
            all_good = False
        
        # Check for frames
        frames_dir = video_dir / "frames_fixed"
        if frames_dir.exists():
            frame_count = len(list(frames_dir.glob("*.jpg")))
            print(f"   ✅ {frame_count} frames extracted")
        else:
            print(f"   ❌ Missing frames")
            all_good = False
        
        # Check for metadata
        meta_file = video_dir / "processed_meta.json"
        if meta_file.exists():
            print(f"   ✅ Metadata generated")
        else:
            print(f"   ⚠️  Missing metadata (non-critical)")
    
    return all_good


def check_phase_3():
    """Check Phase 3: Speech Transcription"""
    print("\n" + "="*80)
    print("PHASE 3: Speech Transcription")
    print("="*80)
    
    processed_dir = Path(config['paths']['processed_dir'])
    video_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    all_good = True
    for video_dir in video_dirs:
        print(f"\n📁 {video_dir.name}")
        
        transcript_file = video_dir / "transcript.json"
        if transcript_file.exists():
            with open(transcript_file) as f:
                data = json.load(f)
            
            segments = data.get('segments', [])
            print(f"   ✅ Transcript with {len(segments)} segments")
            
            if segments:
                print(f"   📝 Sample: \"{segments[0].get('text', '')[:60]}...\"")
        else:
            print(f"   ❌ Missing transcript")
            all_good = False
    
    return all_good


def check_phase_4():
    """Check Phase 4: Vision Understanding"""
    print("\n" + "="*80)
    print("PHASE 4: Vision Understanding")
    print("="*80)
    
    processed_dir = Path(config['paths']['processed_dir'])
    video_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    all_good = True
    for video_dir in video_dirs:
        print(f"\n📁 {video_dir.name}")
        
        vision_file = video_dir / "vision_captions.json"
        if vision_file.exists():
            with open(vision_file) as f:
                data = json.load(f)
            
            captions = data.get('captions', [])
            print(f"   ✅ {len(captions)} frame captions generated")
            
            if captions:
                print(f"   🖼️  Sample: \"{captions[0].get('caption', '')[:60]}\"")
        else:
            print(f"   ❌ Missing vision captions")
            all_good = False
    
    return all_good


def check_phase_5():
    """Check Phase 5: Multimodal Alignment"""
    print("\n" + "="*80)
    print("PHASE 5: Multimodal Alignment")
    print("="*80)
    
    processed_dir = Path(config['paths']['processed_dir'])
    video_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    all_good = True
    for video_dir in video_dirs:
        print(f"\n📁 {video_dir.name}")
        
        context_file = video_dir / "context_blocks.json"
        if context_file.exists():
            with open(context_file) as f:
                data = json.load(f)
            
            blocks = data.get('context_blocks', [])
            print(f"   ✅ {len(blocks)} context blocks created")
            
            if blocks:
                block = blocks[0]
                speech = block.get('speech', {}).get('text', '')
                visual = block.get('visual', {}).get('caption', '')
                print(f"   🔗 Sample block:")
                print(f"      Speech: \"{speech[:50]}...\"")
                print(f"      Visual: \"{visual[:50]}\"")
        else:
            print(f"   ❌ Missing context blocks")
            all_good = False
    
    return all_good


def check_phase_6():
    """Check Phase 6: Semantic Structuring"""
    print("\n" + "="*80)
    print("PHASE 6: Semantic Structuring")
    print("="*80)
    
    processed_dir = Path(config['paths']['processed_dir'])
    video_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    all_good = True
    for video_dir in video_dirs:
        print(f"\n📁 {video_dir.name}")
        
        semantic_file = video_dir / "semantic_structure.json"
        if semantic_file.exists():
            with open(semantic_file) as f:
                data = json.load(f)
            
            blocks = data.get('semantic_blocks', [])
            print(f"   ✅ {len(blocks)} semantic blocks extracted")
            
            # Aggregate topics, decisions, tasks
            all_topics = set()
            all_decisions = []
            all_tasks = []
            
            for block in blocks:
                all_topics.update(block.get('topics', []))
                all_decisions.extend(block.get('decisions', []))
                all_tasks.extend(block.get('tasks', []))
            
            print(f"   🏷️  Topics: {len(all_topics)}")
            if all_topics:
                print(f"      {', '.join(list(all_topics)[:5])}")
            
            print(f"   🎯 Decisions: {len(all_decisions)}")
            if all_decisions:
                print(f"      {all_decisions[0][:60]}")
            
            print(f"   ✅ Tasks: {len(all_tasks)}")
            if all_tasks:
                print(f"      {all_tasks[0][:60]}")
        else:
            print(f"   ❌ Missing semantic structure")
            all_good = False
    
    return all_good


def check_phase_7():
    """Check Phase 7: Embeddings & Vector Storage"""
    print("\n" + "="*80)
    print("PHASE 7: Embeddings & Vector Storage")
    print("="*80)
    
    memory_dir = Path(config['paths']['memory_dir']) / "vector_db"
    
    index_file = memory_dir / "faiss_index.bin"
    metadata_file = memory_dir / "metadata.json"
    
    all_good = True
    
    if index_file.exists():
        print(f"   ✅ FAISS index created")
        
        # Try to load it
        try:
            import faiss
            index = faiss.read_index(str(index_file))
            print(f"   📊 Vector count: {index.ntotal}")
        except Exception as e:
            print(f"   ⚠️  Could not load index: {e}")
    else:
        print(f"   ❌ Missing FAISS index")
        all_good = False
    
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        print(f"   ✅ Metadata stored ({len(metadata)} entries)")
        
        if metadata:
            print(f"   💾 Sample metadata:")
            sample = metadata[0]
            print(f"      Video: {sample.get('video_id')}")
            print(f"      Topics: {sample.get('topics', [])}")
    else:
        print(f"   ❌ Missing metadata")
        all_good = False
    
    return all_good


def main():
    """Run all phase checks."""
    print("\n" + "="*80)
    print("SLM PIPELINE - END-TO-END TEST")
    print("="*80)
    
    results = [
        ("Phase 1: Capture", check_phase_1()),
        ("Phase 2: Preprocessing", check_phase_2()),
        ("Phase 3: Transcription", check_phase_3()),
        ("Phase 4: Vision", check_phase_4()),
        ("Phase 5: Alignment", check_phase_5()),
        ("Phase 6: Semantics", check_phase_6()),
        ("Phase 7: Embeddings", check_phase_7()),
    ]
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 All phases completed successfully!")
    else:
        print(f"\n⚠️  {total_count - passed_count} phase(s) failed or incomplete")
    
    print("="*80)


if __name__ == "__main__":
    main()
