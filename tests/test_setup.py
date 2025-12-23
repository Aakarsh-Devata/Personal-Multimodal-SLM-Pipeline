#!/usr/bin/env python3
"""
Quick test script to verify the pipeline setup.
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import yaml
        print("  ✅ pyyaml")
    except ImportError:
        print("  ❌ pyyaml - run: pip3 install pyyaml")
        return False
    
    try:
        import torch
        print(f"  ✅ torch ({torch.__version__})")
    except ImportError:
        print("  ❌ torch - run: pip3 install torch")
        return False
    
    try:
        import transformers
        print(f"  ✅ transformers ({transformers.__version__})")
    except ImportError:
        print("  ❌ transformers - run: pip3 install transformers")
        return False
    
    try:
        import sentence_transformers
        print("  ✅ sentence-transformers")
    except ImportError:
        print("  ❌ sentence-transformers - run: pip3 install sentence-transformers")
        return False
    
    try:
        import faiss
        print("  ✅ faiss-cpu")
    except ImportError:
        print("  ❌ faiss-cpu - run: pip3 install faiss-cpu")
        return False
    
    try:
        import fastapi
        print("  ✅ fastapi")
    except ImportError:
        print("  ❌ fastapi - run: pip3 install fastapi")
        return False
    
    return True


def test_ollama():
    """Test Ollama connection."""
    print("\n🧪 Testing Ollama connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"  ✅ Ollama is running")
            print(f"     Available models: {', '.join([m.get('name', 'unknown') for m in models])}")
            return True
        else:
            print("  ❌ Ollama returned error")
            return False
    except Exception as e:
        print(f"  ❌ Cannot connect to Ollama: {e}")
        print("     Make sure Ollama is running: ollama serve")
        return False


def test_config():
    """Test configuration loading."""
    print("\n🧪 Testing configuration...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from slm_pipeline.config import config
        
        print("  ✅ Configuration loaded")
        print(f"     Processed dir: {config['paths']['processed_dir']}")
        print(f"     Vision model: {config['models']['vision']['name']}")
        print(f"     LLM model: {config['models']['slm']['name']}")
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False


def test_data():
    """Test if processed data exists."""
    print("\n🧪 Testing processed data...")
    
    processed_dir = Path("/Users/aakarsh/Desktop/ProjectRoot/phone_videos/processed")
    
    if not processed_dir.exists():
        print("  ❌ Processed directory not found")
        return False
    
    video_folders = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    if not video_folders:
        print("  ❌ No processed videos found")
        return False
    
    print(f"  ✅ Found {len(video_folders)} processed video(s)")
    
    for video_dir in video_folders:
        transcript = video_dir / "transcript.json"
        if transcript.exists():
            print(f"     ✅ {video_dir.name} has transcript")
        else:
            print(f"     ⚠️  {video_dir.name} missing transcript")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SLM Pipeline - System Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Ollama", test_ollama()))
    results.append(("Config", test_config()))
    results.append(("Data", test_data()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(p for _, p in results)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready.")
        print("\nNext step: Run the pipeline")
        print("  python slm_pipeline/orchestrator.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
