#!/usr/bin/env python3
"""
CLI Interface for SLM Pipeline
Query your personal memory system from the command line.
"""

import sys
import argparse
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))
from config import config
from agents.memory_agent import MemoryAgent
from agents.summary_agent import SummaryAgent


def query_command(args):
    """Handle query command."""
    agent = MemoryAgent()
    results = agent.search_memory(args.query, k=args.limit, video_id=args.video)
    
    print(f"\n🔍 Search Results for: '{args.query}'\n")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Video: {result.get('video_id')}")
        print(f"    Time: {result.get('time_range', [0, 0])[0]:.1f}s - {result.get('time_range', [0, 0])[1]:.1f}s")
        print(f"    Speech: {result.get('speech_text', '')[:200]}")
        print(f"    Visual: {result.get('visual_caption', '')}")
        if result.get('topics'):
            print(f"    Topics: {', '.join(result.get('topics', []))}")
        if result.get('decisions'):
            print(f"    Decisions: {', '.join(result.get('decisions', []))}")
        print(f"    Similarity: {result.get('similarity', 0):.3f}")
    
    print("\n" + "=" * 80)


def summarize_command(args):
    """Handle summarize command."""
    agent = SummaryAgent()
    
    if args.meeting:
        result = agent.summarize_meeting(args.meeting)
        
        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
            return
        
        print(f"\n📝 Summary of: {result['video_id']}\n")
        print("=" * 80)
        print(f"\n{result['summary']}\n")
        print(f"Topics: {', '.join(result.get('topics', []))}")
        print(f"Decisions: {len(result.get('decisions', []))}")
        print(f"Tasks: {len(result.get('tasks', []))}")
        print("\n" + "=" * 80)
    
    elif args.all:
        result = agent.summarize_all_videos()
        
        print(f"\n📚 All Video Summaries ({result['total_videos']} videos)\n")
        print("=" * 80)
        
        for summary in result['summaries']:
            print(f"\n🎬 {summary['video_id']}")
            print(f"   {summary['summary'][:200]}...")
            print(f"   Topics: {', '.join(summary.get('topics', [])[:5])}")
        
        print("\n" + "=" * 80)


def tasks_command(args):
    """Handle tasks command."""
    agent = MemoryAgent()
    tasks = agent.get_all_tasks()
    
    print(f"\n✅ All Tasks ({len(tasks)} found)\n")
    print("=" * 80)
    
    for i, task_info in enumerate(tasks[:args.limit], 1):
        print(f"\n[{i}] {task_info['task']}")
        print(f"    Video: {task_info['video_id']}")
        print(f"    Time: {task_info.get('time_range', [0, 0])[0]:.1f}s")
        print(f"    Context: {task_info.get('context', '')[:150]}")
    
    print("\n" + "=" * 80)


def decisions_command(args):
    """Handle decisions command."""
    agent = MemoryAgent()
    decisions = agent.get_all_decisions()
    
    print(f"\n🎯 All Decisions ({len(decisions)} found)\n")
    print("=" * 80)
    
    for i, decision_info in enumerate(decisions[:args.limit], 1):
        print(f"\n[{i}] {decision_info['decision']}")
        print(f"    Video: {decision_info['video_id']}")
        print(f"    Time: {decision_info.get('time_range', [0, 0])[0]:.1f}s")
        print(f"    Context: {decision_info.get('context', '')[:150]}")
    
    print("\n" + "=" * 80)


def topics_command(args):
    """Handle topics command."""
    agent = MemoryAgent()
    topics = agent.get_topics(k=args.limit)
    
    print(f"\n🏷️  Most Common Topics\n")
    print("=" * 80)
    
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic}")
    
    print("\n" + "=" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SLM Pipeline CLI - Query your personal memory system"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Search memories with natural language')
    query_parser.add_argument('query', type=str, help='Search query')
    query_parser.add_argument('--limit', type=int, default=5, help='Number of results')
    query_parser.add_argument('--video', type=str, help='Filter results by video ID')
    
    # Summarize command
    summary_parser = subparsers.add_parser('summarize', help='Summarize meetings or videos')
    summary_parser.add_argument('--meeting', type=str, help='Video ID to summarize')
    summary_parser.add_argument('--all', action='store_true', help='Summarize all videos')
    
    # Tasks command
    tasks_parser = subparsers.add_parser('tasks', help='List all tasks')
    tasks_parser.add_argument('--limit', type=int, default=20, help='Number of tasks to show')
    
    # Decisions command
    decisions_parser = subparsers.add_parser('decisions', help='List all decisions')
    decisions_parser.add_argument('--limit', type=int, default=20, help='Number of decisions to show')
    
    # Topics command
    topics_parser = subparsers.add_parser('topics', help='List common topics')
    topics_parser.add_argument('--limit', type=int, default=20, help='Number of topics to show')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to command handler
    if args.command == 'query':
        query_command(args)
    elif args.command == 'summarize':
        summarize_command(args)
    elif args.command == 'tasks':
        tasks_command(args)
    elif args.command == 'decisions':
        decisions_command(args)
    elif args.command == 'topics':
        topics_command(args)


if __name__ == "__main__":
    main()
