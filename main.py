#!/usr/bin/env python
"""
CLI entry-point.  Run:

    python main.py train --semesters 2
"""
import argparse
from rich import print  # rich is tiny makes CLI pleasant

from src.pipeline import TrainingPipeline

parser = argparse.ArgumentParser(description="Student Outcome Prediction pipeline")
sub = parser.add_subparsers(dest="command", required=True)

train_p = sub.add_parser("train", help="train & evaluate all models")
train_p.add_argument("--semesters", type=int, default=2, choices=(0, 1, 2))

if __name__ == "__main__":
    args = parser.parse_args()
    if args.command == "train":
        pipeline = TrainingPipeline()
        results = pipeline.run(semesters=args.semesters)
        print("\n[b] Leaderboard[/b]")
        for r in results:
            print(f"{r['model']:<18} -> accuracy = {r['accuracy']:.3f}")