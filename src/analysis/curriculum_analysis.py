import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def analyze_curriculum(results_dir: str):
    base_path = Path(results_dir)
    if not base_path.exists():
        print(f"Results directory {results_dir} not found.")
        return

    records = []

    # 1. Load results
    for run_dir in base_path.iterdir():
        if not run_dir.is_dir():
            continue

        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue

        with open(results_file, 'r') as f:
            data = json.load(f)

        config = data['config']

        # Calculate time-averaged w
        # For our schedules (linear from 0 to 1, or 1 to 0; step at 50%; cosine from 0 to 1)
        # the theoretical time-average is 0.5.
        # Constant schedules are just their constant w.
        avg_w = 0.5
        sched = config.get('curriculum_schedule')
        if sched == 'constant':
            avg_w = config.get('curriculum_start_w', 0.0)

        records.append({
            'name': config['condition_name'],
            'schedule': sched,
            'start_w': config.get('curriculum_start_w'),
            'end_w': config.get('curriculum_end_w'),
            'avg_w': avg_w,
            'grokked': data['grokked'],
            'grokking_step': data['grokking_step'],
            'final_test_acc': data['final_test_acc'],
            'final_weight_norm': data['final_weight_norm']
        })

    if not records:
        print("No results found.")
        return

    df = pd.DataFrame(records)

    # 2. Compare against average mixtures
    print("=== CONSTANT BASELINES ===")
    constants = df[df['schedule'] == 'constant'].groupby('avg_w').agg({
        'grokked': 'mean',
        'grokking_step': 'mean',
        'final_test_acc': 'mean'
    }).reset_index()
    print(constants.to_string(index=False))

    print("\n=== SCHEDULES (Average w = 0.5) ===")
    schedules_05 = df[(df['schedule'] != 'constant') & (df['avg_w'] == 0.5)]

    def direction(row):
        if row['start_w'] > row['end_w']:
            return "Collapse -> Pure"
        else:
            return "Pure -> Collapse"

    if len(schedules_05) > 0:
        schedules_05['direction'] = schedules_05.apply(direction, axis=1)
        sched_stats = schedules_05.groupby(['schedule', 'direction']).agg({
            'grokked': 'mean',
            'grokking_step': 'mean',
            'final_test_acc': 'mean'
        }).reset_index()
        print(sched_stats.to_string(index=False))

    # 3. Explicit answers
    print("\n=== MECHANISTIC HYPOTHESIS EVALUATION ===")
    print("Does late exposure to pure data recover grokking?")
    # Find 'Collapse -> Pure'
    recover_runs = schedules_05[schedules_05['direction'] == 'Collapse -> Pure']
    if len(recover_runs) > 0:
        recover_rate = recover_runs['grokked'].mean()
        if recover_rate > 0:
            print(f"-> YES. {recover_rate*100:.1f}% of 'Collapse -> Pure' runs grokked.")
        else:
            print(f"-> NO. 0% of 'Collapse -> Pure' runs grokked. Collapse irreversibly poisoned the optimization path.")

    print("\nDoes early exposure to pure data protect against later collapsed data?")
    # Find 'Pure -> Collapse'
    protect_runs = schedules_05[schedules_05['direction'] == 'Pure -> Collapse']
    if len(protect_runs) > 0:
        protect_rate = protect_runs['grokked'].mean()
        if protect_rate > 0:
            print(f"-> YES. {protect_rate*100:.1f}% of 'Pure -> Collapse' runs grokked.")
        else:
            print(f"-> NO. Early pure data does not inoculate against later collapse.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/curriculum")
    args = parser.parse_args()

    analyze_curriculum(args.results_dir)
