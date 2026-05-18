#!/usr/bin/env python3
import subprocess
import argparse
import sys
import re
from pathlib import Path
import os
from typing import List, Dict

def run_cmd(cmd: List[str]) -> str:
    """Run a command and return its standard output."""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        return e.output

def get_running_jobs(prefix: str = None) -> List[Dict]:
    """Get active jobs from squeue."""
    # Attempt to use squeue, if not available return empty list or mock info
    try:
        # Check if squeue exists
        subprocess.check_call(["which", "squeue"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Warning: 'squeue' command not found. Slurm might not be available.")
        return []

    cmd = ["squeue", "--me", "--format=%.18i %.30j %.8T %.10M %.10l"]
    output = run_cmd(cmd)

    jobs = []
    lines = output.strip().split("\n")[1:] # Skip header
    for line in lines:
        if not line.strip(): continue

        parts = line.split()
        if len(parts) >= 5:
            job_id, name, state, elapsed, requested = parts[0], parts[1], parts[2], parts[3], parts[4]
            if prefix is None or name.startswith(prefix):
                jobs.append({
                    "id": job_id,
                    "name": name,
                    "state": state,
                    "elapsed": elapsed,
                    "requested": requested
                })
    return jobs

def get_failed_jobs(prefix: str = None, hours: int = 24) -> List[Dict]:
    """Get failed jobs from sacct in the last N hours."""
    try:
        subprocess.check_call(["which", "sacct"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return []

    cmd = ["sacct", f"--starttime=now-{hours}hours", "--format=JobID,JobName%30,State", "-X"]
    output = run_cmd(cmd)

    jobs = []
    lines = output.strip().split("\n")[2:] # Skip header and line
    for line in lines:
        if not line.strip(): continue

        parts = line.split()
        if len(parts) >= 3:
            job_id, name, state = parts[0], parts[1], parts[2]
            if state in ["FAILED", "TIMEOUT", "OUT_OF_MEMORY"] and (prefix is None or name.startswith(prefix)):
                jobs.append({
                    "id": job_id,
                    "name": name,
                    "state": state
                })
    return jobs

def show_recent_logs(log_dir: str, lines: int = 5, failed_only: bool = False):
    """Show recent lines from output logs."""
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Log directory '{log_dir}' does not exist.")
        return

    # Find recent logs
    logs = list(log_path.glob("*.err" if failed_only else "*.out"))
    if not logs:
        print(f"No {'error' if failed_only else 'output'} logs found in {log_dir}")
        return

    # Sort by modification time, newest first
    logs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    print(f"\n--- Recent {'Error' if failed_only else 'Output'} Logs (Top 5 files) ---")
    for log_file in logs[:5]:
        print(f"\n[{log_file.name}]")
        try:
            with open(log_file, 'r') as f:
                content = f.readlines()
                if not content:
                    print("  (Empty file)")
                else:
                    for line in content[-lines:]:
                        print(f"  {line.rstrip()}")
        except Exception as e:
            print(f"  Error reading file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Slurm Job Monitoring Dashboard")
    parser.add_argument("--prefix", type=str, default=None, help="Filter jobs by name prefix")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory containing job logs")
    parser.add_argument("--tail", type=int, default=5, help="Number of lines to show from logs")
    parser.add_argument("--failed-logs", action="store_true", help="Show error logs instead of output logs")

    args = parser.parse_args()

    print("=" * 60)
    print(" SLURM JOB DASHBOARD ")
    print("=" * 60)

    # Active Jobs
    running_jobs = get_running_jobs(args.prefix)
    print(f"\n--- Active Jobs ({len(running_jobs)}) ---")
    if running_jobs:
        print(f"{'JOB ID':<15} {'NAME':<30} {'STATE':<10} {'TIME':<12} {'REQUESTED':<12}")
        print("-" * 80)
        for job in running_jobs:
            print(f"{job['id']:<15} {job['name']:<30} {job['state']:<10} {job['elapsed']:<12} {job['requested']:<12}")
    else:
        print("No active jobs found.")

    # Failed Jobs
    failed_jobs = get_failed_jobs(args.prefix)
    print(f"\n--- Recent Failed Jobs ({len(failed_jobs)}) ---")
    if failed_jobs:
        print(f"{'JOB ID':<15} {'NAME':<30} {'STATE':<15}")
        print("-" * 65)
        for job in failed_jobs:
            # Highlight failed states with color
            state_str = f"\033[91m{job['state']}\033[0m"
            print(f"{job['id']:<15} {job['name']:<30} {state_str}")
    else:
        print("No recently failed jobs.")

    # Logs
    show_recent_logs(args.log_dir, args.tail, args.failed_logs)
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
