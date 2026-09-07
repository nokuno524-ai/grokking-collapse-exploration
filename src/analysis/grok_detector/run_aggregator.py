import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import os

class MalformedRunError(Exception):
    pass

def parse_log_file(filepath: Path) -> Dict[str, Any]:
    """Parses a JSON, JSONL, or CSV log file into a standard format."""
    ext = filepath.suffix.lower()

    if ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
            _validate_schema(data, filepath)
            return data

    elif ext == '.jsonl':
        data = {'history': []}
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                if 'config' in record:
                    data['config'] = record['config']
                elif 'step' in record and 'test_acc' in record:
                    data['history'].append(record)
        _validate_schema(data, filepath)
        return data

    elif ext == '.csv':
        data = {'history': []}
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    record = {
                        'step': int(row['step']),
                        'test_acc': float(row['test_acc'])
                    }
                    data['history'].append(record)
                except (KeyError, ValueError) as e:
                    pass # ignore lines without valid step/test_acc

        # For CSVs, config is often passed separately or needs a dummy one if it doesn't exist
        # We will assume a companion config.json exists or it's provided by the caller
        config_path = filepath.parent / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                data['config'] = json.load(f)
        else:
            # Try to infer severity from directory name if no config exists
            severity = 0.0
            for part in filepath.parts:
                if 'noise' in part:
                    try:
                        severity = float(part.replace('noise', ''))
                    except ValueError:
                        pass
                if 'level' in part:
                    try:
                        severity = float(part.replace('level', ''))
                    except ValueError:
                        pass

            data['config'] = {'collapse_severity': severity, 'noise_fraction': severity}

        _validate_schema(data, filepath)
        return data

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def _validate_schema(data: Dict[str, Any], filepath: Path) -> None:
    if 'config' not in data:
        raise MalformedRunError(f"Missing 'config' dictionary in {filepath}")
    if 'history' not in data:
        raise MalformedRunError(f"Missing 'history' list in {filepath}")

    config = data['config']
    if 'collapse_severity' not in config and 'noise_fraction' not in config:
        raise MalformedRunError(f"Missing severity key (collapse_severity or noise_fraction) in config of {filepath}")

    if not data['history']:
        raise MalformedRunError(f"Empty 'history' in {filepath}")

    for i, record in enumerate(data['history']):
        if 'step' not in record or 'test_acc' not in record:
            raise MalformedRunError(f"Missing 'step' or 'test_acc' in history index {i} of {filepath}")

def load_runs_from_directories(directories: List[Union[str, Path]]) -> List[Dict[str, Any]]:
    """Loads all valid run logs from a list of directories."""
    runs = []

    for d in directories:
        dir_path = Path(d)
        if not dir_path.exists() or not dir_path.is_dir():
            continue

        # Search for results.json, results.jsonl, or history.csv
        for filepath in dir_path.rglob('*'):
            if filepath.is_file() and filepath.name in ['results.json', 'results.jsonl', 'history.csv', 'metrics.csv']:
                try:
                    run_data = parse_log_file(filepath)
                    run_data['filepath'] = str(filepath)
                    runs.append(run_data)
                except MalformedRunError as e:
                    print(f"Warning: Skipping {filepath}: {e}")
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    return runs

def align_runs_on_step(runs: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Groups runs by their severity condition."""
    grouped = {}
    for run in runs:
        config = run['config']
        severity = config.get('noise_fraction', config.get('collapse_severity', 0.0))
        # Use a small round to avoid float key issues
        sev_key = round(float(severity), 4)
        if sev_key not in grouped:
            grouped[sev_key] = []
        grouped[sev_key].append(run)
    return grouped
