import json
import sys

def validate_registry(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Registry file {filepath} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Registry file {filepath} is not valid JSON.")
        sys.exit(1)

    if not isinstance(data, list):
        print("Registry must be a JSON array of objects.")
        sys.exit(1)

    seen_hashes = set()
    for idx, entry in enumerate(data):
        if 'config_hash' not in entry:
            print(f"Entry {idx} missing 'config_hash'.")
            sys.exit(1)

        h = entry['config_hash']
        if h in seen_hashes:
            print(f"Duplicate config_hash found: {h}")
            sys.exit(1)
        seen_hashes.add(h)

        # Check required fields
        for field in ['metrics', 'effect_sizes', 'CIs']:
            if field not in entry:
                print(f"Entry {idx} (hash {h}) missing required field: {field}")
                sys.exit(1)

    print(f"Registry valid. {len(data)} entries found.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python registry_validator.py <registry.json>")
        sys.exit(1)
    validate_registry(sys.argv[1])
