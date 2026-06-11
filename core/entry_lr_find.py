import argparse
import json
import sys
import os
from pathlib import Path

# Add project root to sys.path to allow importing core modules
# This script is at h:\REapp\core\entry_lr_find.py
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description="Worker Entry: LR Finder")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    try:
        # Load config
        with open(args.config, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # Import AI libraries here (ALLOWED in worker)
        from core.lulynx_trainer.lr_finder import LRFinder
        from core.lulynx_trainer.config_adapter import ConfigAdapter

        # Convert config
        lulynx_config = ConfigAdapter.from_frontend_dict(config_dict)

        # Initialize LRFinder
        finder = LRFinder(config=lulynx_config)

        # Define progress callback
        def on_step(step, epoch, loss, lr):
            # Print JSON progress to stdout
            # We use a special prefix so the host can easily distinguish progress JSON from other logs
            msg = json.dumps({
                'type': 'progress',
                'step': step,
                'loss': float(loss),
                'lr': float(lr)
            })
            print(f"PROGRESS_JSON:{msg}", flush=True)

        finder.set_callbacks(on_step=on_step)

        # Run search
        results = finder.find(start_lr=1e-7, end_lr=1e-1, num_steps=100)
        suggested = finder.suggest_lr()

        # Print final result
        # Print final result
        msg = json.dumps({
            'type': 'result',
            'status': 'completed',
            'suggested_lr': float(suggested) if suggested else None,
            'results': results
        })
        print(f"PROGRESS_JSON:{msg}", flush=True)

    except Exception as e:
        import traceback
        msg = json.dumps({
            'type': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        })
        print(f"PROGRESS_JSON: {msg}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
