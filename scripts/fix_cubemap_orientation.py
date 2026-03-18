from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.cubemap_dataset import rotate_dataset_orientation_180


def main() -> int:
    parser = argparse.ArgumentParser(description="Rotate cubemap images and masks by 180 degrees in-place")
    parser.add_argument(
        "--dataset-root",
        default=r"C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\mandarabyyoo",
        help="Dataset root containing perspective_views and masks",
    )
    parser.add_argument("--force", action="store_true", help="Rotate again even if marker file exists")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be rotated without modifying files")
    args = parser.parse_args()

    result = rotate_dataset_orientation_180(
        dataset_root=Path(args.dataset_root),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())