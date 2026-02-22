"""
deploy_gazetteers.py

Copies all .geojson files from output/ into the web data directory.
If a file with the same name already exists at the destination, the existing
file is moved to a "backup" sub-folder with a timestamp appended to its name
(e.g. empire_settlements_20260222_120810.geojson) before the new file is copied across.

Usage:
  python scripts/deploy_gazetteers.py              # deploy all GeoJSON files
  python scripts/deploy_gazetteers.py --dry-run    # preview actions, no writes
  python scripts/deploy_gazetteers.py empire_settlements   # deploy a single file by stem
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

SOURCE_DIR = REPO_ROOT / "output"
DEST_DIR = Path(r"C:\Users\toddc\dev\personal\old-world-atlas\oldworldatlas-web\data")
BACKUP_DIR = DEST_DIR / "backup"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def backup_existing(dest_file: Path, timestamp: str, dry_run: bool) -> None:
    """Move *dest_file* into BACKUP_DIR with a timestamp suffix."""
    backup_name = f"{dest_file.stem}_{timestamp}{dest_file.suffix}"
    backup_path = BACKUP_DIR / backup_name
    log.info("    Backing up existing file → backup/%s", backup_name)
    if not dry_run:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dest_file), backup_path)


def deploy_file(source: Path, timestamp: str, dry_run: bool) -> bool:
    """
    Copy *source* to DEST_DIR, backing up any pre-existing file first.

    Returns True on success, False on error.
    """
    dest_file = DEST_DIR / source.name
    log.info("─── %s ───", source.name)

    try:
        if dest_file.exists():
            backup_existing(dest_file, timestamp, dry_run)

        log.info("    Copying → %s", dest_file)
        if not dry_run:
            DEST_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest_file)

        return True

    except Exception as exc:
        log.error("    Failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deploy gazetteer CSVs to the web data directory.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="STEM",
        help=(
            "Stem(s) of GeoJSON files to deploy (e.g. empire_settlements karaz_ankor). "
            "Omit to deploy all GeoJSON files in output/."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without copying or moving any files.",
    )
    args = parser.parse_args(argv)

    # Collect source files
    if args.files:
        sources: list[Path] = []
        for stem in args.files:
            candidate = SOURCE_DIR / f"{stem}.geojson"
            if not candidate.exists():
                log.error("Source file not found: %s", candidate)
                return 1
            sources.append(candidate)
    else:
        sources = sorted(SOURCE_DIR.glob("*.geojson"))
        if not sources:
            log.warning("No CSV files found in %s", SOURCE_DIR)
            return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log.info("Source : %s", SOURCE_DIR.relative_to(REPO_ROOT))
    log.info("Dest   : %s", DEST_DIR)
    log.info("Deploying %d GeoJSON file(s) …", len(sources))
    if args.dry_run:
        log.info("(dry-run mode — no files will be written)")

    results = {src.name: deploy_file(src, timestamp, args.dry_run) for src in sources}

    passed = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]

    log.info("")
    log.info("Summary: %d deployed, %d failed.", len(passed), len(failed))
    if failed:
        log.error("Failed: %s", failed)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
