"""
sync_gazetteers.py

Downloads the six gazetteer sheets from a publicly-accessible Google Spreadsheet
and replaces the corresponding CSV files in input/gazetteers/.

Each sheet is fetched via the Google Sheets CSV export endpoint:
  https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid={GID}

Validation rules (applied once per sheet, the first time the script is run):
  - The downloaded CSV must have at least one data row (beyond the header).
  - The header columns must match the expected columns for that file exactly.
  - If validation fails the existing file is left untouched and the error is reported.

Usage:
  python scripts/sync_gazetteers.py              # update all sheets
  python scripts/sync_gazetteers.py --dry-run    # download & validate only, no writes
  python scripts/sync_gazetteers.py empire       # update a single sheet by name
"""

import argparse
import csv
import io
import logging
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPREADSHEET_ID = "1NjA45QfX9vfy97ZA8a7HcN-Zc-TKdr-nWGhKvaMRpVQ"
BASE_URL = (
    f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}"
    "/export?format=csv&gid={gid}"
)

# Maps the output filename (without extension) to its sheet GID and expected header.
SHEETS: dict[str, dict] = {
    "empire": {
        "gid": "1095068342",
        "expected_header": [
            "Settlement", "Population", "Estate", "Trade", "Tags", "Notes",
            "Coordinates", "Province_2515", "Province_2512", "Province_2276",
            "Ruler_2515", "Ruler_2512", "Ruler_2276",
            "wiki_url", "wiki_title", "wiki_description", "wiki_image",
        ],
    },
    "westerland": {
        "gid": "1817375421",
        "expected_header": [
            "Settlement", "Population", "Estate", "Trade", "Tags", "Notes",
            "Coordinates", "Province_2515", "Province_2512", "Province_2276",
            "Ruler_2515", "Ruler_2512", "Ruler_2276",
            "wiki_url", "wiki_title", "wiki_description", "wiki_image",
        ],
    },
    "bretonnia": {
        "gid": "890230032",
        "expected_header": [
            "Settlement", "Population", "Estate", "Trade", "Tags", "Notes",
            "Coordinates", "Province_2515", "Province_2512", "Province_2276",
            "Ruler_2515", "Ruler_2512", "Ruler_2276",
            "wiki_url", "wiki_title", "wiki_description", "wiki_image",
        ],
    },
    "karaz_ankor": {
        "gid": "164637166",
        "expected_header": [
            "Settlement", "Type", "Estate", "Trade", "Tags", "Notes",
            "Coordinates", "Province_2515", "Province_2512", "Province_2276",
            "Ruler_2515", "Ruler_2512", "Ruler_2276",
            "wiki_url", "wiki_title", "wiki_description", "wiki_image",
        ],
    },
    "wood_elves": {
        "gid": "904575690",
        "expected_header": [
            "Settlement", "Type", "Estate", "Trade", "Tags", "Notes",
            "Coordinates", "Province_2515", "Province_2512", "Province_2276",
            "Ruler_2515", "Ruler_2512", "Ruler_2276",
            "wiki_url", "wiki_title", "wiki_description", "wiki_image",
        ],
    },
    "provinces": {
        "gid": "2020870558",
        "expected_header": [
            "name", "formal_title", "info_description", "info_image",
            "info_wiki_url", "part_of", "population", "province_type",
        ],
    },
}

# Directory paths (relative to this script file)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GAZETTEER_DIR = REPO_ROOT / "input" / "gazetteers"
BACKUP_DIR = REPO_ROOT / "input" / "gazetteers" / "_backups"

# Sentinel file that records whether first-run validation has already passed
VALIDATED_SENTINEL = GAZETTEER_DIR / ".sync_validated"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def build_url(gid: str) -> str:
    """Return the CSV export URL for the given sheet GID."""
    return BASE_URL.format(gid=gid)


def fetch_csv(url: str) -> str:
    """Download the CSV at *url* and return its content as a string."""
    log.debug("Fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")


def parse_header(csv_text: str) -> list[str]:
    """Return the first row of *csv_text* as a list of column names."""
    reader = csv.reader(io.StringIO(csv_text))
    return next(reader)


def count_data_rows(csv_text: str) -> int:
    """Return the number of non-header rows in *csv_text*."""
    reader = csv.reader(io.StringIO(csv_text))
    next(reader)  # skip header
    return sum(1 for _ in reader)


def validate(name: str, csv_text: str, expected_header: list[str]) -> list[str]:
    """
    Validate the downloaded CSV content.

    Returns a (possibly empty) list of human-readable error strings.
    """
    errors: list[str] = []

    actual_header = parse_header(csv_text)
    if actual_header != expected_header:
        errors.append(
            f"[{name}] Header mismatch.\n"
            f"  Expected : {expected_header}\n"
            f"  Got      : {actual_header}"
        )

    n_rows = count_data_rows(csv_text)
    if n_rows == 0:
        errors.append(f"[{name}] Downloaded sheet contains no data rows.")

    return errors


def backup_existing(dest_path: Path) -> None:
    """Copy the current file at *dest_path* into a timestamped backup."""
    if not dest_path.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{dest_path.stem}_{timestamp}.csv"
    shutil.copy2(dest_path, backup_path)
    log.info("Backed up existing file → %s", backup_path.relative_to(REPO_ROOT))


def write_csv(dest_path: Path, csv_text: str) -> None:
    """Write *csv_text* to *dest_path*, ensuring Windows-style line endings are
    normalised to Unix line endings (consistent with the existing files)."""
    normalised = csv_text.replace("\r\n", "\n").replace("\r", "\n")
    dest_path.write_text(normalised, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main sync logic
# ---------------------------------------------------------------------------


def sync_sheet(
    name: str,
    config: dict,
    *,
    dry_run: bool = False,
    force_validate: bool = False,
) -> bool:
    """
    Download, (optionally) validate, and save one sheet.

    Returns True on success, False on failure.
    """
    dest_path = GAZETTEER_DIR / f"{name}.csv"
    url = build_url(config["gid"])

    log.info("─── %s ───", name)
    log.info("  Downloading from GID %s …", config["gid"])

    try:
        csv_text = fetch_csv(url)
    except Exception as exc:
        log.error("  Download failed: %s", exc)
        return False

    n_rows = count_data_rows(csv_text)
    log.info("  Downloaded %d data rows.", n_rows)

    # Validation: run on first sync OR when explicitly requested.
    needs_validation = force_validate or not VALIDATED_SENTINEL.exists()
    if needs_validation:
        log.info("  Validating format against expected schema …")
        errors = validate(name, csv_text, config["expected_header"])
        if errors:
            for err in errors:
                log.error(err)
            log.error("  Validation failed — %s will NOT be updated.", dest_path.name)
            return False
        log.info("  Validation passed ✓")

    if dry_run:
        log.info("  Dry-run mode: skipping write.")
        return True

    backup_existing(dest_path)
    write_csv(dest_path, csv_text)
    log.info("  Saved → %s", dest_path.relative_to(REPO_ROOT))
    return True


def mark_validated() -> None:
    """Create the sentinel file so subsequent runs skip slow validation."""
    VALIDATED_SENTINEL.write_text(
        f"First-run validation passed at {datetime.now().isoformat()}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync gazetteer CSVs from Google Sheets.",
    )
    parser.add_argument(
        "sheets",
        nargs="*",
        metavar="SHEET",
        help=(
            "Names of sheets to sync (e.g. empire bretonnia). "
            "Omit to sync all sheets."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and validate but do not write files.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        dest="force_validate",
        help="Force re-validation of headers even if already validated once.",
    )
    args = parser.parse_args(argv)

    # Determine which sheets to process
    if args.sheets:
        unknown = [s for s in args.sheets if s not in SHEETS]
        if unknown:
            log.error("Unknown sheet(s): %s. Valid names: %s", unknown, list(SHEETS))
            return 1
        target_sheets = {k: SHEETS[k] for k in args.sheets}
    else:
        target_sheets = SHEETS

    log.info("Syncing %d sheet(s) …", len(target_sheets))
    if args.dry_run:
        log.info("(dry-run mode — no files will be written)")

    first_run = not VALIDATED_SENTINEL.exists()
    results: dict[str, bool] = {}

    for name, config in target_sheets.items():
        results[name] = sync_sheet(
            name,
            config,
            dry_run=args.dry_run,
            force_validate=args.force_validate,
        )

    # After all sheets pass validation for the first time, write the sentinel.
    if not args.dry_run and first_run and all(results.values()):
        mark_validated()
        log.info("First-run validation complete — future runs will skip header checks.")

    # Summary
    passed = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]

    log.info("")
    log.info("Summary: %d succeeded, %d failed.", len(passed), len(failed))
    if failed:
        log.error("Failed sheets: %s", failed)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
