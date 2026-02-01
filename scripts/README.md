# Wiki Metadata Scripts

This directory contains scripts for downloading and managing wiki metadata for settlements from the Warhammer Fantasy Fandom Wiki.

## Scripts

### download_wiki_metadata.py

Main script for downloading wiki metadata (URL, title, description, image) for settlements from any gazetteer CSV file.

**Usage:**
```bash
# Download metadata for empire.csv (default)
python scripts/download_wiki_metadata.py

# Download metadata for westerland.csv
python scripts/download_wiki_metadata.py westerland.csv

# Download metadata for any other CSV file
python scripts/download_wiki_metadata.py <filename>.csv
```

**Features:**
- Queries the MediaWiki API for settlement information
- Extracts article descriptions (first 3 sentences)
- Handles non-latin characters (e.g., Bögenhafen)
- Rate limiting (0.5 seconds between requests)
- Saves results to `output/<name>_wiki_metadata.csv`
- Creates log file at `logs/<name>_wiki_download.log`
- Optionally updates the original CSV file with metadata

### extract_descriptions.py

Utility script for extracting article descriptions for settlements that already have wiki URLs populated in the CSV files.

**Usage:**
```bash
python scripts/extract_descriptions.py
```

**Features:**
- Only processes settlements with existing wiki_url values
- Extracts and updates wiki_description column
- Processes both empire.csv and westerland.csv
- Faster than re-running full download

### process_map_svg.py

Script for processing SVG map files (separate functionality).

## Requirements

Install required packages:
```bash
pip install requests beautifulsoup4
```

## CSV Format

The gazetteer CSV files must have the following columns:
- `Settlement` - Name of the settlement
- `wiki_url` - URL to wiki article (populated by scripts)
- `wiki_title` - Title from wiki (populated by scripts)
- `wiki_description` - Article description (populated by scripts)
- `wiki_image` - Image URL from wiki (populated by scripts)

## Output Files

- `output/<name>_wiki_metadata.csv` - Extracted metadata for all settlements
- `logs/<name>_wiki_download.log` - Detailed log of processing

## Notes

- The script respects the wiki's rate limits with 0.5 second delays between requests
- Non-existent wiki pages will have empty metadata fields
- Descriptions are extracted from the first 3 sentences of articles
- References and formatting are stripped from descriptions
