"""
Merge two Empire gazette CSV files into one combined file.

The "empire_combined.csv" file is the master. For each settlement, if a field is
empty in the master but filled in the "empire.csv" file, use the data from
"empire.csv". Settlement name is the unique identifier.
"""

import csv
from pathlib import Path
from collections import defaultdict

INPUT_DIR = Path(__file__).parent.parent / "input" / "gazetteers"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

MASTER_FILE = INPUT_DIR / "AtlasoftheOldWorld - empire_combined.csv"
SECONDARY_FILE = INPUT_DIR / "AtlasoftheOldWorld - empire.csv"
OUTPUT_FILE = OUTPUT_DIR / "AotOW-combined.csv"

def read_csv(filepath):
    """Read CSV file and return list of dicts and fieldnames."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    return rows, fieldnames

def merge_gazetteers():
    """Merge master and secondary gazetteers."""
    print(f"Reading master file: {MASTER_FILE.name}")
    master_rows, fieldnames = read_csv(MASTER_FILE)
    
    print(f"Reading secondary file: {SECONDARY_FILE.name}")
    secondary_rows, secondary_fieldnames = read_csv(SECONDARY_FILE)
    
    # Verify same fieldnames
    if fieldnames != secondary_fieldnames:
        print("WARNING: Field names differ between files!")
        print(f"  Master: {fieldnames}")
        print(f"  Secondary: {secondary_fieldnames}")
    
    # Index both files by settlement name
    master_dict = {}
    for row in master_rows:
        settlement_name = row.get('Settlement', '').strip()
        if settlement_name:
            master_dict[settlement_name] = row
    
    secondary_dict = {}
    for row in secondary_rows:
        settlement_name = row.get('Settlement', '').strip()
        if settlement_name:
            secondary_dict[settlement_name] = row
    
    print(f"\nMaster file: {len(master_rows)} settlements ({len(master_dict)} unique)")
    print(f"Secondary file: {len(secondary_rows)} settlements ({len(secondary_dict)} unique)")
    
    # Merge data: start with all master settlements
    merged_rows = []
    filled_from_secondary = 0
    unique_from_secondary = 0
    
    # First pass: process all master settlements, filling from secondary where needed
    for settlement_name, master_row in master_dict.items():
        # Make a copy of the master row
        merged_row = dict(master_row)
        
        # If this settlement exists in secondary, fill any empty fields
        if settlement_name in secondary_dict:
            secondary_row = secondary_dict[settlement_name]
            
            # For each field, if master is empty but secondary has data, use secondary
            for field in fieldnames:
                master_value = merged_row.get(field, '').strip()
                secondary_value = secondary_row.get(field, '').strip()
                
                # If master is empty and secondary has data, use secondary
                if not master_value and secondary_value:
                    merged_row[field] = secondary_value
                    filled_from_secondary += 1
        
        merged_rows.append(merged_row)
    
    # Second pass: add any settlements unique to secondary file
    for settlement_name, secondary_row in secondary_dict.items():
        if settlement_name not in master_dict:
            # This settlement only exists in secondary, add it
            merged_rows.append(dict(secondary_row))
            unique_from_secondary += 1
    
    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    
    print(f"\nMerge complete!")
    print(f"Output file: {OUTPUT_FILE.name}")
    print(f"Total merged settlements: {len(merged_rows)}")
    print(f"Settlements from master: {len(master_dict)}")
    print(f"Unique settlements from secondary: {unique_from_secondary}")
    print(f"Fields filled from secondary data: {filled_from_secondary}")

if __name__ == "__main__":
    merge_gazetteers()
