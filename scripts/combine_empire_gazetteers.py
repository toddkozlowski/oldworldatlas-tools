"""
Combine empire.csv and madalfred_empire.csv into a single comprehensive file.

This script merges data from two gazetteer sources:
- empire.csv: Main gazetteer with full column structure
- madalfred_empire.csv: Additional settlements from Madalfred's gazetteers

Rules:
1. Do NOT overwrite Population, Estate, or Trade from empire.csv
2. For existing settlements, only populate EMPTY fields with madalfred data
3. For new settlements, add "source:MadAlfred" tag
4. Handle non-latin characters (ö) by checking both original and latin equivalents (o, oe)

Author: Generated script
Date: February 14, 2026
"""

import csv
import re
import unicodedata
from pathlib import Path


def normalize_for_comparison(text):
    """
    Normalize text for comparison, handling non-latin characters.
    
    Returns a list of variations to check:
    - Original text
    - Text with diacritics removed
    - Text with ö -> o substitution
    - Text with ö -> oe substitution
    
    Args:
        text: Settlement name to normalize
        
    Returns:
        List of normalized variations
    """
    if not text:
        return []
    
    text = text.strip().lower()
    variations = [text]
    
    # Remove diacritics (accents)
    # Normalize to NFD (decomposed form) then remove combining characters
    nfd = unicodedata.normalize('NFD', text)
    without_diacritics = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    if without_diacritics != text:
        variations.append(without_diacritics)
    
    # Special handling for common German characters
    replacements = [
        ('ö', 'o'),
        ('ö', 'oe'),
        ('ä', 'a'),
        ('ä', 'ae'),
        ('ü', 'u'),
        ('ü', 'ue'),
        ('ß', 'ss'),
    ]
    
    for original, replacement in replacements:
        if original in text:
            variant = text.replace(original, replacement)
            if variant not in variations:
                variations.append(variant)
    
    return variations


def find_matching_settlement(settlement_name, empire_settlements):
    """
    Find if a settlement exists in empire data, considering variations.
    
    Args:
        settlement_name: Name to search for
        empire_settlements: Dictionary of empire settlements (normalized_name -> row_data)
        
    Returns:
        Matching settlement name from empire data, or None
    """
    variations = normalize_for_comparison(settlement_name)
    
    for variant in variations:
        if variant in empire_settlements:
            return variant
    
    return None


def merge_field(empire_value, madalfred_value):
    """
    Merge a field value, only using madalfred if empire is empty.
    
    Args:
        empire_value: Value from empire.csv
        madalfred_value: Value from madalfred_empire.csv
        
    Returns:
        Merged value
    """
    # Check if empire value is empty (None, empty string, or just whitespace)
    if not empire_value or not empire_value.strip():
        return madalfred_value if madalfred_value else ''
    return empire_value


def add_tag(existing_tags, new_tag):
    """
    Add a tag to existing tags field.
    
    Args:
        existing_tags: Current tags value
        new_tag: Tag to add
        
    Returns:
        Updated tags string
    """
    if not existing_tags or not existing_tags.strip():
        return f'"{new_tag}"'
    
    # Remove quotes if present
    tags = existing_tags.strip().strip('"')
    
    # Check if tag already exists
    if new_tag in tags:
        return existing_tags
    
    # Add new tag
    if tags:
        return f'"{tags},{new_tag}"'
    else:
        return f'"{new_tag}"'


def main():
    """
    Main function to combine empire.csv and madalfred_empire.csv.
    """
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    empire_file = project_root / 'input' / 'gazetteers' / 'empire.csv'
    madalfred_file = project_root / 'input' / 'gazetteers' / 'madalfred_empire.csv'
    output_file = project_root / 'input' / 'gazetteers' / 'empire_combined.csv'
    
    print(f"Reading empire.csv from: {empire_file}")
    print(f"Reading madalfred_empire.csv from: {madalfred_file}")
    
    # Read empire.csv
    empire_data = []
    empire_settlements = {}  # normalized_name -> index in empire_data
    
    with open(empire_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        empire_fieldnames = reader.fieldnames
        
        for row in reader:
            empire_data.append(row)
            # Store all variations for this settlement
            variations = normalize_for_comparison(row['Settlement'])
            for variant in variations:
                if variant not in empire_settlements:
                    empire_settlements[variant] = len(empire_data) - 1
    
    print(f"Loaded {len(empire_data)} settlements from empire.csv")
    
    # Read madalfred_empire.csv
    madalfred_data = []
    
    with open(madalfred_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            madalfred_data.append(row)
    
    print(f"Loaded {len(madalfred_data)} settlements from madalfred_empire.csv")
    
    # Process madalfred data
    new_settlements = 0
    updated_fields = 0
    
    for mad_row in madalfred_data:
        settlement_name = mad_row['Settlement']
        
        # Check if settlement exists in empire
        matching_key = find_matching_settlement(settlement_name, empire_settlements)
        
        if matching_key is not None:
            # Settlement exists - update only empty fields
            empire_idx = empire_settlements[matching_key]
            empire_row = empire_data[empire_idx]
            
            # Fields we can update (NOT Population, Estate, Trade)
            update_fields = {
                'Notes': mad_row.get('Notes', ''),
                'Coordinates': mad_row.get('Coordinates', ''),
                'Ruler_2515': mad_row.get('Ruler', ''),  # Map Ruler to Ruler_2515
            }
            
            # Also update Province_2515 if empty
            if not empire_row.get('Province_2515', '').strip():
                update_fields['Province_2515'] = mad_row.get('Province_2515', '')
            
            for field, mad_value in update_fields.items():
                if mad_value and not empire_row.get(field, '').strip():
                    empire_row[field] = mad_value
                    updated_fields += 1
        
        else:
            # New settlement - add it with source:MadAlfred tag
            new_row = {}
            
            # Initialize all empire columns
            for field in empire_fieldnames:
                new_row[field] = ''
            
            # Populate from madalfred data (but NOT Population, Estate, Trade per requirements)
            new_row['Settlement'] = settlement_name
            new_row['Notes'] = mad_row.get('Notes', '')
            new_row['Coordinates'] = mad_row.get('Coordinates', '')
            new_row['Province_2515'] = mad_row.get('Province_2515', '')
            new_row['Ruler_2515'] = mad_row.get('Ruler', '')
            
            # Add source tag
            new_row['Tags'] = '"source:MadAlfred"'
            
            empire_data.append(new_row)
            new_settlements += 1
    
    print(f"\nMerge complete:")
    print(f"  - {new_settlements} new settlements added")
    print(f"  - {updated_fields} empty fields populated from madalfred data")
    print(f"  - Total settlements in combined file: {len(empire_data)}")
    
    # Write combined data
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=empire_fieldnames)
        writer.writeheader()
        writer.writerows(empire_data)
    
    print(f"\nOutput written to: {output_file}")
    print("Done!")


if __name__ == '__main__':
    main()
