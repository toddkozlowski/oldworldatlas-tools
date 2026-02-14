"""
Parse Madalfred gazetteer text files and convert them to CSV format.

This script reads text files from the input/gazetteers/madalfred directory,
extracts settlement data from the tables, and combines them into a single
CSV file structured similarly to empire.csv.

Author: Generated script
Date: February 14, 2026
"""

import csv
import os
import re
from pathlib import Path


def normalize_settlement_name(name):
    """Convert settlement name to proper case (Title Case)."""
    if not name:
        return ""
    return name.strip().title()


def clean_field(text):
    """Clean and normalize a field value."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    if text == '-' or text == '-/-':
        return ""
    return text


def extract_province_name(filename):
    """Extract province name from filename."""
    return Path(filename).stem.title()


def tokenize_text(text):
    """
    Tokenize the text into meaningful chunks.
    Splits on whitespace but preserves structure.
    """
    # Split lines and tokenize each
    tokens = []
    for line in text.split('\n'):
        line_stripped = line.strip()
        if line_stripped:
            # Split on multiple spaces or tabs, but keep single-space phrases together
            parts = re.split(r'\s{2,}|\t+', line_stripped)
            for part in parts:
                if part.strip():
                    tokens.append(part.strip())
    return tokens


def parse_gazetteer_file(filepath):
    """
    Parse a single gazetteer text file using token-based extraction.
    
    Strategy:
    1. Read entire file as text
    2. Split into blocks at settlement boundaries
    3. Extract data from each block using patterns
    """
    province = extract_province_name(filepath.name)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find the start of data (after GAZETTEER OF THE... header)
    pattern = r'GAZETTEER OF THE .+?\n.+?\n'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    
    # Get text after the header
    data_section = content[match.end():]
    
    # Split into lines for processing
    lines = data_section.split('\n')
    
    settlements = []
    current_block = []
    size_codes = ['CS', 'C', 'T', 'ST', 'V', 'F', 'M']
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip column headers that repeat
        if 'Settlement Name' in line or line_stripped in ['Size', 'Ruler', 'Pop', 'Wealth', 'Source', 'Garrison/', 'Militia', 'Notes']:
            continue
        
        # Skip completely empty lines
        if not line_stripped:
            continue
        
        # Check if this starts a new settlement
        # A new settlement typically has: Name + Size_Code + possibly more data
        tokens = line_stripped.split()
        if len(tokens) < 2:
            # Not enough tokens, add to current block
            if current_block:
                current_block.append(line_stripped)
            continue
        
        first_token = tokens[0]
        second_token = tokens[1]
        
        # Check if line appears to start a new settlement
        # Pattern: SettlementName SizeCode ... (e.g., "AVERHEIM T Grand Count")
        # Pattern: SettlementName SizeCode ... (e.g., "Friedendorf V Grand Count")
        is_new_settlement = False
        
        # Check if second token is a size code (V, T, ST, C, CS, F, M)
        if second_token in size_codes:
            # Check that first token looks like a name (starts with letter, not all digits)
            if first_token and first_token[0].isalpha() and not first_token.isdigit():
                is_new_settlement = True
        # Special case for ST (Small Town) which might be split
        elif second_token == 'Town' and len(tokens) > 2:
            # Check if this is "Small Town" or just "Town"
            if first_token == 'Small':
                # Skip this, it's likely a continuation 
                pass
            elif first_token[0].isalpha():
                # This could be "SettlementName Town ..." which means size is T
                is_new_settlement = True
        
        if is_new_settlement:
            # Process previous block if it exists
            if current_block:
                settlement = extract_settlement_data(current_block, province)
                if settlement and settlement.get('settlement'):
                    settlements.append(settlement)
            
            # Start new block
            current_block = [line_stripped]
        else:
            # Add to current block
            if current_block:  # Only add if we've started collecting
                current_block.append(line_stripped)
    
    # Don't forget the last block
    if current_block:
        settlement = extract_settlement_data(current_block, province)
        if settlement and settlement.get('settlement'):
            settlements.append(settlement)
    
    return settlements


def extract_settlement_data(block, province):
    """
    Extract settlement data from a block of text lines.
    
    Args:
        block: List of text lines belonging to one settlement
        province: Province name
        
    Returns:
        Dictionary with settlement data
    """
    # Combine block into single text for easier regex matching
    text = ' '.join(block)
    
    # Limit text length to avoid accumulating too much data
    if len(text) > 500:
        text = text[:500]
    
    # Initialize settlement data
    settlement = {
        'settlement': '',
        'size': '',
        'ruler': '',
        'pop': '',
        'wealth': '',
        'source': '',
        'garrison': '',
        'notes': '',
        'province': province
    }
    
    # Split first line to get name and size
    first_line_parts = block[0].split()
    
    if not first_line_parts:
        return settlement
    
    # First token is usually the settlement name
    settlement['settlement'] = first_line_parts[0]
    
    # Look for size code (CS, C, T, ST, V, F, M)
    size_codes = ['CS', 'C', 'T', 'ST', 'V', 'F', 'M']
    for part in first_line_parts[1:4]:  # Check first few tokens
        if part in size_codes:
            settlement['size'] = part
            break
    # Also check for 'Town', 'Village', etc.
    if 'Town' in text[:30] and not settlement['size']:
        settlement['size'] = 'T'
    elif 'Village' in text[:30] and not settlement['size']:
        settlement['size'] = 'V'
    
    # Extract population (should be a 1-6 digit number, standalone)
    # Look for pattern: whitespace + number + whitespace
    pop_matches = re.findall(r'\s(\d{1,6})\s', ' ' + text + ' ')
    if pop_matches:
        # Take the first number that looks like a population (not too small)
        for match in pop_matches:
            pop_val = int(match)
            if pop_val > 0:  # Any positive number could be population
                settlement['pop'] = match
                break
    
    # Extract wealth (single digit 1-5, appears after population usually)
    # Look for standalone single digit
    if settlement['pop']:
        # Look after the population number
        pop_pos = text.find(settlement['pop'])
        remaining = text[pop_pos + len(settlement['pop']):]
        wealth_match = re.search(r'\s([1-5])\s', ' ' + remaining[:100] + ' ')
        if wealth_match:
            settlement['wealth'] = wealth_match.group(1)
    
    # Extract garrison/militia (pattern like "35a & 80b/ 350c" or "15b/36c" or "-/10c")
    garrison_match = re.search(r'([\d]+[abc][\s&]*)+/?[\s]*([\d]+[abc]*)?', text, re.IGNORECASE)
    if garrison_match:
        garrison_text = garrison_match.group(0)
        # Make sure it's not just a number followed by 'c' for another purpose
        if '/' in garrison_text or re.search(r'\d[abc]', garrison_text, re.IGNORECASE):
            settlement['garrison'] = garrison_text
    
    # Extract ruler (titles like Grand Count, Baron, Count, etc.)
    ruler_patterns = [
        r'((?:Grand\s+)?(?:Count|Baron|Emperor|Graf)(?:ess)?)\s+([\w\s]+?)(?=\s+\d{1,6}\s|\s+\d{1,3},\d{3}\s)',
        r'(Assembly)',  # For Mootland
        r'(Abbot|Grandmaster|Margrave|Duke|Duchess)\s+([\w\s]+?)(?=\s+\d{1,6}\s)',
    ]
    for pattern in ruler_patterns:
        ruler_match = re.search(pattern, text, re.IGNORECASE)
        if ruler_match:
            ruler_text = ruler_match.group(0).strip()
            # Limit ruler length
            if len(ruler_text) < 60:
                settlement['ruler'] = ruler_text
                break
    
    # Extract source/trade (keywords like Trade, Agriculture, etc.)
    trade_keywords = ['Trade', 'Agriculture', 'Cattle', 'Government', 'Timber', 'Fishing', 
                      'Mining', 'Subsistence', 'Wine', 'Wool', 'Sheep', 'Metal', 'Ore',
                      'Pottery', 'Textiles', 'Fur', 'Beer', 'Cheese', 'Tobacco', 'Woodcraft']
    
    trades = []
    for keyword in trade_keywords:
        if keyword in text:
            if keyword not in trades:
                trades.append(keyword)
    settlement['source'] = ', '.join(trades[:5]) if trades else ''  # Limit to first 5
    
    # Extract notes (typically appears after garrison or at end)
    # Look for descriptive text (not just keywords)
    notes_patterns = [
        r'(Provincial [Cc]apital[^.]*\.)',
        r'(Ferry (?:across|over)[^.]*\.)',
        r'(Bridge (?:across|over)[^.]*\.)',
        r'(Known for[^.]*\.)',
        r'(Fortified[^.]*\.)',
        r'(Former[^.]*\.)',
        r'(Wiped out[^.]*\.)',
        r'(Guards[^.]*\.)',
        r'(Marks[^.]*\.)',
        r'(Largest[^.]*\.)',
        r'(Dominated by[^.]*\.)',
        r'(Abbey[^.]*\.)',
        r'(Monastery[^.]*\.)',
        r'(Temple[^.]*\.)',
        r'(Centre[^.]*\.)',
        r'(Center[^.]*\.)',
    ]
    
    notes_parts = []
    for pattern in notes_patterns:
        for notes_match in re.finditer(pattern, text, re.IGNORECASE):
            note = notes_match.group(1).strip()
            if note and note not in notes_parts and len(note) < 200:
                notes_parts.append(note)
                if len(notes_parts) >= 2:  # Limit to 2 notes
                    break
        if len(notes_parts) >= 2:
            break
    
    settlement['notes'] = ' '.join(notes_parts)
    
    return settlement


def convert_to_output_format(settlements):
    """
    Convert settlements from parsed format to output CSV format.
    
    Args:
        settlements: List of settlement dictionaries from parsing
        
    Returns:
        List of dictionaries ready for CSV output
    """
    output_data = []
    
    for s in settlements:
        output_row = {
            'Settlement': normalize_settlement_name(s['settlement']),
            'Population': clean_field(s['pop']),
            'Estate': clean_field(s['size']),
            'Ruler': clean_field(s['ruler']),
            'Trade': clean_field(s['source']),
            'Tags': '',  # Leave empty as requested
            'Notes': clean_field(s['notes']),
            'Coordinates': '',  # Leave empty as requested
            'Province_2515': s['province']
        }
        output_data.append(output_row)
    
    return output_data


def main():
    """
    Main function to process all Madalfred gazetteer files.
    """
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_dir = project_root / 'input' / 'gazetteers' / 'madalfred'
    output_file = project_root / 'input' / 'gazetteers' / 'madalfred_empire.csv'
    
    print(f"Processing Madalfred gazetteer files from: {input_dir}")
    
    # Get all .txt files in the madalfred directory
    txt_files = sorted(input_dir.glob('*.txt'))
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(txt_files)} files to process")
    
    # Process each file
    all_settlements = []
    for txt_file in txt_files:
        print(f"  Processing {txt_file.name}...")
        settlements = parse_gazetteer_file(txt_file)
        print(f"    Found {len(settlements)} settlements")
        all_settlements.extend(settlements)
    
    # Convert to output format
    output_data = convert_to_output_format(all_settlements)
    
    print(f"\nTotal settlements extracted: {len(output_data)}")
    
    # Write to CSV
    fieldnames = [
        'Settlement', 'Population', 'Estate', 'Ruler', 'Trade',
        'Tags', 'Notes', 'Coordinates', 'Province_2515'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)
    
    print(f"\nOutput written to: {output_file}")
    print("Done!")


if __name__ == '__main__':
    main()
