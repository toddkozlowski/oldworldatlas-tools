"""
Script to query Warhammer Fandom Wiki for settlement metadata.

This script reads settlement names from CSV files in the gazetteers directory,
queries the Warhammer Fandom Wiki for each settlement, and extracts metadata
(URL, title, description, image) using the MediaWiki API.

Usage:
    python download_wiki_metadata.py [csv_filename]
    
    If no filename is provided, defaults to 'empire.csv'
    
Examples:
    python download_wiki_metadata.py empire.csv
    python download_wiki_metadata.py westerland.csv
"""

import csv
import time
import unicodedata
import re
import os
from typing import Dict, Optional
import requests
from bs4 import BeautifulSoup


def normalize_name_to_latin(name: str) -> str:
    """
    Convert non-latin characters to their closest latin equivalents.
    
    Args:
        name: Settlement name that may contain non-latin characters
        
    Returns:
        Name with non-latin characters replaced by latin equivalents
    """
    # Normalize to NFD (decomposed form) and filter out combining characters
    normalized = unicodedata.normalize('NFD', name)
    latin_name = ''.join(
        char for char in normalized
        if unicodedata.category(char) != 'Mn'  # Mn = Nonspacing_Mark
    )
    return latin_name


def extract_name_variants(name: str) -> list:
    """
    Extract name variants from a settlement name that may contain parentheses.
    
    For names like "Karak Ungor (Red-Eye Mountain)", returns:
    - The original name: "Karak Ungor (Red-Eye Mountain)"
    - Name without parentheses: "Karak Ungor"
    - Content inside parentheses: "Red-Eye Mountain"
    
    Args:
        name: Settlement name that may contain parentheses
        
    Returns:
        List of name variants to try
    """
    variants = [name]  # Always try original name first
    
    # Check if name contains parentheses
    paren_match = re.search(r'(.+?)\s*\((.+?)\)', name)
    if paren_match:
        # Extract name without parentheses (strip whitespace)
        name_without_paren = paren_match.group(1).strip()
        # Extract content inside parentheses
        paren_content = paren_match.group(2).strip()
        
        # Add name without parentheses
        if name_without_paren and name_without_paren != name:
            variants.append(name_without_paren)
        
        # Add parentheses content
        if paren_content:
            variants.append(paren_content)
    
    return variants


def get_article_description(page_title: str) -> str:
    """
    Extract opening sentences from a Fandom wiki article.
    
    Args:
        page_title: Title of the wiki page
        
    Returns:
        String containing up to 3 opening sentences from the article
    """
    api_url = "https://warhammerfantasy.fandom.com/api.php"
    
    params = {
        'action': 'parse',
        'format': 'json',
        'page': page_title,
        'prop': 'text',
        'disabletoc': True
    }
    
    try:
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'parse' not in data:
            return ""
        
        html_content = data['parse']['text']['*']
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove aside/infobox elements
        for aside in soup.find_all(['aside', 'table']):
            aside.decompose()
        
        # Remove reference tags and citations
        for ref in soup.find_all(['sup', 'span'], class_=lambda x: x and 'reference' in str(x)):
            ref.decompose()
        
        # Remove quote blocks if present at the start
        for quote in soup.find_all('blockquote'):
            quote.decompose()
        
        # Find the main content paragraphs
        paragraphs = soup.find_all('p')
        
        text_parts = []
        for p in paragraphs:
            # Get text with separator to handle links properly
            text = p.get_text(separator=' ', strip=True)
            
            # Skip if starts with quote or is very short
            if text.startswith('"') or text.startswith("'") or len(text) < 20:
                continue
                
            text_parts.append(text)
            
            # Stop after we have enough content (at least 200 chars)
            if len(' '.join(text_parts)) > 200:
                break
        
        if not text_parts:
            return ""
        
        full_text = ' '.join(text_parts)
        
        # Clean up the text
        # Remove citation markers like [1], [2], [1a], [1b], etc.
        full_text = re.sub(r'\[\d+[a-z]?\]', '', full_text)
        # Remove multiple spaces
        full_text = re.sub(r'\s+', ' ', full_text)
        # Fix common spacing issues around punctuation
        full_text = re.sub(r'\s+([.,!?;:])', r'\1', full_text)
        
        # Extract up to 3 sentences
        # Split by sentence endings (period, exclamation, question mark followed by space and capital)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', full_text)
        
        # Take up to 3 sentences
        description = ' '.join(sentences[:3]).strip()
        
        return description
        
    except Exception:
        return ""


def fetch_wiki_metadata(settlement_name: str) -> Optional[Dict[str, str]]:
    """
    Fetch metadata from Warhammer Fandom Wiki using MediaWiki API.
    
    Args:
        settlement_name: Name of the settlement to query
        
    Returns:
        Dictionary with keys: url, title, description, image
        Returns None if the page doesn't exist
    """
    api_url = "https://warhammerfantasy.fandom.com/api.php"
    
    # Extract name variants (handles parentheses)
    names_to_try = extract_name_variants(settlement_name)
    
    # For each base variant, also try latin equivalent if needed
    expanded_names = []
    for name_variant in names_to_try:
        expanded_names.append(name_variant)
        latin_name = normalize_name_to_latin(name_variant)
        if latin_name != name_variant:
            expanded_names.append(latin_name)
    
    names_to_try = expanded_names
    
    for name_variant in names_to_try:
        try:
            # Query the MediaWiki API
            params = {
                'action': 'query',
                'format': 'json',
                'titles': name_variant,
                'prop': 'info|pageimages|pageprops',
                'inprop': 'url',
                'piprop': 'original',
                'ppprop': 'description'
            }
            
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            # Check if page exists (page_id != -1 means it exists)
            for page_id, page_data in pages.items():
                if page_id != '-1':  # Page exists
                    title = page_data.get('title', '')
                    url = page_data.get('fullurl', '')
                    
                    # Get description by extracting article text
                    description = get_article_description(title)
                    
                    # Get image
                    image = ''
                    if 'original' in page_data:
                        image = page_data['original'].get('source', '')
                    
                    return {
                        'url': url,
                        'title': title,
                        'description': description,
                        'image': image
                    }
            
        except requests.RequestException:
            # Try next name variant
            continue
    
    # No page found for any variant
    return None


def process_settlements(input_csv: str, output_csv: str, log_file: str):
    """
    Process all settlements from the input CSV and fetch wiki metadata.
    
    Args:
        input_csv: Path to input CSV file with settlement names
        output_csv: Path to output CSV file for wiki metadata
        log_file: Path to log file for errors
    """
    # Read settlements from CSV
    settlements = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            settlements.append(row['Settlement'])
    
    csv_name = os.path.basename(input_csv)
    total_settlements = len(settlements)
    print(f"\n{'='*70}")
    print(f"Starting wiki metadata download for {total_settlements} settlements")
    print(f"Source: {csv_name}")
    print(f"{'='*70}\n")
    
    # Statistics
    processed = 0
    found = 0
    errors = []
    
    # Output data
    results = []
    
    # Open log file
    interrupted = False
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"Wiki Metadata Download Log - {csv_name}\n")
        log.write("=" * 70 + "\n\n")
        
        try:
            for idx, settlement in enumerate(settlements, 1):
                processed += 1
                
                # Progress indicator
                progress_pct = (idx / total_settlements) * 100
                print(f"[{idx}/{total_settlements} - {progress_pct:.1f}%] Processing: {settlement}...", end=' ')
                
                try:
                    metadata = fetch_wiki_metadata(settlement)
                    
                    if metadata:
                        found += 1
                        results.append({
                            'settlement': settlement,
                            'url': metadata['url'],
                            'title': metadata['title'],
                            'description': metadata['description'],
                            'image': metadata['image']
                        })
                        print(f"✓ Found (Total: {found})")
                        log.write(f"SUCCESS [{idx}/{total_settlements}]: {settlement}\n")
                        log.write(f"  URL: {metadata['url']}\n\n")
                    else:
                        results.append({
                            'settlement': settlement,
                            'url': '',
                            'title': '',
                            'description': '',
                            'image': ''
                        })
                        print(f"✗ Not found")
                        log.write(f"NOT FOUND [{idx}/{total_settlements}]: {settlement}\n\n")
                        
                except Exception as e:
                    error_msg = f"Error processing {settlement}: {str(e)}"
                    errors.append(error_msg)
                    results.append({
                        'settlement': settlement,
                        'url': '',
                        'title': '',
                        'description': '',
                        'image': ''
                    })
                    print(f"✗ Error")
                    log.write(f"ERROR [{idx}/{total_settlements}]: {settlement}\n")
                    log.write(f"  {str(e)}\n\n")
                
                # Rate limiting: wait 0.5 seconds between requests
                if idx < total_settlements:
                    time.sleep(0.5)
                    
        except KeyboardInterrupt:
            interrupted = True
            print("\n\n⚠️  Process interrupted by user")
            log.write("\n" + "="*70 + "\n")
            log.write("PROCESS INTERRUPTED BY USER\n")
            log.write("="*70 + "\n")
    
    # Write results to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['settlement', 'url', 'title', 'description', 'image']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary report
    print(f"\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}")
    if interrupted:
        print("⚠️  PROCESS WAS INTERRUPTED")
    print(f"Total settlements processed: {processed}")
    print(f"Wiki pages found: {found}")
    print(f"Wiki pages not found: {processed - found}")
    print(f"Errors encountered: {len(errors)}")
    if interrupted and processed < total_settlements:
        print(f"Remaining settlements: {total_settlements - processed}")
    print(f"\nOutput saved to: {output_csv}")
    print(f"Log saved to: {log_file}")
    print(f"{'='*70}\n")
    
    if errors:
        print("Errors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more (see log file)")


def update_original_csv(original_csv: str, metadata_csv: str):
    """
    Update the original CSV with wiki metadata from the output CSV.
    
    Args:
        original_csv: Path to original CSV file to update
        metadata_csv: Path to CSV file with wiki metadata
    """
    print(f"\n{'='*70}")
    print("UPDATING ORIGINAL CSV")
    print(f"{'='*70}\n")
    
    # Read metadata
    metadata_dict = {}
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata_dict[row['settlement']] = {
                'wiki_url': row['url'],
                'wiki_title': row['title'],
                'wiki_description': row['description'],
                'wiki_image': row['image']
            }
    
    # Read original CSV
    rows = []
    with open(original_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            settlement = row['Settlement']
            if settlement in metadata_dict:
                row.update(metadata_dict[settlement])
            rows.append(row)
    
    # Write updated CSV
    with open(original_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Updated {original_csv} with wiki metadata")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    # Default to empire.csv, but allow specifying a different CSV file
    # Optional --yes / -y flag skips the interactive update prompt
    args = sys.argv[1:]
    auto_yes = '--yes' in args or '-y' in args
    csv_args = [a for a in args if not a.startswith('-')]
    
    if csv_args:
        csv_name = csv_args[0]
    else:
        csv_name = "empire.csv"
    
    input_csv = f"input/gazetteers/{csv_name}"
    base_name = os.path.splitext(csv_name)[0]
    output_csv = f"output/{base_name}_wiki_metadata.csv"
    log_file = f"logs/{base_name}_wiki_download.log"
    
    print("\n" + "="*70)
    print("WARHAMMER FANDOM WIKI METADATA DOWNLOADER")
    print("="*70)
    
    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"\n❌ Error: Input file not found: {input_csv}")
        print(f"Usage: python download_wiki_metadata.py [csv_filename] [-y|--yes]")
        sys.exit(1)
    
    # Step 1: Download wiki metadata
    process_settlements(input_csv, output_csv, log_file)
    
    # Step 2: Update the original CSV (auto if -y/--yes, otherwise prompt)
    if auto_yes:
        update_original_csv(input_csv, output_csv)
        print("✓ All done! The original CSV has been updated with wiki metadata.")
    else:
        print("\nWould you like to update the original CSV file with the wiki metadata?")
        print(f"This will update: {input_csv}")
        response = input("Enter 'yes' to proceed, or press Enter to skip: ").strip().lower()
        
        if response == 'yes':
            update_original_csv(input_csv, output_csv)
            print("✓ All done! The original CSV has been updated with wiki metadata.")
        else:
            print(f"\nSkipped updating original CSV.")
            print(f"You can manually update it later using the data in: {output_csv}")
    
    print("\n" + "="*70)
    print("Process complete!")
    print("="*70 + "\n")
