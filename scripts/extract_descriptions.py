"""
Script to extract descriptions for settlements that already have wiki URLs.

This script reads the CSV files and only processes settlements that have
a wiki_url already populated, extracting the article description text.
"""

import csv
import time
import re
from typing import Dict, List
import requests
from bs4 import BeautifulSoup


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
        
    except Exception as e:
        print(f"  Error extracting description: {e}")
        return ""


def update_descriptions_for_csv(csv_file: str):
    """
    Update descriptions for settlements that already have wiki URLs.
    
    Args:
        csv_file: Path to the CSV file to update
    """
    print(f"\n{'='*70}")
    print(f"Processing: {csv_file}")
    print(f"{'='*70}\n")
    
    # Read the CSV
    rows = []
    settlements_to_process = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            rows.append(row)
            # Check if this settlement has a wiki_url but no description
            if row.get('wiki_url') and row['wiki_url'].strip():
                if not row.get('wiki_description') or not row['wiki_description'].strip():
                    settlements_to_process.append((len(rows) - 1, row))
    
    total = len(settlements_to_process)
    print(f"Found {total} settlements with wiki URLs that need descriptions\n")
    
    if total == 0:
        print("No settlements need processing!")
        return
    
    # Process each settlement
    for idx, (row_idx, row) in enumerate(settlements_to_process, 1):
        settlement_name = row['Settlement']
        wiki_title = row['wiki_title']
        
        print(f"[{idx}/{total} - {(idx/total)*100:.1f}%] Processing: {settlement_name}...", end=' ')
        
        try:
            description = get_article_description(wiki_title)
            
            if description:
                rows[row_idx]['wiki_description'] = description
                print(f"✓ Description extracted ({len(description)} chars)")
            else:
                print(f"✗ No description found")
            
            # Rate limiting
            if idx < total:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Write updated CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n{'='*70}")
    print(f"✓ Updated {csv_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("WIKI DESCRIPTION EXTRACTOR")
    print("Extracting descriptions for settlements with existing wiki URLs")
    print("="*70)
    
    # Process both CSV files
    csv_files = [
        "input/gazetteers/empire.csv",
        "input/gazetteers/westerland.csv"
    ]
    
    for csv_file in csv_files:
        try:
            update_descriptions_for_csv(csv_file)
        except FileNotFoundError:
            print(f"\n⚠️  File not found: {csv_file}")
        except Exception as e:
            print(f"\n⚠️  Error processing {csv_file}: {e}")
    
    print("\n" + "="*70)
    print("Process complete!")
    print("="*70 + "\n")
