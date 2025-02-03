#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from datasets import Dataset
import glob
import html
import re
import string
import os

def clean_xml_content(content):
    """
    Clean and prepare XML content for parsing.
    """
    # First, escape any unescaped & characters that aren't part of an entity
    content = re.sub(r'&(?!(amp|lt|gt|apos|quot|#\d+|#x[0-9a-fA-F]+);)', '&amp;', content)
    
    # Add root element if not present
    if not content.strip().startswith('<?xml'):
        content = '<?xml version="1.0" encoding="UTF-8"?>\n<root>\n' + content + '\n</root>'
    
    return content

def process_xml_files():
    """Process XML files and extract abstracts."""
    abstracts = []
    
    xml_files = glob.glob('DATA/*.xml')
    print(f"Found {len(xml_files)} XML files")
    
    if not xml_files:
        print("No XML files found in DATA directory. Exiting.")
        return
    
    for file in xml_files:
        try:
            # Read file content
            with open(file, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            # Clean XML content
            xml_content = clean_xml_content(xml_content)
            
            # Save cleaned XML for debugging
            cleaned_path = f"{file}.cleaned.xml"
            with open(cleaned_path, "w", encoding="utf-8") as debug_file:
                debug_file.write(xml_content)
            
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError as e:
                print(f"XML parsing error in {file}: {e}")
                print("Attempting alternative parsing method...")
                try:
                    # Alternative parsing attempt with more aggressive cleaning
                    xml_content = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]', '', xml_content)
                    root = ET.fromstring(xml_content)
                except ET.ParseError as e2:
                    print(f"Alternative parsing also failed: {e2}")
                    continue
            
            # Process each publication
            for pub in root.findall('.//publication'):
                abstract_text = pub.get('abstract', '')
                if abstract_text:
                    # Decode HTML entities
                    abstract_text = html.unescape(abstract_text)
                    # Remove HTML tags
                    abstract_text = re.sub(r'<[^>]+>', ' ', abstract_text)
                    # Clean up whitespace
                    abstract_text = ' '.join(abstract_text.split())
                    
                    formatted_text = f"Summarize this UK Biobank abstract:\n\n{abstract_text}\n\n"
                    abstracts.append({'text': formatted_text})
                    print(f"Successfully extracted abstract from {file}")
                    print(f"Abstract length: {len(abstract_text)} characters")
                    print(f"Preview: {abstract_text[:100]}...")
        
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    # Save the dataset if we have any abstracts
    if abstracts:
        dataset = Dataset.from_list(abstracts)
        os.makedirs('ANALYSIS', exist_ok=True)
        dataset.save_to_disk('ANALYSIS/processed_data')
        print(f"\nSuccessfully processed {len(abstracts)} abstracts")
    else:
        print("\nNo valid abstracts extracted. Dataset not saved.")

if __name__ == "__main__":
    process_xml_files()
