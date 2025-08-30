#!/usr/bin/env python3
"""
CSV to JSON Converter Script
Reads CSV files and converts them to JSON format with proper data types and error handling.
"""

import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
import argparse


def clean_column_names(df):
    """
    Clean column names by removing extra spaces and special characters
    """
    df.columns = df.columns.str.strip()
    return df


def convert_csv_to_json(csv_file_path, output_file_path=None, indent=2, orient='records'):
    """
    Convert CSV file to JSON format
    
    Parameters:
    - csv_file_path: Path to the CSV file
    - output_file_path: Path for output JSON file (optional)
    - indent: JSON indentation for pretty printing
    - orient: JSON orientation ('records', 'index', 'values', etc.)
    
    Returns:
    - Dictionary containing the converted data and metadata
    """
    try:
        print(f"Reading CSV file: {csv_file_path}")
        
        # Read CSV file with pandas
        df = pd.read_csv(csv_file_path)
        
        # Clean column names
        df = clean_column_names(df)
        
        # Get basic information about the dataset
        info = {
            'filename': os.path.basename(csv_file_path),
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'conversion_timestamp': datetime.now().isoformat(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        print(f"Dataset info:")
        print(f"  - Rows: {info['rows']}")
        print(f"  - Columns: {info['columns']}")
        print(f"  - Column names: {', '.join(info['column_names'][:5])}{'...' if len(info['column_names']) > 5 else ''}")
        
        # Handle missing values by converting them to None (null in JSON)
        df = df.where(pd.notnull(df), None)
        
        # Convert DataFrame to JSON-compatible format
        if orient == 'records':
            data = df.to_dict(orient='records')
        elif orient == 'index':
            data = df.to_dict(orient='index')
        elif orient == 'columns':
            data = df.to_dict(orient='dict')
        else:
            data = df.to_dict(orient=orient)
        
        # Create the complete JSON structure
        json_output = {
            'metadata': info,
            'data': data
        }
        
        # Determine output file path
        if output_file_path is None:
            base_name = Path(csv_file_path).stem
            output_file_path = f"{base_name}.json"
        
        # Write to JSON file
        print(f"Writing JSON file: {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_output, json_file, indent=indent, ensure_ascii=False, default=str)
        
        print(f"✅ Successfully converted {csv_file_path} to {output_file_path}")
        return json_output
        
    except FileNotFoundError:
        print(f"❌ Error: CSV file '{csv_file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"❌ Error: CSV file '{csv_file_path}' is empty.")
        return None
    except pd.errors.ParserError as e:
        print(f"❌ Error parsing CSV file: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def convert_multiple_csvs(csv_files, output_dir=None, combine=False):
    """
    Convert multiple CSV files to JSON
    
    Parameters:
    - csv_files: List of CSV file paths
    - output_dir: Directory for output files
    - combine: Whether to combine all data into one JSON file
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_data = {}
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"⚠️  Warning: File '{csv_file}' not found. Skipping...")
            continue
        
        # Determine output path
        if output_dir:
            base_name = Path(csv_file).stem
            output_path = os.path.join(output_dir, f"{base_name}.json")
        else:
            output_path = None
        
        # Convert individual file
        result = convert_csv_to_json(csv_file, output_path)
        
        if result and combine:
            file_key = Path(csv_file).stem
            all_data[file_key] = result
        
        print("-" * 50)
    
    # Create combined file if requested
    if combine and all_data:
        combined_output = "combined_datasets.json"
        if output_dir:
            combined_output = os.path.join(output_dir, combined_output)
        
        combined_data = {
            'metadata': {
                'combined_datasets': list(all_data.keys()),
                'total_files': len(all_data),
                'combination_timestamp': datetime.now().isoformat()
            },
            'datasets': all_data
        }
        
        print(f"Writing combined JSON file: {combined_output}")
        with open(combined_output, 'w', encoding='utf-8') as json_file:
            json.dump(combined_data, json_file, indent=2, ensure_ascii=False, default=str)
        print(f"✅ Successfully created combined file: {combined_output}")


def main():
    """
    Main function to handle command line arguments and execute conversion
    """
    parser = argparse.ArgumentParser(description='Convert CSV files to JSON format')
    parser.add_argument('csv_files', nargs='+', help='CSV file(s) to convert')
    parser.add_argument('-o', '--output-dir', help='Output directory for JSON files')
    parser.add_argument('-c', '--combine', action='store_true', help='Combine all datasets into one JSON file')
    parser.add_argument('--orient', default='records', choices=['records', 'index', 'columns', 'values'],
                       help='JSON orientation (default: records)')
    parser.add_argument('--indent', type=int, default=2, help='JSON indentation (default: 2)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CSV TO JSON CONVERTER")
    print("=" * 60)
    
    if len(args.csv_files) == 1:
        # Single file conversion
        csv_file = args.csv_files[0]
        output_path = None
        if args.output_dir:
            base_name = Path(csv_file).stem
            output_path = os.path.join(args.output_dir, f"{base_name}.json")
        
        convert_csv_to_json(csv_file, output_path, args.indent, args.orient)
    else:
        # Multiple files conversion
        convert_multiple_csvs(args.csv_files, args.output_dir, args.combine)
    
    print("=" * 60)
    print("Conversion completed!")


# Example usage for your specific files
if __name__ == "__main__":
    # If running directly, convert the specific files mentioned
    csv_files = [
        "renewable_hydrogen_dataset.csv",
        "renewable_hydrogen_dataset_2535.csv"
    ]
    
    print("=" * 60)
    print("CSV TO JSON CONVERTER - DIRECT EXECUTION")
    print("=" * 60)
    
    # Check if files exist before processing
    existing_files = [f for f in csv_files if os.path.exists(f)]
    
    if existing_files:
        print(f"Found {len(existing_files)} CSV file(s) to process:")
        for f in existing_files:
            print(f"  - {f}")
        print()
        
        # Convert files
        convert_multiple_csvs(existing_files, output_dir="json_output", combine=True)
    else:
        print("No CSV files found in the current directory.")
        print("Usage examples:")
        print("1. Convert single file: python script.py file.csv")
        print("2. Convert multiple files: python script.py file1.csv file2.csv")
        print("3. Convert with output directory: python script.py -o output_folder file.csv")
        print("4. Combine multiple files: python script.py -c file1.csv file2.csv")
    
    print("=" * 60)