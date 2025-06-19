#!/usr/bin/env python3
"""
Test script to verify that the parsing fix works for your Schema files
"""

import pandas as pd
import sys
import os

def test_manual_parsing():
    """
    Test the manual parsing approach for Schema 27
    """
    print("🧪 Testing manual parsing fix for Schema 27...")

    file_path = "DATA/schema_27.txt"

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    rows = []
    header = None
    skipped_lines = 0
    problem_lines = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Split by tab
            parts = line.split('\t')

            if line_num == 0:
                header = parts
                print(f"  Header: {header}")
                print(f"  Expected fields: {len(header)}")
                continue

            # Handle rows with wrong number of columns
            original_field_count = len(parts)

            if len(parts) != len(header):
                if len(parts) > len(header):
                    # Record this as a problem line we're fixing
                    if len(problem_lines) < 5:  # Only record first 5 for display
                        problem_lines.append({
                            'line_num': line_num + 1,
                            'original_fields': original_field_count,
                            'content_preview': parts[:3] + ['...'] if len(parts) > 3 else parts
                        })

                    # Too many columns - merge the extra ones into the last column
                    fixed_parts = parts[:len(header)-1] + ['\t'.join(parts[len(header)-1:])]
                    parts = fixed_parts
                elif len(parts) < len(header):
                    # Too few columns - pad with empty strings
                    parts.extend([''] * (len(header) - len(parts)))

            # Final check
            if len(parts) != len(header):
                skipped_lines += 1
                continue

            rows.append(parts)

            # Stop after processing enough to validate
            if line_num > 2000:  # Process enough to test line 1710
                break

    print(f"\n📊 Parsing Results:")
    print(f"  Total rows processed: {len(rows)}")
    print(f"  Skipped bad lines: {skipped_lines}")
    print(f"  Problem lines fixed: {len(problem_lines)}")

    if problem_lines:
        print(f"\n🔧 Examples of fixed lines:")
        for prob in problem_lines:
            print(f"  Line {prob['line_num']}: {prob['original_fields']} → {len(header)} fields")
            print(f"    Preview: {prob['content_preview']}")

    # Create DataFrame
    df = pd.DataFrame(rows, columns=header)
    print(f"\n✅ Successfully created DataFrame: {df.shape}")

    # Check that line 1710 data is included
    if len(df) >= 1710:
        print(f"\n🎯 Line 1710 check:")
        row_1710 = df.iloc[1708]  # 0-indexed, so 1710 -> 1708
        print(f"  app_id: {row_1710['app_id']}")
        print(f"  title: {row_1710['title'][:50]}...")
        print(f"  institution: {row_1710['institution']}")
        print(f"  notes preview: {str(row_1710['notes'])[:100]}...")

    return df

def test_pandas_skip_approach():
    """
    Test pandas with skip bad lines
    """
    print(f"\n🧪 Testing pandas 'skip bad lines' approach...")

    try:
        df = pd.read_csv("DATA/schema_27.txt", sep='\t', on_bad_lines='skip', low_memory=False)
        print(f"✅ Pandas skip approach successful: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Pandas skip approach failed: {e}")
        return None

def main():
    """
    Test both approaches and recommend the best one
    """
    print("🔍 TESTING PARSING FIXES FOR UK BIOBANK SCHEMA FILES")
    print("=" * 60)

    # Test both approaches
    manual_df = test_manual_parsing()
    pandas_df = test_pandas_skip_approach()

    # Compare results
    print(f"\n📊 COMPARISON:")
    print(f"  Manual parsing: {manual_df.shape if manual_df is not None else 'Failed'}")
    print(f"  Pandas skip:   {pandas_df.shape if pandas_df is not None else 'Failed'}")

    if manual_df is not None and pandas_df is not None:
        if len(manual_df) > len(pandas_df):
            print(f"  🏆 Manual parsing recovered more data ({len(manual_df)} vs {len(pandas_df)} rows)")
        else:
            print(f"  🏆 Pandas skip is simpler and sufficient ({len(pandas_df)} rows)")

    # Test Schema 19 as well
    print(f"\n🧪 Testing Schema 19 parsing...")
    try:
        schema19_df = pd.read_csv("DATA/schema_19.txt", sep='\t', low_memory=False)
        print(f"✅ Schema 19 parsed successfully: {schema19_df.shape}")
    except Exception as e:
        print(f"❌ Schema 19 parsing failed: {e}")
        schema19_df = None

    # Final recommendation
    print(f"\n💡 RECOMMENDATION:")
    if manual_df is not None or pandas_df is not None:
        print(f"  ✅ Parsing fix successful! You can now run baseline evaluation.")
        if pandas_df is not None:
            print(f"  📝 Use the 'skip bad lines' approach for simplicity")
        else:
            print(f"  📝 Use the manual parsing approach for maximum data recovery")
    else:
        print(f"  ⚠️  Both approaches failed. Manual inspection needed.")

    return manual_df, pandas_df

if __name__ == "__main__":
    manual_result, pandas_result = main()
