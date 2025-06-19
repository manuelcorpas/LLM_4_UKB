#!/usr/bin/env python3
"""
Even simpler debug script without f-string issues
"""

def check_schema_files():
    print("Checking Schema files...")

    # Check schema_27.txt around line 1710
    try:
        with open("DATA/schema_27.txt", 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        print("Schema 27 info:")
        print("  Total lines:", len(lines))

        if len(lines) > 0:
            header = lines[0].strip()
            print("  Header:", header)
            print("  Header fields:", len(header.split('\t')))
            print("  Header columns:", header.split('\t'))

        if len(lines) > 1710:
            problem_line = lines[1709]  # 0-indexed
            print("\nLine 1710:")
            print("  Content:", repr(problem_line[:200]))
            print("  Tab count:", problem_line.count('\t'))
            print("  Field count:", len(problem_line.split('\t')))

        # Check a few lines around it
        print("\nSample lines around 1710:")
        for i in range(max(0, 1708), min(len(lines), 1713)):
            line = lines[i]
            field_count = len(line.split('\t'))
            print("  Line", i+1, ":", field_count, "fields")

    except Exception as e:
        print("Error reading schema_27.txt:", e)

    # Quick check of schema_19.txt too
    try:
        with open("DATA/schema_19.txt", 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()

        print("\nSchema 19 info:")
        print("  Header:", first_line[:100] + "...")
        print("  Header fields:", len(first_line.split('\t')))

    except Exception as e:
        print("Error reading schema_19.txt:", e)

if __name__ == "__main__":
    check_schema_files()
