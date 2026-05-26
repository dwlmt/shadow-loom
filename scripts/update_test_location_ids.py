#!/usr/bin/env python3
"""Update all test files to include location id= parameter.

P2-FIX: Add Location.id to all test fixtures for consistency with mandatory field.
"""
import re
from pathlib import Path

def update_location_ids_in_tests(file_path: Path) -> tuple[int, bool]:
    """Add id= parameter to all Location constructors in test files.
    
    Handles patterns like:
      "LOC_A": Location(name="A", description="A", ...)
    
    Returns:
        (count_updated, file_modified)
    """
    content = file_path.read_text()
    original = content
    
    # Pattern: "LOC_SOMETHING": Location(name=...
    # We want to add: id="LOC_SOMETHING",
    pattern = r'"(LOC_[A-Z_0-9]+)":\s*Location\(\s*name='
    
    def add_id(match):
        loc_id = match.group(1)
        # Add id as first parameter
        return f'"{loc_id}": Location(\n                id="{loc_id}",\n                name='
    
    updated_content = re.sub(pattern, add_id, content)
    
    if updated_content != original:
        file_path.write_text(updated_content)
        count = len(re.findall(pattern, content))
        return count, True
    return 0, False


def main():
    """Update all test files."""
    test_dir = Path(__file__).parent.parent / "tests"
    files = sorted(test_dir.glob("test_*.py"))
    
    total_updated = 0
    files_modified = 0
    
    for file_path in files:
        count, modified = update_location_ids_in_tests(file_path)
        if modified:
            print(f"✓ {file_path.name}: {count} locations updated")
            total_updated += count
            files_modified += 1
    
    if files_modified == 0:
        print("No test files needed updating")
    else:
        print(f"\nTotal: {total_updated} locations in {files_modified} test files")


if __name__ == "__main__":
    main()
