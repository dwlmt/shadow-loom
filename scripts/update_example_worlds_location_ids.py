#!/usr/bin/env python3
"""Update all example worlds to include location id= parameter.

P2-FIX: Add Location.id to all example world fixtures for consistency with architecture docs.
"""
import re
from pathlib import Path

def update_location_ids(file_path: Path) -> tuple[int, bool]:
    """Add id= parameter to all Location constructors.
    
    Returns:
        (count_updated, file_modified)
    """
    content = file_path.read_text()
    original = content
    
    # Pattern: "LOC_SOMETHING": Location(
    # We want to add: id="LOC_SOMETHING",
    pattern = r'"(LOC_[A-Z_0-9]+)":\s*Location\('
    
    def add_id(match):
        loc_id = match.group(1)
        # Add id as first parameter
        return f'"{loc_id}": Location(\n            id="{loc_id}",'
    
    updated_content = re.sub(pattern, add_id, content)
    
    if updated_content != original:
        file_path.write_text(updated_content)
        count = len(re.findall(pattern, content))
        return count, True
    return 0, False


def main():
    """Update all example worlds."""
    example_dir = Path(__file__).parent.parent / "example_worlds"
    files = sorted(example_dir.glob("*.py"))
    files = [f for f in files if f.name != "__init__.py"]
    
    total_updated = 0
    files_modified = 0
    
    for file_path in files:
        count, modified = update_location_ids(file_path)
        if modified:
            print(f"✓ {file_path.name}: {count} locations updated")
            total_updated += count
            files_modified += 1
        else:
            print(f"  {file_path.name}: no locations found or already updated")
    
    print(f"\nTotal: {total_updated} locations in {files_modified} files")


if __name__ == "__main__":
    main()
