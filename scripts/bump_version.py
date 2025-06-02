#!/usr/bin/env python3
"""
Version bumping utility for semiq-ml package.
Usage: python scripts/bump_version.py [major|minor|patch]
"""

import re
import sys
import os
from pathlib import Path

def update_version_in_file(file_path, current_version, new_version):
    """Update version string in a file."""
    with open(file_path, 'r') as file:
        content = file.read()
    
    updated_content = content.replace(f'"{current_version}"', f'"{new_version}"')
    updated_content = updated_content.replace(f"'{current_version}'", f"'{new_version}'")
    
    with open(file_path, 'w') as file:
        file.write(updated_content)
    
    print(f"Updated version in {file_path} from {current_version} to {new_version}")

def bump_version(version_type):
    """Bump version based on semver (major.minor.patch)."""
    # Read current version from __init__.py
    init_file = Path("semiq_ml/__init__.py")
    with open(init_file, 'r') as f:
        content = f.read()
    
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not version_match:
        print("Version not found in __init__.py")
        sys.exit(1)
    
    current_version = version_match.group(1)
    major, minor, patch = map(int, current_version.split('.'))
    
    # Update version number based on type
    if version_type == "major":
        new_version = f"{major + 1}.0.0"
    elif version_type == "minor":
        new_version = f"{major}.{minor + 1}.0"
    elif version_type == "patch":
        new_version = f"{major}.{minor}.{patch + 1}"
    else:
        print("Invalid version type. Use 'major', 'minor', or 'patch'")
        sys.exit(1)
    
    # Update files
    update_version_in_file(init_file, current_version, new_version)
    update_version_in_file("pyproject.toml", current_version, new_version)
    
    print(f"\nVersion bumped from {current_version} to {new_version}")
    print("To release:")
    print(f"1. Commit changes: git commit -am 'Bump version to {new_version}'")
    print(f"2. Tag release: git tag v{new_version}")
    print(f"3. Push changes: git push && git push --tags")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["major", "minor", "patch"]:
        print("Usage: python scripts/bump_version.py [major|minor|patch]")
        sys.exit(1)
    
    bump_version(sys.argv[1])
