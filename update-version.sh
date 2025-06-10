#!/bin/bash
# update-version.sh - Automated version update script for semiq-ml
# Updates version in files, creates git tag, and triggers GitHub workflows for PyPI publishing

set -e  # Exit immediately if any command fails

# Check if a version argument was provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <new_version>"
  echo "Example: $0 0.3.0"
  exit 1
fi

NEW_VERSION=$1

# Validate version format (basic semver format)
if ! [[ $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
  echo "Error: Version must follow semantic versioning (e.g., 1.2.3 or 1.2.3-alpha)"
  exit 1
fi

echo "Updating to version: $NEW_VERSION"

# Update version in pyproject.toml
echo "Updating pyproject.toml..."
sed -i "s/^version = \"[^\"]*\"/version = \"$NEW_VERSION\"/" pyproject.toml

# Update version in setup.py
echo "Updating setup.py..."
sed -i "s/version=\"[^\"]*\"/version=\"$NEW_VERSION\"/" setup.py

# Update version in __init__.py
echo "Updating semiq_ml/__init__.py..."
sed -i "s/__version__ = \"[^\"]*\"/__version__ = \"$NEW_VERSION\"/" semiq_ml/__init__.py

# Commit the changes
echo "Committing changes..."
git add pyproject.toml setup.py semiq_ml/__init__.py
git commit -m "Bump version to $NEW_VERSION"

# Create and push tag
echo "Creating git tag v$NEW_VERSION..."
git tag -a "v$NEW_VERSION" -m "Version $NEW_VERSION"

# Get current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Push changes and tag to remote
echo "Pushing changes and tag to GitHub..."
git push origin "$CURRENT_BRANCH"
git push origin "v$NEW_VERSION"

echo "Version $NEW_VERSION has been updated, committed, and pushed to GitHub."
echo "GitHub workflows should now handle PyPI deployment automatically."