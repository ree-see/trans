#!/bin/bash
# Quick setup script for trans CLI

set -e

echo "Installing Python dependencies..."
uv pip install -e ".[all]"

echo ""
echo "Setup complete!"
echo ""
echo "Usage: trans \"https://www.youtube.com/watch?v=VIDEO_ID\""
echo ""
echo "Or run directly: python trans_cli.py <url>"
