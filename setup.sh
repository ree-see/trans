#!/bin/bash
# Quick setup script for transcribe tool

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "Usage: ./transcribe \"https://www.youtube.com/watch?v=VIDEO_ID\""
echo ""
echo "Optional: Add to PATH by adding this to ~/.zshrc:"
echo "export PATH=\"\$PATH:$(pwd)\""
