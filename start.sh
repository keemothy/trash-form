#!/bin/bash

# RecycleBnB Start Script
echo "Starting RecycleBnB..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed. Please install Python 3.7+ first."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "ERROR: pip is not installed. Please install pip first."
    exit 1
fi

# Use pip3 if available, otherwise pip
PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

echo "Installing Python dependencies..."
$PIP_CMD install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies. Please check your Python environment."
    exit 1
fi

echo "Dependencies installed successfully."
echo ""

# Check for Claude API key
if [ -z "$CLAUDE_API_KEY" ] && [ ! -f ".env" ]; then
    echo "Claude API key not found."
    read -p "Enter your Claude API key (or press Enter to skip): " api_key
    if [ ! -z "$api_key" ]; then
        echo "CLAUDE_API_KEY=$api_key" > .env
        export CLAUDE_API_KEY=$api_key
        echo "API key saved to .env file"
    else
        echo "WARNING: Running without Claude API key. AI features may not work."
    fi
fi

echo ""
echo "Starting RecycleBnB server..."
echo "Open http://localhost:9000 in your browser"
echo ""

# Start the server
cd web && $PYTHON_CMD server.py
