#!/bin/bash

# RecycleBnB Start Script
echo "🌍 Starting RecycleBnB..."

# Check for Claude API key
if [ -z "$CLAUDE_API_KEY" ] && [ ! -f ".env" ]; then
    read -p "Enter your Claude API key (or press Enter to skip): " api_key
    if [ ! -z "$api_key" ]; then
        echo "CLAUDE_API_KEY=$api_key" > .env
        export CLAUDE_API_KEY=$api_key
    fi
fi

# Start the server
cd web && python server.py
