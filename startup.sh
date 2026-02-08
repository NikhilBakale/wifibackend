#!/bin/bash

# Azure App Service startup script for Flask application
echo "Starting Bat Monitoring Backend..."

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p static/spectrograms
mkdir -p static/audio

# Set environment variable for production
export FLASK_ENV=production

# Start Gunicorn with Flask app
# Bind to PORT from environment or default to 8000
gunicorn --bind=0.0.0.0:${PORT:-8000} --timeout 600 --workers=2 --threads=4 app:app
