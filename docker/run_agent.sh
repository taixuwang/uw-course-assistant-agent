#!/usr/bin/env bash
set -e
cd /app

if [ ! -f "courses.json" ]; then
    echo "ERROR: courses.json not found in /app."
    echo "Add courses.json to the project before building or deploying."
    exit 1
fi

if [ ! -d "./uw_chroma_db" ] || [ -z "$(ls -A ./uw_chroma_db 2>/dev/null)" ]; then
    echo "=========================================="
    echo "First run: building vector database..."
    echo "This may take 5-15 minutes. Please wait."
    echo "=========================================="
    python build_vector_db.py
fi

exec python -u app.py
