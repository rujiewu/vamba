#!/bin/bash

# Get current script directory (i.e., `rsync.sh ../` directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get parent directory of `SCRIPT_DIR`
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Remote server details
REMOTE_USER="wurujie"
REMOTE_HOST="lambda"
REMOTE_PATH="/home/wurujie/workspace/code"
REMOTE_VAMBA_SCRIPT_DIR="$REMOTE_PATH/$(basename "$SCRIPT_DIR")"

# Check argument to determine upload or download
if [ "$1" == "upload" ]; then
    echo "Uploading local to remote server..."
    rsync -avzP --progress --exclude='.*' "$SCRIPT_DIR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    echo "Upload completed!"

elif [ "$1" == "download" ]; then
    echo "Downloading from remote server to local..."
    rsync -avzP --progress --exclude='.*' "$REMOTE_USER@$REMOTE_HOST:$REMOTE_VAMBA_SCRIPT_DIR" "$PARENT_DIR"
    echo "Download completed!"

else
    echo "bash sync.sh upload   # Upload local to remote server"
    echo "bash sync.sh download # Download from remote server to local"
    exit 1
fi
