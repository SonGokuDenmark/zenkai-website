#!/bin/bash
# Sync local alphatrader scripts to shared server location
# Run this after making changes to keep Fips in sync

SERVER="goku@192.168.0.160"
DEST="/home/shared/alphatrader/scripts"

echo "Syncing alphatrader scripts to server..."

# Main scripts
scp train_lstm.py analyze_lstm.py validate_lstm.py $SERVER:$DEST/

# Source modules
scp -r src $SERVER:$DEST/

echo "Done! Scripts synced to $DEST"
echo ""
echo "Fips can run from server with:"
echo "  cd /home/shared/alphatrader/scripts"
echo "  python3 train_lstm.py --help"
