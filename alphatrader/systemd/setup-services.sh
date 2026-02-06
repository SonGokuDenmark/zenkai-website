#!/bin/bash
# Setup script for AlphaTrader data collector services
# Run as root: sudo bash setup-services.sh

set -e

echo "Setting up AlphaTrader data collector services..."

# Create log directory
mkdir -p /var/log/alphatrader
chown goku:alphatrader /var/log/alphatrader
chmod 775 /var/log/alphatrader

# Copy service files
cp binance-collector.service /etc/systemd/system/
cp stock-collector.service /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable services (start on boot)
systemctl enable binance-collector.service
systemctl enable stock-collector.service

echo ""
echo "Services installed. To start:"
echo "  sudo systemctl start binance-collector"
echo "  sudo systemctl start stock-collector"
echo ""
echo "To check status:"
echo "  sudo systemctl status binance-collector"
echo "  sudo systemctl status stock-collector"
echo ""
echo "To view logs:"
echo "  tail -f /var/log/alphatrader/binance-collector.log"
echo "  tail -f /var/log/alphatrader/stock-collector.log"
echo ""
echo "Done!"
