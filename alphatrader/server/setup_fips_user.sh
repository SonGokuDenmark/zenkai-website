#!/bin/bash
# Setup read-only user 'fips' on zenkaiserver
# Run this as: sudo bash setup_fips_user.sh

set -e

echo "=== Creating read-only user 'fips' ==="

# 1. Create Linux user with restricted shell
echo ""
echo "Step 1: Creating Linux user..."
useradd -m -s /bin/rbash fips
echo "fips:fips_ssh_2024" | chpasswd
echo "  Linux user 'fips' created with password: fips_ssh_2024"

# 2. Setup restricted environment
echo ""
echo "Step 2: Setting up restricted shell environment..."
mkdir -p /home/fips/bin

# Symlink only allowed commands
ln -sf /usr/bin/psql /home/fips/bin/psql
ln -sf /bin/ls /home/fips/bin/ls
ln -sf /bin/cat /home/fips/bin/cat
ln -sf /bin/grep /home/fips/bin/grep
ln -sf /usr/bin/less /home/fips/bin/less 2>/dev/null || ln -sf /bin/less /home/fips/bin/less
ln -sf /usr/bin/head /home/fips/bin/head
ln -sf /usr/bin/tail /home/fips/bin/tail

# Lock down .bashrc
cat > /home/fips/.bashrc << 'EOF'
export PATH=/home/fips/bin
export PS1="fips@goku:\w$ "
echo "Welcome fips - read-only access"
echo "Available commands: ls, cat, grep, head, tail, less, psql"
EOF

chown root:root /home/fips/.bashrc
chmod 644 /home/fips/.bashrc
chattr +i /home/fips/.bashrc 2>/dev/null || echo "  (chattr not available, bashrc still protected by ownership)"

echo "  Restricted shell configured"

# 3. Create PostgreSQL user
echo ""
echo "Step 3: Creating PostgreSQL read-only user..."
sudo -u postgres psql << 'EOF'
-- Create user (ignore if exists)
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'fips') THEN
    CREATE USER fips WITH PASSWORD 'fips_readonly_2024';
  END IF;
END $$;

-- Grant connect
GRANT CONNECT ON DATABASE zenkai_data TO fips;
EOF

sudo -u postgres psql -d zenkai_data << 'EOF'
-- Grant usage on schema
GRANT USAGE ON SCHEMA public TO fips;

-- Grant SELECT on all existing tables
GRANT SELECT ON ALL TABLES IN SCHEMA public TO fips;

-- Grant SELECT on future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO fips;

-- Grant SELECT on sequences
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO fips;
EOF

echo "  PostgreSQL user 'fips' created with SELECT only"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Credentials for fips:"
echo "  SSH:        fips@goku (192.168.0.160) / fips_ssh_2024"
echo "  PostgreSQL: fips / fips_readonly_2024"
echo ""
echo "Still need to create Grafana user manually via web UI:"
echo "  1. Go to http://192.168.0.160:3000"
echo "  2. Login as admin"
echo "  3. Administration > Users > New user"
echo "  4. Username: fips, Role: Viewer"
echo ""
echo "Test with:"
echo "  ssh fips@goku"
echo "  PGPASSWORD=fips_readonly_2024 psql -h localhost -U fips -d zenkai_data -c 'SELECT COUNT(*) FROM ohlcv;'"
