#!/usr/bin/env bash
set -e

export DISPLAY=:99
PORT="${PORT:-10000}"

echo "Starting virtual display for Playwright (headless=False)..."
Xvfb :99 -screen 0 1280x900x24 &
sleep 2

echo "Starting VNC server and noVNC..."
x11vnc -display :99 -forever -nopw -shared -rfbport 5900 -bg -o /tmp/x11vnc.log
websockify --web /usr/share/novnc 6080 localhost:5900 &

echo "Starting web terminal (ttyd)..."
chmod +x /app/docker/run_agent.sh
ttyd -p 7681 -W -t disableLeaveAlert=true /app/docker/run_agent.sh &

echo "Configuring nginx on port ${PORT}..."
rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true

cat > /etc/nginx/conf.d/default.conf <<EOF
server {
    listen ${PORT};
    server_name _;

    location /health {
        return 200 'ok';
        add_header Content-Type text/plain;
    }

    location /vnc/ {
        proxy_pass http://127.0.0.1:6080/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_read_timeout 86400;
    }

    location / {
        proxy_pass http://127.0.0.1:7681;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_read_timeout 86400;
    }
}
EOF

echo "UW Course Agent is starting."
echo "  Terminal: http://localhost:${PORT}/"
echo "  noVNC:    http://localhost:${PORT}/vnc/vnc.html?autoconnect=true&resize=scale"
echo "  Health:   http://localhost:${PORT}/health"

exec nginx -g 'daemon off;'
