FROM mcr.microsoft.com/playwright/python:v1.49.0-noble

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    ttyd \
    nginx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py build_vector_db.py uw_course_scraper.py courses.json ./
COPY docker/ ./docker/

RUN chmod +x /app/docker/start.sh /app/docker/run_agent.sh

ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:99

EXPOSE 10000

CMD ["/app/docker/start.sh"]
