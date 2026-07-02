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

ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY app.py build_vector_db.py uw_course_scraper.py courses.json ./
COPY docker/ ./docker/

# Strip Windows CRLF so shebangs work in Linux (common on Windows dev machines)
RUN sed -i 's/\r$//' /app/docker/start.sh /app/docker/run_agent.sh \
    && chmod +x /app/docker/start.sh /app/docker/run_agent.sh

ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:99

EXPOSE 10000

CMD ["/app/docker/start.sh"]
