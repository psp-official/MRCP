FROM python:3.11-slim

# FFmpeg၊ HarfBuzz နှင့် Google Noto Sans Myanmar + Padauk Font သွင်းခြင်း
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fontconfig \
    fonts-noto-core \
    fonts-padauk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY fonts/ /usr/share/fonts/truetype/custom/
RUN fc-cache -f -v

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DATA_DIR=/var/data/recap_jobs
RUN mkdir -p /var/data/recap_jobs

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
