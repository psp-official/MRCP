FROM python:3.11-slim

# FFmpeg၊ Fontconfig နှင့် မြန်မာ Unicode Font သွင်းခြင်း
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fontconfig \
    fonts-padauk \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Custom မြန်မာဖောင့်များ (ဥပမာ A13 WenYoe) ကို System Font အဖြစ် သွင်းခြင်း
COPY fonts/ /usr/share/fonts/truetype/custom/
RUN fc-cache -f -v

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render Paid Plan Persistent Disk လမ်းကြောင်း (/var/data) သို့မဟုတ် local temp
ENV DATA_DIR=/var/data/recap_jobs
RUN mkdir -p /var/data/recap_jobs

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
