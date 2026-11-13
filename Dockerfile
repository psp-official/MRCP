FROM python:3.11-slim

# FFmpeg၊ Fontconfig နှင့် မြန်မာ Unicode Fonts (Noto Sans Myanmar + SIL Padauk) သွင်းခြင်း
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fontconfig \
    fonts-noto-core \
    fonts-sil-padauk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Repo ထဲတွင် fonts/ folder ပါလာပါက System Font အဖြစ် သွင်းပြီး မပါလာပါကလည်း Error မတက်အောင် ပြုလုပ်ခြင်း
RUN mkdir -p /usr/share/fonts/truetype/custom && \
    if [ -d "fonts" ]; then cp -rf fonts/* /usr/share/fonts/truetype/custom/ 2>/dev/null || true; fi && \
    fc-cache -f -v

# Render Paid Plan Persistent Disk လမ်းကြောင်း (/var/data)
ENV DATA_DIR=/var/data/recap_jobs
RUN mkdir -p /var/data/recap_jobs

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
