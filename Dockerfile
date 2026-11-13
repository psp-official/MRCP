FROM python:3.12-slim

# FFmpeg + fontconfig (မြန်မာ font ကို ကိုယ်တိုင် copy လုပ်မည်)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fontconfig \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Custom font ကို စနစ်ထဲ register လုပ်ခြင်း
COPY fonts/ /usr/share/fonts/truetype/a13wenyeo/
RUN fc-cache -fv && fc-list | grep -i wenyeo || true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
