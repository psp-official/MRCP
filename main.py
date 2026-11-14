import os
import re
import json
import uuid
import time
import shutil
import asyncio
import unicodedata
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx
import edge_tts
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="AI Movie Recap Narrator (Smooth Blur & Fast Server Edition)")

DATA_DIR = Path(os.getenv("DATA_DIR", "./recap_jobs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

GEMINI_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-2.5-flash-lite",
]

VOICE_MAP = {
    "Thiha": "my-MM-ThihaNeural",
    "Nilar": "my-MM-NilarNeural",
    "Guy": "en-US-GuyNeural",
    "Jenny": "en-US-JennyNeural",
    "Niwat": "th-TH-NiwatNeural",
    "Premwadee": "th-TH-PremwadeeNeural",
}

JOBS: Dict[str, Dict[str, Any]] = {}


# -------------------------------------------------------------------------
# ATOMIC DISK JOB STATE & LIVE LOGS
# -------------------------------------------------------------------------
def save_job_state(job_id: str):
    if job_id in JOBS:
        job_dir = DATA_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = job_dir / "status.json.tmp"
        status_file = job_dir / "status.json"
        tmp_file.write_text(json.dumps(JOBS[job_id], ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_file, status_file)


def load_job_state(job_id: str) -> Optional[Dict[str, Any]]:
    status_file = DATA_DIR / job_id / "status.json"
    if status_file.exists():
        try:
            data = json.loads(status_file.read_text(encoding="utf-8"))
            JOBS[job_id] = data
            return data
        except Exception:
            pass
    return JOBS.get(job_id)


def append_job_log(job_id: str, msg: str):
    if job_id not in JOBS:
        return
    elapsed = max(0, int(time.time() - JOBS[job_id]["started_at"]))
    m = elapsed // 60
    s = elapsed % 60
    line = f"{m}:{s:02d} {msg}"
    logs = JOBS[job_id].setdefault("logs", [])
    if not logs or logs[-1] != line:
        logs.append(line)
    save_job_state(job_id)


# -------------------------------------------------------------------------
# 1. MYANMAR UNICODE CLEANER & PHRASE / SRT SPLITTERS
# -------------------------------------------------------------------------
def clean_burmese_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(
        r"^(ရင်ခုန်စရာ|သဘာဝကျသော|မြန်မာသဘာဝဖြင့်|ပြန်လည်ရေးသားပါ|ဘာသာပြန်|Narration|Translation)[^:：]*[:：]\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    for ch in ["\u200b", "\u200c", "\u200d", "\ufeff", "*", "#", "_", "`", "~", '"', "'"]:
        text = text.replace(ch, "")

    text = text.replace("\u1036\u102f", "\u102f\u1036")
    text = text.replace("\u1036\u1030", "\u1030\u1036")
    text = text.replace("\u1037\u103a", "\u103a\u1037")
    return re.sub(r"\s+", " ", text).strip()


def split_into_short_phrases(
    text: str, words_per_caption: int = 9, caption_width_pct: int = 96, text_size: int = 32
) -> List[str]:
    text = clean_burmese_text(text)
    if not text:
        return [""]

    width_factor = max(0.5, caption_width_pct / 96.0)
    size_factor = max(0.6, 32.0 / max(16, text_size))
    base_chars = int(words_per_caption) * 2 + 4
    max_chars = max(10, min(28, int(base_chars * width_factor * size_factor)))

    clauses = re.split(r"(?<=[၊။])\s*", text)
    phrases: List[str] = []

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) <= max_chars:
            phrases.append(clause)
            continue

        words = clause.split(" ")
        cur = ""
        for w in words:
            if not w:
                continue
            cand = f"{cur} {w}".strip() if cur else w
            if len(cand) <= max_chars:
                cur = cand
            else:
                if cur:
                    phrases.append(cur)
                while len(w) > max_chars:
                    cut = max_chars
                    while cut < len(w) and ("\u102b" <= w[cut] <= "\u103f"):
                        cut += 1
                    phrases.append(w[:cut])
                    w = w[cut:]
                cur = w
        if cur:
            phrases.append(cur)

    return phrases if phrases else [text]


def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_srt_subtitle_lines(text: str, max_line_chars: int = 24) -> str:
    if not text:
        return ""
    raw_lines = [clean_burmese_text(line).rstrip("။၊") for line in text.split("\n") if line.strip()]
    joined = "\n".join(raw_lines)

    if "\n" not in joined and len(joined) > max_line_chars and " " in joined:
        words = joined.split(" ")
        mid = len(joined) // 2
        cur_len = 0
        line1, line2 = [], []
        for w in words:
            if cur_len < mid or not line1:
                line1.append(w)
                cur_len += len(w) + 1
            else:
                line2.append(w)
        if line1 and line2:
            return f"{' '.join(line1)}\n{' '.join(line2)}"

    return joined


def build_srt_file(segments: List[Dict[str, Any]], srt_path: Path):
    blocks = []
    for idx, seg in enumerate(segments, start=1):
        st = format_srt_time(seg["start"])
        en = format_srt_time(seg["end"])
        raw_txt = seg.get("narration") or seg.get("text", "")
        txt = format_srt_subtitle_lines(raw_txt)
        if txt:
            blocks.append(f"{idx}\n{st} --> {en}\n{txt}\n")
    srt_path.write_text("\n".join(blocks), encoding="utf-8")


# -------------------------------------------------------------------------
# 2. FFMPEG HELPERS & SMOOTH GAUSSIAN BLUR BUILDER (NO DELOGO STREAKS!)
# -------------------------------------------------------------------------
def run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg Error: {proc.stderr[-800:]}")
    return proc.stdout


def get_duration(file_path: Path) -> float:
    out = run_cmd([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
    ])
    return float(out.strip() or 0.0)


def get_video_dimensions(file_path: Path) -> tuple[int, int]:
    out = run_cmd([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", str(file_path)
    ])
    parts = out.strip().split("x")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return 1280, 720


def cleanup_old_jobs(max_age_seconds: int = 3600):
    now = time.time()
    for folder in DATA_DIR.iterdir():
        if folder.is_dir():
            try:
                if now - folder.stat().st_mtime > max_age_seconds:
                    shutil.rmtree(folder, ignore_errors=True)
            except Exception:
                pass


def hex_to_ass_color(hex_color: str, alpha: int = 0) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        hex_color = "FFFF00"
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H{alpha:02X}{b.upper()}{g.upper()}{r.upper()}"


def format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_smooth_blur_filter(cfg: Dict[str, Any], out_w: int, out_h: int) -> Optional[str]:
    """
    delogo ကြောင့် ဒေါင်လိုက်အစင်းကြောင်းများ လှုပ်ခါနေခြင်းကို အပြီးသတ်ဖယ်ရှားပြီး
    တည်ငြိမ်ချောမွေ့သော 2D Gaussian Blur (gblur) ဖြင့် စာတန်းထိုးနေရာကို ဖုံးအုပ်ပေးခြင်း
    """
    want_band = cfg.get("full_band", True)
    blur_boxes = int(cfg.get("blur_boxes", 0))
    if not want_band and blur_boxes <= 0:
        return None

    blur_val = max(2, min(30, int(cfg.get("band_blur", 14))))
    sigma = round( blur_val * 1.6 + 4.0, 1 )

    if want_band:
        b_height_pct = max(5, min(45, int(cfg.get("band_height", 18))))
        b_bottom_pct = max(0, min(45, int(cfg.get("band_from_bottom", 0))))
        band_h = max(16, int(out_h * (b_height_pct / 100.0)))
        band_h -= band_h % 2
        band_y = max(0, out_h - band_h - int(out_h * (b_bottom_pct / 100.0)))
        band_y -= band_y % 2
        if band_y + band_h > out_h:
            band_h = out_h - band_y
        band_x = 0
        band_w = out_w
    else:
        # Custom blur box default region
        band_w = int(out_w * 0.76)
        band_w -= band_w % 2
        band_h = max(20, int(out_h * 0.14))
        band_h -= band_h % 2
        band_x = (out_w - band_w) // 2
        band_y = int(out_h * 0.80)

    # Smooth 2D Gaussian blur (gblur) on the cropped subtitle strip only (Zero flickering, <2MB RAM)
    return (
        f"split[vmain][vsub];"
        f"[vsub]crop={band_w}:{band_h}:{band_x}:{band_y},gblur=sigma={sigma}:steps=2[vblur];"
        f"[vmain][vblur]overlay={band_x}:{band_y}"
    )


def update_job_step(job_id: str, step_name: str, progress: int, log_msg: Optional[str] = None):
    step_order = [
        "extract_audio",
        "upload",
        "transcribe",
        "translate",
        "voice",
        "timeline_captions",
        "render_video",
    ]
    job = JOBS[job_id]
    job["current_step"] = step_name
    job["progress"] = progress
    passed = True
    for s in step_order:
        if s == step_name:
            job["steps"][s] = "running"
            passed = False
        elif passed:
            job["steps"][s] = "done"
        else:
            job["steps"][s] = "pending"
    if log_msg:
        append_job_log(job_id, log_msg)
    else:
        save_job_state(job_id)


# -------------------------------------------------------------------------
# 3. STEP 2 & 3: GROQ WHISPER TRANSCRIPTION
# -------------------------------------------------------------------------
async def groq_whisper_transcribe(
    audio_path: Path,
    groq_key: str,
    source_lang: str,
    job_id: str,
    total_video_dur: float,
    mode: str = "AI Recap",
) -> List[Dict[str, Any]]:
    update_job_step(job_id, "upload", 14, "အသံဖိုင်ကို ဆာဗာဆီ ပို့နေပါတယ်... 0%")
    headers = {"Authorization": f"Bearer {groq_key.strip()}"}

    data_fields: Dict[str, Any] = {
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
        "temperature": "0.0",
    }
    if source_lang and source_lang != "Auto-detect":
        data_fields["language"] = source_lang

    update_job_step(job_id, "transcribe", 22, "စာသားထုတ်နေပါတယ်")
    async with httpx.AsyncClient(timeout=180.0) as client:
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f.read(), "audio/mpeg")}
        res = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers=headers,
            data=data_fields,
            files=files,
        )
        if res.status_code != 200:
            raise RuntimeError(f"Groq Whisper Error ({res.status_code}): {res.text}")
        res_json = res.json()

    raw_segments = res_json.get("segments") or []
    cleaned_segments: List[Dict[str, Any]] = []

    hallucination_words = {"字幕", "谢谢观看", "Subtitles", "Thank you for watching", "Amara.org"}
    for seg in raw_segments:
        txt = (seg.get("text") or "").strip()
        if not txt or any(hw in txt for hw in hallucination_words):
            continue
        if float(seg.get("no_speech_prob", 0.0)) > 0.88:
            continue
        st = max(0.0, float(seg.get("start", 0.0)))
        en = min(total_video_dur, max(st + 0.4, float(seg.get("end", st + 1.5))))
        cleaned_segments.append({"start": st, "end": en, "text": txt})

    # MM SRT SUBTITLE သို့မဟုတ် Subtitle (Original + subs) Mode တွင် မူရင်းအသံအချိန်အတိုင်း အတိအကျထားသည်
    if mode in ("MM SRT SUBTITLE", "Subtitle"):
        final_segs = cleaned_segments
    else:
        final_segs = []
        for seg in cleaned_segments:
            if not final_segs:
                final_segs.append(seg)
                continue
            prev = final_segs[-1]
            gap = seg["start"] - prev["end"]
            combined_dur = seg["end"] - prev["start"]
            if gap < 0.35 and (prev["end"] - prev["start"] < 2.2 or combined_dur <= 5.5):
                prev["end"] = seg["end"]
                prev["text"] = f"{prev['text']} {seg['text']}".strip()
            else:
                final_segs.append(seg)

    if not final_segs:
        final_segs = [{"start": 0.0, "end": min(5.0, total_video_dur), "text": res_json.get("text", "...")}]

    for i, m in enumerate(final_segs):
        m["index"] = i
        m["start"] = round(m["start"], 2)
        m["end"] = round(m["end"], 2)

    return final_segs


# -------------------------------------------------------------------------
# 4. STEP 4: GEMINI 3.8 NATIVE JSON TRANSLATION
# -------------------------------------------------------------------------
async def gemini_generate_recap(
    segments: List[Dict[str, Any]],
    gemini_key: str,
    gemini_model: str,
    mode: str,
    output_lang: str,
    want_hashtags: bool,
    default_hashtags: str,
    add_fyp: bool,
) -> Dict[str, Any]:
    lang_map = {
        "မြန်မာ": "natural, engaging SPOKEN BURMESE (မြန်မာစကားပြောဟန်)",
        "English": "natural, engaging spoken English",
        "ไทย": "natural, engaging spoken Thai (ภาษาไทย)",
    }
    target_lang_desc = lang_map.get(output_lang, lang_map["မြန်မာ"])

    if mode in ("MM SRT SUBTITLE", "Subtitle"):
        system_instruction = f"""You are a professional Burmese Donghua (Xianxia/Cultivation) & Drama Fansub Subtitle Translator.
Translate each input dialogue line into 100% natural, emotional {target_lang_desc} subtitles matching the exact character tone.

STRICT FANSUB RULES:
1. CHARACTER DIALOGUE & PRONOUNS: Use natural context-appropriate pronouns ("ငါ", "မင်း", "ရှင်", "ရှင့်ရဲ့", "ညီမတော်", "သခင်လေး") and natural spoken endings ("ပါလား", "တော့", "တာလဲ", "ပြီပဲ!", "ပါဦး", "လိုက်တာ", "ရစေ").
2. XIANXIA / CULTIVATION GLOSSARY:
   - 法则之力 / 法则 -> နိယာမစွမ်းအင် / နိယာမတွေ
   - 准帝 -> အင်ပါယာလောင်းအဆင့်, 大帝 -> မဟာအင်ပါယာအဆင့်
   - 修为 / 境界 -> ကျင့်ကြံမှုအဆင့်, 威压 / 气息 -> ဖိအား / အရှိန်အဝါ
   - 融合 -> ပေါင်းစည်းတယ်, 突破 -> အဆင့်တက်သွားပြီ
   - 圣女 -> သန့်ရှင်းသူမလေး, 圣子 -> သန့်ရှင်းသူသား, 宗主 / 掌门 -> ဂိုဏ်းချုပ်, 灵石 -> ဝိညာဉ်ကျောက်တုံး
3. SUBTITLE FORMATTING:
   - For "MM SRT SUBTITLE": Do NOT put "။" at the end of lines, and split lines longer than 22 chars with "\\n".
   - For "Subtitle": Put spaces between phrases so captions display cleanly on screen.
4. Keep the exact same `index` for every segment."""
    else:
        pro_instruction = ""
        if mode == "Pro AI Recap":
            pro_instruction = "\n- Since Pro AI Recap mode is active, make the storytelling extra dramatic, smooth, and emotionally engaging."

        system_instruction = f"""You are an expert translator and voiceover scriptwriter for Chinese Donghua (Xianxia/Cultivation) and short dramas.
Translate each input segment directly and accurately into {target_lang_desc} so that it matches the exact meaning, emotion, and timing of the scene.{pro_instruction}

STRICT BURMESE RULES:
1. DIRECT SPOKEN STYLE: Translate each segment naturally as spoken Burmese dialogue/narration (e.g., "ရှောင်ဟွာ မင်းဘာလုပ်နေတာလဲ။", "သန့်ရှင်းသူမလေး ရောက်လာပြီ။").
2. NATURAL SPOKEN ENDINGS: Always use spoken particles ("တယ်", "လဲ", "ပြီ", "တာပေါ့", "ပါတယ်", "လား", "နော်"). NEVER use stiff literary words ("သည်", "၏", "၌", "၎င်း").
3. XIANXIA / CULTIVATION TERMINOLOGY:
   - 圣女 -> သန့်ရှင်းသူမလေး, 圣子 -> သန့်ရှင်းသူသား, 掌门 / 宗主 -> ဂိုဏ်းချုပ်
   - 师姐 -> အစ်မတော် / စီနီယာအစ်မ, 师兄 -> အစ်ကိုတော် / စီနီယာအစ်ကို
   - 少爷 / 公子 -> သခင်လေး, 宿主 -> သခင်, 统子 / 系统 -> တုံစီ / စနစ်
   - 灵石 -> ဝိညာဉ်ကျောက်တုံး, 炼气 -> လျန်ချီအဆင့်, 筑基 -> အခြေတည်အဆင့်, 元丹 / 金丹 -> ယွမ်တန်အဆင့်
4. PHRASE SPACING: Put spaces between Burmese phrases and use "၊" and "။" so subtitles can break into clean single-line captions.
5. Keep the exact same `index` for every segment."""

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "segments": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "index": {"type": "INTEGER"},
                        "narration": {"type": "STRING"},
                    },
                    "required": ["index", "narration"],
                },
            },
            "tiktok_caption": {"type": "STRING"},
        },
        "required": ["segments", "tiktok_caption"],
    }

    preferred = gemini_model or DEFAULT_GEMINI_MODEL
    if preferred in ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"):
        preferred = "gemini-3.8-flash"
    models_to_try = [preferred] + [m for m in GEMINI_FALLBACK_MODELS if m != preferred]

    narr_map: Dict[int, str] = {}
    tiktok_cap = "ဇာတ်လမ်းကောင်းလေးကို အဆုံးထိ အားပေးကြပါဦးနော် 🎬"
    batch_size = 35

    async with httpx.AsyncClient(timeout=240.0) as client:
        for b_start in range(0, len(segments), batch_size):
            batch = segments[b_start : b_start + batch_size]
            prompt_payload = [
                {
                    "index": x["index"],
                    "duration_sec": round(x["end"] - x["start"], 1),
                    "text": x["text"],
                }
                for x in batch
            ]

            body = {
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": json.dumps({"segments": prompt_payload}, ensure_ascii=False)}],
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": response_schema,
                },
            }

            parsed = None
            last_err = ""
            for m_name in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={gemini_key.strip()}"
                res = await client.post(url, json=body)
                if res.status_code == 200:
                    data = res.json()
                    try:
                        raw_txt = data["candidates"][0]["content"]["parts"][0]["text"]
                        parsed = json.loads(raw_txt, strict=False)
                        break
                    except Exception as parse_err:
                        last_err = str(parse_err)
                else:
                    last_err = f"[{m_name}] {res.status_code}: {res.text}"
                await asyncio.sleep(1.5)

            if not parsed:
                raise RuntimeError(f"Gemini API Error: {last_err}")

            for item in parsed.get("segments", []):
                raw_narr = item.get("narration", "")
                if mode == "MM SRT SUBTITLE":
                    narr_map[int(item["index"])] = format_srt_subtitle_lines(raw_narr)
                else:
                    narr_map[int(item["index"])] = clean_burmese_text(raw_narr)
            if parsed.get("tiktok_caption"):
                tiktok_cap = clean_burmese_text(parsed["tiktok_caption"])

    for seg in segments:
        seg["narration"] = narr_map.get(seg["index"]) or clean_burmese_text(seg["text"])

    caption_out = ""
    if want_hashtags:
        tags = default_hashtags.strip()
        if add_fyp and "#fyp" not in tags.lower():
            tags = f"{tags} #fyp".strip()
        caption_out = f"{tiktok_cap}\n{tags}".strip()

    return {"segments": segments, "tiktok_caption": caption_out}


# -------------------------------------------------------------------------
# 5. STEP 6: ASS SUBTITLE BUILDER
# -------------------------------------------------------------------------
def build_ass_subtitles(
    caption_events: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    play_res_x: int,
    play_res_y: int,
    ass_path: Path,
):
    req_font = cfg.get("font", "Padauk")
    if Path("/usr/share/fonts/truetype/custom/A13WenYoe.ttf").exists() and req_font == "A13 WenYoe":
        font_name = "A13 WenYoe"
    elif req_font == "Noto Sans Myanmar":
        font_name = "Noto Sans Myanmar"
    else:
        font_name = "Padauk"

    raw_font_size = int(cfg.get("text_size", 32))
    scale_ratio = play_res_y / 380.0
    font_size = max(16, int(round(raw_font_size * scale_ratio)))

    primary_col = hex_to_ass_color(cfg.get("text_color", "#FFFF00"), 0)
    outline_col = hex_to_ass_color(cfg.get("outline_color", "#000000"), 0)
    back_col = hex_to_ass_color("#000000", 140)

    raw_outline = float(cfg.get("outline_width", 4))
    outline_w = round(max(0.0, raw_outline * (play_res_y / 480.0)), 1)

    bold_flag = -1 if cfg.get("bold", True) else 0
    border_style = 3 if cfg.get("box_bg", False) else 1

    vert_pct = max(8, min(96, int(cfg.get("vertical_position", 89))))
    margin_v = max(12, int(play_res_y * (100 - vert_pct) / 100.0))

    width_pct = max(40, min(100, int(cfg.get("caption_width", 96))))
    side_margin = max(12, int(play_res_x * (100 - width_pct) / 200.0))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_col},{primary_col},{outline_col},{back_col},{bold_flag},0,0,0,100,100,0,0,{border_style},{outline_w},1,2,{side_margin},{side_margin},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for ev in caption_events:
        st = format_ass_time(ev["start"])
        en = format_ass_time(ev["end"])
        txt = clean_burmese_text(ev["text"])
        if txt:
            lines.append(f"Dialogue: 0,{st},{en},Default,,0,0,0,,{txt}")

    ass_path.write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------------------
# 6. MAIN 7-STEP PIPELINE WORKER (FAST SUBTITLE MODE + SMOOTH GBLUR)
# -------------------------------------------------------------------------
async def process_recap_pipeline(
    job_id: str, video_path: Path, cfg: Dict[str, Any], logo_path: Optional[Path]
):
    job_dir = DATA_DIR / job_id
    try:
        cleanup_old_jobs()

        groq_key = cfg.get("groq_key") or DEFAULT_GROQ_KEY
        gemini_key = cfg.get("gemini_key") or DEFAULT_GEMINI_KEY
        gemini_model = cfg.get("gemini_model") or DEFAULT_GEMINI_MODEL
        mode = cfg.get("mode", "AI Recap")

        if not groq_key or not gemini_key:
            raise RuntimeError("Groq API Key နှင့် Gemini API Key များကို 'Own key' တွင် ထည့်သွင်းပေးပါ။")

        orig_w, orig_h = await asyncio.to_thread(get_video_dimensions, video_path)
        total_video_dur = await asyncio.to_thread(get_duration, video_path)

        hd_1080 = cfg.get("tiktok_hd", False)
        ratio = cfg.get("ratio", "Original")
        base_dim = 1080 if hd_1080 else 720

        if ratio == "9:16":
            out_w, out_h = base_dim, int(base_dim * 16 / 9)
        elif ratio == "1:1":
            out_w, out_h = base_dim, base_dim
        elif ratio == "4:5":
            out_w, out_h = base_dim, int(base_dim * 5 / 4)
        elif ratio == "16:9":
            out_w, out_h = int(base_dim * 16 / 9), base_dim
        else:
            max_h = 1080 if hd_1080 else 720
            scale_factor = (max_h / orig_h) if orig_h > max_h else 1.0
            out_w = int(orig_w * scale_factor)
            out_h = int(orig_h * scale_factor)

        out_w -= out_w % 2
        out_h -= out_h % 2

        # 1. Extract audio
        update_job_step(job_id, "extract_audio", 5, "ဗီဒီယိုထဲက အသံကို ထုတ်ယူနေပါတယ်...")
        extracted_audio = job_dir / "source_audio.mp3"
        await asyncio.to_thread(
            run_cmd,
            [
                "ffmpeg", "-y", "-threads", "2", "-i", str(video_path),
                "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
                str(extracted_audio),
            ],
        )

        if JOBS.get(job_id, {}).get("status") == "cancelled":
            return

        # 2. Upload & 3. Transcribe
        segments = await groq_whisper_transcribe(
            extracted_audio,
            groq_key,
            cfg.get("source_language", "Auto-detect"),
            job_id,
            total_video_dur,
            mode,
        )

        if JOBS.get(job_id, {}).get("status") == "cancelled":
            return

        # 4. Translate
        update_job_step(job_id, "translate", 38, "ဘာသာပြန်နေပါတယ်")
        recap_data = await gemini_generate_recap(
            segments=segments,
            gemini_key=gemini_key,
            gemini_model=gemini_model,
            mode=mode,
            output_lang=cfg.get("output_language", "မြန်မာ"),
            want_hashtags=cfg.get("ai_hashtags", False),
            default_hashtags=cfg.get("default_hashtags", ""),
            add_fyp=cfg.get("add_fyp", True),
        )
        segments = recap_data["segments"]
        JOBS[job_id]["tiktok_caption"] = recap_data["tiktok_caption"]
        save_job_state(job_id)

        # -----------------------------------------------------------------
        # OPTION A: MM SRT SUBTITLE MODE (Instant .srt file output)
        # -----------------------------------------------------------------
        if mode == "MM SRT SUBTITLE":
            srt_file = job_dir / "subtitles.srt"
            build_srt_file(segments, srt_file)
            for k in JOBS[job_id]["steps"]:
                JOBS[job_id]["steps"][k] = "done"
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["output_url"] = f"/api/download/{job_id}.srt"
            append_job_log(job_id, "မြန်မာ .srt စာတန်းထိုးဖိုင် ထုတ်ပြီးပါပြီ!")
            return

        # Geometry Filter (Crop + Zoom + Scale)
        geom_filters = []
        cx = float(cfg.get("crop_x", 0))
        cy = float(cfg.get("crop_y", 0))
        cw = float(cfg.get("crop_w", 100))
        ch = float(cfg.get("crop_h", 100))
        if cw < 99.0 or ch < 99.0:
            crop_px_w = max(64, int(orig_w * (cw / 100.0)))
            crop_px_h = max(64, int(orig_h * (ch / 100.0)))
            crop_px_x = max(0, min(orig_w - crop_px_w, int(orig_w * (cx / 100.0))))
            crop_px_y = max(0, min(orig_h - crop_px_h, int(orig_h * (cy / 100.0))))
            crop_px_w -= crop_px_w % 2
            crop_px_h -= crop_px_h % 2
            geom_filters.append(f"crop={crop_px_w}:{crop_px_h}:{crop_px_x}:{crop_px_y}")

        zoom_pct = int(cfg.get("zoom", 100))
        if zoom_pct > 100:
            zf = zoom_pct / 100.0
            geom_filters.append(
                f"crop=trunc(iw/{zf}/2)*2:trunc(ih/{zf}/2)*2:(iw-ow)/2:(ih-oh)/2"
            )

        fit_mode = cfg.get("fit_mode", "crop")
        if ratio != "Original" and fit_mode == "crop":
            geom_filters.append(
                f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h}"
            )
        else:
            geom_filters.append(f"scale={out_w}:{out_h}")

        geom_chain = ",".join(geom_filters)
        words_per_cap = max(3, int(cfg.get("words_per_caption", 9)))
        cap_width_pct = int(cfg.get("caption_width", 96))
        cap_text_size = int(cfg.get("text_size", 32))

        # -----------------------------------------------------------------
        # OPTION B: SUBTITLE (ORIGINAL + SUBS) SERVER CLOUD FAST MODE
        # (Skips TTS & Segment Slicing -> Direct Single-Pass Fast Server Render!)
        # -----------------------------------------------------------------
        if mode == "Subtitle":
            update_job_step(job_id, "voice", 58, "မူရင်းအသံ (Original Audio) ကို အသုံးပြုနေပါတယ်...")
            update_job_step(job_id, "timeline_captions", 72, "စာတန်းထိုး အချိန်မှတ်တမ်းများ ချိန်ညှိနေပါတယ်...")

            caption_events = []
            for seg in segments:
                seg_st = float(seg["start"])
                seg_en = max(seg_st + 0.4, float(seg["end"]))
                seg_dur = seg_en - seg_st
                phrases = split_into_short_phrases(
                    seg["narration"], words_per_cap, cap_width_pct, cap_text_size
                )
                total_chars = max(1, sum(len(p) for p in phrases))
                cur_offset = 0.0
                for ph in phrases:
                    ph_dur = seg_dur * (len(ph) / total_chars)
                    caption_events.append({
                        "start": seg_st + cur_offset,
                        "end": seg_st + cur_offset + ph_dur,
                        "text": ph,
                    })
                    cur_offset += ph_dur

            ass_path = job_dir / "captions.ass"
            build_ass_subtitles(caption_events, cfg, out_w, out_h, ass_path)

            update_job_step(
                job_id,
                "render_video",
                85,
                "Fast render (Server Cloud Original + Subs) နဲ့ အမြန်ထုတ်နေပါတယ်...",
            )

            vf_filters = [geom_chain]
            if cfg.get("random_4px_crop", False):
                vf_filters.append(f"crop=iw-8:ih-8:4:4,scale={out_w}:{out_h}")
            if cfg.get("mirror", False):
                vf_filters.append("hflip")

            smooth_blur = build_smooth_blur_filter(cfg, out_w, out_h)
            if smooth_blur:
                vf_filters.append(smooth_blur)

            if cfg.get("captions_on", True):
                escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
                vf_filters.append(f"ass='{escaped_ass}'")

            if cfg.get("watermark_on", False) and cfg.get("watermark_text"):
                wm_txt = re.sub(r"[^A-Za-z0-9_\-\.\s]", "", cfg["watermark_text"])
                wm_size = int(cfg.get("watermark_size", 17))
                wm_op = int(cfg.get("watermark_opacity", 65)) / 100.0
                if cfg.get("watermark_move", False):
                    x_expr = "(w-text_w)*0.5+sin(t*0.8)*(w*0.35)"
                    y_expr = "(h-text_h)*0.5+cos(t*0.6)*(h*0.35)"
                else:
                    x_expr = "w-tw-24"
                    y_expr = "h*0.82"
                vf_filters.append(
                    f"drawtext=text='{wm_txt}':fontsize={wm_size}:fontcolor=white@{wm_op:.2f}:x={x_expr}:y={y_expr}"
                )

            final_video = job_dir / "final_recap.mp4"
            vf_chain = ",".join(vf_filters)

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-threads", "2",
                "-i", str(video_path),
                "-vf", vf_chain,
                "-map", "0:v:0",
                "-map", "0:a:0?",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-maxrate", "2.8M",
                "-bufsize", "5.6M",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac",
                "-b:a", "128k",
                str(final_video),
            ]
            await asyncio.to_thread(run_cmd, ffmpeg_cmd)

            for k in JOBS[job_id]["steps"]:
                JOBS[job_id]["steps"][k] = "done"
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["output_url"] = f"/api/download/{job_id}"
            append_job_log(job_id, "Original + Subs ဗီဒီယို ထုတ်ပြီးပါပြီ! အလိုအလျောက် ဒေါင်းလုဒ်ဆွဲနေပါတယ်...")
            return

        # -----------------------------------------------------------------
        # OPTION C: AI RECAP & PRO AI RECAP (4x Parallel TTS + Smooth Blur)
        # -----------------------------------------------------------------
        update_job_step(job_id, "voice", 52, "အသံထုတ်နေပါတယ် (Fast Parallel Server TTS)...")
        voice_short = cfg.get("voice_name", "Thiha")
        voice_id = VOICE_MAP.get(voice_short, "my-MM-ThihaNeural")
        speed_val = int(cfg.get("voice_speed", 30))
        pitch_val = int(cfg.get("voice_pitch", 0))
        rate_str = f"{speed_val:+d}%"
        pitch_str = f"{pitch_val:+d}Hz"

        tts_sem = asyncio.Semaphore(4)

        async def generate_single_tts(seg_item: Dict[str, Any]):
            async with tts_sem:
                if JOBS.get(job_id, {}).get("status") == "cancelled":
                    return
                raw_tts = job_dir / f"raw_tts_{seg_item['index']:04d}.mp3"
                trim_tts = job_dir / f"tts_{seg_item['index']:04d}.mp3"
                comm = edge_tts.Communicate(seg_item["narration"], voice_id, rate=rate_str, pitch=pitch_str)
                await comm.save(str(raw_tts))
                await asyncio.to_thread(
                    run_cmd,
                    [
                        "ffmpeg", "-y", "-threads", "1", "-i", str(raw_tts),
                        "-af", "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_silence=0.08:start_threshold=-45dB,areverse",
                        "-ar", "24000", "-ac", "1", "-b:a", "96k",
                        str(trim_tts),
                    ],
                )
                raw_tts.unlink(missing_ok=True)
                seg_item["tts_file"] = trim_tts
                seg_item["tts_duration"] = max(0.35, await asyncio.to_thread(get_duration, trim_tts))

        await asyncio.gather(*(generate_single_tts(s) for s in segments))
        append_job_log(job_id, "ပြီးပါပြီ")

        # 6. Timeline + captions
        update_job_step(job_id, "timeline_captions", 72, "ရုပ်သံဖိုင်တွေ ဒေါင်းလုဒ် လုပ်နေပါတယ်...")
        caption_events = []
        concat_v_list: List[Path] = []
        concat_a_list: List[Path] = []
        timeline_cursor = 0.0
        prev_video_end = 0.0
        clip_counter = 0

        for seg in segments:
            if JOBS.get(job_id, {}).get("status") == "cancelled":
                return
            seg_start = max(prev_video_end, seg["start"])
            seg_end = max(seg_start + 0.4, seg["end"])

            gap_dur = seg_start - prev_video_end
            if gap_dur >= 0.4:
                target_gap_dur = min(3.0, max(0.35, gap_dur / 1.5))
                gap_pts = target_gap_dur / gap_dur
                v_gap = job_dir / f"vclip_{clip_counter:04d}_gap.mp4"
                a_gap = job_dir / f"aclip_{clip_counter:04d}_gap.mp3"
                clip_counter += 1

                await asyncio.to_thread(
                    run_cmd,
                    [
                        "ffmpeg", "-y", "-threads", "1",
                        "-ss", str(prev_video_end), "-t", str(gap_dur),
                        "-i", str(video_path), "-an",
                        "-vf", f"setpts={gap_pts:.4f}*(PTS-STARTPTS),{geom_chain}",
                        "-r", "24", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
                        str(v_gap),
                    ],
                )
                await asyncio.to_thread(
                    run_cmd,
                    [
                        "ffmpeg", "-y", "-threads", "1", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                        "-t", f"{target_gap_dur:.3f}", "-b:a", "96k", str(a_gap),
                    ],
                )
                concat_v_list.append(v_gap)
                concat_a_list.append(a_gap)
                timeline_cursor += target_gap_dur

            orig_dur = max(0.4, seg_end - seg_start)
            tts_dur = seg["tts_duration"]
            pts_factor = tts_dur / orig_dur
            v_seg = job_dir / f"vclip_{clip_counter:04d}_speech.mp4"
            clip_counter += 1

            await asyncio.to_thread(
                run_cmd,
                [
                    "ffmpeg", "-y", "-threads", "1",
                    "-ss", str(seg_start), "-t", str(orig_dur),
                    "-i", str(video_path), "-an",
                    "-vf", f"setpts={pts_factor:.4f}*(PTS-STARTPTS),{geom_chain}",
                    "-r", "24", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
                    str(v_seg),
                ],
            )
            concat_v_list.append(v_seg)
            concat_a_list.append(seg["tts_file"])

            phrases = split_into_short_phrases(
                seg["narration"], words_per_cap, cap_width_pct, cap_text_size
            )
            total_chars = max(1, sum(len(p) for p in phrases))
            cur_offset = 0.0
            for ph in phrases:
                ph_dur = tts_dur * (len(ph) / total_chars)
                caption_events.append({
                    "start": timeline_cursor + cur_offset,
                    "end": timeline_cursor + cur_offset + ph_dur,
                    "text": ph,
                })
                cur_offset += ph_dur

            timeline_cursor += tts_dur
            prev_video_end = seg_end

        v_list_txt = job_dir / "v_list.txt"
        a_list_txt = job_dir / "a_list.txt"
        v_list_txt.write_text("\n".join([f"file '{p.name}'" for p in concat_v_list]), encoding="utf-8")
        a_list_txt.write_text("\n".join([f"file '{p.name}'" for p in concat_a_list]), encoding="utf-8")

        merged_raw = job_dir / "merged_timeline.mp4"
        merged_audio = job_dir / "merged_voice.mp3"
        await asyncio.to_thread(
            run_cmd,
            ["ffmpeg", "-y", "-threads", "1", "-f", "concat", "-safe", "0", "-i", str(v_list_txt), "-c", "copy", str(merged_raw)],
        )
        await asyncio.to_thread(
            run_cmd,
            ["ffmpeg", "-y", "-threads", "1", "-f", "concat", "-safe", "0", "-i", str(a_list_txt), "-c", "copy", str(merged_audio)],
        )

        for p in concat_v_list:
            p.unlink(missing_ok=True)

        ass_path = job_dir / "captions.ass"
        build_ass_subtitles(caption_events, cfg, out_w, out_h, ass_path)

        # 7. Render video (Smooth Gaussian Blur + Captions)
        update_job_step(
            job_id,
            "render_video",
            86,
            "Fast render (server cloud) နဲ့ လုပ်နေပါတယ်...",
        )
        vf_filters = []

        if cfg.get("random_4px_crop", False):
            vf_filters.append(f"crop=iw-8:ih-8:4:4,scale={out_w}:{out_h}")

        if cfg.get("mirror", False):
            vf_filters.append("hflip")

        smooth_blur = build_smooth_blur_filter(cfg, out_w, out_h)
        if smooth_blur:
            vf_filters.append(smooth_blur)

        if cfg.get("captions_on", True):
            escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
            vf_filters.append(f"ass='{escaped_ass}'")

        if cfg.get("watermark_on", False) and cfg.get("watermark_text"):
            wm_txt = re.sub(r"[^A-Za-z0-9_\-\.\s]", "", cfg["watermark_text"])
            wm_size = int(cfg.get("watermark_size", 17))
            wm_op = int(cfg.get("watermark_opacity", 65)) / 100.0
            if cfg.get("watermark_move", False):
                x_expr = "(w-text_w)*0.5+sin(t*0.8)*(w*0.35)"
                y_expr = "(h-text_h)*0.5+cos(t*0.6)*(h*0.35)"
            else:
                x_expr = "w-tw-24"
                y_expr = "h*0.82"
            vf_filters.append(
                f"drawtext=text='{wm_txt}':fontsize={wm_size}:fontcolor=white@{wm_op:.2f}:x={x_expr}:y={y_expr}"
            )

        final_video = job_dir / "final_recap.mp4"
        vf_chain = ",".join(vf_filters) if vf_filters else "null"

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-threads", "2",
            "-i", str(merged_raw),
            "-i", str(merged_audio),
            "-vf", vf_chain,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-maxrate", "2.8M",
            "-bufsize", "5.6M",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            str(final_video),
        ]

        await asyncio.to_thread(run_cmd, ffmpeg_cmd)

        for k in JOBS[job_id]["steps"]:
            JOBS[job_id]["steps"][k] = "done"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["output_url"] = f"/api/download/{job_id}"
        append_job_log(job_id, "ဗီဒီယို ထုတ်လုပ်ပြီးပါပြီ! အလိုအလျောက် ဒေါင်းလုဒ်ဆွဲနေပါတယ်...")

    except Exception as e:
        if job_id in JOBS:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)
            append_job_log(job_id, f"Error: {str(e)}")


# -------------------------------------------------------------------------
# FASTAPI ENDPOINTS
# -------------------------------------------------------------------------
@app.post("/api/voice-preview")
async def voice_preview(
    voice_name: str = Form("Thiha"),
    speed: int = Form(30),
    pitch: int = Form(0),
):
    voice_id = VOICE_MAP.get(voice_name, "my-MM-ThihaNeural")
    sample_text = "မင်္ဂလာပါခင်ဗျာ၊ ဒီနေ့မှာတော့ စိတ်ဝင်စားစရာကောင်းတဲ့ ရုပ်ရှင်ဇာတ်လမ်းလေးကို တင်ဆက်ပေးသွားမှာ ဖြစ်ပါတယ်။"
    out_file = DATA_DIR / f"preview_{voice_name}_{speed}_{pitch}.mp3"
    if not out_file.exists():
        comm = edge_tts.Communicate(sample_text, voice_id, rate=f"{speed:+d}%", pitch=f"{pitch:+d}Hz")
        await comm.save(str(out_file))
    return FileResponse(out_file, media_type="audio/mpeg")


@app.post("/api/start-recap")
async def start_recap(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    config: str = Form(...),
    logo: Optional[UploadFile] = File(None),
):
    job_id = str(uuid.uuid4())[:10]
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / (video.filename or "input.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    logo_path = None
    if logo and logo.filename:
        logo_path = job_dir / logo.filename
        with open(logo_path, "wb") as f:
            shutil.copyfileobj(logo.file, f)

    cfg = json.loads(config)

    JOBS[job_id] = {
        "job_id": job_id,
        "filename": video.filename,
        "mode": cfg.get("mode", "AI Recap"),
        "status": "running",
        "progress": 2,
        "current_step": "extract_audio",
        "started_at": time.time(),
        "steps": {
            "extract_audio": "running",
            "upload": "pending",
            "transcribe": "pending",
            "translate": "pending",
            "voice": "pending",
            "timeline_captions": "pending",
            "render_video": "pending",
        },
        "logs": ["0:00 ဗီဒီယိုထဲက အသံကို ထုတ်ယူနေပါတယ်..."],
        "tiktok_caption": "",
        "output_url": None,
        "error": None,
    }
    save_job_state(job_id)

    background_tasks.add_task(process_recap_pipeline, job_id, video_path, cfg, logo_path)
    return JSONResponse({"job_id": job_id})


@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    job = load_job_state(job_id)
    if job:
        job["status"] = "cancelled"
        job["error"] = "Cancelled by user"
        JOBS[job_id] = job
        save_job_state(job_id)
    return JSONResponse({"ok": True})


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = load_job_state(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    elapsed = int(time.time() - job["started_at"])
    prog = max(1, job["progress"])
    eta = int(elapsed * (100 - prog) / prog) if prog < 100 else 0
    return {**job, "elapsed_sec": elapsed, "eta_sec": eta}


@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    clean_id = job_id.replace(".srt", "")
    job = load_job_state(clean_id) or {}
    orig_name = Path(job.get("filename") or "video.mp4").stem

    if job_id.endswith(".srt") or job.get("mode") == "MM SRT SUBTITLE":
        srt_file = DATA_DIR / clean_id / "subtitles.srt"
        if not srt_file.exists():
            raise HTTPException(status_code=404, detail="SRT file not ready")
        return FileResponse(
            srt_file,
            media_type="application/x-subrip",
            filename=f"{orig_name}_MM.srt",
        )

    final_video = DATA_DIR / clean_id / "final_recap.mp4"
    if not final_video.exists():
        raise HTTPException(status_code=404, detail="Video not ready")
    return FileResponse(
        final_video,
        media_type="video/mp4",
        filename=f"Recap_{orig_name}_{clean_id}.mp4",
        headers={"Cache-Control": "public, max-age=3600"},
    )


Path("static").mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
