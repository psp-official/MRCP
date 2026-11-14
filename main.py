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

app = FastAPI(title="AI Movie Recap Narrator (Groq Whisper + Gemini Edition)")

# Render Paid Plan Persistent Disk လမ်းကြောင်း (/var/data/recap_jobs)
DATA_DIR = Path(os.getenv("DATA_DIR", "./recap_jobs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# 2026 နိုဝင်ဘာလ လက်ရှိ အလုပ်လုပ်နေသော တရားဝင် Gemini Models စာရင်း
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
# 1. MYANMAR UNICODE CLEANER & SINGLE-LINE SHORT PHRASE SPLITTER
# -------------------------------------------------------------------------
def clean_burmese_text(text: str) -> str:
    """မြန်မာစာသားများကို သန့်စင်ပြီး သရအစီအစဉ်မှား၍ [] ပေါ်ခြင်းကို ကာကွယ်ခြင်း"""
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


def split_into_short_phrases(text: str, words_per_caption: int = 9) -> List[str]:
    """
    စာတန်းထိုး တစ်ခါပေါ်လျှင် တစ်ကြောင်းတည်း အနေတော် (၁၄ မှ ၂၂ လုံးခန့်)
    သပ်သပ်ရပ်ရပ် ပေါ်စေရန် ပုဒ်ဖြတ် (၊ ။) နှင့် Space အလိုက် ပိုင်းဖြတ်ပေးခြင်း
    """
    text = clean_burmese_text(text)
    if not text:
        return [""]

    max_chars = max(14, min(25, int(words_per_caption) * 2 + 4))
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


# -------------------------------------------------------------------------
# 2. FFMPEG & SYSTEM HELPERS
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


def cleanup_old_jobs(max_age_seconds: int = 86400):
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


def update_job_step(job_id: str, step_name: str, progress: int):
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


# -------------------------------------------------------------------------
# 3. STEP 2 & 3: GROQ WHISPER-LARGE-V3-TURBO TRANSCRIPTION
# -------------------------------------------------------------------------
async def groq_whisper_transcribe(
    audio_path: Path, groq_key: str, source_lang: str, job_id: str, total_video_dur: float
) -> List[Dict[str, Any]]:
    """
    Groq Whisper Large V3 Turbo သုံးပြီး အသံဖိုင်ကို စက္ကန့်ပိုင်းအတွင်း
    Segment Timestamps အတိအကျဖြင့် ဖမ်းယူခြင်း
    """
    update_job_step(job_id, "upload", 18)
    headers = {"Authorization": f"Bearer {groq_key.strip()}"}

    data_fields: Dict[str, Any] = {
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
        "temperature": "0.0",
    }
    if source_lang and source_lang != "Auto-detect":
        data_fields["language"] = source_lang

    update_job_step(job_id, "transcribe", 28)
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

    # နောက်ခံတီးလုံးသက်သက်ဖြစ်သော အပိုင်းများနှင့် စာသားအလွတ်များကို စစ်ထုတ်ခြင်း
    hallucination_words = {"字幕", "谢谢观看", "Subtitles", "Thank you for watching", "Amara.org"}
    for seg in raw_segments:
        txt = (seg.get("text") or "").strip()
        if not txt or any(hw in txt for hw in hallucination_words):
            continue
        if float(seg.get("no_speech_prob", 0.0)) > 0.88:
            continue
        st = max(0.0, float(seg.get("start", 0.0)))
        en = min(total_video_dur, max(st + 0.5, float(seg.get("end", st + 2.0))))
        cleaned_segments.append({"start": st, "end": en, "text": txt})

    # အရမ်းတိုနေသော စာကြောင်းလေးများ (< 2.2s) ကို ကပ်လျက်စာကြောင်းနှင့် ပေါင်းစပ်ပေးခြင်းဖြင့်
    # မြန်မာလို အသံထုတ်ရာတွင် အရုပ်နှင့်အသံ အချိန်ကိုက် ပိုမိုချောမွေ့စေသည်
    merged: List[Dict[str, Any]] = []
    for seg in cleaned_segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        combined_dur = seg["end"] - prev["start"]
        if gap < 0.35 and (prev["end"] - prev["start"] < 2.2 or combined_dur <= 5.5):
            prev["end"] = seg["end"]
            prev["text"] = f"{prev['text']} {seg['text']}".strip()
        else:
            merged.append(seg)

    if not merged:
        merged = [{"start": 0.0, "end": min(10.0, total_video_dur), "text": res_json.get("text", "...")}]

    for i, m in enumerate(merged):
        m["index"] = i
        m["start"] = round(m["start"], 2)
        m["end"] = round(m["end"], 2)

    return merged


# -------------------------------------------------------------------------
# 4. STEP 4: GEMINI API NATIVE JSON TRANSLATION / RECAP
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
    """
    Google Gemini API ၏ Native JSON Schema Mode ကို သုံး၍
    စာကြောင်းတစ်ကြောင်းချင်းစီကို သဘာဝကျသော မြန်မာစကားပြောအဖြစ် တိကျစွာ ပြောင်းလဲခြင်း
    """
    lang_map = {
        "မြန်မာ": "natural, engaging SPOKEN BURMESE (မြန်မာစကားပြောဟန်)",
        "English": "natural, engaging spoken English",
        "ไทย": "natural, engaging spoken Thai (ภาษาไทย)",
    }
    target_lang_desc = lang_map.get(output_lang, lang_map["မြန်မာ"])

    system_instruction = f"""You are an expert translator and voiceover scriptwriter for Chinese Donghua (Xianxia/Cultivation) and short dramas.
Translate each input segment directly and accurately into {target_lang_desc} so that it matches the exact meaning, emotion, and timing of the scene.

STRICT BURMESE RULES:
1. DIRECT SPOKEN STYLE: Translate each segment naturally as spoken Burmese dialogue/narration (e.g., "ရှောင်ဟွာ မင်းဘာလုပ်နေတာလဲ။", "သန့်ရှင်းသူမလေး ရောက်လာပြီ။", "ရောက်လာတော့လည်း ဘာဖြစ်လဲ။", "ပြောပြီးပြီလား။ မင်းလို အသုံးမကျတဲ့ကောင်က ငါ့ကို ရိုက်ရဲတယ်ပေါ့။").
2. NATURAL SPOKEN ENDINGS: Always use spoken particles ("တယ်", "လဲ", "ပြီ", "တာပေါ့", "ပါတယ်", "လား", "နော်", "ကွ"). NEVER use stiff literary words ("သည်", "၏", "၌", "၎င်း").
3. XIANXIA / CULTIVATION TERMINOLOGY:
   - 圣女 -> သန့်ရှင်းသူမလေး
   - 圣子 -> သန့်ရှင်းသူသား
   - 掌门 / 宗主 -> ဂိုဏ်းချုပ်
   - 师姐 -> အစ်မတော် / စီနီယာအစ်မ
   - 师兄 -> အစ်ကိုတော် / စီနီယာအစ်ကို
   - 少爷 / 公子 -> သခင်လေး
   - 宿主 -> အိမ်ရှင် / သခင်
   - 统子 / 系统 -> တုံစီ / စနစ်
   - 灵石 -> ဝိညာဉ်ကျောက်တုံး (下品=အောက်တန်းစား, 中品=အလယ်အလတ်တန်းစား, 上品=အဆင့်မြင့်)
   - 炼气 -> လျန်ချီအဆင့်, 筑基 -> အခြေတည်အဆင့်, 元丹 / 金丹 -> ယွမ်တန်အဆင့် / ရွှေဗဟိုအဆင့်
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
                narr_map[int(item["index"])] = clean_burmese_text(item.get("narration", ""))
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
# 5. STEP 6: ASS SUBTITLE BUILDER (CLEAN SINGLE-LINE BURMESE CAPTIONS)
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

    font_size = int(cfg.get("text_size", 30))
    primary_col = hex_to_ass_color(cfg.get("text_color", "#FFFF00"), 0)
    outline_col = hex_to_ass_color(cfg.get("outline_color", "#000000"), 0)
    back_col = hex_to_ass_color("#000000", 130)
    outline_w = int(cfg.get("outline_width", 3))
    bold_flag = -1 if cfg.get("bold", True) else 0
    border_style = 3 if cfg.get("box_bg", False) else 1

    vert_pct = int(cfg.get("vertical_position", 89))
    margin_v = max(18, int(play_res_y * (100 - vert_pct) / 100.0))
    width_pct = int(cfg.get("caption_width", 96))
    side_margin = max(16, int(play_res_x * (100 - width_pct) / 200.0))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_col},{primary_col},{outline_col},{back_col},{bold_flag},0,0,0,95,100,0,0,{border_style},{outline_w},1,2,{side_margin},{side_margin},{margin_v},1

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
# 6. MAIN 7-STEP PIPELINE WORKER
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

        if not groq_key or not gemini_key:
            raise RuntimeError("Groq API Key နှင့် Gemini API Key များကို 'Own key' တွင် ထည့်သွင်းပေးပါ။")

        orig_w, orig_h = await asyncio.to_thread(get_video_dimensions, video_path)
        total_video_dur = await asyncio.to_thread(get_duration, video_path)

        # 1. Extract audio (32kbps 16kHz Mono -> Groq 25MB limit အတွင်း ပေါ့ပါးမြန်ဆန်စွာ ပို့နိုင်ရန်)
        update_job_step(job_id, "extract_audio", 8)
        extracted_audio = job_dir / "source_audio.mp3"
        await asyncio.to_thread(
            run_cmd,
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
                str(extracted_audio),
            ],
        )

        # 2. Upload & 3. Transcribe (Groq Whisper Large V3 Turbo)
        segments = await groq_whisper_transcribe(
            extracted_audio,
            groq_key,
            cfg.get("source_language", "Auto-detect"),
            job_id,
            total_video_dur,
        )

        # 4. Translate (Google Gemini API)
        update_job_step(job_id, "translate", 42)
        recap_data = await gemini_generate_recap(
            segments=segments,
            gemini_key=gemini_key,
            gemini_model=gemini_model,
            mode=cfg.get("mode", "AI Recap"),
            output_lang=cfg.get("output_language", "မြန်မာ"),
            want_hashtags=cfg.get("ai_hashtags", True),
            default_hashtags=cfg.get("default_hashtags", ""),
            add_fyp=cfg.get("add_fyp", True),
        )
        segments = recap_data["segments"]
        JOBS[job_id]["tiktok_caption"] = recap_data["tiktok_caption"]

        # 5. Voice (Edge-TTS + Leading/Trailing Silence Removal)
        update_job_step(job_id, "voice", 58)
        voice_short = cfg.get("voice_name", "Thiha")
        voice_id = VOICE_MAP.get(voice_short, "my-MM-ThihaNeural")
        speed_val = int(cfg.get("voice_speed", 30))
        pitch_val = int(cfg.get("voice_pitch", 0))
        rate_str = f"{speed_val:+d}%"
        pitch_str = f"{pitch_val:+d}Hz"

        for seg in segments:
            raw_tts = job_dir / f"raw_tts_{seg['index']:04d}.mp3"
            trim_tts = job_dir / f"tts_{seg['index']:04d}.mp3"
            comm = edge_tts.Communicate(seg["narration"], voice_id, rate=rate_str, pitch=pitch_str)
            await comm.save(str(raw_tts))
            await asyncio.to_thread(
                run_cmd,
                [
                    "ffmpeg", "-y", "-i", str(raw_tts),
                    "-af", "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_silence=0.08:start_threshold=-45dB,areverse",
                    "-ar", "24000", "-ac", "1", "-b:a", "96k",
                    str(trim_tts),
                ],
            )
            seg["tts_file"] = trim_tts
            seg["tts_duration"] = max(0.35, await asyncio.to_thread(get_duration, trim_tts))

        # 6. Timeline + captions
        update_job_step(job_id, "timeline_captions", 70)
        words_per_cap = max(3, int(cfg.get("words_per_caption", 9)))
        caption_events = []
        concat_v_list: List[Path] = []
        concat_a_list: List[Path] = []
        timeline_cursor = 0.0
        prev_video_end = 0.0
        clip_counter = 0

        for seg in segments:
            seg_start = max(prev_video_end, seg["start"])
            seg_end = max(seg_start + 0.4, seg["end"])

            # စကားမပြောသော အက်ရှင်/ကြားကာလ (Gap >= 0.4s) ကို အခန်းမပျောက်စေရန် ထည့်သွင်းခြင်း
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
                        "ffmpeg", "-y",
                        "-ss", str(prev_video_end), "-t", str(gap_dur),
                        "-i", str(video_path), "-an",
                        "-vf", f"setpts={gap_pts:.4f}*(PTS-STARTPTS)",
                        "-r", "24", "-c:v", "libx264", "-preset", "ultrafast",
                        str(v_gap),
                    ],
                )
                await asyncio.to_thread(
                    run_cmd,
                    [
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                        "-t", f"{target_gap_dur:.3f}", "-b:a", "96k", str(a_gap),
                    ],
                )
                concat_v_list.append(v_gap)
                concat_a_list.append(a_gap)
                timeline_cursor += target_gap_dur

            # စကားပြောသော အပိုင်းကို TTS အသံအရှည်နှင့် ကွက်တိချိန်ညှိခြင်း
            orig_dur = max(0.4, seg_end - seg_start)
            tts_dur = seg["tts_duration"]
            pts_factor = tts_dur / orig_dur
            v_seg = job_dir / f"vclip_{clip_counter:04d}_speech.mp4"
            clip_counter += 1

            await asyncio.to_thread(
                run_cmd,
                [
                    "ffmpeg", "-y",
                    "-ss", str(seg_start), "-t", str(orig_dur),
                    "-i", str(video_path), "-an",
                    "-vf", f"setpts={pts_factor:.4f}*(PTS-STARTPTS)",
                    "-r", "24", "-c:v", "libx264", "-preset", "ultrafast",
                    str(v_seg),
                ],
            )
            concat_v_list.append(v_seg)
            concat_a_list.append(seg["tts_file"])

            phrases = split_into_short_phrases(seg["narration"], words_per_cap)
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
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(v_list_txt), "-c", "copy", str(merged_raw)],
        )
        await asyncio.to_thread(
            run_cmd,
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(a_list_txt), "-c", "copy", str(merged_audio)],
        )

        # Output Resolution
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
            scale_factor = (1080 / orig_h) if (hd_1080 and orig_h < 1080) else 1.0
            out_w = int(orig_w * scale_factor)
            out_h = int(orig_h * scale_factor)

        out_w -= out_w % 2
        out_h -= out_h % 2

        ass_path = job_dir / "captions.ass"
        build_ass_subtitles(caption_events, cfg, out_w, out_h, ass_path)

        # 7. Render video
        update_job_step(job_id, "render_video", 85)
        vf_filters = []

        if cfg.get("random_4px_crop", False):
            vf_filters.append("crop=iw-8:ih-8:4:4")

        zoom_pct = int(cfg.get("zoom", 100))
        if zoom_pct > 100:
            zf = zoom_pct / 100.0
            vf_filters.append(f"scale=iw*{zf}:ih*{zf},crop=iw/{zf}:ih/{zf}")

        if cfg.get("mirror", False):
            vf_filters.append("hflip")

        fit_mode = cfg.get("fit_mode", "crop")
        if ratio != "Original":
            if fit_mode == "pad":
                vf_filters.append(
                    f"split[original][copy];"
                    f"[copy]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},boxblur=20:10[blurred];"
                    f"[original]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease[scaled];"
                    f"[blurred][scaled]overlay=(W-w)/2:(H-h)/2"
                )
            else:
                vf_filters.append(
                    f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h}"
                )
        else:
            vf_filters.append(f"scale={out_w}:{out_h}")

        # မူရင်းတရုတ်စာတန်းထိုးနေရာကို သပ်ရပ်စွာ Gaussian Blur ဖုံးအုပ်ခြင်း
        if cfg.get("full_band", True) or int(cfg.get("blur_boxes", 0)) > 0 or cfg.get("captions_on", True):
            band_y = int(out_h * 0.81)
            band_h = max(24, out_h - band_y)
            vf_filters.append(
                f"split[vbase][vsub];"
                f"[vsub]crop={out_w}:{band_h}:0:{band_y},boxblur=20:8[vblur];"
                f"[vbase][vblur]overlay=0:{band_y}"
            )

        if hd_1080:
            vf_filters.append("unsharp=5:5:0.8:3:3:0.4")

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

        if logo_path and logo_path.exists():
            filter_complex = (
                f"[0:v]{vf_chain}[vmain];"
                f"[2:v]scale=96:96,format=yuva420p,"
                f"geq=lum='p(X,Y)':a='if(lte(hypot(X-48,Y-48),48),255,0)'[vlogo];"
                f"[vmain][vlogo]overlay=W-w-24:24[vout]"
            )
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(merged_raw),
                "-i", str(merged_audio),
                "-i", str(logo_path),
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-map", "1:a:0",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20" if hd_1080 else "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest",
                str(final_video),
            ]
        else:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(merged_raw),
                "-i", str(merged_audio),
                "-vf", vf_chain,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20" if hd_1080 else "23",
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

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


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
        "tiktok_caption": "",
        "output_url": None,
        "error": None,
    }

    background_tasks.add_task(process_recap_pipeline, job_id, video_path, cfg, logo_path)
    return JSONResponse({"job_id": job_id})


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    elapsed = int(time.time() - job["started_at"])
    prog = max(1, job["progress"])
    eta = int(elapsed * (100 - prog) / prog) if prog < 100 else 0
    return {**job, "elapsed_sec": elapsed, "eta_sec": eta}


@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    final_video = DATA_DIR / job_id / "final_recap.mp4"
    if not final_video.exists():
        raise HTTPException(status_code=404, detail="Video not ready")
    return FileResponse(final_video, media_type="video/mp4", filename=f"recap_{job_id}.mp4")


Path("static").mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
