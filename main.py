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

app = FastAPI(title="AI Movie Recap Narrator (Exact Pro Sync Edition)")

DATA_DIR = Path(os.getenv("DATA_DIR", "./recap_jobs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
DEFAULT_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")

FREE_MODEL_FALLBACKS = [
    "google/gemma-4-31b-it:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "openrouter/free",
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
# 1. MYANMAR UNICODE CLEANER & SHORT SINGLE-LINE CAPTION SPLITTER
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

    # Fix vowel ordering without regex \u escape issues
    text = text.replace("\u1036\u102f", "\u102f\u1036")
    text = text.replace("\u1036\u1030", "\u1030\u1036")
    text = text.replace("\u1037\u103a", "\u103a\u1037")
    return re.sub(r"\s+", " ", text).strip()


def split_into_short_phrases(text: str, words_per_caption: int = 9) -> List[str]:
    """
    video_1.mp4 ကဲ့သို့ စာတန်းထိုး တစ်ခါပေါ်လျှင် တစ်ကြောင်းတည်း အနေတော် (၁၄ မှ ၂၂ လုံးခန့်)
    သပ်သပ်ရပ်ရပ် ပေါ်စေရန် ပုဒ်ဖြတ် (၊ ။) နှင့် Space အလိုက် ပိုင်းဖြတ်ပေးခြင်း
    """
    text = clean_burmese_text(text)
    if not text:
        return [""]

    # တစ်ကြောင်းတည်း လှပစွာ ပေါ်စေရန် အများဆုံး စာလုံးရေ ~22 လုံး
    max_chars = max(14, min(26, int(words_per_caption) * 2 + 4))

    # ပထမအဆင့် - ပုဒ်ဖြတ် (၊) နှင့် ပုဒ်မ (။) များဖြင့် အရင်ခွဲသည်
    clauses = re.split(r"(?<=[၊။])\s*", text)
    phrases: List[str] = []

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) <= max_chars:
            phrases.append(clause)
            continue

        # Clause က ရှည်နေပါက Space ခြားထားသော စကားလုံးများအလိုက် ထပ်ပိုင်းသည်
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
# 2. FFMPEG HELPERS
# -------------------------------------------------------------------------
def run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr[-1000:]}")
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
# 3. STEP 2 & 3: ASSEMBLYAI TRANSCRIBE (UTTERANCE-LEVEL SYNC)
# -------------------------------------------------------------------------
async def request_assemblyai_transcript(
    client: httpx.AsyncClient,
    headers: Dict[str, str],
    upload_url: str,
    lang_code: Optional[str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "audio_url": upload_url,
        "speech_models": ["universal-3-5-pro", "universal-2"],
    }
    if lang_code:
        payload["language_code"] = lang_code
    else:
        payload["language_detection"] = True
        payload["language_detection_options"] = {
            "expected_languages": ["zh", "en", "th", "ko", "ja", "vi"]
        }

    tr_res = await client.post(
        "https://api.assemblyai.com/v2/transcript", headers=headers, json=payload
    )
    if tr_res.status_code != 200:
        raise RuntimeError(f"AssemblyAI Transcript Error: {tr_res.text}")
    transcript_id = tr_res.json()["id"]

    while True:
        st_res = await client.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers
        )
        data = st_res.json()
        status = data.get("status")
        if status == "completed":
            return data
        elif status == "error":
            raise RuntimeError(f"AssemblyAI Failed: {data.get('error')}")
        await asyncio.sleep(3)


async def assemblyai_upload_and_transcribe(
    audio_path: Path, api_key: str, source_lang: str, job_id: str, total_video_dur: float
) -> List[Dict[str, Any]]:
    headers = {"authorization": api_key.strip()}
    async with httpx.AsyncClient(timeout=300.0) as client:
        update_job_step(job_id, "upload", 15)
        with open(audio_path, "rb") as f:
            up_res = await client.post(
                "https://api.assemblyai.com/v2/upload", headers=headers, content=f.read()
            )
        if up_res.status_code != 200:
            raise RuntimeError(f"AssemblyAI Upload Error: {up_res.text}")
        upload_url = up_res.json()["upload_url"]

        update_job_step(job_id, "transcribe", 25)
        chosen_lang = None if (source_lang == "Auto-detect" or not source_lang) else source_lang

        try:
            data = await request_assemblyai_transcript(client, headers, upload_url, chosen_lang)
            detected = data.get("language_code", "")
            conf = float(data.get("language_confidence") or 1.0)
            # နောက်ခံတီးလုံးကြောင့် တရုတ်ကားကို English လို့ မှားဖမ်းမိပါက Chinese (zh) ဖြင့် ပြန်ဖမ်းသည်
            if chosen_lang is None and detected == "en" and conf < 0.85:
                data = await request_assemblyai_transcript(client, headers, upload_url, "zh")
        except Exception:
            data = await request_assemblyai_transcript(client, headers, upload_url, "zh")

        transcript_id = data["id"]

        # Sentences endpoint ကို အရင်ယူသည်
        sent_res = await client.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}/sentences",
            headers=headers,
        )
        raw_sentences = sent_res.json().get("sentences", []) if sent_res.status_code == 200 else []

        # တရုတ်စာတွင် sentences က အရမ်းရှည်နေပါက words မှ အသံရပ်ချိန် (pause > 0.45s) သို့မဟုတ် ပုဒ်ဖြတ်အလိုက် ခွဲထုတ်သည်
        words = data.get("words") or []
        if words:
            utterances: List[Dict[str, Any]] = []
            cur_w: List[str] = []
            seg_st = words[0]["start"]
            prev_end = words[0]["end"]

            for w in words:
                gap = w["start"] - prev_end
                dur = prev_end - seg_st
                # အသံ 0.45s ခြားသွားလျှင် သို့မဟုတ် 6 စက္ကန့်ပြည့်ပြီး ပုဒ်ဖြတ်ပါလျှင် စာကြောင်းတစ်ကြောင်းအဖြစ် ဖြတ်သည်
                if cur_w and (
                    (gap >= 450 and dur >= 1500)
                    or (dur >= 5500 and w["text"].endswith(("。", "！", "？", "，", ".", "!", "?")))
                    or dur >= 9000
                ):
                    utterances.append({
                        "text": "".join(cur_w) if data.get("language_code", "").startswith("zh") else " ".join(cur_w),
                        "start": seg_st,
                        "end": prev_end,
                    })
                    cur_w = []
                    seg_st = w["start"]

                cur_w.append(w["text"])
                prev_end = w["end"]

            if cur_w:
                utterances.append({
                    "text": "".join(cur_w) if data.get("language_code", "").startswith("zh") else " ".join(cur_w),
                    "start": seg_st,
                    "end": prev_end,
                })
            if utterances:
                return utterances

        if raw_sentences:
            return raw_sentences

        return [{
            "text": data.get("text") or "",
            "start": 0,
            "end": int(total_video_dur * 1000),
        }]


# -------------------------------------------------------------------------
# 4. STEP 4: OPENROUTER SENTENCE-BY-SENTENCE NATURAL BURMESE TRANSLATION
# -------------------------------------------------------------------------
def parse_ai_json_response(raw_content: str) -> Dict[str, Any]:
    cleaned_raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", raw_content)
    json_match = re.search(r"\{.*\}", cleaned_raw, flags=re.DOTALL)
    json_str = json_match.group(0) if json_match else cleaned_raw.strip()

    try:
        return json.loads(json_str, strict=False)
    except Exception:
        fallback_segs = []
        pattern = r'"index"\s*:\s*(\d+)\s*,\s*"narration"\s*:\s*"([^"]+)"'
        for m in re.finditer(pattern, cleaned_raw):
            fallback_segs.append({"index": int(m.group(1)), "narration": m.group(2)})
        cap_match = re.search(r'"tiktok_caption"\s*:\s*"([^"]+)"', cleaned_raw)
        return {
            "segments": fallback_segs,
            "tiktok_caption": cap_match.group(1) if cap_match else "",
        }


async def openrouter_generate_recap(
    sentences: List[Dict[str, Any]],
    api_key: str,
    model: str,
    mode: str,
    output_lang: str,
    want_hashtags: bool,
    default_hashtags: str,
    add_fyp: bool,
) -> Dict[str, Any]:
    """
    video_1.mp4 ကဲ့သို့ စာကြောင်းတစ်ကြောင်းချင်းစီကို မူရင်းအဓိပ္ပာယ်မပျက်၊
    သဘာဝကျသော မြန်မာစကားပြော (Spoken Burmese Dub/Recap style) ဖြင့် တိကျစွာ ဘာသာပြန်ခြင်း
    """
    segments = []
    for idx, s in enumerate(sentences):
        st = round(s["start"] / 1000.0, 2)
        en = round(max(st + 0.8, s["end"] / 1000.0), 2)
        segments.append({
            "index": idx,
            "start": st,
            "end": en,
            "text": s["text"],
        })

    system_prompt = """You are an expert Burmese translator and voiceover scriptwriter for Chinese Donghua (Xianxia/Cultivation) and dramas.
Translate each input sentence directly and accurately into natural, punchy SPOKEN BURMESE (မြန်မာစကားပြောဟန်) that matches the exact meaning and emotion of the character/narrator.

STRICT RULES:
1. DIRECT SPOKEN TRANSLATION: Translate sentence-by-sentence naturally (e.g., "ရှောင်ဟွာ မင်းဘာလုပ်နေတာလဲ။", "သန့်ရှင်းသူမလေး ရောက်လာပြီ။", "ရောက်လာတော့လည်း ဘာဖြစ်လဲ။", "ပြောပြီးပြီလား။").
2. NATURAL SPOKEN ENDINGS: Use natural spoken particles ("တယ်", "လဲ", "ပြီ", "တာပေါ့", "ပါတယ်", "လား", "နော်"). NEVER use stiff literary words ("သည်", "၏", "၌", "၎င်း").
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
4. WORD SPACING FOR SUBTITLES: Put a space between Burmese phrases/words so subtitles can split cleanly, and end sentences with "။" or clauses with "၊".
5. NO PROMPT ECHO: Output ONLY valid single-line JSON:
{"segments": [{"index": 0, "narration": "..."}], "tiktok_caption": "..."}"""

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://recap-app.onrender.com",
        "X-Title": "AI Movie Recap Narrator",
    }
    preferred = model or DEFAULT_OPENROUTER_MODEL
    candidate_models = [preferred] + [m for m in FREE_MODEL_FALLBACKS if m != preferred]

    narr_map: Dict[int, str] = {}
    tiktok_cap = "ဇာတ်လမ်းကောင်းလေးကို အဆုံးထိ အားပေးကြပါဦးနော် 🎬"
    batch_size = 30

    async with httpx.AsyncClient(timeout=240.0) as client:
        for b_start in range(0, len(segments), batch_size):
            batch = segments[b_start : b_start + batch_size]
            payload_items = [{"index": x["index"], "text": x["text"]} for x in batch]
            user_content = json.dumps({"segments": payload_items}, ensure_ascii=False)
            raw_content = None
            last_error = ""

            for try_model in candidate_models:
                body = {
                    "model": try_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                }
                res = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body
                )
                if res.status_code == 200:
                    res_json = res.json()
                    if "choices" in res_json and res_json["choices"]:
                        raw_content = res_json["choices"][0]["message"]["content"]
                        if raw_content:
                            break
                last_error = f"[{try_model}] {res.status_code}: {res.text}"
                await asyncio.sleep(2)

            if not raw_content:
                raise RuntimeError(f"OpenRouter Error: {last_error}")

            parsed = parse_ai_json_response(raw_content)

            for item in parsed.get("segments", []):
                if "index" in item and "narration" in item:
                    narr_map[int(item["index"])] = clean_burmese_text(item["narration"])
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
    # Custom A13WenYoe ဖောင့်ရှိပါက သုံးပြီး မရှိပါက Padauk / Noto Sans Myanmar ကို သုံးသည်
    req_font = cfg.get("font", "Padauk")
    if Path("/usr/share/fonts/truetype/custom/A13WenYoe.ttf").exists() and req_font == "A13WenYoe":
        font_name = "A13WenYoe"
    elif req_font == "Padauk":
        font_name = "Padauk"
    else:
        font_name = "Noto Sans Myanmar"

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
# 6. MAIN 7-STEP PIPELINE WORKER (EXACT VIDEO_1.MP4 TIMELINE & GAP SYNC)
# -------------------------------------------------------------------------
async def process_recap_pipeline(
    job_id: str, video_path: Path, cfg: Dict[str, Any], logo_path: Optional[Path]
):
    job_dir = DATA_DIR / job_id
    try:
        cleanup_old_jobs()

        aai_key = cfg.get("assemblyai_key") or DEFAULT_ASSEMBLYAI_KEY
        or_key = cfg.get("openrouter_key") or DEFAULT_OPENROUTER_KEY
        or_model = cfg.get("openrouter_model") or DEFAULT_OPENROUTER_MODEL

        if not aai_key or not or_key:
            raise RuntimeError("AssemblyAI နှင့် OpenRouter API Key များကို 'Own key' တွင် ထည့်သွင်းပေးပါ။")

        orig_w, orig_h = await asyncio.to_thread(get_video_dimensions, video_path)
        total_video_dur = await asyncio.to_thread(get_duration, video_path)

        # 1. Extract audio
        update_job_step(job_id, "extract_audio", 8)
        extracted_audio = job_dir / "source_audio.mp3"
        await asyncio.to_thread(
            run_cmd,
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
                str(extracted_audio),
            ],
        )

        # 2. Upload & 3. Transcribe
        sentences = await assemblyai_upload_and_transcribe(
            extracted_audio,
            aai_key,
            cfg.get("source_language", "Auto-detect"),
            job_id,
            total_video_dur,
        )

        # 4. Translate
        update_job_step(job_id, "translate", 42)
        recap_data = await openrouter_generate_recap(
            sentences=sentences,
            api_key=or_key,
            model=or_model,
            mode=cfg.get("mode", "AI Recap"),
            output_lang=cfg.get("output_language", "မြန်မာ"),
            want_hashtags=cfg.get("ai_hashtags", True),
            default_hashtags=cfg.get("default_hashtags", ""),
            add_fyp=cfg.get("add_fyp", True),
        )
        segments = recap_data["segments"]
        JOBS[job_id]["tiktok_caption"] = recap_data["tiktok_caption"]

        # 5. Voice (Edge-TTS + Leading/Trailing Silence Trim for Tight Pacing)
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
            # Edge-TTS အသံဖိုင်၏ အစနှင့်အဆုံးရှိ အသံတိတ်နေရာပိုများကို ဖြတ်ထုတ်ပြီး 24kHz Mono တူညီအောင် ညှိသည်
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

        # 6. Timeline + captions (Preserving Action Gaps + Speech Sync like video_1.mp4)
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

            # စကားမပြောသော ကြားကာလ (Action/Silent Gap > 0.35s) ရှိပါက အခန်းမပျောက်စေရန် ထည့်သွင်းပေးခြင်း
            gap_dur = seg_start - prev_video_end
            if gap_dur >= 0.35:
                # Gap အရမ်းရှည်ပါက 1.45x သွက်သွက်လေး ပြပေးသည် (အမြင့်ဆုံး 3.5 စက္ကန့်)
                target_gap_dur = min(3.5, max(0.35, gap_dur / 1.45))
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

            # စကားပြောသော အပိုင်း (Speech Segment) ကို TTS အသံအရှည်နှင့် ကွက်တိချိန်ညှိခြင်း
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

            # စာကြောင်းတစ်ကြောင်းကို စကားစုတိုလေးများ (Short Single-Line Phrases) အဖြစ် ခွဲပြီး အသံနှင့် အချိန်ကိုက်ပြသခြင်း
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

        # 7. Render video (Mirror + Smooth Bottom Subtitle Blur + Single-Line Yellow Captions)
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

        # video_1.mp4 အတိုင်း အောက်ခြေ မူရင်းတရုတ်စာတန်းထိုးနေရာကို ကြည်လင်သပ်ရပ်စွာ Gaussian Blur ဖုံးအုပ်ခြင်း
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
