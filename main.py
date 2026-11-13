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

app = FastAPI(title="AI Movie Recap Narrator (2026 Fixed Edition)")

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
# MYANMAR UNICODE NORMALIZER & TEXT CLEANER (လေးထောင့်ကွက် [] နှင့် Prompt စာသားဖယ်ရှားရန်)
# -------------------------------------------------------------------------
def clean_burmese_text(text: str) -> str:
    """
    ၁။ AI က မှားထည့်လိုက်သော Prompt ညွှန်ကြားချက်များကို ဖယ်ရှားခြင်း
    ၂။ မြန်မာ Unicode သေးသေးတင်/တစ်ချောင်းငင် အစီအစဉ်မှား၍ [] ပေါ်ခြင်းကို ပြင်ဆင်ခြင်း
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    # Remove prompt echo prefixes if any
    text = re.sub(
        r"^(ရင်ခုန်စရာ|သဘာဝကျသော|မြန်မာသဘာဝဖြင့်|ပြန်လည်ရေးသားပါ|Narration|Recap)[^:：]*[:：]\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    # Remove zero-width characters & markdown symbols that cause [] boxes in libass
    text = re.sub(r"[\u200b\u200c\u200d\ufeff*#_`~]", "", text)
    # Fix Myanmar Unicode vowel ordering: Anusvara (ံ U+1036) before U (ု U+102F / ူ U+1030) -> U before Anusvara
    text = re.sub(r"\u1036(\u102f|\u1030)", r"\1\u1036", text)
    # Fix Asat (် U+103A) + Dot below (့ U+1037) ordering
    text = re.sub(r"\u1037\u103a", "\u103a\u1037", text)
    return re.sub(r"\s+", " ", text).strip()


def split_burmese_caption_chunks(text: str, words_per_caption: int) -> List[str]:
    """
    မြန်မာစာကြောင်းရှည်ကြီးကို စာတန်းထိုးတွင် ဖတ်ရလွယ်အောင် တစ်ကြောင်းလျှင်
    စာလုံးရေ (Character/Syllable) အနေတော်အလိုက် သပ်သပ်ရပ်ရပ် ပိုင်းဖြတ်ပေးခြင်း
    """
    text = clean_burmese_text(text)
    if not text:
        return [""]
    # words_per_caption = 9 ဆိုလျှင် မြန်မာစာလုံးရေ ~32 လုံးခန့် (တစ်ကြောင်းစာ အနေတော်)
    max_chars = max(18, int(words_per_caption) * 3 + 5)

    # ပုဒ်ဖြတ်၊ ပုဒ်မ နှင့် Space များမှ အရင်ခွဲထုတ်သည်
    tokens = re.split(r"(?<=[၊။\s])", text)
    chunks: List[str] = []
    current = ""

    for tk in tokens:
        if not tk:
            continue
        if len(current) + len(tk) <= max_chars:
            current += tk
        else:
            if current.strip():
                chunks.append(current.strip())
            # Token တစ်ခုတည်းက အရမ်းရှည်နေပါက မြန်မာအက္ခရာအလိုက် ထပ်ပိုင်းသည်
            while len(tk) > max_chars:
                # Consonant boundary ရှာပြီး ဖြတ်သည်
                cut = max_chars
                while cut < len(tk) and "\u102b" <= tk[cut] <= "\u103f":
                    cut += 1
                chunks.append(tk[:cut].strip())
                tk = tk[cut:]
            current = tk

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


# -------------------------------------------------------------------------
# FFMPEG & HELPER FUNCTIONS
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
# STEP 2 & 3: ASSEMBLYAI TRANSCRIBE (CHINESE / DONGHUA ACCURACY FIX)
# -------------------------------------------------------------------------
async def request_assemblyai_transcript(
    client: httpx.AsyncClient,
    headers: Dict[str, str],
    upload_url: str,
    lang_code: Optional[str],
) -> Dict[str, Any]:
    """AssemblyAI သို့ Transcript တောင်းခံပြီး ပြီးဆုံးသည်အထိ စောင့်ပေးသော helper"""
    payload: Dict[str, Any] = {
        "audio_url": upload_url,
        # universal-2 သည် တရုတ် (zh) နှင့် အာရှဘာသာစကားများတွင် အင်္ဂလိပ်လို မှားမဖမ်းဘဲ အတိကျဆုံးဖြစ်သည်
        "speech_models": ["universal-2"],
    }
    if lang_code:
        payload["language_code"] = lang_code
    else:
        payload["language_detection"] = True
        payload["language_confidence_threshold"] = 0.6

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
        # Step 2: Upload
        update_job_step(job_id, "upload", 15)
        with open(audio_path, "rb") as f:
            up_res = await client.post(
                "https://api.assemblyai.com/v2/upload", headers=headers, content=f.read()
            )
        if up_res.status_code != 200:
            raise RuntimeError(f"AssemblyAI Upload Error: {up_res.text}")
        upload_url = up_res.json()["upload_url"]

        # Step 3: Transcribe
        update_job_step(job_id, "transcribe", 25)
        chosen_lang = None if (source_lang == "Auto-detect" or not source_lang) else source_lang

        try:
            data = await request_assemblyai_transcript(client, headers, upload_url, chosen_lang)
            # Auto-detect က နောက်ခံတီးလုံးကြောင့် တရုတ်ကားကို English (en) လို့ မှားဖမ်းမိပြီး confidence နည်းနေလျှင်
            # သို့မဟုတ် စာသားအလွန်နည်းနေလျှင် Chinese (zh) ဖြင့် အလိုအလျောက် ပြန်ဖမ်းပေးသည်
            detected = data.get("language_code", "")
            conf = float(data.get("language_confidence") or 1.0)
            if chosen_lang is None and detected == "en" and conf < 0.82:
                data = await request_assemblyai_transcript(client, headers, upload_url, "zh")
        except Exception:
            # Threshold မပြည့်ပါက Chinese (zh) ဖြင့် တိုက်ရိုက်ပြန်ဖမ်းသည်
            data = await request_assemblyai_transcript(client, headers, upload_url, "zh")

        # Words array မှ အချိန်အတိအကျဖြင့် Scene Windows (12-18s) တည်ဆောက်ခြင်း
        words = data.get("words") or []
        if words:
            sentences = []
            cur_w = []
            seg_st = words[0]["start"]
            for w in words:
                if not cur_w:
                    seg_st = w["start"]
                cur_w.append(w["text"])
                dur_ms = w["end"] - seg_st
                if dur_ms >= 10000 or w["text"].endswith(("。", "！", "？", ".", "!", "?")):
                    if dur_ms >= 4000:
                        sentences.append({
                            "text": " ".join(cur_w),
                            "start": seg_st,
                            "end": w["end"],
                        })
                        cur_w = []
            if cur_w:
                sentences.append({
                    "text": " ".join(cur_w),
                    "start": seg_st,
                    "end": words[-1]["end"],
                })
            if sentences:
                return sentences

        # Fallback if words list is empty
        return [{
            "text": data.get("text") or "",
            "start": 0,
            "end": int(total_video_dur * 1000),
        }]


# -------------------------------------------------------------------------
# STEP 4: OPENROUTER NATURAL BURMESE MOVIE RECAP GENERATOR
# -------------------------------------------------------------------------
async def openrouter_generate_recap(
    sentences: List[Dict[str, Any]],
    api_key: str,
    model: str,
    mode: str,
    output_lang: str,
    want_hashtags: bool,
    default_hashtags: str,
    add_fyp: bool,
    total_video_dur: float,
) -> Dict[str, Any]:
    """
    ဗီဒီယို အစမှ အဆုံးထိ အခန်းမပြတ်တောက်စေရန် Continuous Scene Segments တည်ဆောက်ပြီး
    သဘာဝကျသော မြန်မာ Movie Recap Narration အဖြစ် ပြောင်းလဲခြင်း
    """
    raw_groups = []
    cur_texts = []
    seg_start = 0.0
    seg_end = 0.0

    for s in sentences:
        st = s["start"] / 1000.0
        en = s["end"] / 1000.0
        if not cur_texts:
            seg_start = st
        cur_texts.append(s["text"])
        seg_end = en
        if (seg_end - seg_start) >= 14.0:
            raw_groups.append({
                "start": seg_start,
                "end": seg_end,
                "text": " ".join(cur_texts),
            })
            cur_texts = []

    if cur_texts:
        raw_groups.append({
            "start": seg_start,
            "end": max(seg_end, seg_start + 5.0),
            "text": " ".join(cur_texts),
        })

    # ဗီဒီယို အခန်းများ လုံးဝမပြတ်တောက်စေရန် 0.0s မှ Video အဆုံးထိ Continuous Timeline ချိတ်ဆက်ခြင်း
    segments = []
    for i, grp in enumerate(raw_groups):
        win_start = 0.0 if i == 0 else raw_groups[i]["start"]
        win_end = total_video_dur if i == len(raw_groups) - 1 else raw_groups[i + 1]["start"]
        win_end = max(win_start + 3.0, win_end)
        segments.append({
            "index": i,
            "start": round(win_start, 2),
            "end": round(win_end, 2),
            "duration_sec": round(win_end - win_start, 1),
            "text": grp["text"],
        })

    system_prompt = """You are a top-tier professional Burmese Movie & Donghua (Xianxia) Recap Narrator.
Your job is to convert the provided transcript segments into an engaging, natural 3rd-person Burmese Movie Recap narration script.

CRITICAL RULES FOR BURMESE NARRATION:
1. USE NATURAL SPOKEN BURMESE ONLY: End sentences naturally with "တယ်", "တာပေါ့", "လိုက်တယ်", "ခဲ့တယ်", "နေတော့တာပေါ့".
2. NEVER USE STIFF LITERARY BURMESE: Strictly forbid using "သည်", "၏", "၌", "၎င်း", "လူကြီးမားသည်", "ပြုလုပ်နေပါသည်".
3. NEVER ECHO INSTRUCTIONS: Do NOT write phrases like "ရင်ခုန်စရာ မြန်မာသဘာဝဖြင့် ပြန်လည်ရေးသားပါ:" or any intro labels. Output ONLY the story narration!
4. CONTEXTUAL CORRECTION: The input transcript is from a Chinese Cultivation (Xianxia) / Drama video and may contain ASR phonetic typos. Infer the true meaning from context (e.g. Sect Master/ဂိုဏ်းချုပ်, Senior Sister/စီနီယာအစ်မ, Maid/အစေခံမလေး, Spirit Stones/ဝိညာဉ်ကျောက်တုံး, Cultivation Realm/ကျင့်ကြံခြင်းအဆင့်, System/စနစ်). NEVER output random English place names like "Arkansas" or "America".
5. LENGTH MATCHING: Match each narration length to its `duration_sec` (approx. 3 to 4 Burmese words per second of duration).
6. OUTPUT FORMAT: Return ONLY a valid JSON object:
{"segments": [{"index": 0, "narration": "..."}], "tiktok_caption": "..."}"""

    # Segment အရေအတွက် များပါက 25 ခုစီ Chunk ခွဲပို့ပြီး JSON မှန်ကန်စွာ ရယူသည်
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
    batch_size = 25

    async with httpx.AsyncClient(timeout=240.0) as client:
        for b_start in range(0, len(segments), batch_size):
            batch = segments[b_start : b_start + batch_size]
            user_content = json.dumps({"segments": batch}, ensure_ascii=False)
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

            # Robust JSON Cleaning to fix 'Invalid control character' error
            cleaned_raw = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", raw_content)
            json_match = re.search(r"\{.*\}", cleaned_raw, flags=re.DOTALL)
            json_str = json_match.group(0) if json_match else cleaned_raw.strip()

            try:
                parsed = json.loads(json_str, strict=False)
            except json.JSONDecodeError:
                # Fallback: fix common quote/newline issues in AI output
                import ast
                try:
                    # Try safe eval if python dict format
                    cleaned_eval = json_str.replace("true", "True").replace("false", "False").replace("null", "None")
                    parsed = ast.literal_eval(cleaned_eval)
                except Exception:
                    raise RuntimeError(f"AI က တရားဝင် JSON format မဟုတ်သော စာသားထုတ်ပေးပါသည်: {json_str[:200]}...")

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
# STEP 6: ASS SUBTITLE BUILDER (NO TOFU BOXES [], SINGLE CLEAN LINE)
# -------------------------------------------------------------------------
def build_ass_subtitles(
    caption_events: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    play_res_x: int,
    play_res_y: int,
    ass_path: Path,
):
    # Noto Sans Myanmar သို့မဟုတ် Padauk ကို သုံးခြင်းဖြင့် မြန်မာစာ [] လုံးဝမပေါ်စေပါ
    req_font = cfg.get("font", "Noto Sans Myanmar")
    font_name = "Padauk" if req_font == "Padauk" else "Noto Sans Myanmar"
    font_size = int(cfg.get("text_size", 30))
    primary_col = hex_to_ass_color(cfg.get("text_color", "#FFFF00"), 0)
    outline_col = hex_to_ass_color(cfg.get("outline_color", "#000000"), 0)
    back_col = hex_to_ass_color("#000000", 130)
    outline_w = int(cfg.get("outline_width", 3))
    bold_flag = -1 if cfg.get("bold", True) else 0
    border_style = 3 if cfg.get("box_bg", False) else 1

    vert_pct = int(cfg.get("vertical_position", 89))
    margin_v = max(16, int(play_res_y * (100 - vert_pct) / 100.0))
    width_pct = int(cfg.get("caption_width", 96))
    side_margin = max(16, int(play_res_x * (100 - width_pct) / 200.0))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 1

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
# MAIN 7-STEP PIPELINE WORKER
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

        # ---------------------------------------------------------
        # 1. Extract audio
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 2. Upload & 3. Transcribe
        # ---------------------------------------------------------
        sentences = await assemblyai_upload_and_transcribe(
            extracted_audio,
            aai_key,
            cfg.get("source_language", "Auto-detect"),
            job_id,
            total_video_dur,
        )

        # ---------------------------------------------------------
        # 4. Translate / AI Recap Script
        # ---------------------------------------------------------
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
            total_video_dur=total_video_dur,
        )
        segments = recap_data["segments"]
        JOBS[job_id]["tiktok_caption"] = recap_data["tiktok_caption"]

        # ---------------------------------------------------------
        # 5. Voice (Edge-TTS Narrator)
        # ---------------------------------------------------------
        update_job_step(job_id, "voice", 58)
        voice_short = cfg.get("voice_name", "Thiha")
        voice_id = VOICE_MAP.get(voice_short, "my-MM-ThihaNeural")
        speed_val = int(cfg.get("voice_speed", 30))
        pitch_val = int(cfg.get("voice_pitch", 0))
        rate_str = f"{speed_val:+d}%"
        pitch_str = f"{pitch_val:+d}Hz"

        for seg in segments:
            tts_file = job_dir / f"tts_{seg['index']:04d}.mp3"
            comm = edge_tts.Communicate(seg["narration"], voice_id, rate=rate_str, pitch=pitch_str)
            await comm.save(str(tts_file))
            seg["tts_file"] = tts_file
            seg["tts_duration"] = await asyncio.to_thread(get_duration, tts_file)

        # ---------------------------------------------------------
        # 6. Timeline + captions (PTS-STARTPTS Fix & Proper Burmese Chunking)
        # ---------------------------------------------------------
        update_job_step(job_id, "timeline_captions", 70)
        words_per_cap = max(3, int(cfg.get("words_per_caption", 9)))
        caption_events = []
        concat_v_list = []
        concat_a_list = []
        timeline_cursor = 0.0

        for seg in segments:
            orig_dur = max(0.5, seg["end"] - seg["start"])
            tts_dur = max(0.5, seg["tts_duration"])
            pts_factor = tts_dur / orig_dur
            v_seg_path = job_dir / f"vseg_{seg['index']:04d}.mp4"

            await asyncio.to_thread(
                run_cmd,
                [
                    "ffmpeg", "-y",
                    "-ss", str(seg["start"]),
                    "-t", str(orig_dur),
                    "-i", str(video_path),
                    "-an",  # မူရင်းရုပ်ရှင်အသံ လုံးဝဖျောက်ပစ်ခြင်း
                    "-vf", f"setpts={pts_factor:.4f}*(PTS-STARTPTS)",
                    "-r", "24", "-c:v", "libx264", "-preset", "ultrafast",
                    str(v_seg_path),
                ],
            )
            concat_v_list.append(v_seg_path)
            concat_a_list.append(seg["tts_file"])

            # မြန်မာစာကြောင်းကို တစ်ကြောင်းစာ အနေတော်အလိုက် အချိန်အတိအကျ ပိုင်းဖြတ်ခြင်း
            chunks = split_burmese_caption_chunks(seg["narration"], words_per_cap)
            total_chars = max(1, sum(len(c) for c in chunks))
            cur_offset = 0.0
            for ch in chunks:
                ch_dur = tts_dur * (len(ch) / total_chars)
                caption_events.append({
                    "start": timeline_cursor + cur_offset,
                    "end": timeline_cursor + cur_offset + ch_dur,
                    "text": ch,
                })
                cur_offset += ch_dur

            timeline_cursor += tts_dur

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

        # ---------------------------------------------------------
        # 7. Render video (Auto Subtitle Blur + Visual Effects + Captions)
        # ---------------------------------------------------------
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

        # မူရင်း တရုတ်စာတန်းထိုးများ မြန်မာစာတန်းထိုးအောက်တွင် မရှုပ်နေစေရန် အောက်ခြေ Subtitle နေရာကို သဘာဝကျကျ Blur ဝါးပေးခြင်း
        if cfg.get("full_band", True) or int(cfg.get("blur_boxes", 0)) > 0 or cfg.get("captions_on", True):
            band_y = int(out_h * 0.82)
            band_h = max(24, int(out_h * 0.14))
            vf_filters.append(
                f"split[vbase][vsub];"
                f"[vsub]crop={out_w}:{band_h}:0:{band_y},boxblur=18:6[vblur];"
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
