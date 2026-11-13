import os
import re
import json
import uuid
import time
import shutil
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx
import edge_tts
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="AI Movie Recap Narrator (2026 Free API Edition)")

# Render Paid Plan Persistent Disk လမ်းကြောင်း (/var/data/recap_jobs)
DATA_DIR = Path(os.getenv("DATA_DIR", "./recap_jobs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 13.11.2026 FREE API CONFIGURATION & AUTO-FALLBACK MODELS
# -------------------------------------------------------------------------
DEFAULT_ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
DEFAULT_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")

# Free Model တစ်ခု Rate Limit (429) ဖြစ်နေပါက အလိုအလျောက် ပြောင်းသုံးမည့် 2026 Free Models စာရင်း
FREE_MODEL_FALLBACKS = [
    "google/gemma-4-31b-it:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "openrouter/free",
]

# Edge-TTS (100% Free) Voice Mapping
VOICE_MAP = {
    "Thiha": "my-MM-ThihaNeural",
    "Nilar": "my-MM-NilarNeural",
    "Guy": "en-US-GuyNeural",
    "Jenny": "en-US-JennyNeural",
    "Niwat": "th-TH-NiwatNeural",
    "Premwadee": "th-TH-PremwadeeNeural",
}

# Job Status များ သိမ်းဆည်းရန် Memory Store
JOBS: Dict[str, Dict[str, Any]] = {}


# -------------------------------------------------------------------------
# HELPER FUNCTIONS (FFMPEG, TIME, ASS SUBTITLES, DISK CLEANUP)
# -------------------------------------------------------------------------
def run_cmd(cmd: List[str]) -> str:
    """FFmpeg နှင့် FFprobe command များကို Run ရန် helper function"""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr[-1000:]}")
    return proc.stdout


def get_duration(file_path: Path) -> float:
    """Video သို့မဟုတ် Audio ဖိုင်၏ ကြာချိန် (စက္ကန့်) ကို တွက်ချက်ခြင်း"""
    out = run_cmd([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
    ])
    return float(out.strip() or 0.0)


def get_video_dimensions(file_path: Path) -> tuple[int, int]:
    """မူရင်း Video ၏ Width နှင့် Height ကို ရယူခြင်း"""
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
    """၂၄ နာရီထက် ကျော်လွန်သော ဖိုင်အဟောင်းများကို Persistent Disk မှ အလိုအလျောက် ရှင်းလင်းခြင်း"""
    now = time.time()
    for folder in DATA_DIR.iterdir():
        if folder.is_dir():
            try:
                if now - folder.stat().st_mtime > max_age_seconds:
                    shutil.rmtree(folder, ignore_errors=True)
            except Exception:
                pass


def hex_to_ass_color(hex_color: str, alpha: int = 0) -> str:
    """#RRGGBB အရောင်ကို .ass စာတန်းထိုးသုံး &HAABBGGRR format သို့ ပြောင်းခြင်း"""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        hex_color = "FFFF00"
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H{alpha:02X}{b.upper()}{g.upper()}{r.upper()}"


def format_ass_time(seconds: float) -> str:
    """စက္ကန့်ကို H:MM:SS.cc (.ass timestamp) သို့ ပြောင်းခြင်း"""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def update_job_step(job_id: str, step_name: str, progress: int):
    """ပုံပါ အဆင့် (၇) ဆင့်၏ အမှန်ခြစ်နှင့် ရာခိုင်နှုန်းကို Update လုပ်ခြင်း"""
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
# STEP 2 & 3: ASSEMBLYAI FREE TIER (UPLOAD & TRANSCRIBE)
# -------------------------------------------------------------------------
async def assemblyai_upload_and_transcribe(
    audio_path: Path, api_key: str, source_lang: str, job_id: str
) -> List[Dict[str, Any]]:
    """အဆင့် ၂ (Upload) နှင့် အဆင့် ၃ (Transcribe) - AssemblyAI Free Tier (185 Hours Free)"""
    headers = {"authorization": api_key.strip()}
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Step 2: Upload Audio
        update_job_step(job_id, "upload", 15)
        with open(audio_path, "rb") as f:
            up_res = await client.post(
                "https://api.assemblyai.com/v2/upload", headers=headers, content=f.read()
            )
        if up_res.status_code != 200:
            raise RuntimeError(f"AssemblyAI Upload Error: {up_res.text}")
        upload_url = up_res.json()["upload_url"]

        # Step 3: Transcribe Audio (AssemblyAI ၏ လက်ရှိ API သတ်မှတ်ချက်အသစ်အရ speech_models ကို သုံးခြင်း)
        update_job_step(job_id, "transcribe", 25)
        payload: Dict[str, Any] = {
            "audio_url": upload_url,
            "speech_models": ["universal-3-5-pro", "universal-2"],  # API အသစ်နှင့် ကိုက်ညီစေရန်
        }
        if source_lang == "Auto-detect" or not source_lang:
            payload["language_detection"] = True
        else:
            payload["language_code"] = source_lang


        tr_res = await client.post(
            "https://api.assemblyai.com/v2/transcript", headers=headers, json=payload
        )
        if tr_res.status_code != 200:
            raise RuntimeError(f"AssemblyAI Transcript Error: {tr_res.text}")
        transcript_id = tr_res.json()["id"]

        # Polling status
        while True:
            st_res = await client.get(
                f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers
            )
            data = st_res.json()
            status = data.get("status")
            if status == "completed":
                break
            elif status == "error":
                raise RuntimeError(f"AssemblyAI Failed: {data.get('error')}")
            await asyncio.sleep(3)

        # Sentences endpoint မှ တိကျသော Timestamps ရယူခြင်း
        sent_res = await client.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}/sentences",
            headers=headers,
        )
        sentences = sent_res.json().get("sentences", [])
        if not sentences and data.get("text"):
            sentences = [{
                "text": data["text"],
                "start": 0,
                "end": int((data.get("audio_duration") or 10) * 1000)
            }]
        return sentences


# -------------------------------------------------------------------------
# STEP 4: OPENROUTER FREE TIER (TRANSLATE / MOVIE RECAP GENERATION)
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
) -> Dict[str, Any]:
    """
    အဆင့် ၄ (Translate / Recap) - OpenRouter Free Tier (50 Requests/Day Free)
    Video ၁ ပုဒ်လုံးကို Request ၁ ကြိမ်တည်းဖြင့် ပို့ပြီး 429 ဖြစ်ပါက အခြား Free Model သို့ အလိုအလျောက် ပြောင်းပေးသည်။
    """
    segments = []
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
        if (seg_end - seg_start) >= 12.0 or len(cur_texts) >= 4:
            segments.append({
                "index": len(segments),
                "start": round(seg_start, 2),
                "end": round(seg_end, 2),
                "text": " ".join(cur_texts),
            })
            cur_texts = []

    if cur_texts:
        segments.append({
            "index": len(segments),
            "start": round(seg_start, 2),
            "end": round(seg_end, 2),
            "text": " ".join(cur_texts),
        })

    lang_instruction = {
        "မြန်မာ": "သဘာဝကျပြီး ဆွဲဆောင်မှုရှိသော မြန်မာစကားပြော (Burmese Movie Recap Narrator style) ဖြင့်",
        "English": "in engaging third-person English Movie Recap narrator style",
        "ไทย": "ในสไตล์ผู้บรรยายสรุปภาพยนตร์ภาษาไทยที่น่าติดตาม",
    }.get(output_lang, "သဘာဝကျသော မြန်မာ Movie Recap Narrator ပုံစံဖြင့်")

    if mode == "Subtitle":
        style_rule = f"မူရင်းစကားပြောအတိုင်း {output_lang} ဘာသာသို့ တိကျသဘာဝကျစွာ ဘာသာပြန်ပေးပါ။"
    else:
        style_rule = (
            f"မူရင်းဇာတ်ကောင်စကားပြော (Dialogue) များကို တိုက်ရိုက်မပြန်ဘဲ ဇာတ်လမ်းပြောပြသူ (3rd-Person Movie Recap Narrator) အသံဖြင့် "
            f"{lang_instruction} အချိတ်အဆက်မိမိ ပြန်လည်ရေးသားပေးပါ။ အပိုင်းတစ်ခုချင်းစီ၏ ကြာချိန်နှင့် ကိုက်ညီသော စာကြောင်းအရှည် ဖြစ်ရပါမည်။"
        )

    system_prompt = (
        "You are an expert Movie Recap Scriptwriter.\n"
        f"{style_rule}\n"
        "IMPORTANT: Respond ONLY with a valid JSON object (no markdown formatting, no extra explanation) in this exact structure:\n"
        '{"segments": [{"index": 0, "narration": "..."}], "tiktok_caption": "..."}\n'
        "Keep the exact same number of segments and matching index values."
    )

    user_content = json.dumps({"segments": segments}, ensure_ascii=False)

    preferred = model or DEFAULT_OPENROUTER_MODEL
    candidate_models = [preferred] + [m for m in FREE_MODEL_FALLBACKS if m != preferred]

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://recap-app.onrender.com",
        "X-Title": "AI Movie Recap Narrator",
    }

    raw_content = None
    last_error = ""

    async with httpx.AsyncClient(timeout=240.0) as client:
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
        raise RuntimeError(f"OpenRouter Free Models အားလုံး ခေတ္တပြည့်နေပါသည်: {last_error}")

    # JSON Parsing (Markdown code block ပါလာလျှင်လည်း သန့်စင်ဖတ်ယူသည်)
    json_match = re.search(r"\{.*\}", raw_content, flags=re.DOTALL)
    cleaned = json_match.group(0) if json_match else raw_content.strip()
    parsed = json.loads(cleaned)

    narr_map = {int(item["index"]): item["narration"] for item in parsed.get("segments", [])}
    for seg in segments:
        seg["narration"] = narr_map.get(seg["index"], seg["text"])

    caption_out = ""
    if want_hashtags:
        base_cap = parsed.get("tiktok_caption", "ဇာတ်လမ်းကောင်းလေးကို အဆုံးထိကြည့်ရှုပေးကြပါဦးနော် 🎬")
        tags = default_hashtags.strip()
        if add_fyp and "#fyp" not in tags.lower():
            tags = f"{tags} #fyp".strip()
        caption_out = f"{base_cap}\n{tags}".strip()

    return {"segments": segments, "tiktok_caption": caption_out}


# -------------------------------------------------------------------------
# STEP 6: ASS SUBTITLE BUILDER
# -------------------------------------------------------------------------
def build_ass_subtitles(
    caption_events: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    play_res_x: int,
    play_res_y: int,
    ass_path: Path,
):
    """အဆင့် ၆ (Timeline + captions) - User ရွေးထားသော ဒီဇိုင်းအတိုင်း .ass စာတန်းထိုးဖိုင် တည်ဆောက်ခြင်း"""
    font_name = cfg.get("font", "A13 WenYoe")
    font_size = int(cfg.get("text_size", 30))
    primary_col = hex_to_ass_color(cfg.get("text_color", "#FFFF00"), 0)
    outline_col = hex_to_ass_color(cfg.get("outline_color", "#000000"), 0)
    back_col = hex_to_ass_color("#000000", 110)
    outline_w = int(cfg.get("outline_width", 3))
    bold_flag = -1 if cfg.get("bold", True) else 0
    border_style = 3 if cfg.get("box_bg", False) else 1

    # Vertical Position (%) နှင့် Caption Width (%) တွက်ချက်ခြင်း
    vert_pct = int(cfg.get("vertical_position", 89))
    margin_v = max(10, int(play_res_y * (100 - vert_pct) / 100.0))
    width_pct = int(cfg.get("caption_width", 96))
    side_margin = max(10, int(play_res_x * (100 - width_pct) / 200.0))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_col},{primary_col},{outline_col},{back_col},{bold_flag},0,0,0,100,100,0,0,{border_style},{outline_w},0,2,{side_margin},{side_margin},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for ev in caption_events:
        st = format_ass_time(ev["start"])
        en = format_ass_time(ev["end"])
        txt = ev["text"].replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{st},{en},Default,,0,0,0,,{txt}")

    ass_path.write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------------------
# MAIN 7-STEP PIPELINE WORKER
# -------------------------------------------------------------------------
async def process_recap_pipeline(
    job_id: str, video_path: Path, cfg: Dict[str, Any], logo_path: Optional[Path]
):
    """ပုံပါ အဆင့် (၇) ဆင့်လုံးကို အစမှအဆုံး အလိုအလျောက် လုပ်ဆောင်ပေးမည့် Main Pipeline"""
    job_dir = DATA_DIR / job_id
    try:
        cleanup_old_jobs()

        aai_key = cfg.get("assemblyai_key") or DEFAULT_ASSEMBLYAI_KEY
        or_key = cfg.get("openrouter_key") or DEFAULT_OPENROUTER_KEY
        or_model = cfg.get("openrouter_model") or DEFAULT_OPENROUTER_MODEL

        if not aai_key or not or_key:
            raise RuntimeError("AssemblyAI နှင့် OpenRouter API Key များကို 'Own key' တွင် ထည့်သွင်းပေးပါ။")

        orig_w, orig_h = await asyncio.to_thread(get_video_dimensions, video_path)

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
        # 2. Upload & 3. Transcribe (AssemblyAI Free Tier)
        # ---------------------------------------------------------
        sentences = await assemblyai_upload_and_transcribe(
            extracted_audio, aai_key, cfg.get("source_language", "Auto-detect"), job_id
        )

        # ---------------------------------------------------------
        # 4. Translate / AI Recap Script (OpenRouter Free Tier)
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
        )
        segments = recap_data["segments"]
        JOBS[job_id]["tiktok_caption"] = recap_data["tiktok_caption"]

        # ---------------------------------------------------------
        # 5. Voice (Edge-TTS Narrator အသံထုတ်ခြင်း)
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
        # 6. Timeline + captions (မူရင်းရုပ်ရှင်အသံကို လုံးဝဖျောက်ပြီး Video Speed နှင့် Captions ချိန်ညှိခြင်း)
        # ---------------------------------------------------------
        update_job_step(job_id, "timeline_captions", 70)
        words_per_cap = max(2, int(cfg.get("words_per_caption", 9)))
        caption_events = []
        concat_v_list = []
        concat_a_list = []
        timeline_cursor = 0.0

        for seg in segments:
            orig_dur = max(0.5, seg["end"] - seg["start"])
            tts_dur = max(0.5, seg["tts_duration"])
            # မူရင်း Video အပိုင်းကို Narrator အသံအရှည်နှင့် ကွက်တိဖြစ်အောင် Speed ညှိပြီး မူရင်းအသံကို လုံးဝဖျောက် (-an) သည်
            pts_factor = tts_dur / orig_dur
            v_seg_path = job_dir / f"vseg_{seg['index']:04d}.mp4"
            await asyncio.to_thread(
                run_cmd,
                [
                    "ffmpeg", "-y",
                    "-ss", str(seg["start"]), "-to", str(seg["end"]),
                    "-i", str(video_path),
                    "-an",  # မူရင်းရုပ်ရှင်အသံ လုံးဝဖျောက်ပစ်ခြင်း
                    "-vf", f"setpts={pts_factor:.4f}*PTS",
                    "-r", "24", "-c:v", "libx264", "-preset", "ultrafast",
                    str(v_seg_path),
                ],
            )
            concat_v_list.append(v_seg_path)
            concat_a_list.append(seg["tts_file"])

            # Words per caption အလိုက် စာတန်းထိုး အချိန်ပိုင်းခြားခြင်း
            words = seg["narration"].split()
            if not words:
                words = [seg["narration"]]
            chunks = [" ".join(words[i : i + words_per_cap]) for i in range(0, len(words), words_per_cap)]
            chunk_dur = tts_dur / len(chunks)
            for idx, ch in enumerate(chunks):
                caption_events.append({
                    "start": timeline_cursor + idx * chunk_dur,
                    "end": timeline_cursor + (idx + 1) * chunk_dur,
                    "text": ch,
                })
            timeline_cursor += tts_dur

        # Video နှင့် Narrator Audio အပိုင်းများကို ဆက်ခြင်း
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

        # Output Resolution တွက်ချက်ခြင်း
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
        # 7. Render video (FFmpeg Visual Effects, Blur, Captions, Watermark, Round Logo)
        # ---------------------------------------------------------
        update_job_step(job_id, "render_video", 85)
        vf_filters = []

        # Random 4px crop
        if cfg.get("random_4px_crop", False):
            vf_filters.append("crop=iw-8:ih-8:4:4")

        # Zoom (%)
        zoom_pct = int(cfg.get("zoom", 100))
        if zoom_pct > 100:
            zf = zoom_pct / 100.0
            vf_filters.append(f"scale=iw*{zf}:ih*{zf},crop=iw/{zf}:ih/{zf}")

        # Mirror (flip left/right)
        if cfg.get("mirror", False):
            vf_filters.append("hflip")

        # Ratio & Fit Mode: Crop (fill) vs Pad (blur bg)
        fit_mode = cfg.get("fit_mode", "crop")
        if ratio != "Original":
            if fit_mode == "pad":
                # Pad (blur bg): နောက်ခံကို Blur ခင်းပြီး အပေါ်မှ မူရင်း Video တင်ခြင်း
                vf_filters.append(
                    f"split[original][copy];"
                    f"[copy]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},boxblur=20:10[blurred];"
                    f"[original]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease[scaled];"
                    f"[blurred][scaled]overlay=(W-w)/2:(H-h)/2"
                )
            else:
                # Crop (fill)
                vf_filters.append(
                    f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h}"
                )
        else:
            vf_filters.append(f"scale={out_w}:{out_h}")

        # Blur boxes သို့မဟုတ် Full band (cover original subtitles)
        if cfg.get("full_band", False) or int(cfg.get("blur_boxes", 0)) > 0:
            vf_filters.append("drawbox=x=0:y=ih*0.78:w=iw:h=ih*0.16:color=black@0.75:t=fill")

        # TikTok HD (1080p) Sharpening
        if hd_1080:
            vf_filters.append("unsharp=5:5:0.8:3:3:0.4")

        # Captions (.ass)
        if cfg.get("captions_on", True):
            escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
            vf_filters.append(f"ass='{escaped_ass}'")

        # Watermark text
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

        # Round Logo ပါဝင်ပါက Overlay ထပ်တင်ခြင်း
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

        # Process အားလုံး ပြီးမြောက်ခြင်း
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
    """Voice Tab ထဲက Play (▶) ခလုတ်နှိပ်လျှင် နမူနာအသံ ဖွင့်ပြရန် Endpoint"""
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
    """▶ Start နှိပ်လိုက်သည့်အခါ Job စတင်ပြီး အဆင့် (၇) ဆင့် Progress ပြန်ပေးမည့် Endpoint"""
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
    """Frontend မှ ၂ စက္ကန့်တစ်ကြိမ် လှမ်းမေးမည့် Progress Status Endpoint"""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    elapsed = int(time.time() - job["started_at"])
    prog = max(1, job["progress"])
    eta = int(elapsed * (100 - prog) / prog) if prog < 100 else 0
    return {**job, "elapsed_sec": elapsed, "eta_sec": eta}


@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    """Render ပြီးသွားသော Final Movie Recap Video ကို Download ဆွဲရန် Endpoint"""
    final_video = DATA_DIR / job_id / "final_recap.mp4"
    if not final_video.exists():
        raise HTTPException(status_code=404, detail="Video not ready")
    return FileResponse(final_video, media_type="video/mp4", filename=f"recap_{job_id}.mp4")


# Frontend UI (static/index.html) ကို Serve လုပ်ခြင်း
Path("static").mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
