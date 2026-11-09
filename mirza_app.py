# -*- coding: utf-8 -*-
# MirZa Tools — self-contained Gradio app (no external repo dependency)
import os, re, uuid
import numpy as np
import gradio as gr
import scipy.io.wavfile as wavfile
import torch

# keep the Colab console clean (hide benign deprecation / logits-processor warnings)
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
try:
    from transformers.utils import logging as _hflog
    _hflog.set_verbosity_error()
except Exception:
    pass

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# works on both Colab (/content) and Kaggle (/kaggle/working)
_BASE_DIR = next((b for b in ("/kaggle/working", "/content") if os.path.isdir(b)), os.getcwd())
OUT_DIR = os.path.join(_BASE_DIR, "mirza_audio")
os.makedirs(OUT_DIR, exist_ok=True)

WA_NUMBER = "0321-2018655"
WA_LINK = "https://wa.me/923212018655"
WEB_URL = "https://mirzatols.blogspot.com/p/mirza-voice-ai.html"
WEB_WA_LINK = "https://wa.me/923212018655?text=" + "Assalam%20o%20alaikum!%20Main%20Colab%20tool%20use%20karta%20hoon%20aur%20Premium%20Web%20App%20ka%20existing-user%20offer%20lena%20chahta%20hoon."
WA_CANVA_LINK = "https://wa.me/923212018655?text=" + "Assalam%20o%20alaikum!%20Mujhe%20Canva%20Pro%20Lifetime%20Access%20chahiye.%20Price%20kya%20hai%3F"
WA_ADS_LINK = "https://wa.me/923212018655?text=" + "Assalam%20o%20alaikum!%20Mujhe%20AdSense%20Approved%20Service%20chahiye."
CHANNEL_URL = "https://whatsapp.com/channel/0029Vb8grPM1SWssTWxLZm3L"

EVENT_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]",
    "[question-ah]", "[question-oh]", "[surprise-ah]", "[surprise-oh]",
    "[surprise-wa]", "[dissatisfaction-hnn]",
]

# Voice Design presets (free + safe — no reference clips needed).
# Each maps a display name to an OmniVoice "instruct" attribute string.
VOICE_PRESETS = {
    "🎲  Default voice": "",
    # --- Female ---
    "👩  Young Woman — Friendly": "Female, Young Adult, Moderate Pitch",
    "😊  Young Woman — Cheerful": "Female, Young Adult, High Pitch",
    "📖  Female Storyteller — Warm": "Female, Middle-aged, Moderate Pitch",
    "👧  Teen Girl": "Female, Teenager, High Pitch",
    "👵  Elderly Woman": "Female, Elderly, Low Pitch",
    "🇬🇧  British Woman": "Female, Young Adult, British Accent",
    "🤫  Soft Whisper — Female": "Female, Young Adult, Whisper",
    # --- Male ---
    "👨  Young Man — Casual": "Male, Young Adult, Moderate Pitch",
    "🎙️  Deep Narrator — Male": "Male, Middle-aged, Very Low Pitch",
    "📖  Male Storyteller": "Male, Middle-aged, Low Pitch",
    "👦  Teen Boy": "Male, Teenager, High Pitch",
    "👴  Elderly Man": "Male, Elderly, Low Pitch",
    "🇬🇧  British Man": "Male, Young Adult, British Accent",
    "🇺🇸  American Man": "Male, Young Adult, American Accent",
    "🇮🇳  Indian Man": "Male, Young Adult, Indian Accent",
    "🤫  Soft Whisper — Male": "Male, Middle-aged, Whisper",
}

print("Loading MirZa Tools model (k2-fsa/OmniVoice) to cuda ...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda",
    dtype=torch.float16,
    load_asr=False,   # extra ASR load nahi -> tez loading + kam GPU memory.
)                     # Clone ke liye reference text faster-whisper se auto ban jata hai (neeche).
SR = model.sampling_rate
print("Model loaded successfully!")

ALL_LANGS = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

_ASR = None  # lazy-loaded Whisper (transformers) for subtitles + auto ref-text


def _get_asr():
    """Load a Whisper ASR pipeline once, via transformers (the stack the tool already
    uses). This avoids faster-whisper's 'metadata_errors' clash with transformers 5.x."""
    global _ASR
    if _ASR is None:
        from transformers import pipeline
        try:
            _ASR = pipeline("automatic-speech-recognition",
                            model="openai/whisper-base", device=0)
        except Exception:
            _ASR = pipeline("automatic-speech-recognition",
                            model="openai/whisper-base", device=-1)
    return _ASR


def _to_mp3(wav_path):
    try:
        from pydub import AudioSegment
        mp3_path = wav_path.replace(".wav", ".mp3")
        AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3", bitrate="320k")
        return mp3_path
    except Exception:
        return wav_path


def _fname(text):
    # Branded, clean download name -> MirZa-Tools-xxxxxxxx.wav
    return "%s/MirZa-Tools-%s.wav" % (OUT_DIR, uuid.uuid4().hex[:8])


def _ts(sec):
    # Hardened: never raises, even on None / bad values (the old subtitle TypeError).
    try:
        sec = float(sec)
    except (TypeError, ValueError):
        sec = 0.0
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    if ms == 1000:
        ms = 0; s += 1
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def _make_srt(wav_path):
    """Transcribe the generated audio -> SRT with timestamps (best-effort)."""
    asr = _get_asr()
    out = asr(wav_path, return_timestamps=True, chunk_length_s=30)
    chunks = out.get("chunks") if isinstance(out, dict) else None
    if not chunks:
        full = (out.get("text") if isinstance(out, dict) else "") or ""
        chunks = [{"text": full, "timestamp": (0.0, max(1.0, len(full) / 15.0))}] if full.strip() else []
    lines, i, clock = [], 0, 0.0
    for seg in chunks:
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        ts = seg.get("timestamp") or (None, None)
        start = ts[0] if len(ts) > 0 else None
        end = ts[1] if len(ts) > 1 else None
        start = clock if start is None else start
        try:
            start = float(start)
        except (TypeError, ValueError):
            start = clock
        if end is None:
            end = start + max(1.0, len(txt) / 15.0)
        try:
            end = float(end)
        except (TypeError, ValueError):
            end = start + 1.5
        if end <= start:
            end = start + 0.8
        clock = end
        i += 1
        lines.append(str(i))
        lines.append("%s --> %s" % (_ts(start), _ts(end)))
        lines.append(txt)
        lines.append("")
    srt_path = wav_path.replace(".wav", ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) if lines else "1\n00:00:00,000 --> 00:00:01,000\n \n")
    return srt_path


def _transcribe(audio_path):
    """Reference audio -> text (Whisper via transformers). Used for clone when the
    user gives no reference text."""
    try:
        asr = _get_asr()
        out = asr(audio_path)
        txt = (out.get("text") or "").strip() if isinstance(out, dict) else ""
        return txt or None
    except Exception as e:
        print("ref transcription failed:", e)
        return None


def _split_text(text, max_chars=240):
    """Split long text into sentence-aware chunks so we can show a real % bar."""
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?。！？؟۔])\s+|\n+', text)
    chunks, cur = [], ""
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= max_chars:
                cur = p
            else:
                for j in range(0, len(p), max_chars):
                    chunks.append(p[j:j + max_chars])
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _run(text, language, ref_audio, ref_text, want_srt,
         speed, duration, steps, denoise, cfg, preprocess, postprocess, mode, instruct=None, progress=None):
    if not text or not text.strip():
        return None, None, "Please enter some text to synthesize."
    lang = language if (language and language != "Auto") else None
    gen_cfg = OmniVoiceGenerationConfig(
        num_step=int(steps or 32),
        guidance_scale=float(cfg) if cfg is not None else 2.0,
        denoise=bool(denoise),
        preprocess_prompt=bool(preprocess),
        postprocess_output=bool(postprocess),
    )
    extra = {}
    if speed is not None and float(speed) != 1.0:
        extra["speed"] = float(speed)

    if mode == "clone":
        if not ref_audio:
            return None, None, "Please upload or record a reference voice (3-10 seconds)."
        rt = ref_text.strip() if (ref_text and ref_text.strip()) else None
        if not rt:
            if progress:
                progress(0.03, desc="Analyzing reference voice…")
            rt = _transcribe(ref_audio)   # ASR off -> faster-whisper se auto ref text
        try:
            extra["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=rt)
        except Exception as e:
            return None, None, "Reference audio error: %s" % e

    # Voice Design: a chosen character/voice preset (only when not cloning)
    if instruct and instruct.strip():
        extra["instruct"] = instruct.strip()

    # --- chunked generation with a real % progress bar ---
    label = "Cloning voice" if mode == "clone" else "Generating voice"
    chunks = _split_text(text)
    n = max(len(chunks), 1)
    single_dur = (n == 1 and duration is not None and float(duration) > 0)
    pieces = []
    for idx, ch in enumerate(chunks):
        if progress:
            progress(idx / n, desc="%s… %d%%  (part %d/%d)" % (label, int((idx / n) * 100), idx + 1, n))
        kw = dict(text=ch, language=lang, generation_config=gen_cfg)
        kw.update(extra)
        if single_dur:
            kw["duration"] = float(duration)
        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, None, "Error on part %d: %s: %s" % (idx + 1, type(e).__name__, e)
        wav = np.asarray(audio[0], dtype=np.float32)
        pieces.append(wav)
        if idx < n - 1:
            pieces.append(np.zeros(int(SR * 0.12), dtype=np.float32))

    if progress:
        progress(0.95, desc="Finalizing audio…")
    wave = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    waveform = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16)
    wav_path = _fname(text)
    wavfile.write(wav_path, SR, waveform)
    mp3_path = _to_mp3(wav_path)

    srt_path, note = None, ""
    if want_srt:
        if progress:
            progress(0.99, desc="Generating subtitles…")
        try:
            srt_path = _make_srt(wav_path)
        except Exception as e:
            note = " (subtitles unavailable: %s)" % type(e).__name__
    return mp3_path, srt_path, "Done! Your audio is ready below." + note


def _pack(mp3, srt, status):
    if not mp3:
        return None, gr.update(visible=False), gr.update(visible=False), "&#9888;&#65039; " + status
    srt_upd = gr.update(value=srt, visible=True) if srt else gr.update(visible=False)
    return mp3, gr.update(value=mp3, visible=True), srt_upd, "&#9989; " + status


def tts_fn(text, language, voice, want_srt, sp, du, ns, dn, gs, pp, po, progress=gr.Progress()):
    instruct = VOICE_PRESETS.get(voice, "")
    mp3, srt, status = _run(text, language, None, None, want_srt, sp, du, ns, dn, gs, pp, po, "tts", instruct=instruct, progress=progress)
    return _pack(mp3, srt, status)


def clone_fn(text, language, ref_audio, ref_text, want_srt, sp, du, ns, dn, gs, pp, po, progress=gr.Progress()):
    mp3, srt, status = _run(text, language, ref_audio, ref_text, want_srt, sp, du, ns, dn, gs, pp, po, "clone", progress=progress)
    return _pack(mp3, srt, status)


theme = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#060d1a", body_background_fill_dark="#060d1a",
    background_fill_primary="#0a1628", background_fill_primary_dark="#0a1628",
    background_fill_secondary="#0d1f38", background_fill_secondary_dark="#0d1f38",
    block_background_fill="#0a1628", block_background_fill_dark="#0a1628",
    block_border_color="rgba(0,180,216,.18)", block_border_color_dark="rgba(0,180,216,.18)",
    block_label_text_color="#a9c6da", block_label_text_color_dark="#a9c6da",
    block_title_text_color="#e0f0ff", block_title_text_color_dark="#e0f0ff",
    button_primary_background_fill="linear-gradient(135deg,#0077b6,#00d4ff)",
    button_primary_background_fill_dark="linear-gradient(135deg,#0077b6,#00d4ff)",
    button_primary_background_fill_hover="linear-gradient(135deg,#0088cc,#33ddff)",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#0d1f38", button_secondary_background_fill_dark="#0d1f38",
    button_secondary_text_color="#00d4ff", button_secondary_text_color_dark="#00d4ff",
    input_background_fill="#0d1f38", input_background_fill_dark="#0d1f38",
    input_border_color="rgba(0,180,216,.18)", input_border_color_dark="rgba(0,180,216,.18)",
    input_border_color_focus="#00d4ff", input_border_color_focus_dark="#00d4ff",
    body_text_color="#e6f1fb", body_text_color_dark="#e6f1fb",
    body_text_color_subdued="#6f93ab", body_text_color_subdued_dark="#6f93ab",
    border_color_primary="rgba(0,180,216,.18)", border_color_primary_dark="rgba(0,180,216,.18)",
)

CSS = """
footer{display:none !important}
/* let the Our Services dropdown escape its Gradio HTML block (so all items show) */
#mz-topbar, #mz-topbar>.html-container{overflow:visible !important}
.block:has(.mz-nav){overflow:visible !important}
.gradio-container{max-width:1180px !important; margin:0 auto !important; padding:0 20px 40px !important;}
.mz-card{background:#0a1628 !important; border:1px solid rgba(0,180,216,.18) !important; border-radius:16px !important; padding:22px !important;}
#mz_gen button, #mz_gen_c button{font-size:1.05rem !important; font-weight:700 !important; padding:15px !important; letter-spacing:.2px;}
.tab-nav{border-bottom:1px solid rgba(0,180,216,.18) !important;}
.tab-nav button{font-size:1.02rem !important; font-weight:600 !important;}
.tab-nav button.selected{color:#00d4ff !important;}
#mz_dl button, #mz_dl_c button{background:linear-gradient(135deg,#0a8f5b,#0fbf78) !important; color:#fff !important; border:none !important; font-weight:700 !important;}
#mz_srt button, #mz_srt_c button{background:#0d1f38 !important; color:#00d4ff !important; border:1px solid rgba(0,180,216,.35) !important; font-weight:700 !important;}
.mz-tags button{min-width:auto !important; padding:5px 11px !important; font-size:.8rem !important; background:#0d1f38 !important; color:#00d4ff !important; border:1px solid rgba(0,180,216,.25) !important;}
.mz-tags button:hover{background:#00d4ff !important; color:#04121f !important;}
label span{font-weight:600 !important;}
/* ===== PRO NAV + DROPDOWN + POPUPS ===== */
.mz-nav{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:12px 18px;margin:8px 0 2px;background:rgba(10,22,40,.7);border:1px solid rgba(0,180,216,.18);border-radius:14px;position:relative;z-index:30}
.mz-brand{font-size:1.25rem;font-weight:800;display:flex;align-items:center;gap:8px;white-space:nowrap}
.mz-brand .g{background:linear-gradient(135deg,#00d4ff,#0077b6);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.mz-menu{display:flex;align-items:center;gap:6px;flex-wrap:wrap;justify-content:flex-end}
.mz-mi{color:#a9c6da;text-decoration:none;font-weight:600;font-size:.92rem;padding:9px 13px;border-radius:9px;background:none;border:none;cursor:pointer;font-family:inherit}
.mz-mi:hover{color:#00d4ff;background:rgba(0,180,216,.08)}
.mz-dd{position:relative}
.mz-ddbtn{display:inline-flex;align-items:center;gap:7px}
.mz-car{color:#00d4ff;font-size:.8rem;transition:transform .2s;display:inline-block}
.mz-dd.open .mz-car{transform:rotate(180deg)}
.mz-ddmenu{display:none;position:absolute;top:calc(100% + 8px);right:0;width:290px;background:#0a1628;border:1px solid rgba(0,180,216,.4);border-radius:14px;padding:8px;box-shadow:0 20px 50px rgba(0,0,0,.55);z-index:40}
.mz-dd.open .mz-ddmenu{display:block}
.mz-ddi{display:flex;align-items:center;gap:12px;padding:11px 12px;border-radius:10px;text-decoration:none;cursor:pointer}
.mz-ddi:hover{background:#0d1f38}
.mz-ddi .mz-di{font-size:1.4rem;width:40px;height:40px;display:flex;align-items:center;justify-content:center;border-radius:10px;background:#0d1f38;flex-shrink:0}
.mz-ddi b{display:block;font-size:.9rem;color:#e6f1fb}
.mz-ddi small{display:block;font-size:.75rem;color:#6f93ab;margin-top:2px}
/* ===== Pricing dropdown ===== */
.mz-priceddmenu{width:290px !important}
.mz-pricehdr{font-size:.68rem;font-weight:800;letter-spacing:.5px;color:#f59e0b;padding:7px 11px 9px}
.mz-priceitem{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:11px 12px;margin:2px 0;border-radius:11px;text-decoration:none;cursor:pointer;border:1px solid transparent}
.mz-priceitem:hover{background:#0d1f38;border-color:rgba(0,180,216,.3)}
.mz-priceitem.best{background:linear-gradient(170deg,rgba(0,180,216,.12),rgba(10,22,40,.25));border:1px solid rgba(0,212,255,.42)}
.mz-priceleft b{display:block;font-size:.95rem;color:#e6f1fb;font-weight:700}
.mz-priceleft small{display:block;font-size:.72rem;color:#6f93ab;margin-top:2px}
.mz-priceright{display:flex;flex-direction:column;align-items:flex-end;gap:3px}
.mz-priceval{font-size:1.1rem;font-weight:800;color:#fff}
.mz-pricebadge{font-size:.55rem;font-weight:800;letter-spacing:.4px;color:#04121f;background:linear-gradient(135deg,#00d4ff,#0077b6);padding:2px 7px;border-radius:100px;white-space:nowrap}
.mz-pricenote{text-align:center;color:#6f93ab;font-size:.67rem;padding:9px 6px 4px;line-height:1.4}
.mz-wabtn{display:inline-flex;align-items:center;gap:7px;background:linear-gradient(135deg,#25d366,#128c7e);color:#fff !important;text-decoration:none;font-weight:700;font-size:.88rem;padding:10px 16px;border-radius:10px;white-space:nowrap}
.mz-hero{text-align:center;padding:24px 16px 4px}
.mz-hero h1{font-size:1.85rem;font-weight:800;color:#e6f1fb;margin:0}
.mz-hero h1 .g{background:linear-gradient(135deg,#00d4ff,#0077b6);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.mz-hero p{color:#a9c6da;margin-top:6px;font-size:.98rem}
.mz-badges{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:14px}
.mz-badge{font-size:.78rem;color:#00d4ff;background:rgba(0,180,216,.08);border:1px solid rgba(0,180,216,.2);padding:5px 13px;border-radius:100px}
.mz-ov{position:fixed;inset:0;background:rgba(3,8,16,.74);backdrop-filter:blur(5px);-webkit-backdrop-filter:blur(5px);display:none;align-items:center;justify-content:center;padding:20px;z-index:99999}
.mz-ov.show{display:flex}
.mz-modal{position:relative;width:410px;max-width:100%;background:linear-gradient(170deg,#0c1d33,#0a1628);border:1px solid rgba(0,180,216,.4);border-radius:20px;padding:30px 28px 26px;box-shadow:0 30px 80px rgba(0,0,0,.6);font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif}
.mz-x{position:absolute;top:14px;right:14px;width:32px;height:32px;border-radius:50%;border:1px solid rgba(0,180,216,.2);background:#0d1f38;color:#a9c6da;font-size:1.05rem;cursor:pointer;display:flex;align-items:center;justify-content:center}
.mz-x:hover{border-color:#00d4ff;color:#00d4ff}
.mz-ico{width:60px;height:60px;border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:1.9rem;margin-bottom:14px}
.mz-tag{display:inline-block;font-size:.68rem;font-weight:800;letter-spacing:.6px;color:#fff;padding:4px 12px;border-radius:100px;margin-bottom:10px}
.mz-modal h2{font-size:1.4rem;font-weight:800;line-height:1.2;color:#e6f1fb;margin:0}
.mz-modal h2 b{background:linear-gradient(135deg,#a98bff,#00d4ff);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.mz-modal p.mz-sub{color:#a9c6da;font-size:.9rem;margin:13px 0 16px;line-height:1.6}
.mz-feats{list-style:none;display:grid;gap:9px;margin:4px 0 20px;padding:0}
.mz-feats li{display:flex;gap:10px;font-size:.9rem;color:#e6f1fb}
.mz-feats .mz-ck{color:#25d366;font-weight:800;flex-shrink:0}
.mz-cta{display:flex;align-items:center;justify-content:center;gap:10px;width:100%;padding:13px;text-decoration:none;border-radius:12px;font-weight:700;line-height:1.25;color:#fff !important;box-sizing:border-box}
.mz-cta.grn{background:linear-gradient(135deg,#25d366,#128c7e)}
.mz-cta.blu{background:linear-gradient(135deg,#0077b6,#00d4ff)}
.mz-cta .mz-m{font-size:1rem}
.mz-cta .mz-n{font-size:.82rem;font-weight:600;opacity:.95}
.mz-later{display:block;width:100%;text-align:center;margin-top:10px;background:none;border:none;color:#6f93ab;font-size:.86rem;font-weight:600;cursor:pointer;font-family:inherit;padding:6px}
.mz-note{text-align:center;color:#6f93ab;font-size:.74rem;margin-top:12px;line-height:1.5}
.mz-offer{background:linear-gradient(135deg,rgba(245,158,11,.16),rgba(10,22,40,.4));border:1px dashed rgba(245,158,11,.6);border-radius:11px;padding:11px 14px;margin:2px 0 16px;font-size:.86rem;color:#ffe9c2;line-height:1.5}
@media(max-width:640px){.mz-menu{gap:2px}.mz-mi{padding:8px 8px;font-size:.82rem}.mz-ddmenu{width:258px}.mz-brand{font-size:1.05rem}}
"""

HEADER = (
    "<div style='text-align:center;padding:22px 10px 6px;'>"
    "<div style='font-family:system-ui,sans-serif;font-size:2.6em;font-weight:800;letter-spacing:-.02em;"
    "background:linear-gradient(135deg,#00d4ff,#0077b6);-webkit-background-clip:text;"
    "background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;'>&#127908; MirZa Tools</div>"
    "<div style='color:#a9c6da;font-size:1.05em;font-family:system-ui,sans-serif;margin-top:4px;'>"
    "Advanced AI Voice Studio &mdash; Voice Cloning &amp; Text-to-Speech in 600+ Languages</div>"
    "<div style='display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:14px;'>"
    "<span style='font-size:.8rem;color:#00d4ff;background:rgba(0,180,216,.08);border:1px solid rgba(0,180,216,.2);padding:5px 13px;border-radius:100px;'>&#127760; 600+ Languages</span>"
    "<span style='font-size:.8rem;color:#00d4ff;background:rgba(0,180,216,.08);border:1px solid rgba(0,180,216,.2);padding:5px 13px;border-radius:100px;'>&#127917; Voice Cloning</span>"
    "<span style='font-size:.8rem;color:#00d4ff;background:rgba(0,180,216,.08);border:1px solid rgba(0,180,216,.2);padding:5px 13px;border-radius:100px;'>&#128190; 320 kbps MP3</span>"
    "<span style='font-size:.8rem;color:#00d4ff;background:rgba(0,180,216,.08);border:1px solid rgba(0,180,216,.2);padding:5px 13px;border-radius:100px;'>&#128221; SRT Subtitles</span>"
    "</div></div>"
)

# Promotional notification for our second service (self-served ad)
BANNER = (
    "<div style='background:linear-gradient(135deg,rgba(0,119,182,.20),rgba(10,22,40,.75));"
    "border:1px solid rgba(0,200,240,.4);border-left:4px solid #f59e0b;border-radius:14px;"
    "padding:18px 24px;margin:12px auto 6px;position:relative;'>"
    "<span style='position:absolute;top:14px;right:16px;background:linear-gradient(135deg,#f59e0b,#f97316);"
    "color:#fff;font-size:.66rem;font-weight:800;letter-spacing:.6px;padding:4px 11px;border-radius:100px;"
    "font-family:system-ui,sans-serif;'>SPECIAL SERVICE</span>"
    "<div style='display:flex;align-items:center;gap:18px;flex-wrap:wrap;'>"
    "<div style='flex:1;min-width:270px;'>"
    "<div style='font-family:system-ui,sans-serif;font-size:1.3em;font-weight:800;color:#ffffff;'>"
    "&#128640; AdSense Approved Service &mdash; Available!</div>"
    "<div style='color:#bfe6f5;font-size:.9rem;margin-top:8px;line-height:1.7;font-family:system-ui,sans-serif;'>"
    "&#9989; 100% Organic &amp; Safe Method &nbsp;&bull;&nbsp; &#9989; AdSense Approved (Proper Process) &nbsp;&bull;&nbsp; &#9989; Safe &amp; Secure &mdash; No Risk<br>"
    "&#9989; Fast &amp; Reliable Updates &nbsp;&bull;&nbsp; &#9989; Full Support at Every Step</div></div>"
    "<a href='" + WA_LINK + "' target='_blank' style='background:linear-gradient(135deg,#25d366,#128c7e);"
    "color:#fff;text-decoration:none;padding:13px 26px;border-radius:10px;font-weight:700;font-size:1rem;"
    "font-family:system-ui,sans-serif;box-shadow:0 4px 16px rgba(37,211,102,.35);white-space:nowrap;text-align:center;'>"
    "&#128172; Get it on WhatsApp<br><span style='font-size:.86em;font-weight:600;'>" + WA_NUMBER + "</span></a>"
    "</div></div>"
)

# Premium Web App promo (upgrade path for existing Colab users)
WEB_PROMO = (
    "<div style='background:linear-gradient(135deg,rgba(0,119,182,.28),rgba(10,22,40,.85));"
    "border:1px solid rgba(0,200,240,.55);border-left:4px solid #00d4ff;border-radius:14px;"
    "padding:18px 24px;margin:8px auto 6px;position:relative;'>"
    "<span style='position:absolute;top:14px;right:16px;background:linear-gradient(135deg,#00d4ff,#0077b6);"
    "color:#fff;font-size:.66rem;font-weight:800;letter-spacing:.6px;padding:4px 11px;border-radius:100px;"
    "font-family:system-ui,sans-serif;'>&#10024; PREMIUM WEB APP</span>"
    # headline + subtext
    "<div style='font-family:system-ui,sans-serif;font-size:1.3em;font-weight:800;color:#ffffff;padding-right:120px;'>"
    "&#128640; Now Live &mdash; MirZa Voice AI Premium Web App</div>"
    "<div style='color:#bfe6f5;font-size:.9rem;margin-top:8px;line-height:1.7;font-family:system-ui,sans-serif;'>"
    "No Colab, no setup &mdash; sign in on any <b>browser or phone</b> and create instantly. Faster, always-on, same studio quality.</div>"
    # benefit chips
    "<div style='display:flex;gap:7px;flex-wrap:wrap;margin-top:12px;font-family:system-ui,sans-serif;'>"
    "<span style='font-size:.76rem;color:#00d4ff;background:rgba(0,180,216,.10);border:1px solid rgba(0,180,216,.25);padding:4px 11px;border-radius:100px;'>&#9889; Instant</span>"
    "<span style='font-size:.76rem;color:#00d4ff;background:rgba(0,180,216,.10);border:1px solid rgba(0,180,216,.25);padding:4px 11px;border-radius:100px;'>&#128241; Works on phone</span>"
    "<span style='font-size:.76rem;color:#00d4ff;background:rgba(0,180,216,.10);border:1px solid rgba(0,180,216,.25);padding:4px 11px;border-radius:100px;'>&#127917; Voice cloning</span>"
    "<span style='font-size:.76rem;color:#00d4ff;background:rgba(0,180,216,.10);border:1px solid rgba(0,180,216,.25);padding:4px 11px;border-radius:100px;'>&#127760; 600+ languages</span>"
    "<span style='font-size:.76rem;color:#00d4ff;background:rgba(0,180,216,.10);border:1px solid rgba(0,180,216,.25);padding:4px 11px;border-radius:100px;'>&#128221; SRT subtitles</span>"
    "</div>"
    # offer highlight box
    "<div style='background:linear-gradient(135deg,rgba(245,158,11,.16),rgba(10,22,40,.4));"
    "border:1px dashed rgba(245,158,11,.6);border-radius:11px;padding:12px 16px;margin-top:14px;"
    "font-family:system-ui,sans-serif;'>"
    "<span style='font-size:.72rem;font-weight:800;color:#fbbf24;letter-spacing:.5px;'>&#127873; EXISTING COLAB USERS &mdash; BONUS OFFER</span>"
    "<div style='color:#ffe9c2;font-size:.95rem;margin-top:5px;line-height:1.6;'>"
    "Get <b style='color:#fff;'>50,000 bonus credits FREE</b> &mdash; that&rsquo;s "
    "<b style='color:#fff;'>350,000 credits</b> "
    "<span style='color:#9db4c7;font-size:.85em;'>(instead of 300,000)</span> "
    "for <b style='color:#fff;'>Rs 2,500/month</b> &nbsp;&bull;&nbsp; "
    "<span style='color:#bfe6f5;'>Limited time</span></div></div>"
    # buttons
    "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:15px;'>"
    "<a href='" + WEB_URL + "' target='_blank' style='background:linear-gradient(135deg,#00d4ff,#0077b6);"
    "color:#fff;text-decoration:none;padding:13px 24px;border-radius:10px;font-weight:700;font-size:1rem;"
    "font-family:system-ui,sans-serif;box-shadow:0 4px 16px rgba(0,150,199,.35);'>"
    "&#127760; Open Premium Web App &rarr;</a>"
    "<a href='" + WEB_WA_LINK + "' target='_blank' style='background:linear-gradient(135deg,#25d366,#128c7e);"
    "color:#fff;text-decoration:none;padding:13px 24px;border-radius:10px;font-weight:700;font-size:1rem;"
    "font-family:system-ui,sans-serif;box-shadow:0 4px 16px rgba(37,211,102,.30);'>"
    "&#128172; Claim Offer on WhatsApp</a>"
    "</div></div>"
)

FOOTER = (
    "<div style='text-align:center;color:#6f93ab;font-size:.82rem;font-family:system-ui,sans-serif;"
    "padding:22px 10px 6px;margin-top:10px;border-top:1px solid rgba(0,180,216,.12);'>"
    "&copy; 2026 MirZa Tools &middot; Support: WhatsApp " + WA_NUMBER + " &middot; Use voices responsibly &mdash; only clone voices you have permission to use."
    "</div>"
)


ex = (
    "https://wa.me/923212018655?text="
    "Assalam%20o%20alaikum!%20Mujhe%20Canva%20Pro%20Lifetime%20Access%20chahiye.%20Price%20kya%20hai%3F"
)
# Canva Pro Lifetime offer — professional popup shown once when the UI loads.
POPUP_JS = r"""
() => {
  try {
    if (window.__mzCanvaShown) return;
    window.__mzCanvaShown = true;
    var css = `
      .cv-overlay{position:fixed;inset:0;background:rgba(3,8,16,.72);backdrop-filter:blur(5px);-webkit-backdrop-filter:blur(5px);display:flex;align-items:center;justify-content:center;padding:20px;z-index:99999;font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif}
      .cv-modal{position:relative;width:410px;max-width:100%;background:linear-gradient(170deg,#0c1d33,#0a1628);border:1px solid rgba(0,180,216,.4);border-radius:20px;padding:30px 28px 26px;box-shadow:0 30px 80px rgba(0,0,0,.6),0 0 50px rgba(0,180,216,.1);animation:cvpop .3s ease}
      @keyframes cvpop{from{transform:scale(.92) translateY(10px);opacity:0}to{transform:none;opacity:1}}
      .cv-x{position:absolute;top:14px;right:14px;width:32px;height:32px;border-radius:50%;border:1px solid rgba(0,180,216,.2);background:#0d1f38;color:#a9c6da;font-size:1.05rem;cursor:pointer;line-height:1;display:flex;align-items:center;justify-content:center}
      .cv-x:hover{border-color:#00d4ff;color:#00d4ff}
      .cv-ico{width:60px;height:60px;border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:1.9rem;background:linear-gradient(135deg,#7c5cff,#00d4ff);box-shadow:0 10px 26px rgba(124,92,255,.35);margin-bottom:14px}
      .cv-tag{display:inline-block;font-size:.68rem;font-weight:800;letter-spacing:.6px;color:#fff;background:linear-gradient(135deg,#f59e0b,#f97316);padding:4px 12px;border-radius:100px;margin-bottom:10px}
      .cv-modal h2{font-size:1.5rem;font-weight:800;line-height:1.2;color:#e6f1fb;margin:0}
      .cv-modal h2 b{background:linear-gradient(135deg,#a98bff,#00d4ff);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
      .cv-life{color:#a9c6da;font-size:.92rem;margin:16px 0 18px}
      .cv-life b{color:#25d366}
      .cv-feats{list-style:none;display:grid;gap:9px;margin:4px 0 22px;padding:0}
      .cv-feats li{display:flex;align-items:flex-start;gap:10px;font-size:.92rem;color:#e6f1fb}
      .cv-feats .ck{color:#25d366;font-weight:800;flex-shrink:0}
      .cv-wa{display:flex;align-items:center;justify-content:center;gap:10px;width:100%;padding:13px;text-decoration:none;background:linear-gradient(135deg,#25d366,#128c7e);color:#fff;border-radius:12px;font-weight:700;line-height:1.25;box-shadow:0 10px 26px rgba(37,211,102,.3);box-sizing:border-box}
      .cv-wa .m{font-size:1.02rem}
      .cv-wa .n{font-size:.82rem;font-weight:600;opacity:.95}
      .cv-later{display:block;width:100%;text-align:center;margin-top:10px;background:none;border:none;color:#6f93ab;font-size:.86rem;font-weight:600;cursor:pointer;font-family:inherit;padding:6px}
      .cv-later:hover{color:#a9c6da}
      .cv-note{text-align:center;color:#6f93ab;font-size:.74rem;margin-top:12px;line-height:1.5}
    `;
    var st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
    var o = document.createElement('div'); o.className = 'cv-overlay';
    o.innerHTML =
      '<div class="cv-modal">'
      + '<button class="cv-x" aria-label="Close">✕</button>'
      + '<div class="cv-ico">🎨</div>'
      + '<span class="cv-tag">⚡ LIMITED TIME OFFER</span>'
      + '<h2>Canva Pro — <b>100% Lifetime Access</b></h2>'
      + '<div class="cv-life"><b>Lifetime access</b> — one-time payment, no monthly fees ever. Saare Pro features unlock.</div>'
      + '<ul class="cv-feats">'
      + '<li><span class="ck">✓</span> All Pro features unlocked</li>'
      + '<li><span class="ck">✓</span> Premium templates &amp; elements</li>'
      + '<li><span class="ck">✓</span> Background Remover &amp; Magic AI tools</li>'
      + '<li><span class="ck">✓</span> Brand Kit &amp; 100+ GB cloud storage</li>'
      + '</ul>'
      + '<a class="cv-wa" href="__WA__" target="_blank" rel="noopener">💬 <span><span class="m">Get it on WhatsApp</span><br><span class="n">0321-2018655</span></span></a>'
      + '<button class="cv-later">Maybe later</button>'
      + '<div class="cv-note">One-time payment · 100% lifetime · All Pro features included</div>'
      + '</div>';
    document.body.appendChild(o);
    var close = function(){ try{ o.remove(); }catch(e){} };
    o.addEventListener('click', function(e){ if(e.target===o) close(); });
    o.querySelector('.cv-x').addEventListener('click', close);
    o.querySelector('.cv-later').addEventListener('click', close);
  } catch(e) { console.log('canva popup error', e); }
}
""".replace("__WA__", ex)


def _gen_settings():
    with gr.Accordion("Generation Settings (optional)", open=False):
        sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed", info="1.0 = normal. >1 faster, <1 slower.")
        du = gr.Number(value=0, label="Duration (seconds)", info="Fixed duration to override speed (0 = off).")
        ns = gr.Slider(4, 64, value=20, step=1, label="Inference Steps (Speed vs Quality)", info="20 = Fast (default). Zyada quality chahiye to 32+ karein (thora slow).")
        gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale (CFG)", info="Higher = closer to reference/prompt.")
        dn = gr.Checkbox(value=True, label="Denoise")
        pp = gr.Checkbox(value=True, label="Preprocess Prompt (silence removal + trims reference)")
        po = gr.Checkbox(value=True, label="Postprocess Output (removes long silences)")
    return sp, du, ns, dn, gs, pp, po


def _tag_row(target):
    gr.Markdown("**Add emotion / event tags:**")
    with gr.Row(elem_classes="mz-tags"):
        for tag in EVENT_TAGS:
            b = gr.Button(tag, size="sm")
            b.click(lambda cur, t=tag: ((cur + " " + t).strip() if cur else t),
                    inputs=[target], outputs=[target])


# ===== Professional nav header + hero + service popups =====
TOPBAR = '''
<div class="mz-nav">
  <div class="mz-brand">&#127897;&#65039; <span class="g">MirZa Tools</span></div>
  <div class="mz-menu">
    <button class="mz-mi" id="mz-home" type="button">Home</button>
    <div class="mz-dd" id="mz-dd-price">
      <button class="mz-mi mz-ddbtn" type="button">&#128176; Pricing <span class="mz-car">&#9662;</span></button>
      <div class="mz-ddmenu mz-priceddmenu">
        <div class="mz-pricehdr">&#9889; DISCOUNT &mdash; COLAB TOOL ACCESS</div>
        <a class="mz-priceitem best" href="__WA6M__" target="_blank" rel="noopener">
          <div class="mz-priceleft"><b>6 Months</b><small>only Rs 500 / month</small></div>
          <div class="mz-priceright"><span class="mz-pricebadge">&#9733; BEST</span><span class="mz-priceval">Rs 3,000</span></div>
        </a>
        <a class="mz-priceitem" href="__WA3M__" target="_blank" rel="noopener">
          <div class="mz-priceleft"><b>3 Months</b><small>Rs 667 / month</small></div>
          <div class="mz-priceright"><span class="mz-priceval">Rs 2,000</span></div>
        </a>
        <div class="mz-pricenote">Limited-time &middot; Pay via WhatsApp &middot; Instant activation</div>
      </div>
    </div>
    <div class="mz-dd" id="mz-dd">
      <button class="mz-mi mz-ddbtn" id="mz-ddbtn" type="button">Our Services <span class="mz-car">&#9662;</span></button>
      <div class="mz-ddmenu">
        <a class="mz-ddi" data-svc="cv"><span class="mz-di">&#127912;</span><div><b>Canva Pro &mdash; Lifetime</b><small>100% lifetime &middot; all Pro features</small></div></a>
        <a class="mz-ddi" data-svc="ads"><span class="mz-di">&#127919;</span><div><b>AdSense Approved Service</b><small>Safe &amp; proper process</small></div></a>
        <a class="mz-ddi" data-svc="web"><span class="mz-di">&#128640;</span><div><b>Premium Web App</b><small>No Colab &mdash; browser &amp; phone</small></div></a>
        <a class="mz-ddi" data-svc="chan"><span class="mz-di">&#128226;</span><div><b>WhatsApp Channel</b><small>Free tips, tools &amp; offers</small></div></a>
      </div>
    </div>
    <a class="mz-wabtn" href="__WALINK__" target="_blank" rel="noopener">&#128172; WhatsApp</a>
  </div>
</div>
<div class="mz-hero">
  <h1>Advanced <span class="g">AI Voice Studio</span></h1>
  <p>Voice Cloning &amp; Text-to-Speech in 600+ Languages</p>
  <div class="mz-badges">
    <span class="mz-badge">&#127760; 600+ Languages</span>
    <span class="mz-badge">&#127917; Voice Cloning</span>
    <span class="mz-badge">&#128190; 320 kbps MP3</span>
    <span class="mz-badge">&#128221; SRT Subtitles</span>
  </div>
</div>

<div class="mz-ov" id="mz-ov-chan">
  <div class="mz-modal">
    <button class="mz-x" data-close="chan" type="button">&#10005;</button>
    <div class="mz-ico" style="background:linear-gradient(135deg,#25d366,#128c7e);box-shadow:0 10px 26px rgba(37,211,102,.35)">&#128226;</div>
    <span class="mz-tag" style="background:linear-gradient(135deg,#f59e0b,#f97316)">&#127873; FREE UPDATES</span>
    <h2>Join our <b>WhatsApp Channel</b></h2>
    <p class="mz-sub">Hamare official channel se juriye aur <b style="color:#25d366">sabse pehle</b> paayiye:</p>
    <ul class="mz-feats">
      <li><span class="mz-ck">&#10003;</span> &#9889; Automation &amp; AI tips</li>
      <li><span class="mz-ck">&#10003;</span> &#127897;&#65039; Voice-tool updates &amp; new features</li>
      <li><span class="mz-ck">&#10003;</span> &#127873; Special offers &amp; free giveaways</li>
      <li><span class="mz-ck">&#10003;</span> &#128218; Tutorials &amp; quick guides</li>
    </ul>
    <a class="mz-cta grn" href="__CHURL__" target="_blank" rel="noopener"><span>&#128226; <span class="mz-m">Join the Channel</span></span></a>
    <button class="mz-later" data-close="chan" type="button">Maybe later</button>
    <div class="mz-note">Free to join &middot; No spam &middot; Leave anytime</div>
  </div>
</div>

<div class="mz-ov" id="mz-ov-cv">
  <div class="mz-modal">
    <button class="mz-x" data-close="cv" type="button">&#10005;</button>
    <div class="mz-ico" style="background:linear-gradient(135deg,#7c5cff,#00d4ff);box-shadow:0 10px 26px rgba(124,92,255,.35)">&#127912;</div>
    <span class="mz-tag" style="background:linear-gradient(135deg,#f59e0b,#f97316)">&#9889; LIMITED TIME OFFER</span>
    <h2>Canva Pro &mdash; <b>100% Lifetime Access</b></h2>
    <p class="mz-sub"><b style="color:#25d366">Lifetime access</b> &mdash; one-time payment, no monthly fees ever. Saare Pro features unlock.</p>
    <ul class="mz-feats">
      <li><span class="mz-ck">&#10003;</span> All Pro features unlocked</li>
      <li><span class="mz-ck">&#10003;</span> Premium templates &amp; Background Remover</li>
      <li><span class="mz-ck">&#10003;</span> Brand Kit &amp; 100+ GB storage</li>
    </ul>
    <a class="mz-cta grn" href="__WACANVA__" target="_blank" rel="noopener"><span>&#128172; <span class="mz-m">Get it on WhatsApp</span><br><span class="mz-n">0321-2018655</span></span></a>
    <button class="mz-later" data-close="cv" type="button">Maybe later</button>
  </div>
</div>

<div class="mz-ov" id="mz-ov-ads">
  <div class="mz-modal">
    <button class="mz-x" data-close="ads" type="button">&#10005;</button>
    <div class="mz-ico" style="background:linear-gradient(135deg,#f59e0b,#f97316);box-shadow:0 10px 26px rgba(245,158,11,.35)">&#127919;</div>
    <span class="mz-tag" style="background:linear-gradient(135deg,#f59e0b,#f97316)">&#11088; SPECIAL SERVICE</span>
    <h2>AdSense <b>Approved Service</b></h2>
    <p class="mz-sub">Apni website / channel ko <b style="color:#fbbf24">AdSense approve</b> karwaiye &mdash; 100% safe aur proper method se.</p>
    <ul class="mz-feats">
      <li><span class="mz-ck">&#10003;</span> 100% Organic &amp; Safe Method</li>
      <li><span class="mz-ck">&#10003;</span> AdSense Approved (Proper Process)</li>
      <li><span class="mz-ck">&#10003;</span> Safe &amp; Secure &mdash; No Risk</li>
      <li><span class="mz-ck">&#10003;</span> Full Support at Every Step</li>
    </ul>
    <a class="mz-cta grn" href="__WAADS__" target="_blank" rel="noopener"><span>&#128172; <span class="mz-m">Get it on WhatsApp</span><br><span class="mz-n">0321-2018655</span></span></a>
    <button class="mz-later" data-close="ads" type="button">Maybe later</button>
  </div>
</div>

<div class="mz-ov" id="mz-ov-web">
  <div class="mz-modal">
    <button class="mz-x" data-close="web" type="button">&#10005;</button>
    <div class="mz-ico" style="background:linear-gradient(135deg,#0077b6,#00d4ff);box-shadow:0 10px 26px rgba(0,150,199,.35)">&#128640;</div>
    <span class="mz-tag" style="background:linear-gradient(135deg,#0077b6,#00d4ff)">&#10024; PREMIUM WEB APP</span>
    <h2>MirZa Voice AI &mdash; <b>Premium Web App</b></h2>
    <p class="mz-sub">No Colab, no setup &mdash; kisi bhi <b>browser ya phone</b> pe sign in karo aur instant awaz banao. Tez, always-on, wahi studio quality.</p>
    <ul class="mz-feats">
      <li><span class="mz-ck">&#10003;</span> &#9889; Instant &middot; &#128241; Works on phone</li>
      <li><span class="mz-ck">&#10003;</span> &#127917; Voice cloning &middot; &#127760; 600+ languages</li>
      <li><span class="mz-ck">&#10003;</span> &#128221; SRT subtitles &middot; full quality controls</li>
    </ul>
    <div class="mz-offer">&#127873; <b style="color:#fbbf24">Existing users offer:</b> 50,000 bonus credits FREE &mdash; total <b style="color:#fff">350,000 credits</b> for <b style="color:#fff">Rs 2,500/month</b>.</div>
    <a class="mz-cta blu" href="__WEBURL__" target="_blank" rel="noopener"><span class="mz-m">&#127760; Open Premium Web App &rarr;</span></a>
    <a class="mz-cta grn" style="margin-top:10px" href="__WEBWA__" target="_blank" rel="noopener"><span>&#128172; <span class="mz-m">Ask on WhatsApp</span><br><span class="mz-n">0321-2018655</span></span></a>
    <button class="mz-later" data-close="web" type="button">Maybe later</button>
  </div>
</div>

'''.replace("__WALINK__", WA_LINK).replace("__WACANVA__", WA_CANVA_LINK).replace("__WAADS__", WA_ADS_LINK).replace("__WEBURL__", WEB_URL).replace("__WEBWA__", WEB_WA_LINK).replace("__CHURL__", CHANNEL_URL).replace("__WA6M__", "https://wa.me/923212018655?text=Assalam%20o%20alaikum!%20Mujhe%20Colab%20Tool%20ka%206-Month%20Discount%20Plan%20(Rs%203000)%20chahiye.").replace("__WA3M__", "https://wa.me/923212018655?text=Assalam%20o%20alaikum!%20Mujhe%20Colab%20Tool%20ka%203-Month%20Discount%20Plan%20(Rs%202000)%20chahiye.")


# Wire the nav dropdown + service popups, and auto-show the Channel popup once.
NAV_JS = r"""
() => {
  try {
    if (window.__mzNav) return;
    window.__mzNav = true;
    // support multiple dropdowns (Pricing + Our Services)
    function closeAllDd(){ document.querySelectorAll('.mz-dd').forEach(function(d){ d.classList.remove('open'); }); }
    document.querySelectorAll('.mz-dd').forEach(function(dd){
      var btn = dd.querySelector('.mz-ddbtn');
      if (btn) btn.addEventListener('click', function(e){ e.stopPropagation();
        document.querySelectorAll('.mz-dd').forEach(function(o){ if(o!==dd) o.classList.remove('open'); });
        dd.classList.toggle('open'); });
    });
    document.addEventListener('click', function(e){ document.querySelectorAll('.mz-dd').forEach(function(dd){ if(!dd.contains(e.target)) dd.classList.remove('open'); }); });
    var home = document.getElementById('mz-home');
    if (home) home.addEventListener('click', function(){ window.scrollTo({top:0,behavior:'smooth'}); closeAllDd(); });
    document.querySelectorAll('.mz-ddi').forEach(function(a){
      a.addEventListener('click', function(){ var k=a.getAttribute('data-svc'); var ov=document.getElementById('mz-ov-'+k); if(ov) ov.classList.add('show'); closeAllDd(); });
    });
    document.querySelectorAll('.mz-priceitem').forEach(function(a){
      a.addEventListener('click', function(){ closeAllDd(); });
    });
    document.querySelectorAll('[data-close]').forEach(function(b){
      b.addEventListener('click', function(){ var k=b.getAttribute('data-close'); var ov=document.getElementById('mz-ov-'+k); if(ov) ov.classList.remove('show'); });
    });
    document.querySelectorAll('.mz-ov').forEach(function(ov){
      ov.addEventListener('click', function(e){ if(e.target===ov) ov.classList.remove('show'); });
    });
    // auto-show WhatsApp Channel popup once, shortly after load
    var chan = document.getElementById('mz-ov-chan');
    if (chan) setTimeout(function(){ chan.classList.add('show'); }, 800);
  } catch(e) { console.log('mz nav error', e); }
}
"""

with gr.Blocks(theme=theme, css=CSS, title="MirZa Tools", analytics_enabled=False, js=NAV_JS) as demo:
    gr.HTML(TOPBAR, elem_id="mz-topbar")

    with gr.Tabs():
        # ---------------- TEXT TO SPEECH ----------------
        with gr.TabItem("🔊  Text to Speech"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes="mz-card"):
                    t_text = gr.Textbox(label="Text to Synthesize", lines=7,
                                        placeholder="Type or paste your text here, then press Generate…")
                    _tag_row(t_text)
                    t_voice = gr.Dropdown(label="🎭 Voice / Character",
                                          choices=list(VOICE_PRESETS.keys()),
                                          value="🎲  Default voice",
                                          info="Pick a ready character voice (male/female, young/old, storyteller, whisper…).")
                    with gr.Row():
                        t_lang = gr.Dropdown(label="Language", choices=ALL_LANGS, value="Auto", scale=3)
                        t_srt = gr.Checkbox(label="Generate subtitles (SRT)", value=False, scale=2)
                    t_sp, t_du, t_ns, t_dn, t_gs, t_pp, t_po = _gen_settings()
                    t_btn = gr.Button("Generate Voice", variant="primary", elem_id="mz_gen")
                with gr.Column(scale=4, elem_classes="mz-card"):
                    t_audio = gr.Audio(label="Output Audio", type="filepath", interactive=False)
                    t_dl = gr.DownloadButton("⬇  Download MP3 (320 kbps)", visible=False, elem_id="mz_dl")
                    t_srt_dl = gr.DownloadButton("⬇  Download Subtitles (SRT)", visible=False, elem_id="mz_srt")
                    t_status = gr.Markdown("Your generated audio will appear here.")
            t_btn.click(tts_fn,
                        [t_text, t_lang, t_voice, t_srt, t_sp, t_du, t_ns, t_dn, t_gs, t_pp, t_po],
                        [t_audio, t_dl, t_srt_dl, t_status])

        # ---------------- VOICE CLONE ----------------
        with gr.TabItem("🎭  Voice Clone"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes="mz-card"):
                    c_ref = gr.Audio(label="Reference Voice (3–10 seconds)", type="filepath")
                    c_reftext = gr.Textbox(label="Reference Text (optional)", lines=2,
                                           placeholder="What the reference says. Leave blank to auto-detect.")
                    c_text = gr.Textbox(label="Text to Synthesize", lines=5,
                                        placeholder="Type or paste your text here…")
                    _tag_row(c_text)
                    with gr.Row():
                        c_lang = gr.Dropdown(label="Language", choices=ALL_LANGS, value="Auto", scale=3)
                        c_srt = gr.Checkbox(label="Generate subtitles (SRT)", value=False, scale=2)
                    c_sp, c_du, c_ns, c_dn, c_gs, c_pp, c_po = _gen_settings()
                    c_btn = gr.Button("Generate in Cloned Voice", variant="primary", elem_id="mz_gen_c")
                with gr.Column(scale=4, elem_classes="mz-card"):
                    c_audio = gr.Audio(label="Output Audio", type="filepath", interactive=False)
                    c_dl = gr.DownloadButton("⬇  Download MP3 (320 kbps)", visible=False, elem_id="mz_dl_c")
                    c_srt_dl = gr.DownloadButton("⬇  Download Subtitles (SRT)", visible=False, elem_id="mz_srt_c")
                    c_status = gr.Markdown("Your cloned audio will appear here.")
            c_btn.click(clone_fn,
                        [c_text, c_lang, c_ref, c_reftext, c_srt, c_sp, c_du, c_ns, c_dn, c_gs, c_pp, c_po],
                        [c_audio, c_dl, c_srt_dl, c_status])

    gr.HTML(FOOTER)

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
