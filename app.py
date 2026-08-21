import streamlit as st
import requests
import time

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Speech-to-Text | CNN + BiGRU + CTC",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-dark:     #0d0f14;
    --bg-card:     #13161e;
    --bg-card2:    #1a1e28;
    --accent:      #4f9eff;
    --accent2:     #a78bfa;
    --success:     #34d399;
    --warn:        #fbbf24;
    --text:        #e2e8f0;
    --text-muted:  #64748b;
    --border:      #1e2433;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text);
}

.stApp { background-color: var(--bg-dark); }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

.hero {
    background: linear-gradient(135deg, #0d0f14 0%, #13161e 40%, #1a1128 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(79,158,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(167,139,250,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.02em;
}

.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 1rem 0;
}

.metric-row { display: flex; gap: 1rem; margin-top: 0.5rem; }
.metric-chip {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1.1rem;
    flex: 1;
    text-align: center;
}
.metric-chip .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--success);
}
.metric-chip .label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.15rem;
}

.transcript-box {
    background: var(--bg-card2);
    border: 1px solid var(--accent);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 1.15rem;
    font-weight: 400;
    color: var(--text);
    line-height: 1.7;
    margin-top: 0.8rem;
    box-shadow: 0 0 20px rgba(79,158,255,0.06);
}

.pipeline {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.5rem 0;
}
.pipe-step {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent2);
}
.pipe-arrow {
    color: var(--text-muted);
    font-size: 0.85rem;
}

[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #3b7dd8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    width: 100%;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

[data-testid="stFileUploader"] {
    background: var(--bg-card2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important;
}

.info-box {
    background: rgba(79,158,255,0.07);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: var(--text-muted);
    margin: 0.8rem 0;
}
.warn-box {
    background: rgba(251,191,36,0.07);
    border-left: 3px solid var(--warn);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: var(--text-muted);
    margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# API CALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def check_api_health(api_url: str):
    try:
        resp = requests.get(f"{api_url}/health", timeout=10)
        if resp.status_code == 200:
            return True, resp.json()
        return False, None
    except requests.exceptions.RequestException:
        return False, None


def call_transcribe_api(api_url: str, file_bytes: bytes, filename: str):
    """
    Sends the uploaded audio file to the deployed FastAPI /transcribe
    endpoint and returns the parsed JSON response.
    Render free tier can take up to ~50s to wake from sleep, so timeout
    is set generously.
    """
    files = {"file": (filename, file_bytes)}
    resp = requests.post(f"{api_url}/transcribe", files=files, timeout=90)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <p style="font-family:'Space Mono',monospace;font-size:0.7rem;
              letter-spacing:0.12em;color:#4f9eff;text-transform:uppercase;
              margin-bottom:1rem;">⚙ Configuration</p>
    """, unsafe_allow_html=True)

    api_url = st.text_input(
        "Backend API URL",
        value="https://voice-to-text-cjqh.onrender.com",
        help="Base URL of the deployed FastAPI service"
    ).rstrip("/")

    st.markdown("---")

    st.markdown("""
    <p style="font-family:'Space Mono',monospace;font-size:0.7rem;
              letter-spacing:0.12em;color:#4f9eff;text-transform:uppercase;
              margin-bottom:0.8rem;">📐 Architecture</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pipeline">
        <span class="pipe-step">Log-Mel</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">ResCNN×3</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">BiGRU×5</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">CTC</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">Greedy</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <p style="font-family:'Space Mono',monospace;font-size:0.7rem;
              letter-spacing:0.12em;color:#4f9eff;text-transform:uppercase;
              margin-bottom:0.8rem;">📊 Trained On</p>
    <div style="font-size:0.82rem;color:#64748b;line-height:1.8;">
        Dataset &nbsp;&nbsp;&nbsp; LibriSpeech<br>
        Split &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; dev-clean<br>
        Duration &nbsp;&nbsp; ~5.4 hours<br>
        Val WER &nbsp;&nbsp;&nbsp; 0.597<br>
        Val CER &nbsp;&nbsp;&nbsp; 0.234<br>
        Epochs &nbsp;&nbsp;&nbsp;&nbsp; 30
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Live API health status
    healthy, health_data = check_api_health(api_url)
    if healthy:
        st.markdown(f"""
        <p style="font-family:'Space Mono',monospace;font-size:0.72rem;
                  color:#34d399;">● API online — epoch {health_data.get('epoch')}</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <p style="font-family:'Space Mono',monospace;font-size:0.72rem;
                  color:#fbbf24;">● API waking up / offline — first
                  request may take up to 50s</p>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">🎙️ Speech-to-Text</p>
    <p class="hero-sub">
        CNN + Bidirectional GRU + CTC Loss &nbsp;·&nbsp;
        Trained from scratch on LibriSpeech &nbsp;·&nbsp;
        Served via FastAPI on Render
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    This UI calls a live backend API (FastAPI + Docker, deployed on Render)
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""<p class="card-title">📂 Upload Audio</p>""",
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a WAV or FLAC file",
        type=["wav", "flac"],
        label_visibility="collapsed"
    )

    if uploaded:
        st.markdown("""<p class="card-title" style="margin-top:1rem;">
                    🎵 Playback</p>""", unsafe_allow_html=True)
        st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")

        st.markdown("<br>", unsafe_allow_html=True)
        transcribe_btn = st.button("▶  Transcribe", use_container_width=True)
    else:
        st.markdown("""
        <div class="info-box">
            Upload a <strong>.wav</strong> or <strong>.flac</strong> file
            to begin. The file is sent to the deployed API for processing.
        </div>
        """, unsafe_allow_html=True)
        transcribe_btn = False

with col_right:
    st.markdown("""<p class="card-title">📄 Transcription</p>""",
                unsafe_allow_html=True)

    if uploaded and transcribe_btn:
        with st.spinner("Transcribing..."):
            try:
                file_bytes = uploaded.getvalue()
                result = call_transcribe_api(api_url, file_bytes, uploaded.name)

                transcript = result.get("transcript", "")
                duration   = result.get("duration_sec", 0.0)
                rtf        = result.get("real_time_factor", 0.0)
                word_count = result.get("word_count", 0)

                if transcript.strip():
                    st.markdown(f"""
                    <div class="transcript-box">"{transcript}"</div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="transcript-box" style="color:#64748b;font-style:italic;">
                        [no speech detected — try a clearer recording]
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-row" style="margin-top:1rem;">
                    <div class="metric-chip">
                        <div class="value">{duration:.1f}s</div>
                        <div class="label">Duration</div>
                    </div>
                    <div class="metric-chip">
                        <div class="value">{rtf:.2f}x</div>
                        <div class="label">RTF</div>
                    </div>
                    <div class="metric-chip">
                        <div class="value">{word_count}</div>
                        <div class="label">Words</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except requests.exceptions.Timeout:
                st.error("Request timed out. The API may still be waking "
                         "up — please try again in a moment.")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API: {e}")
            except Exception as e:
                st.error(f"Transcription failed: {e}")

    else:
        st.markdown("""
        <div class="transcript-box" style="color:#64748b;font-style:italic;
             min-height:80px;display:flex;align-items:center;">
            Transcription will appear here after you upload a file and
            click Transcribe.
        </div>
        """, unsafe_allow_html=True)
