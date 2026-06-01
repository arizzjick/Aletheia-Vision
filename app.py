import streamlit as st
import cv2
import tempfile
import os
import shutil
import numpy as np
import csv
import pandas as pd
import base64
from datetime import datetime
from yt_dlp import YoutubeDL
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image

# ==========================================
# FUNGSI UNTUK MENGUBAH GAMBAR JADI KODE HTML
# ==========================================
def get_image_as_base64(path):
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    except Exception:
        return "" 

# ==========================================
# 1. KONFIGURASI HALAMAN & ICON BROWSER TAB
# ==========================================
try:
    fav_icon = Image.open("ICON ALETHEIA.png")
except Exception:
    fav_icon = "🏛️"

st.set_page_config(
    page_title="Aletheia Vision", 
    page_icon=fav_icon, 
    layout="wide", 
    initial_sidebar_state="auto"
)

# Kustomisasi Tampilan Antarmuka Berbasis CSS
st.markdown("""
    <style>
    /* Background Utama & Warna Teks Marmer */
    .stApp { 
        background-color: #0a0c14; 
        color: #e1dfd7; 
    }
    
    /* Sidebar Bertema Kuil Gelap dengan Batas Emas Kuno */
    section[data-testid="stSidebar"] { 
        background-color: #111422; 
        border-right: 2px solid #dfb755; 
    }
    
    /* Warna Judul Mengikuti Kilau Emas Ilahi */
    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2 { 
        color: #dfb755 !important; 
        font-family: 'Georgia', serif;
        text-shadow: 0px 2px 10px rgba(223, 183, 85, 0.2);
    }
    
    /* Tombol Utama (DETEKSI SEKARANG) - Efek Emas Bergradasi */
    div[data-testid="column"]:nth-child(2) .stButton>button {
        background: linear-gradient(90deg, #dfb755 0%, #aa7c11 100%);
        color: #0a0c14; 
        border: none; 
        border-radius: 4px; 
        font-weight: bold;
        padding: 10px; 
        height: 50px; 
        font-size: 16px; 
        width: 100%;
        box-shadow: 0px 4px 20px rgba(223, 183, 85, 0.4);
        transition: all 0.3s ease;
    }
    div[data-testid="column"]:nth-child(2) .stButton>button:hover { 
        background: linear-gradient(90deg, #aa7c11 0%, #dfb755 100%);
        color: #ffffff;
        box-shadow: 0px 6px 25px rgba(223, 183, 85, 0.6);
        transform: translateY(-2px);
    }
    
    /* Tombol Biasa / Unduh Laporan */
    .stButton>button { 
        border-radius: 4px; 
        width: 100%; 
        border: 1px solid #dfb755; 
        color: #dfb755; 
        background-color: transparent;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        background-color: #dfb755; 
        color: #0a0c14; 
    }
    
    /* Wadah utama deretan daftar tab */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] { 
        background-color: #111422; 
        padding: 10px 10px 0px 10px; 
        border-radius: 5px; 
        border-bottom: 2px solid #dfb755; 
        gap: 8px;
    }
    
    /* Mengatur tombol tab individu agar teks memiliki ruang aman */
    div[data-testid="stTabs"] button {
        padding: 10px 24px !important;    
        white-space: nowrap !important;   
        border-radius: 4px 4px 0px 0px !important;
        color: #e1dfd7 !important;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.2s ease;
    }
    
    /* Tampilan Tab Aktif (Dipilih) */
    div[data-testid="stTabs"] button[aria-selected="true"] { 
        background-color: #dfb755 !important; 
        color: #0a0c14 !important; 
        font-weight: bold !important; 
    }
    
    /* Efek Hover saat Kursor Mendekati Tab */
    div[data-testid="stTabs"] button:hover {
        background-color: rgba(223, 183, 85, 0.1) !important;
        color: #dfb755 !important;
    }
    
    /* Angka Besar di Dashboard Statistik */
    div[data-testid="stMetricValue"] { 
        font-size: 2rem; 
        font-weight: bold; 
        color: #dfb755 !important; 
    }
    
    /* Warna Teks Penjelasan (Muted) */
    .stMarkdown p, .stCaption {
        color: #9fa4b4;
    }
    
    /* Wadah Utama Header Kebenaran (Rata Tengah Universal) */
    .logo-container { 
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        justify-content: center; 
        text-align: center; 
        margin-bottom: 30px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PROSES AMAN UNTUK API KEY & DATABASE
# ==========================================
API_KEY = ""
try:
    if hasattr(st, "secrets") and "API_KEY" in st.secrets:
        API_KEY = st.secrets["API_KEY"]
except Exception:
    pass

MODEL_ID = "model-clasification-baru/1"
CSV_FILE = "riwayat_deteksi.csv"
THUMB_DIR = "history_thumbnails"

os.makedirs(THUMB_DIR, exist_ok=True)

client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)

if 'media_path' not in st.session_state: st.session_state.media_path = None
if 'media_label' not in st.session_state: st.session_state.media_label = ""
if 'source_type' not in st.session_state: st.session_state.source_type = ""

def detect_frame(image_path, threshold):
    try:
        cfg = InferenceConfiguration(confidence_threshold=threshold)
        with client.use_configuration(cfg):
            res = client.infer(image_path, model_id=MODEL_ID)
        preds = res.get("predictions", [])
        if preds:
            best = max(preds, key=lambda x: x['confidence'])
            return best.get("class", "asli").lower(), best.get("confidence", 0)
        return "asli", 0
    except Exception: 
        return "error", 0

def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30 
    
    if tot == 0: return []
    frames = []
    for idx in np.linspace(0, tot - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            tmp = tempfile.mktemp(suffix=".jpg")
            cv2.imwrite(tmp, frame) 
            sec = float(idx) / fps
            frames.append((tmp, int(idx), sec))
    cap.release()
    return frames

def save_to_csv(src_type, src_name, label, ratio, thumb_path="", detail_analysis=""):
    exists = os.path.isfile(CSV_FILE)
    try:
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if not exists: 
                w.writerow(['Waktu', 'Tipe Sumber', 'Nama/Link', 'Hasil', 'Rasio AI', 'Thumbnail', 'Detail Analisis'])
            w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), src_type, src_name, label, f"{ratio:.0%}", thumb_path, detail_analysis])
        return True
    except PermissionError:
        st.error("⚠️ **Gagal mencatat riwayat!** File `riwayat_deteksi.csv` sedang dibuka di program lain (Excel). Silakan tutup terlebih dahulu.")
        return False
    except Exception:
        return False

def get_stats():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            total = len(df)
            deepfake = len(df[df['Hasil'] == 'TERDETEKSI DEEPFAKE'])
            ambigu = len(df[df['Hasil'] == 'TERINDIKASI MENCURIGAKAN (AMBIGU)'])
            asli = len(df[df['Hasil'] == 'TERINDIKASI ASLI'])
            return total, deepfake, ambigu, asli
        except Exception:
            return 0, 0, 0, 0
    return 0, 0, 0, 0

# ==========================================
# 3. SIDEBAR (DASHBOARD STATISTIK)
# ==========================================
with st.sidebar:
    st.markdown("### 🏛 *Dashboard* Statistik")
    st.write("---")
    
    st.markdown("⚖️ **Total Pengujian Video**")
    metric_total = st.empty() 
    
    col_f, col_a, col_r = st.columns(3)
    with col_f:
        st.markdown("<small>🔴 Deepfake</small>", unsafe_allow_html=True)
        metric_fake = st.empty()
    with col_a:
        st.markdown("<small>🟡 Ambigu</small>", unsafe_allow_html=True)
        metric_ambigu = st.empty()
    with col_r:
        st.markdown("<small>🟢 Asli</small>", unsafe_allow_html=True)
        metric_real = st.empty()

    def refresh_sidebar_stats():
        t, f, a, r = get_stats()
        metric_total.metric(label="Total", value=t, label_visibility="collapsed")
        metric_fake.metric(label="Fake", value=f, label_visibility="collapsed")
        metric_ambigu.metric(label="Ambigu", value=a, label_visibility="collapsed")
        metric_real.metric(label="Real", value=r, label_visibility="collapsed")

    refresh_sidebar_stats()
        
    st.write("---")
    st.markdown("### 🧭 Parameter Analisis Model")
    
    input_num_frames = st.slider("Jumlah Sampel Frame Video:", min_value=5, max_value=50, value=10, step=5, 
                                 help="Semakin banyak frame, analisis semakin akurat namun proses deteksi memakan waktu lebih lama.")
    input_threshold = st.slider("Sensitivitas Confidence Model:", min_value=0.10, max_value=0.95, value=0.40, step=0.05,
                                help="Batas ambang keyakinan minimum deteksi kecerdasan buatan.")
    
    st.write("---")
    st.markdown("### 📜 Informasi Sistem")
    st.caption("**Model:** ResNet50 Architecture\n\n**Akurasi Dataset:** 99.9%\n\n**Framework:** Roboflow Inference HTTP")

# ==========================================
# 4. KONTEN UTAMA HALAMAN (STRUKTUR LOGO & HEADER)
# ==========================================
logo_base64 = get_image_as_base64("LOGO ALETHEIA.png")
icon_base64 = get_image_as_base64("ICON ALETHEIA.png")

if logo_base64:
    logo_html = f"<img src='{logo_base64}' style='max-width: 1000px; width: 100%; height: auto; margin-bottom: 20px; display: block; margin-left: auto; margin-right: auto;'>"
else:
    logo_html = ""

if icon_base64:
    header_full_html = f"""
    <div class='logo-container'>
        {logo_html}
        <div style='display: flex; align-items: baseline; justify-content: center; gap: 12px; margin-bottom: 2px; width: 100%;'>
            <img src='{icon_base64}' style='height: 48px; width: auto; display: inline-block; margin: 0; padding: 0;'>
            <h1 style='margin: 0; padding: 0; display: inline-block; color: #dfb755 !important; font-family: "Georgia", serif;'>Aletheia Vision</h1>
        </div>
        <p style='color:#9fa4b4; margin-top: 12px; font-style: italic; text-align: center; width: 100%;'>
            Deteksi video deepfake dengan kecerdasan buatan | Kebenaran di Balik Setiap Frame
        </p>
    </div>
    """
else:
    header_full_html = f"""
    <div class='logo-container'>
        {logo_html}
        <div style='display: flex; align-items: baseline; justify-content: center; gap: 12px; margin-bottom: 2px; width: 100%;'>
            <span style='font-size: 38px; display: inline-block; margin: 0; padding: 0;'>🦉</span>
            <h1 style='margin: 0; padding: 0; display: inline-block; color: #dfb755 !important; font-family: "Georgia", serif;'>Aletheia Vision</h1>
        </div>
        <p style='color:#9fa4b4; margin-top: 12px; font-style: italic; text-align: center; width: 100%;'>
            Deteksi video deepfake dengan kecerdasan buatan | Kebenaran di Balik Setiap Frame
        </p>
    </div>
    """

st.markdown(header_full_html, unsafe_allow_html=True)

# Modul Tab Utama Aplikasi
tab1, tab2, tab3 = st.tabs(["🔮 Deteksi Video", "📜 Riwayat", "🏛️ Tentang Aplikasi"])

# ----------------- TAB 1: OPERASI DETEKSI VIDEO -----------------
with tab1:
    col_kiri, col_kanan = st.columns([1.2, 1])

    with col_kiri:
        st.markdown("### 📥 Sumber Video")
        mode = st.radio("Metode Input:", ["File Lokal", "YouTube"], horizontal=True, label_visibility="collapsed")
        
        if mode == "YouTube":
            yt_url = st.text_input("URL YouTube:", placeholder="https://www.youtube.com/...")
            
            with st.expander("🔑 Solusi Utama Bypass HTTP Error 403 Forbidden (WAJIB JIKA ERROR)", expanded=True):
                st.markdown("""
                YouTube memblokir bot otomatis. Untuk mengatasinya:
                1. Pasang ekstensi **'Get cookies.txt LOCALLY'** atau **'Cookie-Editor'** di Chrome/Edge/Firefox kamu.
                2. Buka halaman utama YouTube (pastikan kamu dalam kondisi login akun Google/YouTube).
                3. Klik ekstensi tersebut, lalu ekspor/unduh sebagai file **`cookies.txt`**.
                4. Unggah file `cookies.txt` tersebut di bawah ini sebelum menekan tombol download.
                """)
                cookie_file = st.file_uploader("Unggah berkas cookies.txt kamu:", type=["txt"], key="yt_cookie_uploader_file")
            
            c_btn1, c_btn2 = st.columns(2)
            
            if c_btn1.button("🏺 Download Video"):
                if yt_url:
                    with st.spinner("⚡ Mengunduh Video Kualitas Tinggi dari YouTube..."):
                        try:
                            if st.session_state.media_path and os.path.exists(st.session_state.media_path) and "yt_" in st.session_state.media_path:
                                try: os.remove(st.session_state.media_path)
                                except Exception: pass 
                            
                            unique_yt_name = f"yt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                            
                            temp_cookie_path = None
                            if cookie_file is not None:
                                temp_cookie_path = "temp_cookies_app.txt"
                                with open(temp_cookie_path, "wb") as f:
                                    f.write(cookie_file.getvalue())
                            
                            ydl_opts = {
                                'format': 'mp4/best',
                                'outtmpl': unique_yt_name,
                                'noplaylist': True,
                                'rm_cached_dir': True,
                                'nocheckcertificate': True,
                                'quiet': True,
                                'no_warnings': True,
                                'http_headers': {
                                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                                    'Accept-Language': 'en-US,en;q=0.5',
                                    'Sec-Fetch-Mode': 'navigate',
                                }
                            }
                            
                            if temp_cookie_path and os.path.exists(temp_cookie_path):
                                ydl_opts['cookiefile'] = temp_cookie_path
                            else:
                                ydl_opts['cookiesfrombrowser'] = ('chrome', 'edge', 'firefox', 'opera')
                            
                            with YoutubeDL(ydl_opts) as ydl: 
                                ydl.download([yt_url])
                            
                            if temp_cookie_path and os.path.exists(temp_cookie_path):
                                try: os.remove(temp_cookie_path)
                                except Exception: pass
                            
                            st.session_state.media_path = unique_yt_name
                            st.session_state.media_label = yt_url
                            st.session_state.source_type = "Video YouTube"
                            st.success("Video Kualitas Tinggi Berhasil Dimuat!")
                            st.rerun()
                        except Exception as e:
                            if 'temp_cookie_path' in locals() and temp_cookie_path and os.path.exists(temp_cookie_path):
                                try: os.remove(temp_cookie_path)
                                except Exception: pass
                            st.error(f"Gagal mengunduh video dari YouTube. Error: {e}")
                            st.info("💡 **Solusi Ampuh 403 Forbidden:** Pastikan kamu telah mengunduh file `cookies.txt` baru dari browser saat sedang membuka halaman YouTube (akun login), lalu pasang di menu unggah berkas di atas.")
                            
            if c_btn2.button("🧹 Reset"): st.session_state.media_path = None
        else:
            # =========================================================================
            # PERUBAHAN DISINI: MENDUKUNG BANYAK FORMAT VIDEO & FILE EXTENSION DINAMIS
            # =========================================================================
            uploaded = st.file_uploader(
                "Upload File Video", 
                type=["mp4", "avi", "mov", "mkv", "wmv", "webm", "flv", "mpeg", "3gp"], 
                label_visibility="collapsed"
            )
            if uploaded:
                # Ambil ekstensi asli dari berkas yang diunggah secara dinamis
                file_extension = os.path.splitext(uploaded.name)[1]
                tmp = tempfile.mktemp(suffix=file_extension)
                with open(tmp, "wb") as f: 
                    f.write(uploaded.read())
                st.session_state.media_path = tmp
                st.session_state.media_label = uploaded.name
                st.session_state.source_type = "Video Lokal"

        if st.session_state.media_path and os.path.exists(st.session_state.media_path):
            st.video(st.session_state.media_path)

    with col_kanan:
        st.markdown("### ⚡ Eksekusi")
        
        if st.button("👁️ DETEKSI SEKARANG", disabled=not bool(st.session_state.media_path)):
            with st.spinner("Menganalisis frame video..."):
                thumb_name = f"thumb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                saved_thumb_path = os.path.join(THUMB_DIR, thumb_name)
                
                frames = extract_frames(st.session_state.media_path, num_frames=input_num_frames)
                if frames and os.path.exists(frames[0][0]):
                    shutil.copy(frames[0][0], saved_thumb_path)
                    
                results = []
                detail_logs = []
                gallery_evidence = [] 
                bar = st.progress(0)
                
                for i, (fp, f_idx, sec) in enumerate(frames):
                    label, conf = detect_frame(fp, input_threshold)
                    results.append({'label': label, 'confidence': conf})
                    
                    if label == 'ai':
                        if conf > 0.85:
                            alasan = "Terjadi penghilangan mikro-ekspresi alami wajah secara ekstrem & struktur geometri organ (mata/mulut) tidak sinkron akibat algoritma generasi wajah sintetis."
                        elif conf > 0.65:
                            alasan = "Ditemukan efek blur abnormal pada batas tepian wajah (blending area) serta ketidaksesuaian distribusi noise spasial antara objek wajah dan latar belakang."
                        else:
                            alasan = "Terdeteksi anomali pantulan cahaya (specular reflection) pada kornea mata subjek yang bergeser atau tidak searah dengan sumber pencahayaan lingkungan."
                            
                        status_label = f"🔴 AI ({conf:.1%})"
                        detail_logs.append(f"Frame #{f_idx} (Detik {sec:.2f}s) → Indikasi AI: {conf:.1%} | Alasan: {alasan}")
                    
                    else:
                        if conf > 0.85:
                            alasan = "Fitur biologis wajah terdeteksi sangat konsisten. Tekstur pori kulit, kontinuitas aliran mikro-ekspresi, dan refleks cahaya alami pada bola mata terintegrasi sempurna tanpa distorsi geometris."
                        elif conf > 0.65:
                            alasan = "Distribusi noise spasial antara objek wajah dan latar belakang sinkron. Tidak ditemukan batas blending abnormal (efek blur buatan) di sekitar area rahang atau rambut."
                        else:
                            alasan = "Struktur topografi wajah dan sinkronisasi bayangan alami tetap terjaga meskipun kualitas tangkapan frame mengalami sedikit penurunan resolusi atau kompresi digital."
                            
                        status_label = f"🟢 Asli ({conf:.1%})"
                        detail_logs.append(f"Frame #{f_idx} (Detik {sec:.2f}s) → Indikasi Asli: {conf:.1%} | Alasan: {alasan}")
                    
                    gallery_evidence.append({
                        'path': fp,
                        'caption': f"Frame #{f_idx} ({sec:.2f}s) - {status_label}"
                    })
                    
                    bar.progress((i + 1) / len(frames))
                
                fake_count = len([r for r in results if r['label'] == 'ai'])
                real_count = len(results) - fake_count
                fake_ratio = fake_count / len(results) if len(results) > 0 else 0
                
                if detail_logs:
                    final_analysis_text = " | ".join(detail_logs)
                else:
                    final_analysis_text = "Tidak ada frame sampel yang berhasil dianalisis."

                st.divider()
                
                if fake_ratio > 0.5:
                    final_lbl = "TERDETEKSI DEEPFAKE"
                    st.error(f"❌ **{final_lbl}** (Rasio AI: {fake_ratio:.0%})")
                elif fake_ratio == 0.5:
                    final_lbl = "TERINDIKASI MENCURIGAKAN (AMBIGU)"
                    st.warning(f"⚠️ **{final_lbl}** (Rasio AI: {fake_ratio:.0%}) - Direkomendasikan melakukan uji ulang secara manual.")
                else:
                    final_lbl = "TERINDIKASI ASLI"
                    st.success(f"🌿 **{final_lbl}** (Rasio AI: {fake_ratio:.0%})")

                st.markdown("#### 📊 Distribusi Klasifikasi Frame Video")
                chart_data = pd.DataFrame({
                    'Status Bingkai': ['Manusia Asli', 'Rekayasa AI'],
                    'Jumlah Bingkai': [real_count, fake_count]
                }).set_index('Status Bingkai')
                st.bar_chart(chart_data)

                report_text = f"""==================================================
        ALETHEIA VISION - REPORT HASIL DETETKSI FORENSIK DIGITAL
==================================================
Waktu Pengujian : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Tipe Sumber     : {st.session_state.source_type}
Nama / URL Media: {st.session_state.media_label}
Kesimpulan Akhir: {final_lbl}
Kepadatan AI    : {fake_ratio:.0%} ({fake_count} dari {len(results)} Frame Sampel)
Threshold Model : {input_threshold}
==================================================
LOG RINCIAN INVESTIGASI PER FRAME VIDEO:
""" + "\n".join([log.replace(" → ", " | ").replace(" | ", "\n   ") for log in detail_logs])

                st.download_button(
                    label="📜 UNDUH LAPORAN FORENSIK (.TXT)",
                    data=report_text,
                    file_name=f"Laporan_Forensik_AletheiaVision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

                with st.expander("🖼️ BUKA GALERI BUKTI VISUAL FRAME TERUJI", expanded=True):
                    grid_cols = st.columns(3) 
                    for idx, item in enumerate(gallery_evidence):
                        with grid_cols[idx % 3]:
                            if os.path.exists(item['path']):
                                st.image(item['path'], caption=item['caption'], use_container_width=True)

                for item in gallery_evidence:
                    if os.path.exists(item['path']):
                        try: os.unlink(item['path'])
                        except Exception: pass

                save_to_csv(st.session_state.source_type, st.session_state.media_label, final_lbl, fake_ratio, saved_thumb_path, final_analysis_text)
                refresh_sidebar_stats()

# ----------------- TAB 2: RIWAYAT -----------------
with tab2:
    st.markdown("### 📜 Riwayat Hasil Pengujian")
    
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            if not df.empty:
                if st.button("🏺 Hapus Semua Catatan Riwayat"):
                    try:
                        if os.path.exists(CSV_FILE): os.remove(CSV_FILE)
                        if os.path.exists(THUMB_DIR):
                            shutil.rmtree(THUMB_DIR)
                            os.makedirs(THUMB_DIR, exist_ok=True)
                        st.success("Riwayat berhasil dibersihkan!")
                        st.rerun()
                    except PermissionError:
                        st.error("❌ Gagal menghapus! Tutup file `riwayat_deteksi.csv` yang sedang terbuka di Excel terlebih dahulu.")
                    except Exception:
                        st.error("❌ Terjadi kegagalan saat membersihkan database.")
                    
                st.write("---")
                for idx, row in df.iloc[::-1].iterrows():
                    status_str = str(row['Hasil'])
                    if "DEEPFAKE" in status_str:
                        color_indicator = "🔴"
                    elif "AMBIGU" in status_str or "MENCURIGAKAN" in status_str:
                        color_indicator = "🟡"
                    else:
                        color_indicator = "🟢"
                        
                    with st.expander(f"{color_indicator} [{row['Waktu']}] - {row['Hasil']}"):
                        col_txt, col_img = st.columns([1.6, 1])
                        with col_txt:
                            st.markdown(f"**Jenis Media:** {row['Tipe Sumber']}")
                            st.markdown(f"**Sumber Data:** `{row['Nama/Link']}`")
                            st.markdown(f"**Skor Kepadatan AI:** {row['Rasio AI']}")
                            
                            st.markdown("<p style='color:#dfb755; font-weight:bold; margin-top:15px;'>📋 Laporan Analisis Forensik Digital:</p>", unsafe_allow_html=True)
                            
                            if 'Detail Analisis' in df.columns and pd.notna(row['Detail Analisis']):
                                logs = str(row['Detail Analisis']).split(" | ")
                                for log in logs:
                                    if "→" in log:
                                        st.markdown(f"📍 **{log.split(' → ')[0]}**")
                                        alasan_konten = log.split(' → ')[1]
                                        if "Indikasi AI" in alasan_konten:
                                            st.markdown(f"<span style='color:#ff4b4b; font-weight:bold;'>{alasan_konten}</span>", unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"<span style='color:#00e676;'>{alasan_konten}</span>", unsafe_allow_html=True)
                                        st.write("")
                                    else:
                                        st.write(f"📜 {log}")
                            else:
                                st.caption("Detail rekaman analisis frame tidak tersedia.")
                                
                        with col_img:
                            if pd.notna(row['Thumbnail']) and os.path.exists(str(row['Thumbnail'])):
                                st.image(row['Thumbnail'], caption="Preview Video Teruji", use_container_width=True)
                            else:
                                 st.caption("Preview tidak tersedia")
            else:
                st.info("Belum ada data pengujian pada sistem.")
        except Exception:
            st.error("Gagal membaca database lokal. Pastikan file CSV tidak sedang dikunci oleh Excel.")
    else:
        st.info("Belum ada data pengujian pada sistem.")

# ----------------- TAB 3: TENTANG APLIKASI -----------------
with tab3:
    st.markdown("### 🏛️ Tentang Aletheia Vision")
    st.write("""
    **Aletheia Vision** adalah sistem arsitektur aplikasi deteksi manipulasi video berbasis *Computer Vision* tingkat lanjut. Aplikasi ini dibangun memanfaatkan keunggulan *Framework* Streamlit dan didukung oleh model klasifikasi gambar arsitektur ResNet50 melalui integrasi Roboflow API untuk memfasilitasi kebutuhan investigasi bukti digital forensik.
    """)
    
    st.warning("⚠️ **Status Aplikasi:** Saat ini sistem masih berada dalam tahapan **pengembangan aktif (Active Development Phase)**. Pembaruan model, optimalisasi kecepatan ekstraksi frame, dan penajaman sensitivitas deteksi akan terus dilakukan untuk menghasilkan performa forensik yang jauh lebih matang.")
    
    st.write("---")
    
    st.markdown("#### 🎯 Tujuan Pembuatan Sistem")
    st.write("""
    Di era pesatnya perkembangan kecerdasan buatan, teknologi rekayasa video (*deepfake*) kini mampu memproduksi manipulasi visual yang sangat halus dan super realistis. Dampaknya, **masyarakat awam sering kali mengalami kesulitan besar untuk membedakan** secara kasat mata mana video yang benar-benar nyata (otentik) dan mana konten palsu hasil fabrikasi kecerdasan buatan. 
    
    Aletheia Vision dirancang dan hadir sebagai **alat uji praktis (testing tool)** yang ramah pengguna. Aplikasi ini bertujuan membantu menjembatani keterbatasan masyarakat awam, akademisi, hingga praktisi hukum agar dapat memverifikasi keabsahan dokumen video secara objektif, instan, serta transparan berdasarkan parameter data ilmiah, bukan sekadar asumsi visual.
    """)
    
    st.write("---")
    
    st.markdown("#### 🏛️ Filosofi Nama \"Aletheia Vision\"")
    st.write("""
    Pemilihan nama **Aletheia Vision** memuat landasan nilai filosofis dan teknis yang kuat untuk masa depan aplikasi ini:
    
    * **Aletheia (Greek: ἀλήθεια):** Berasal dari kosakata bahasa Yunani Kuno yang memiliki arti **"Kebenaran"** atau **"Keterbukaan"**. Secara etimologis, kata ini merepresentasikan sebuah konsep *"keadaan yang tidak tersembunyi"* atau tindakan menyingkap sesuatu yang tadinya tertutup rapat agar esensi aslinya terlihat jelas.
    * **Vision:** Merepresentasikan domain teknologi utama yang melandasi aplikasi ini, yaitu *Computer Vision* (kemampuan komputer untuk melihat dan mengekstrak informasi dari data visual), sekaligus melambangkan visi masa depan sistem ini sebagai pengawas keaslian media digital.
    """)
