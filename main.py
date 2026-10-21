import urllib.parse
import math
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_geolocation import streamlit_geolocation

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="FestiNesia - Cultural & Art Events", layout="wide")

# --- KUSTOMISASI TAMPILAN / THEME (CSS) ---
st.markdown("""
    <style>
    /* 1. Latar Belakang Aplikasi & Warna Teks Utama */
    .stApp {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }

    /* 2. Warna Teks Paragraf & Label */
    p, span, label, div {
        color: #333333 !important;
    }

    /* 3. Header Utama & Sub-header */
    h1 {
        color: #FF6600 !important;
        border-bottom: 3px solid #FF6600;
        padding-bottom: 8px;
    }

    h2, h3, h4 {
        color: #E65100 !important;
    }

    /* 4. Top Bar Streamlit */
    header[data-testid="stHeader"] {
        background-color: #FFF3E0 !important;
        border-bottom: 2px solid #FF6600;
    }

    /* 5. Tombol / Button (Latar Putih, Border & Teks Oranye) */
    div.stButton > button {
        background-color: #FFFFFF !important; /* Isi Putih */
        color: #FF6600 !important;            /* Teks Oranye */
        border: 1.5px solid #FF6600 !important;/* Border Oranye Samar */
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
    }

    div.stButton > button:hover {
        background-color: #FFF3E0 !important; /* Soft Cream/Oranye Muda saat Hover */
        color: #E65100 !important;
        border-color: #E65100 !important;
    }

    /* 6. Expander / Card Event dengan Outline Samar (Box Outline) */
    .streamlit-expanderHeader {
        background-color: #FAFAFA !important;
        color: #333333 !important;
        border: 1px solid #E0E0E0 !important; /* Outline samar tipis */
        border-left: 4px solid #FF6600 !important; /* Akses Oranye di Kiri */
        border-radius: 6px !important;
        padding: 10px 14px !important;
    }
    
    /* Container/Content di dalam Expander */
    div[data-testid="stExpanderDetails"] {
        border: 1px solid #E0E0E0 !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
        background-color: #FFFFFF !important;
        padding: 14px !important;
    }
    
    /* Box Outline Samar untuk Container Umum / Form */
    div[data-testid="stForm"], div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important; /* Bayangan sangat samar */
    }
    </style>
""", unsafe_allow_html=True)

# --- INISIALISASI SESSION STATE UNTUK NAVIGASI & LIMIT ---
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Beranda"
if 'jelajah_page' not in st.session_state:
    st.session_state.jelajah_page = 1

def set_tab(tab_name):
    st.session_state.active_tab = tab_name

def reset_jelajah_page():
    st.session_state.jelajah_page = 1

def prev_page():
    st.session_state.jelajah_page -= 1

def next_page():
    st.session_state.jelajah_page += 1

# 1. Load Dataset Pergelaran Seni & Budaya
@st.cache_data
def load_data():
    file_path = 'data_penyelenggaraan_seni_budaya_5_kategori.xlsx'
    try:
        excel_file = pd.read_excel(file_path, sheet_name=None)
        all_sheets = [sheet_df for _, sheet_df in excel_file.items()]
        df = pd.concat(all_sheets, ignore_index=True)
    except Exception as e:
        st.error(f"Gagal membaca file dataset Excel: {e}")
        return pd.DataFrame()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df

df = load_data()

# 2. Fungsi Rekomendasi
def recommend_events(wilayah=None, keyword=""):
    filtered_df = df.copy()
    if wilayah and wilayah != "Semua":
        filtered_df = filtered_df[filtered_df['wilayah'].astype(str).str.lower() == wilayah.lower()]
    if filtered_df.empty:
        return filtered_df
    
    if keyword:
        nama_col = 'nama_kegiatan' if 'nama_kegiatan' in filtered_df.columns else filtered_df.columns[0]
        jenis_col = 'jenis_kesenian' if 'jenis_kesenian' in filtered_df.columns else nama_col
        search_corpus = filtered_df[nama_col].fillna('') + " " + filtered_df[jenis_col].fillna('')
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(search_corpus)
        query_vec = vectorizer.transform([keyword])
        
        filtered_df['score'] = cosine_similarity(query_vec, tfidf_matrix).flatten()
        results = filtered_df.sort_values(by='score', ascending=False)
    else:
        results = filtered_df
    return results

# --- TAMPILAN FRONT-END STREAMLIT ---
page = st.session_state.active_tab

if page == "Beranda":
    st.title("FestiNesia")
    st.caption("Platform informasi pergelaran seni, budaya, dan festival lokal.")
    
    st.write("""
    **FestiNesia** adalah platform digital andalan untuk membantumu menemukan berbagai pertunjukan seni, festival budaya, dan pergelaran lokal secara praktis.
    
    - **Atraksi:** Temukan beragam kurasi acara seni dan festival menarik.
    - **Aksesibilitas:** Dapatkan rekomendasi terdekat dan navigasi rute instan langsung ke lokasi via Google Maps.
    - **Amenitas:** Nikmati kemudahan eksplorasi fasilitas pendukung dan titik menarik di sekitar tempat acara.
    """)
    st.markdown("---")
    
    st.subheader("Navigasi Cepat")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Jelajah Semua Event", use_container_width=True):
            st.session_state.active_tab = "Jelajah Event"
            st.rerun()
    with col_btn2:
        if st.button("Buka Trip Planner", use_container_width=True):
            st.session_state.active_tab = "Trip Planner"
            st.rerun()

# (Sisa logika halaman Jelajah Event dan Trip Planner tetap sama)
