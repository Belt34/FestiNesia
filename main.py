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
    /* Latar Belakang Aplikasi & Warna Teks Utama */
    .stApp {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }

    /* Warna Teks Paragraf & Label */
    p, span, label, div {
        color: #333333 !important;
    }

    /* Header Utama & Sub-header dengan Warna Oranye */
    h1 {
        color: #FF6600 !important;
        border-bottom: 3px solid #FF6600;
        padding-bottom: 8px;
    }

    h2, h3, h4 {
        color: #E65100 !important;
    }

    /* Top Bar Streamlit */
    header[data-testid="stHeader"] {
        background-color: #FFF3E0 !important;
        border-bottom: 2px solid #FF6600;
    }

    /* Tombol / Button */
    div.stButton > button {
        background-color: #FF6600 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: bold !important;
    }

    div.stButton > button:hover {
        background-color: #E65100 !important;
        color: #FFFFFF !important;
    }

    /* Expander / Card Event */
    .streamlit-expanderHeader {
        background-color: #F8F9FA !important;
        color: #333333 !important;
        border-left: 4px solid #FF6600 !important;
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
