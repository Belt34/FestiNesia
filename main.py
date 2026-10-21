import urllib.parse
import math
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(page_title="FestiNesia - Cultural & Art Events", layout="wide")

# --- KUSTOMISASI TAMPILAN / THEME (CSS) ---
st.markdown("""
    <style>
    /* 1. Latar Belakang Aplikasi (Oranye Krem Lembut dari Header) */
    .stApp {
        background-color: #FFF3E0 !important;
        color: #333333 !important;
    }

    /* 2. Warna Teks Paragraf, Label & Title */
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

    /* 5. Custom Dropdown & Input Box */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    .stSelectbox div[role="combobox"] {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border: 1.5px solid #FF6600 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] *, 
    div[data-baseweb="input"] *,
    input {
        color: #333333 !important;
        background-color: transparent !important;
    }

    div[data-baseweb="popover"], 
    ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
        border: 1px solid #FF6600 !important;
        border-radius: 8px !important;
    }

    ul[data-baseweb="menu"] li {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }

    ul[data-baseweb="menu"] li:hover {
        background-color: #FFF3E0 !important;
        color: #FF6600 !important;
    }

    /* 6. Custom Tombol Utama & Widget Lokasi */
    div.stButton > button, 
    button[title="Get Location"],
    .stButton button {
        background-color: #FFFFFF !important;
        color: #FF6600 !important;
        border: 1.5px solid #FF6600 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
    }

    div.stButton > button:hover,
    button[title="Get Location"]:hover {
        background-color: #FFE0B2 !important;
        color: #E65100 !important;
        border-color: #E65100 !important;
    }

    /* 7. Info Box & Alert Container */
    div[data-testid="stAlert"],
    div[data-baseweb="notification"] {
        background-color: #FFE0B2 !important;
        color: #D84315 !important;
        border: 1px solid #FFB74D !important;
        border-radius: 8px !important;
    }

    /* 8. Expander / Card Container */
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border: 1px solid #FFE0B2 !important;
        border-left: 4px solid #FF6600 !important;
        border-radius: 6px !important;
    }
    
    div[data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
        border: 1px solid #FFE0B2 !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
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
    if st.session_state.jelajah_page > 1:
        st.session_state.jelajah_page -= 1

def next_page():
    st.session_state.jelajah_page += 1

# 2. Load Dataset Pergelaran Seni & Budaya
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

# 3. Fungsi Rekomendasi / Pencarian
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

# --- 1. HALAMAN BERANDA ---
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

# --- 2. HALAMAN JELAJAH EVENT (LENGKAP) ---
elif page == "Jelajah Event":
    if st.button("← Kembali ke Beranda"):
        st.session_state.active_tab = "Beranda"
        st.rerun()
        
    st.title("Jelajah Semua Event")
    st.caption("Temukan seluruh daftar pergelaran seni dan festival di Indonesia.")
    st.markdown("---")
    
    # Filter dan Pencarian Event
    col_search, col_wilayah = st.columns([2, 1])
    with col_search:
        keyword = st.text_input("🔍 Cari Event / Jenis Seni:", placeholder="Contoh: Tari, Musik, Karnaval...")
    with col_wilayah:
        daftar_wilayah = ["Semua"]
        if not df.empty and 'wilayah' in df.columns:
            daftar_wilayah += sorted(df['wilayah'].dropna().unique().tolist())
        wilayah_pilihan = st.selectbox("📍 Pilih Wilayah:", daftar_wilayah)
        
    results = recommend_events(
        wilayah=None if wilayah_pilihan == "Semua" else wilayah_pilihan,
        keyword=keyword
    )
    
    if not results.empty:
        # Pagination Setup (10 Event per Halaman)
        items_per_page = 10
        total_items = len(results)
        total_pages = math.ceil(total_items / items_per_page)
        
        # Validasi batas halaman
        if st.session_state.jelajah_page > total_pages:
            st.session_state.jelajah_page = 1
            
        start_idx = (st.session_state.jelajah_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_results = results.iloc[start_idx:end_idx]
        
        st.write(f"Menampilkan **{start_idx + 1} - {min(end_idx, total_items)}** dari **{total_items}** event:")
        
        # Tampilkan List Card Event
        for idx, row in page_results.iterrows():
            nama = row.get('nama_kegiatan', row.iloc[0])
            wilayah = row.get('wilayah', 'Indonesia')
            jenis = row.get('jenis_kesenian', '-')
            penyelenggara = row.get('penyelenggara', '-')
            
            with st.expander(f"🎭 {nama} ({wilayah})"):
                st.write(f"**Jenis Kesenian:** {jenis}")
                st.write(f"**Wilayah:** {wilayah}")
                st.write(f"**Penyelenggara:** {penyelenggara}")
                query_maps = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(str(nama) + ' ' + str(wilayah))}"
                st.markdown(f"[🗺️ Buka Rute di Google Maps]({query_maps})")
                
        # Kontrol Pagination
        st.markdown("---")
        col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
        with col_p1:
            if st.session_state.jelajah_page > 1:
                st.button("◀ Halaman Sebelumnya", on_click=prev_page)
        with col_p2:
            st.write(f"<p style='text-align: center;'>Halaman {st.session_state.jelajah_page} dari {total_pages}</p>", unsafe_allow_html=True)
        with col_p3:
            if st.session_state.jelajah_page < total_pages:
                st.button("Halaman Selanjutnya ▶", on_click=next_page)
    else:
        st.info("Tidak ditemukan event yang sesuai dengan pencarianmu.")

# --- 3. HALAMAN TRIP PLANNER (LENGKAP) ---
elif page == "Trip Planner":
    if st.button("← Kembali ke Beranda"):
        st.session_state.active_tab = "Beranda"
        st.rerun()
        
    st.title("Trip Planner - Cultural Route")
    st.caption("Rencanakan kunjungan ke tempat acara seni budaya berdasarkan minat dan lokasimu.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        kategori_minat = st.selectbox("Pilih Jenis Kesenian / Minat:", ["Semua", "Seni Tari", "Seni Musik", "Seni Teater", "Seni Rupa", "Karnaval / Budaya"])
    with col2:
        daftar_wilayah = ["Semua"]
        if not df.empty and 'wilayah' in df.columns:
            daftar_wilayah += sorted(df['wilayah'].dropna().unique().tolist())
        wilayah_pilihan = st.selectbox("Wilayah / Kota:", daftar_wilayah)
        
    if st.button("Buat Rencana Kunjungan", use_container_width=True):
        st.success("Rencana perjalanan berhasil dibuat!")
        results = recommend_events(
            wilayah=None if wilayah_pilihan == "Semua" else wilayah_pilihan,
            keyword="" if kategori_minat == "Semua" else kategori_minat
        )
        if not results.empty:
            for idx, row in results.head(5).iterrows():
                nama = row.get('nama_kegiatan', row.iloc[0])
                wilayah = row.get('wilayah', 'Indonesia')
                with st.expander(f"📍 {nama} ({wilayah})"):
                    st.write(f"**Wilayah:** {wilayah}")
                    query_maps = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(str(nama) + ' ' + str(wilayah))}"
                    st.markdown(f"[🗺️ Buka Rute di Google Maps]({query_maps})")
        else:
            st.info("Tidak ditemukan event yang sesuai dengan kriteria filter.")
