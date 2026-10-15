import math
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_geolocation import streamlit_geolocation

# --- INISIALISASI SESSION STATE UNTUK TOMBOL "LAINNYA" ---
if 'display_limit' not in st.session_state:
    st.session_state.display_limit = 3

def reset_limit():
    st.session_state.display_limit = 3

def load_more():
    st.session_state.display_limit += 3

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="FestiNesia - Jakarta Cultural Events", layout="wide")

# 1. Load Dataset Pergelaran Seni & Budaya Jakarta
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

    TEMPAT_COORDS = {
        "TAMAN ISMAIL MARZUKI": (-6.1901, 106.8398),
        "KAWASAN MONAS SISI UTARA": (-6.1754, 106.8272),
        "PLAZA SISI SELATAN MONAS": (-6.1772, 106.8272),
        "JAKARTA INTERNATIONAL EQUESTRIAN PARK PULOMAS": (-6.1883, 106.8875),
        "ATRIUM LAGUNA DAN TRIBECA PARK": (-6.1783, 106.7925),
        "BALAI PERTEMUAN BLOK G BALAIKOTA PROVINSI DKI JAKARTA": (-6.1831, 106.8267),
        "BALAI AGUNG, BALAIKOTA PROVINSI DKI JAKARTA": (-6.1831, 106.8267),
        "CANDI BENTAR ANCOL JAKARTA UTARA": (-6.1256, 106.8361),
        "GELANGGANG KEC. TANJUNG PRIOK JAKARTA UTARA": (-6.1325, 106.8906),
        "JCC SENAYAN": (-6.2163, 106.8016)
    }

    DEFAULT_COORD = (-6.1754, 106.8272)

    if 'tempat_acara' in df.columns:
        df['tempat_acara_clean'] = df['tempat_acara'].astype(str).str.strip().str.upper()
        df['latitude'] = df['tempat_acara_clean'].map(lambda x: TEMPAT_COORDS.get(x, DEFAULT_COORD)[0])
        df['longitude'] = df['tempat_acara_clean'].map(lambda x: TEMPAT_COORDS.get(x, DEFAULT_COORD)[1])
    else:
        df['latitude'] = DEFAULT_COORD[0]
        df['longitude'] = DEFAULT_COORD[1]
    
    return df

df = load_data()

# 2. Perhitungan Jarak Garis Lurus (Sangat Cepat - Tidak Menggunakan API Internet)
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 1)

# 3. Fungsi Rekomendasi (Mengembalikan Seluruh Data Tersortir)
def recommend_events(user_lat=None, user_lng=None, wilayah=None, keyword=""):
    filtered_df = df.copy()
    
    if wilayah and wilayah != "Semua":
        filtered_df = filtered_df[filtered_df['wilayah'].astype(str).str.lower() == wilayah.lower()]
        
    if filtered_df.empty:
        return filtered_df

    # Hitung jarak dengan cepat menggunakan Pandas Apply
    if user_lat is not None and user_lng is not None:
        filtered_df['jarak_km'] = filtered_df.apply(lambda row: calculate_distance(user_lat, user_lng, row['latitude'], row['longitude']), axis=1)

    if keyword:
        nama_col = 'nama_kegiatan' if 'nama_kegiatan' in filtered_df.columns else filtered_df.columns[0]
        jenis_col = 'jenis_kesenian' if 'jenis_kesenian' in filtered_df.columns else nama_col
        
        search_corpus = filtered_df[nama_col].fillna('') + " " + filtered_df[jenis_col].fillna('')
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(search_corpus)
        query_vec = vectorizer.transform([keyword])
        filtered_df['score'] = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        if user_lat is not None and user_lng is not None:
            results = filtered_df.sort_values(by=['score', 'jarak_km'], ascending=[False, True])
        else:
            results = filtered_df.sort_values(by='score', ascending=False)
    else:
        if user_lat is not None and user_lng is not None:
            results = filtered_df.sort_values(by='jarak_km', ascending=True)
        else:
            results = filtered_df

    return results

# --- TAMPILAN FRONT-END STREAMLIT ---
col_logo, col_nav = st.columns([2, 6])
with col_logo:
    st.markdown("### 🌟 FestiNesia Jakarta")
with col_nav:
    page = st.radio("", ["Beranda", "Jelajah Event", "AI Planner"], horizontal=True)

st.divider()

if page == "Beranda":
    st.title("Pergelaran Seni & Budaya DKI Jakarta")
    st.caption("Temukan pertunjukan seni, bazar budaya, dan festival lokal terdekat dari lokasimu.")
    
    st.subheader("📍 Deteksi Lokasi Real-Time")
    location = streamlit_geolocation()
    
    user_lat, user_lng = None, None
    if location and location.get('latitude') and location.get('longitude'):
        user_lat = location['latitude']
        user_lng = location['longitude']
        st.success(f"Lokasi kamu terdeteksi: **{user_lat:.4f}, {user_lng:.4f}**")
    else:
        st.info("Klik tombol lokasi di atas untuk mengizinkan akses GPS real-time browser kamu.")

    # Form Input (Menggunakan on_change untuk mereset limit tampilan ke 3)
    with st.container():
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            if 'jenis_kesenian' in df.columns:
                jenis_kesenian_list = ["Semua"] + sorted([str(x).strip() for x in df['jenis_kesenian'].dropna().unique() if str(x).strip() != 'nan'])
            else:
                jenis_kesenian_list = ["Semua"]
            keyword_input = st.selectbox("Pilih Minat / Jenis Kesenian", jenis_kesenian_list, on_change=reset_limit)
        with c2:
            wilayah_list = ["Semua"] + list(df['wilayah'].dropna().unique()) if 'wilayah' in df.columns else ["Semua"]
            wilayah_input = st.selectbox("Wilayah Administrasi", wilayah_list, on_change=reset_limit)
        with c3:
            st.write("")
            search_btn = st.button("Cari Event", use_container_width=True, on_click=reset_limit)
            
    if search_btn or keyword_input != "Semua":
        selected_keyword = "" if keyword_input == "Semua" else keyword_input
        active_lat = user_lat if user_lat else -6.1754
        active_lng = user_lng if user_lng else 106.8272
        
        # Ambil SEMUA data yang cocok
        results = recommend_events(
            user_lat=active_lat, 
            user_lng=active_lng, 
            wilayah=wilayah_input, 
            keyword=selected_keyword
        )
        
        st.subheader("Hasil Rekomendasi Event")
        if results.empty:
            st.warning("Tidak ditemukan event seni/budaya yang sesuai.")
        else:
            # Batasi tampilan sesuai limit saat ini (awal: 3)
            displayed_results = results.head(st.session_state.display_limit)
            
            cols = st.columns(3)
            for idx, (_, row) in enumerate(displayed_results.iterrows()):
                with cols[idx % 3]:
                    nama_event = row.get('nama_kegiatan', row.get('nama_event', 'Event Seni Budaya'))
                    st.info(f"**{nama_event}**")
                    st.write(f"🏛️ **Tempat:** {row.get('tempat_acara', 'Jakarta')}")
                    st.write(f"📍 **Wilayah:** {row.get('wilayah', 'DKI Jakarta')}")
                    st.write(f"🎭 **Kesenian:** {row.get('jenis_kesenian', '-')}")
                    
                    if 'jarak_km' in row:
                        st.caption(f"📏 Jarak: **{row['jarak_km']} km**")
                        
                    # Tombol Langsung ke Google Maps (1x Klik)
                    maps_url = f"https://www.google.com/maps/dir/?api=1&origin={active_lat},{active_lng}&destination={row['latitude']},{row['longitude']}"
                    st.markdown(f"[🗺️ Buka Rute di Google Maps]({maps_url})")
                    st.markdown("---")
            
            # Tombol "Tampilkan Lainnya" jika masih ada data yang belum ditampilkan
            if len(results) > st.session_state.display_limit:
                st.button("Tampilkan Lainnya 👇", on_click=load_more, use_container_width=True)

elif page == "Jelajah Event":
    st.title("Daftar Pergelaran Seni di Jakarta")
    
    wilayah_list = list(df['wilayah'].dropna().unique()) if 'wilayah' in df.columns else []
    selected_wilayah = st.selectbox("Pilih Wilayah", wilayah_list)
    
    if selected_wilayah:
        filtered_res = df[df['wilayah'] == selected_wilayah]
        st.write(f"Menampilkan **{len(filtered_res)}** acara di wilayah **{selected_wilayah}**:")
        
        for _, row in filtered_res.iterrows():
            nama_event = row.get('nama_kegiatan', row.get('nama_event', 'Event Seni Budaya'))
            with st.expander(f"{nama_event} — {row.get('tempat_acara', 'Lokasi Acara')}"):
                st.write(f"**Tanggal/Waktu:** {row.get('waktu', row.get('bulan', '-'))}")
                st.write(f"**Jenis Kesenian:** {row.get('jenis_kesenian', '-')}")
                st.write(f"**Wilayah:** {row.get('wilayah', '-')}")

elif page == "AI Planner":
    st.title("AI Trip Planner - Jakarta Cultural Route")
    st.write("Rencanakan kunjungan ke tempat acara seni budaya di Jakarta berdasarkan minatmu.")
    
    with st.form("ai_form"):
        if 'jenis_kesenian' in df.columns:
            opsi_minat = sorted([str(x).strip() for x in df['jenis_kesenian'].dropna().unique() if str(x).strip() != 'nan'])
        else:
            opsi_minat = ["Semua"]
            
        minat_selected = st.selectbox("Pilih jenis pertunjukan yang ingin kamu kunjungi:", opsi_minat)
        submitted = st.form_submit_button("Buat Rencana Kunjungan")
        
        if submitted:
            active_lat = user_lat if user_lat else -6.1754
            active_lng = user_lng if user_lng else 106.8272
            
            # Tampilkan top 3 untuk AI Planner
            res = recommend_events(user_lat=active_lat, user_lng=active_lng, keyword=minat_selected).head(3)
            st.success("Rekomendasi Berhasil Dibuat!")
            for _, row in res.iterrows():
                nama_event = row.get('nama_kegiatan', row.get('nama_event', 'Event Seni Budaya'))
                maps_url = f"https://www.google.com/maps/dir/?api=1&origin={active_lat},{active_lng}&destination={row['latitude']},{row['longitude']}"
                
                st.markdown(
                    f"- **{nama_event}**\n"
                    f"  - 🏛️ Venue: *{row.get('tempat_acara', '-')}, {row.get('wilayah', '-')}*\n"
                    f"  - 🎭 Kesenian: *{row.get('jenis_kesenian', '-')}*\n"
                    f"  - 📏 Jarak: *{row.get('jarak_km', 0)} km* — [Buka Rute di Google Maps]({maps_url})"
                )
