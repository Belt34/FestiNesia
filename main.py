import math
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_geolocation import streamlit_geolocation

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="FestiNesia - Jakarta Cultural Events", layout="wide")

# 1. Load Dataset Pergelaran Seni & Budaya Jakarta
@st.cache_data
def load_data():
    file_path = 'data_penyelenggaraan_seni_budaya_5_kategori.xlsx'
    
    try:
        # Membaca seluruh tab/sheet sekaligus dan menggabungkannya
        excel_file = pd.read_excel(file_path, sheet_name=None)
        all_sheets = [sheet_df for _, sheet_df in excel_file.items()]
        df = pd.concat(all_sheets, ignore_index=True)
    except Exception as e:
        st.error(f"Gagal membaca file dataset Excel: {e}")
        return pd.DataFrame()

    # Membersihkan nama kolom agar seragam (lowercase & hapus spasi berlebih)
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Pemetaan Titik Koordinat Presisi Berdasarkan 'tempat_acara' Spesifik di Jakarta
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

# 2. Perhitungan Jarak Haversine (Cadangan jika API Rute OSRM offline)
def calculate_haversine(lat1, lon1, lat2, lon2, avg_speed_kmh=30):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = R * c
    
    travel_time_hours = distance_km / avg_speed_kmh
    hours = int(travel_time_hours)
    minutes = int((travel_time_hours - hours) * 60)
    time_str = f"{hours} jam {minutes} menit" if hours > 0 else f"{minutes} menit"
    return round(distance_km, 1), time_str

# 3. Perhitungan Jarak Rute Jalan Raya Real-Time via OSRM API
def get_realtime_route(lat1, lon1, lat2, lon2):
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get("routes"):
                distance_km = round(data["routes"][0]["distance"] / 1000, 1)
                duration_min = round(data["routes"][0]["duration"] / 60)
                
                hours = duration_min // 60
                minutes = duration_min % 60
                time_str = f"{hours} jam {minutes} menit" if hours > 0 else f"{minutes} menit"
                return distance_km, time_str
    except Exception:
        pass
    
    return calculate_haversine(lat1, lon1, lat2, lon2)

# 4. Fungsi Rekomendasi Pergelaran Seni & Budaya
def recommend_events(user_lat=None, user_lng=None, wilayah=None, keyword="", top_n=6):
    filtered_df = df.copy()
    
    if wilayah and wilayah != "Semua":
        filtered_df = filtered_df[filtered_df['wilayah'].astype(str).str.lower() == wilayah.lower()]
        
    if filtered_df.empty:
        return filtered_df

    if user_lat is not None and user_lng is not None:
        distances = []
        times = []
        for _, row in filtered_df.iterrows():
            dist, time_est = get_realtime_route(user_lat, user_lng, row['latitude'], row['longitude'])
            distances.append(dist)
            times.append(time_est)
        filtered_df['jarak_km'] = distances
        filtered_df['estimasi_waktu'] = times

    if keyword:
        # Menggabungkan nama kegiatan dan jenis kesenian jika kolomnya ada
        nama_col = 'nama_kegiatan' if 'nama_kegiatan' in filtered_df.columns else filtered_df.columns[0]
        jenis_col = 'jenis_kesenian' if 'jenis_kesenian' in filtered_df.columns else nama_col
        
        search_corpus = filtered_df[nama_col].fillna('') + " " + filtered_df[jenis_col].fillna('')
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(search_corpus)
        query_vec = vectorizer.transform([keyword])
        similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        filtered_df['score'] = similarity_scores
        
        if user_lat is not None and user_lng is not None:
            results = filtered_df.sort_values(by=['score', 'jarak_km'], ascending=[False, True])
        else:
            results = filtered_df.sort_values(by='score', ascending=False)
    else:
        if user_lat is not None and user_lng is not None:
            results = filtered_df.sort_values(by='jarak_km', ascending=True)
        else:
            results = filtered_df

    return results.head(top_n)

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
    
    # Widget Geolocation Browser Real-Time
    st.subheader("📍 Deteksi Lokasi Real-Time")
    location = streamlit_geolocation()
    
    user_lat, user_lng = None, None
    if location and location.get('latitude') and location.get('longitude'):
        user_lat = location['latitude']
        user_lng = location['longitude']
        st.success(f"Lokasi kamu terdeteksi: **{user_lat:.4f}, {user_lng:.4f}**")
    else:
        st.info("Klik tombol lokasi di atas untuk mengizinkan akses GPS real-time browser kamu.")

    # Form Input Pencarian Menggunakan Dropdown (Tanpa Tab)
    with st.container():
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            if 'jenis_kesenian' in df.columns:
                jenis_kesenian_list = ["Semua"] + sorted([str(x).strip() for x in df['jenis_kesenian'].dropna().unique() if str(x).strip() != 'nan'])
            else:
                jenis_kesenian_list = ["Semua"]
            keyword_input = st.selectbox("Pilih Minat / Jenis Kesenian", jenis_kesenian_list)
        with c2:
            wilayah_list = ["Semua"] + list(df['wilayah'].dropna().unique()) if 'wilayah' in df.columns else ["Semua"]
            wilayah_input = st.selectbox("Wilayah Administrasi", wilayah_list)
        with c3:
            st.write("")
            search_btn = st.button("Cari Event", use_container_width=True)
            
    if search_btn or keyword_input != "Semua":
        selected_keyword = "" if keyword_input == "Semua" else keyword_input
        active_lat = user_lat if user_lat else -6.1754
        active_lng = user_lng if user_lng else 106.8272
        
        results = recommend_events(
            user_lat=active_lat, 
            user_lng=active_lng, 
            wilayah=wilayah_input, 
            keyword=selected_keyword, 
            top_n=6
        )
        
        st.subheader("Hasil Rekomendasi Event")
        if results.empty:
            st.warning("Tidak ditemukan event seni/budaya yang sesuai.")
        else:
            cols = st.columns(3)
            for idx, (_, row) in enumerate(results.iterrows()):
                with cols[idx % 3]:
                    nama_event = row.get('nama_kegiatan', row.get('nama_event', 'Event Seni Budaya'))
                    st.info(f"**{nama_event}**")
                    st.write(f"🏛️ **Tempat:** {row.get('tempat_acara', 'Jakarta')}")
                    st.write(f"📍 **Wilayah:** {row.get('wilayah', 'DKI Jakarta')}")
                    st.write(f"🎭 **Kesenian:** {row.get('jenis_kesenian', '-')}")
                    
                    if 'jarak_km' in row:
                        st.caption(f"🚗 Rute Jalan: **{row['jarak_km']} km** ({row['estimasi_waktu']})")
                    st.markdown("---")

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
            
            res = recommend_events(user_lat=active_lat, user_lng=active_lng, keyword=minat_selected, top_n=3)
            st.success("Rekomendasi Berhasil Dibuat!")
            for _, row in res.iterrows():
                nama_event = row.get('nama_kegiatan', row.get('nama_event', 'Event Seni Budaya'))
                st.markdown(
                    f"- **{nama_event}**\n"
                    f"  - 🏛️ Venue: *{row.get('tempat_acara', '-')}, {row.get('wilayah', '-')}*\n"
                    f"  - 🎭 Kesenian: *{row.get('jenis_kesenian', '-')}*\n"
                    f"  - 🚗 Estimasi Jarak: *{row.get('jarak_km', 0)} km ({row.get('estimasi_waktu', '-')})*"
                )
