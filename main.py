import math
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_geolocation import streamlit_geolocation

st.set_page_config(page_title="FestiNesia", layout="wide")

# 1. Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('KEN_2026_Event_Kemenpar.csv')
    
    KOTA_COORDS = {
        "Kota Banda Aceh": (5.5483, 95.3238),
        "Kota Medan": (3.5952, 98.6722),
        "Kab. Samosir": (2.6053, 98.7186),
        "Kota Pariaman": (-0.6264, 100.1189),
        "Kota Bandung": (-6.9175, 107.6191),
        "Kota Jakarta Pusat": (-6.1805, 106.8284)
    }
    
    df['latitude'] = df['kabupaten_kota'].map(lambda x: KOTA_COORDS.get(x, (-6.1805, 106.8284))[0])
    df['longitude'] = df['kabupaten_kota'].map(lambda x: KOTA_COORDS.get(x, (-6.1805, 106.8284))[1])
    return df

df = load_data()

# 2. Perhitungan Jarak Haversine (Fallback jika OSRM gagal/offline)
def calculate_haversine(lat1, lon1, lat2, lon2, avg_speed_kmh=50):
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
    # Format OSRM: /driving/{longitude1},{latitude1};{longitude2},{latitude2}
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
    
    # Jika API OSRM error/timeout, gunakan rumus Haversine sebagai cadangan
    return calculate_haversine(lat1, lon1, lat2, lon2)

# 4. Fungsi Rekomendasi Festival
def recommend_festivals(user_lat=None, user_lng=None, provinsi=None, bulan=None, keyword="", top_n=5):
    filtered_df = df.copy()
    if provinsi:
        filtered_df = filtered_df[filtered_df['provinsi'].str.lower() == provinsi.lower()]
    if bulan:
        filtered_df = filtered_df[filtered_df['bulan_mulai'].str.lower() == bulan.lower()]
    
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
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(filtered_df['nama_event'])
        query_vec = vectorizer.transform([keyword])
        similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        filtered_df['score'] = similarity_scores
        filtered_df['score'] += filtered_df['top_10_ken'].apply(lambda x: 0.2 if x == 'Ya' else 0.0)
        
        if user_lat is not None and user_lng is not None:
            results = filtered_df.sort_values(by=['jarak_km', 'score'], ascending=[True, False])
        else:
            results = filtered_df.sort_values(by='score', ascending=False)
    else:
        if user_lat is not None and user_lng is not None:
            results = filtered_df.sort_values(by='jarak_km', ascending=True)
        else:
            results = filtered_df

    return results.head(top_n)

# --- FRONT-END STREAMLIT ---
col_logo, col_nav = st.columns([2, 6])
with col_logo:
    st.markdown("### 🌟 FestiNesia")
with col_nav:
    page = st.radio("", ["Beranda", "Jelajah", "AI Planner"], horizontal=True)

st.divider()

if page == "Beranda":
    st.title("Temukan Festival, Jelajahi Indonesia.")
    st.caption("Platform wisata budaya berbasis AI untuk membantu kamu menemukan festival dan destinasi terbaik.")
    
    # Widget Geolocation Browser Real-Time
    st.subheader("📍 Dapatkan Lokasi Real-Time Kamu")
    location = streamlit_geolocation()
    
    user_lat, user_lng = None, None
    if location and location.get('latitude') and location.get('longitude'):
        user_lat = location['latitude']
        user_lng = location['longitude']
        st.success(f"Lokasi terdeteksi: {user_lat:.4f}, {user_lng:.4f}")
    else:
        st.info("Klik tombol di atas untuk membagikan lokasi GPS kamu.")

    # Input Pencarian Cepat
    with st.container():
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            keyword_input = st.text_input("Cari Berdasarkan Minat / Nama Event", placeholder="Misal: Music, Coffee, Budaya...")
        with c2:
            provinsi_input = st.selectbox("Provinsi (Opsional)", ["Semua"] + list(df['provinsi'].dropna().unique()))
        with c3:
            st.write("")
            search_btn = st.button("Cari Event", use_container_width=True)
            
    if search_btn or keyword_input:
        prov_filter = None if provinsi_input == "Semua" else provinsi_input
        
        # Fallback jika user belum klik tombol lokasi: Gunakan koordinat default Medan
        active_lat = user_lat if user_lat else 3.5952
        active_lng = user_lng if user_lng else 98.6722
        
        results = recommend_festivals(user_lat=active_lat, user_lng=active_lng, provinsi=prov_filter, keyword=keyword_input, top_n=6)
        
        st.subheader("Hasil Pencarian Festival")
        if results.empty:
            st.warning("Tidak ditemukan event yang sesuai.")
        else:
            cols = st.columns(3)
            for idx, (_, row) in enumerate(results.iterrows()):
                with cols[idx % 3]:
                    st.info(f"**{row['nama_event']}**")
                    st.write(f"📍 {row['kabupaten_kota']}, {row['provinsi']}")
                    if 'jarak_km' in row:
                        st.caption(f"🚗 Rute Jalan: **{row['jarak_km']} km** ({row['estimasi_waktu']})")
                    st.markdown("---")
