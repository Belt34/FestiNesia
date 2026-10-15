import urllib.parse
import math
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_geolocation import streamlit_geolocation

# --- INISIALISASI SESSION STATE UNTUK NAVIGASI & LIMIT ---
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Beranda"

if 'display_limit' not in st.session_state:
    st.session_state.display_limit = 3

if 'jelajah_page' not in st.session_state:
    st.session_state.jelajah_page = 1

def set_tab(tab_name):
    st.session_state.active_tab = tab_name

def reset_limit():
    st.session_state.display_limit = 3

def load_more():
    st.session_state.display_limit += 3

def reset_jelajah_page():
    st.session_state.jelajah_page = 1

def prev_page():
    st.session_state.jelajah_page -= 1

def next_page():
    st.session_state.jelajah_page += 1

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="FestiNesia - Cultural & Art Events", layout="wide")

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
    
    # Standarisasi kolom tanggal untuk filter rentang waktu
    date_col = 'tanggal_kegiatan' if 'tanggal_kegiatan' in df.columns else ('waktu' if 'waktu' in df.columns else None)
    if date_col:
        df['parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        df['parsed_date'] = pd.NaT
        
    return df

df = load_data()

# 2. Fungsi Rekomendasi (Dengan Filter Rentang Waktu Max 1 Minggu untuk Multi-Event)
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

def get_multi_event_package(results_df):
    """Mengambil maksimal 3 event yang tanggalnya berdekatan (dalam rentang waktu maksimal 7 hari)"""
    if results_df.empty:
        return results_df
        
    # Urutkan berdasarkan tanggal jika ada
    valid_dates = results_df.dropna(subset=['parsed_date']).sort_values(by='parsed_date')
    invalid_dates = results_df[results_df['parsed_date'].isna()]
    
    sorted_df = pd.concat([valid_dates, invalid_dates])
    
    if len(sorted_df) <= 1:
        return sorted_df.head(3)
        
    # Ambil baris pertama sebagai acuan tanggal utama
    first_date = sorted_df.iloc[0]['parsed_date']
    
    if pd.isna(first_date):
        return sorted_df.head(3)
        
    # Filter event lain yang selisih waktunya maksimal 7 hari dari event pertama
    time_filtered = sorted_df[
        (sorted_df['parsed_date'].isna()) | 
        ((sorted_df['parsed_date'] - first_date).dt.days.abs() <= 7)
    ]
    
    return time_filtered.head(3)

# --- TAMPILAN FRONT-END STREAMLIT ---
page = st.session_state.active_tab

if page == "Beranda":
    st.title("Selamat Datang di FestiNesia 🎭")
    st.caption("Platform informasi pergelaran seni, budaya, dan festival lokal.")
    
    st.markdown("---")
    st.subheader("🚀 Navigasi Cepat")
    st.write("Silakan pilih menu di bawah ini untuk mulai menjelajahi acara atau merencanakan perjalanan budayamu:")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗺️ Jelajah Semua Event", use_container_width=True):
            st.session_state.active_tab = "Jelajah Event"
            st.rerun()
    with col_btn2:
        if st.button("🤖 Buka AI Trip Planner", use_container_width=True):
            st.session_state.active_tab = "AI Planner"
            st.rerun()
            
    st.markdown("---")
    st.info("💡 **Tips:** Gunakan **AI Planner** untuk mencari rekomendasi pertunjukan spesifik berdasarkan minat dan wilayah, serta buat rute multi-event berdekatan dalam seminggu!")

elif page == "Jelajah Event":
    if st.button("🏠 Kembali ke Beranda"):
        st.session_state.active_tab = "Beranda"
        st.rerun()
        
    st.title("Daftar Pergelaran Seni & Budaya")
    
    wilayah_list = list(df['wilayah'].dropna().unique()) if 'wilayah' in df.columns else []
    selected_wilayah = st.selectbox("Pilih Wilayah", wilayah_list, on_change=reset_jelajah_page)
    
    if selected_wilayah:
        filtered_res = df[df['wilayah'] == selected_wilayah]
        total_items = len(filtered_res)
        
        st.write(f"Menampilkan total **{total_items}** acara di wilayah **{selected_wilayah}**:")
        
        items_per_page = 10
        total_pages = math.ceil(total_items / items_per_page) if total_items > 0 else 1
        
        if st.session_state.jelajah_page > total_pages:
            st.session_state.jelajah_page = total_pages
        if st.session_state.jelajah_page < 1:
            st.session_state.jelajah_page = 1
            
        start_idx = (st.session_state.jelajah_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        displayed_res = filtered_res.iloc[start_idx:end_idx]
        
        for _, row in displayed_res.iterrows():
            nama_event = row.get('nama_kegiatan', row.get('nama_event', 'Event Seni Budaya'))
            tanggal_val = row.get('tanggal_kegiatan', row.get('waktu', row.get('bulan', '-')))
            
            with st.expander(f"{nama_event} — {row.get('tempat_acara', 'Lokasi Acara')}"):
                st.write(f"**Tanggal:** {tanggal_val}")
                st.write(f"**Jenis Kesenian:** {row.get('jenis_kesenian', '-')}")
                st.write(f"**Wilayah:** {row.get('wilayah', '-')}")
                
        if total_pages > 1:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.button("⬅️ Sebelumnya", on_click=prev_page, disabled=(st.session_state.jelajah_page == 1), use_container_width=True)
                
            with col2:
                st.markdown(f"<div style='text-align: center; padding-top: 10px;'><b>Halaman {st.session_state.jelajah_page} dari {total_pages}</b></div>", unsafe_allow_html=True)
                
            with col3:
                st.button("Selanjutnya ➡️", on_click=next_page, disabled=(st.session_state.jelajah_page == total_pages), use_container_width=True)

elif page == "AI Planner":
    if st.button("🏠 Kembali ke Beranda"):
        st.session_state.active_tab = "Beranda"
        st.rerun()
        
    st.title("AI Trip Planner - Cultural Route")
    st.write("Rencanakan kunjungan ke tempat acara seni budaya berdasarkan minat dan lokasimu.")
    
    st.subheader("📍 Deteksi Lokasi Real-Time")
    location = streamlit_geolocation()
    
    user_lat, user_lng = None, None
    if location and location.get('latitude') and location.get('longitude'):
        user_lat = location['latitude']
        user_lng = location['longitude']
        st.success(f"Lokasi kamu terdeteksi: **{user_lat:.4f}, {user_lng:.4f}**")
    else:
        st.info("Klik tombol lokasi di atas agar Rute Google Maps mengarah dari posisimu saat ini.")

    with st.form("ai_form"):
        c1, c2 = st.columns(2)
        with c1:
            if 'jenis_kesenian' in df.columns:
                jenis_kesenian_list = ["Semua"] + sorted([str(x).strip() for x in df['jenis_kesenian'].dropna().unique() if str(x).strip() != 'nan'])
            else:
                jenis_kesenian_list = ["Semua"]
            minat_selected = st.selectbox("Pilih Jenis Kesenian / Minat:", jenis_kesenian_list)
        with c2:
            wilayah_list = ["Semua"] + list(df['wilayah'].dropna().unique()) if 'wilayah' in df.columns else ["Semua"]
            wilayah_input = st.selectbox("Wilayah / Kota", wilayah_list)
            
        submitted = st.form_submit_button("Buat Rencana Kunjungan", use_container_width=True)
        
        if submitted:
            selected_keyword = "" if minat_selected == "Semua" else minat_selected
            active_lat = user_lat if user_lat else -6.1754
            active_lng = user_lng if user_lng else 106.8272
            
            results = recommend_events(
                wilayah=wilayah_input, 
                keyword=selected_keyword
            )
            
            st.success("Rekomendasi Berhasil Dibuat!")
            if results.empty:
                st.warning("Tidak ditemukan event seni/budaya yang sesuai dengan filter tersebut.")
            else:
                tab_single, tab_multi = st.tabs(["✨ Rekomendasi Single Event", "🗺️ Paket Multi-Event (Waktu Berdekatan)"])
                
                with tab_single:
                    displayed_results = results.head(3)
                    cols = st.columns(3)
                    for idx, (_, row) in enumerate(displayed_results.iterrows()):
                        with cols[idx % 3]:
                            nama_event = row.get('nama_kegiatan', row.get('nama_event', 'Event Seni Budaya'))
                            tempat_acara = str(row.get('tempat_acara', '')).strip()
                            wilayah_acara = str(row.get('wilayah', 'Indonesia')).strip()
                            tanggal_val = row.get('tanggal_kegiatan', row.get('waktu', row.get('bulan', '-')))
                            
                            st.info(f"**{nama_event}**")
                            st.write(f"🏛️ **Tempat:** {tempat_acara if tempat_acara and tempat_acara.lower() != 'nan' else '-'}")
                            st.write(f"📍 **Wilayah:** {wilayah_acara}")
                            st.write(f"📅 **Tanggal:** {tanggal_val}")
                            st.write(f"🎭 **Kesenian:** {row.get('jenis_kesenian', '-')}")
                            
                            if tempat_acara and tempat_acara.lower() != 'nan':
                                tujuan_query = urllib.parse.quote_plus(f"{tempat_acara}, {wilayah_acara}")
                            else:
                                tujuan_query = urllib.parse.quote_plus(wilayah_acara)
                                
                            maps_url = f"https://www.google.com/maps/dir/?api=1&origin={active_lat},{active_lng}&destination={tujuan_query}"
                            st.markdown(f"[🗺️ Buka Rute di Google Maps]({maps_url})")
                            st.markdown("---")
                            
                with tab_multi:
                    st.write("### 🧭 Paket Kunjungan Multi-Event (Rentang Waktu Maksimal 1 Minggu)")
                    st.caption("Rangkaian 3 event di wilayah yang sama dengan jadwal pelaksanaan yang berdekatan.")
                    
                    multi_results = get_multi_event_package(results)
                    waypoints_query = []
                    
                    for i, (_, row) in enumerate(multi_results.iterrows(), 1):
                        nama_event = row.get('nama_kegiatan', row.get('nama_event', 'Event Seni Budaya'))
                        tempat_acara = str(row.get('tempat_acara', '')).strip()
                        wilayah_acara = str(row.get('wilayah', 'Indonesia')).strip()
                        tanggal_val = row.get('tanggal_kegiatan', row.get('waktu', row.get('bulan', '-')))
                        
                        venue_str = tempat_acara if tempat_acara and tempat_acara.lower() != 'nan' else wilayah_acara
                        waypoints_query.append(f"{venue_str}, {wilayah_acara}")
                        
                        st.markdown(f"**{i}. {nama_event}**")
                        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp; 🏛️ *Venue:* {venue_str}")
                        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp; 📅 *Tanggal:* {tanggal_val}")
                        
                    if len(waypoints_query) > 0:
                        destination_final = urllib.parse.quote_plus(waypoints_query[-1])
                        
                        if len(waypoints_query) > 1:
                            waypoints_list = "|".join([urllib.parse.quote_plus(wp) for wp in waypoints_query[:-1]])
                            multi_maps_url = f"https://www.google.com/maps/dir/?api=1&origin={active_lat},{active_lng}&destination={destination_final}&waypoints={waypoints_list}"
                        else:
                            multi_maps_url = f"https://www.google.com/maps/dir/?api=1&origin={active_lat},{active_lng}&destination={destination_final}"
                            
                        st.markdown("---")
                        st.markdown(f"👉 **[🗺️ Buka Rute Lengkap Multi-Event di Google Maps]({multi_maps_url})**")
