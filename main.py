import urllib.parse
import math
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_geolocation import streamlit_geolocation

# --- INISIALISASI SESSION STATE UNTUK PAGINATION & LIMIT ---
if 'display_limit' not in st.session_state:
    st.session_state.display_limit = 3

if 'jelajah_page' not in st.session_state:
    st.session_state.jelajah_page = 1

def reset_limit():
    st.session_state.display_limit = 3

def load_more():
    st.session_state.display_limit += 3

# Fungsi navigasi pagination untuk tab Jelajah
def reset_jelajah_page():
    st.session_state.jelajah_page = 1

def prev_page():
    st.session_state.jelajah_page -= 1

def next_page():
    st.session_state.jelajah_page += 1

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
    return df

df = load_data()

# 2. Fungsi Rekomendasi (Hanya Berdasarkan TF-IDF Keyword & Filter)
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
col_logo, col_nav = st.columns([2, 6])
with col_logo:
    st.markdown("### 🌟 FestiNesia Jakarta")
with col_nav:
    page = st.radio("", ["Beranda", "Jelajah Event", "AI Planner"], horizontal=True)

st.divider()

if page == "Beranda":
    st.title("Pergelaran Seni & Budaya DKI Jakarta")
    st.caption("Temukan pertunjukan seni, bazar budaya, dan festival lokal di sekitarmu.")
    
    st.subheader("📍 Deteksi Lokasi Real-Time")
    location = streamlit_geolocation()
    
    user_lat, user_lng = None, None
    if location and location.get('latitude') and location.get('longitude'):
        user_lat = location['latitude']
        user_lng = location['longitude']
        st.success(f"Lokasi kamu terdeteksi: **{user_lat:.4f}, {user_lng:.4f}**")
    else:
        st.info("Klik tombol lokasi di atas agar Rute Google Maps mengarah dari posisimu saat ini.")

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
        
        results = recommend_events(
            wilayah=wilayah_input, 
            keyword=selected_keyword
        )
        
        st.subheader("Hasil Rekomendasi Event")
        if results.empty:
            st.warning("Tidak ditemukan event seni/budaya yang sesuai.")
        else:
            displayed_results = results.head(st.session_state.display_limit)
            cols = st.columns(3)
            for idx, (_, row) in enumerate(displayed_results.iterrows()):
                with cols[idx % 3]:
                    nama_event = row.get('nama_kegiatan', row.get('nama_event', 'Event Seni Budaya'))
                    tempat_acara = str(row.get('tempat_acara', '')).strip()
                    wilayah_acara = str(row.get('wilayah', 'DKI Jakarta')).strip()
                    
                    st.info(f"**{nama_event}**")
                    st.write(f"🏛️ **Tempat:** {tempat_acara if tempat_acara and tempat_acara.lower() != 'nan' else '-'}")
                    st.write(f"📍 **Wilayah:** {wilayah_acara}")
                    st.write(f"🎭 **Kesenian:** {row.get('jenis_kesenian', '-')}")
                    
                    if tempat_acara and tempat_acara.lower() != 'nan':
                        tujuan_query = urllib.parse.quote_plus(f"{tempat_acara}, {wilayah_acara}")
                    else:
                        tujuan_query = urllib.parse.quote_plus(wilayah_acara)
                        
                    maps_url = f"https://www.google.com/maps/dir/?api=1&origin={active_lat},{active_lng}&destination={tujuan_query}"
                    st.markdown(f"[🗺️ Buka Rute di Google Maps]({maps_url})")
                    st.markdown("---")
            
            if len(results) > st.session_state.display_limit:
                st.button("Tampilkan Lainnya 👇", on_click=load_more, use_container_width=True)

elif page == "Jelajah Event":
    st.title("Daftar Pergelaran Seni di Jakarta")
    
    wilayah_list = list(df['wilayah'].dropna().unique()) if 'wilayah' in df.columns else []
    selected_wilayah = st.selectbox("Pilih Wilayah", wilayah_list, on_change=reset_jelajah_page)
    
    if selected_wilayah:
        filtered_res = df[df['wilayah'] == selected_wilayah]
        total_items = len(filtered_res)
        
        st.write(f"Menampilkan total **{total_items}** acara di wilayah **{selected_wilayah}**:")
        
        # --- LOGIKA PAGINATION ---
        items_per_page = 10
        total_pages = math.ceil(total_items / items_per_page) if total_items > 0 else 1
        
        # Pengaman nilai halaman agar tidak error
        if st.session_state.jelajah_page > total_pages:
            st.session_state.jelajah_page = total_pages
        if st.session_state.jelajah_page < 1:
            st.session_state.jelajah_page = 1
            
        start_idx = (st.session_state.jelajah_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        # Memotong dataframe sesuai halaman saat ini (maks 10 baris)
        displayed_res = filtered_res.iloc[start_idx:end_idx]
        
        for _, row in displayed_res.iterrows():
            nama_event = row.get('nama_kegiatan', row.get('nama_event', 'Event Seni Budaya'))
            with st.expander(f"{nama_event} — {row.get('tempat_acara', 'Lokasi Acara')}"):
                st.write(f"**Tanggal:** {row.get('waktu', row.get('bulan', '-'))}")
                st.write(f"**Jenis Kesenian:** {row.get('jenis_kesenian', '-')}")
                st.write(f"**Wilayah:** {row.get('wilayah', '-')}")
                
        # --- KONTROL TOMBOL PAGINATION DI BAWAH ---
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
            active_lat = st.session_state.get('user_lat', -6.1754) if 'user_lat' in st.session_state else -6.1754
            active_lng = st.session_state.get('user_lng', 106.8272) if 'user_lng' in st.session_state else 106.8272
            
            res = recommend_events(keyword=minat_selected).head(3)
            st.success("Rekomendasi Berhasil Dibuat!")
            for _, row in res.iterrows():
                nama_event = row.get('nama_kegiatan', row.get('nama_event', 'Event Seni Budaya'))
                tempat_acara = str(row.get('tempat_acara', '')).strip()
                wilayah_acara = str(row.get('wilayah', 'DKI Jakarta')).strip()
                
                if tempat_acara and tempat_acara.lower() != 'nan':
                    tujuan_query = urllib.parse.quote_plus(f"{tempat_acara}, {wilayah_acara}")
                else:
                    tujuan_query = urllib.parse.quote_plus(wilayah_acara)
                    
                maps_url = f"https://www.google.com/maps/dir/?api=1&origin={active_lat},{active_lng}&destination={tujuan_query}"
                
                st.markdown(
                    f"- **{nama_event}**\n"
                    f"  - 🏛️ Venue: *{tempat_acara if tempat_acara and tempat_acara.lower() != 'nan' else '-'}, {wilayah_acara}*\n"
                    f"  - 🎭 Kesenian: *{row.get('jenis_kesenian', '-')}*\n"
                    f"  - [🗺️ Buka Rute di Google Maps]({maps_url})"
                )
