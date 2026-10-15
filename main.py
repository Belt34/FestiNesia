import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Dataset
df = pd.read_csv('/KEN_2026_Event_Kemenpar.csv')

# --- MOCK DATA COORDINATES (Tambahkan kolom lat & lng ke dataset) ---
# Pada praktik nyata, koordinat ini bisa kamu peroleh melalui proses Geocoding.
# Di sini dibuat contoh koordinat default untuk beberapa kota utama:
KOTA_COORDS = {
    "Kota Banda Aceh": (5.5483, 95.3238),
    "Kota Medan": (3.5952, 98.6722),
    "Kab. Samosir": (2.6053, 98.7186),
    "Kota Pariaman": ( -0.6264, 100.1189),
    "Kota Bandung": (-6.9175, 107.6191),
    "Kota Jakarta Pusat": (-6.1805, 106.8284)
}

# Asign latitude & longitude ke dataframe (default koordinat Jakarta Pusat jika belum ada di mapping)
df['latitude'] = df['kabupaten_kota'].map(lambda x: KOTA_COORDS.get(x, (-6.1805, 106.8284))[0])
df['longitude'] = df['kabupaten_kota'].map(lambda x: KOTA_COORDS.get(x, (-6.1805, 106.8284))[1])


# 2. Fungsi Perhitungan Jarak (Haversine Formula) & Waktu
def calculate_distance_and_time(lat1, lon1, lat2, lon2, avg_speed_kmh=50):
    """
    Menghitung jarak garis lurus (dalam km) dan estimasi waktu perjalanan darat.
    """
    R = 6371.0  # Radius bumi dalam Kilometer
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = R * c

    # Estimasi waktu (Kecepatan rata-rata kendaraan darat/mobil = 50 km/jam)
    travel_time_hours = distance_km / avg_speed_kmh
    hours = int(travel_time_hours)
    minutes = int((travel_time_hours - hours) * 60)

    if hours > 0:
        time_str = f"{hours} jam {minutes} menit"
    else:
        time_str = f"{minutes} menit"

    return round(distance_km, 1), time_str


# 3. Fungsi Rekomendasi Festival
def recommend_festivals(user_lat=None, user_lng=None, provinsi=None, bulan=None, keyword="", top_n=5):
    filtered_df = df.copy()

    # Filter Lokasi (opsional)
    if provinsi:
        filtered_df = filtered_df[filtered_df['provinsi'].str.lower() == provinsi.lower()]

    # Filter Bulan (opsional)
    if bulan:
        filtered_df = filtered_df[filtered_df['bulan_mulai'].str.lower() == bulan.lower()]

    if filtered_df.empty:
        print("Tidak ditemukan event yang sesuai dengan filter lokasi/bulan.")
        return

    # Hitung Jarak & Estimasi Waktu jika koordinat user dimasukkan
    if user_lat is not None and user_lng is not None:
        distances = []
        times = []
        for _, row in filtered_df.iterrows():
            dist, time_est = calculate_distance_and_time(user_lat, user_lng, row['latitude'], row['longitude'])
            distances.append(dist)
            times.append(time_est)

        filtered_df['jarak_km'] = distances
        filtered_df['estimasi_waktu'] = times

    # Search Similarity berbasis Keyword / Minat
    if keyword:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(filtered_df['nama_event'])
        query_vec = vectorizer.transform([keyword])

        # Hitung similarity score
        similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        filtered_df['score'] = similarity_scores

        # Berikan bonus skor untuk Top 10 KEN
        filtered_df['score'] += filtered_df['top_10_ken'].apply(lambda x: 0.2 if x == 'Ya' else 0.0)

        # Pengurutan: gabungkan aspek jarak dan kesesuaian keyword
        if user_lat is not None and user_lng is not None:
            # Urutkan berdasarkan jarak terdekat dan skor tertinggi
            results = filtered_df.sort_values(by=['jarak_km', 'score'], ascending=[True, False])
        else:
            results = filtered_df.sort_values(by='score', ascending=False)
    else:
        if user_lat is not None and user_lng is not None:
            results = filtered_df.sort_values(by='jarak_km', ascending=True)
        else:
            results = filtered_df

    # --- FORMAT OUTPUT SESUAI REQUEST ---
    print("festival, lokasi")
    print("-" * 50)
    for _, row in results.head(top_n).iterrows():
        nama = row['nama_event']
        lokasi = f"{row['kabupaten_kota']}, {row['provinsi']}"

        if user_lat is not None and user_lng is not None:
            info_jarak = f"({row['jarak_km']} km / {row['estimasi_waktu']})"
        else:
            info_jarak = ""

        print(f"{nama} - {lokasi} {info_jarak}")


# --- CONTOH PENGGUNAAN ---
# Misal User berada di Kota Medan (Lat: 3.5952, Lng: 98.6722)
user_lat_medan = 3.5952
user_lng_medan = 98.6722

recommend_festivals(
    user_lat=user_lat_medan,
    user_lng=user_lng_medan,
    keyword="Music Coffee",
    top_n=5
)