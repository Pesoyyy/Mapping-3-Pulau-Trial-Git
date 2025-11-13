import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# =============================================================================
# KONFIGURASI HALAMAN Streamlit
# =============================================================================
st.set_page_config(
    page_title="Pinpoint Map 3 Pulau - Full Data",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# KONFIGURASI & UTILITIES - BATAS SANGAT KETAT UNTUK INDONESIA
# =============================================================================
class Config:
    # Batas SANGAT KETAT untuk Indonesia (hanya wilayah daratan utama)
    # Menghindari Malaysia, Singapura, Brunei, Timor Leste, dan Papua Nugini
    INDONESIA_STRICT_BOUNDS = {
        # Sumatera: hindari Malaysia dan Singapura
        'sumatera': {
            'min_lat': -6.0, 'max_lat': 5.5,
            'min_lon': 95.0, 'max_lon': 105.8  # Hindari Malaysia barat
        },
        # Jawa: relatif aman
        'jawa': {
            'min_lat': -8.9, 'max_lat': -5.5,
            'min_lon': 105.5, 'max_lon': 114.5
        },
        # Kalimantan: hindari Malaysia dan Brunei
        'kalimantan': {
            'min_lat': -4.2, 'max_lat': 4.0,    # Batasi utara untuk hindari Malaysia/Brunei
            'min_lon': 108.5, 'max_lon': 119.0
        },
        # Bali dan Nusa Tenggara
        'bali_nt': {
            'min_lat': -9.5, 'max_lat': -8.0,
            'min_lon': 115.0, 'max_lon': 119.0
        },
        # Sulawesi
        'sulawesi': {
            'min_lat': -6.5, 'max_lat': 1.5,
            'min_lon': 118.5, 'max_lon': 125.0
        }
    }
    
    # Area yang DILARANG (negara tetangga dan laut)
    FORBIDDEN_AREAS = [
        # Malaysia Barat (Semenanjung Malaya)
        {'min_lat': 1.0, 'max_lat': 7.0, 'min_lon': 99.0, 'max_lon': 105.0},
        # Singapura
        {'min_lat': 1.2, 'max_lat': 1.5, 'min_lon': 103.5, 'max_lon': 104.2},
        # Malaysia Timur (Sabah & Sarawak) dan Brunei
        {'min_lat': 4.0, 'max_lat': 8.0, 'min_lon': 109.0, 'max_lon': 120.0},
        # Filipina selatan
        {'min_lat': 4.0, 'max_lat': 7.0, 'min_lon': 116.0, 'max_lon': 127.0},
        # Timor Leste
        {'min_lat': -9.5, 'max_lat': -8.0, 'min_lon': 124.0, 'max_lon': 127.5},
        # Papua Nugini
        {'min_lat': -11.0, 'max_lat': -1.0, 'min_lon': 140.0, 'max_lon': 155.0}
    ]
    
    PARQUET_FILES = {
        'matched': 'src/esb_3pulau_ultra_precise_matches.parquet',
        'esb': 'src/Tarikan_data_ESB_3_Pulau_2025.parquet', 
        'scraping': 'src/data_3_pulau_final.parquet'
    }

def is_in_forbidden_area(lat, lon):
    """Cek apakah koordinat berada di area terlarang (negara tetangga)"""
    for area in Config.FORBIDDEN_AREAS:
        if (area['min_lat'] <= lat <= area['max_lat'] and 
            area['min_lon'] <= lon <= area['max_lon']):
            return True
    return False

def validate_indonesia_strict(lat, lon):
    """Validasi KOORDINAT INDONESIA dengan SANGAT KETAT"""
    if pd.isna(lat) or pd.isna(lon):
        return False
    
    # Validasi range dasar
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return False
    
    # Cek apakah di area terlarang (negara tetangga)
    if is_in_forbidden_area(lat, lon):
        return False
    
    # Cek apakah berada di salah satu wilayah Indonesia yang diizinkan
    for region, bounds in Config.INDONESIA_STRICT_BOUNDS.items():
        if (bounds['min_lat'] <= lat <= bounds['max_lat'] and 
            bounds['min_lon'] <= lon <= bounds['max_lon']):
            return True
    
    return False

def detect_and_fix_coordinate_issues(df, lat_col, lon_col, dataset_name):
    """Deteksi dan perbaiki masalah koordinat dengan lebih agresif"""
    if df.empty:
        return df
    
    issues_found = 0
    corrections_made = 0
    
    st.write(f"🔍 **Analisis mendalam {dataset_name}:**")
    
    # Analisis statistik koordinat
    lat_stats = df[lat_col].describe()
    lon_stats = df[lon_col].describe()
    
    st.write(f"   - Latitude: min={lat_stats['min']:.4f}, max={lat_stats['max']:.4f}")
    st.write(f"   - Longitude: min={lon_stats['min']:.4f}, max={lon_stats['max']:.4f}")
    
    # Deteksi koordinat yang jelas-jelas salah
    clearly_wrong = df[
        (df[lat_col].abs() > 90) | 
        (df[lon_col].abs() > 180) |
        (df[lat_col] == 0) | (df[lon_col] == 0)  # Koordinat (0,0) biasanya salah
    ]
    
    if len(clearly_wrong) > 0:
        st.warning(f"   ⚠️ Ditemukan {len(clearly_wrong)} koordinat yang jelas-jelas salah")
        issues_found += len(clearly_wrong)
    
    # Deteksi koordinat di negara tetangga
    in_neighbor_countries = df[df.apply(
        lambda row: is_in_forbidden_area(row[lat_col], row[lon_col]), axis=1
    )]
    
    if len(in_neighbor_countries) > 0:
        st.error(f"   ❌ Ditemukan {len(in_neighbor_countries)} koordinat di negara tetangga:")
        # Tampilkan sample
        sample = in_neighbor_countries[[lat_col, lon_col]].head(5)
        for idx, row in sample.iterrows():
            st.write(f"      - Lat: {row[lat_col]:.4f}, Lon: {row[lon_col]:.4f}")
        issues_found += len(in_neighbor_countries)
    
    # Hanya kembalikan data yang valid
    valid_mask = df.apply(lambda row: validate_indonesia_strict(row[lat_col], row[lon_col]), axis=1)
    valid_data = df[valid_mask].copy()
    
    removed_count = len(df) - len(valid_data)
    if removed_count > 0:
        st.success(f"   ✅ Diproses: {len(df)} → {len(valid_data)} records ({removed_count} dihapus)")
    
    return valid_data

def clean_and_validate_coordinates_strict(df, lat_col, lon_col, dataset_name):
    """Bersihkan dan validasi koordinat dengan pendekatan sangat ketat"""
    if df.empty:
        st.warning(f"⚠️ {dataset_name}: Dataframe kosong")
        return df
    
    # Pastikan kolom ada
    if lat_col not in df.columns or lon_col not in df.columns:
        st.error(f"❌ {dataset_name}: Kolom {lat_col} atau {lon_col} tidak ditemukan")
        return pd.DataFrame()
    
    # Drop rows dengan koordinat null
    df_clean = df.dropna(subset=[lat_col, lon_col]).copy()
    
    if df_clean.empty:
        st.warning(f"⚠️ {dataset_name}: Tidak ada data setelah dropna")
        return df_clean
    
    # Konversi ke numeric
    df_clean[lat_col] = pd.to_numeric(df_clean[lat_col], errors='coerce')
    df_clean[lon_col] = pd.to_numeric(df_clean[lon_col], errors='coerce')
    
    # Hapus NaN setelah konversi
    df_clean = df_clean.dropna(subset=[lat_col, lon_col])
    
    if df_clean.empty:
        st.warning(f"⚠️ {dataset_name}: Tidak ada data setelah konversi numeric")
        return df_clean
    
    # Validasi sangat ketat
    st.info(f"🧹 **Memproses {dataset_name}:** {len(df)} records awal")
    df_valid = detect_and_fix_coordinate_issues(df_clean, lat_col, lon_col, dataset_name)
    
    return df_valid

# =============================================================================
# FUNGSI LOAD DATA YANG LEBIH KETAT
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def load_and_process_data_strict():
    """Load dan proses data dengan validasi SANGAT KETAT"""
    
    # Check files
    st.write("🔍 **Memeriksa file...**")
    for file_type, file_path in Config.PARQUET_FILES.items():
        if os.path.exists(file_path):
            st.success(f"✅ {file_type}: {file_path}")
        else:
            st.error(f"❌ {file_type}: {file_path} - FILE TIDAK DITEMUKAN")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    try:
        # Load data
        st.write("📥 **Loading data...**")
        df_matched = pd.read_parquet(Config.PARQUET_FILES['matched'])
        df_esb = pd.read_parquet(Config.PARQUET_FILES['esb'])
        df_scraping = pd.read_parquet(Config.PARQUET_FILES['scraping'])
        
        st.success(f"📊 Data loaded - Matched: {len(df_matched):,}, ESB: {len(df_esb):,}, Scraping: {len(df_scraping):,}")
        
    except Exception as e:
        st.error(f"❌ Error loading files: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # =========================================================================
    # PROSES DATA MATCHED DENGAN VALIDASI SANGAT KETAT
    # =========================================================================
    df_matched_clean = clean_and_validate_coordinates_strict(
        df_matched, 'latitude_esb', 'longitude_esb', "Matched Data"
    )
    
    green_data = pd.DataFrame()
    if not df_matched_clean.empty:
        try:
            green_data = df_matched_clean[[
                'brand_pulau', 'longitude_esb', 'latitude_esb', 
                'address_esb', 'address_pulau', 'final_score', 'match_confidence'
            ]].copy()
            
            green_data = green_data.rename(columns={
                'brand_pulau': 'nama_restoran',
                'latitude_esb': 'lat',
                'longitude_esb': 'lon',
                'address_esb': 'alamat_esb',
                'address_pulau': 'alamat_pulau',
                'final_score': 'similarity_score',
                'match_confidence': 'confidence'
            })
            
            green_data['kategori'] = 'Match'
            green_data['color'] = 'green'
            green_data['source'] = 'matched'
            
        except Exception as e:
            st.error(f"❌ Error processing matched data: {e}")
    
    # =========================================================================
    # PROSES DATA ESB DENGAN VALIDASI SANGAT KETAT
    # =========================================================================
    df_esb_clean = clean_and_validate_coordinates_strict(
        df_esb, 'latitude', 'longitude', "ESB Data"
    )
    
    orange_data = pd.DataFrame()
    if not df_esb_clean.empty:
        try:
            orange_data = df_esb_clean[[
                'branchName', 'brandName', 'latitude', 'longitude', 
                'address', 'cityName', 'provinceName'
            ]].copy()
            
            orange_data['nama_restoran'] = orange_data.apply(
                lambda row: f"{row['brandName']} - {row['branchName']}" 
                if pd.notna(row['branchName']) and str(row['branchName']).strip() != '' 
                else row['brandName'],
                axis=1
            )
            
            orange_data = orange_data.rename(columns={
                'latitude': 'lat',
                'longitude': 'lon',
                'address': 'alamat'
            })
            
            orange_data['kategori'] = 'Hanya ESB'
            orange_data['color'] = 'orange'
            orange_data['source'] = 'esb'
            
            # Filter out data yang sudah ada di matched
            if not green_data.empty:
                original_count = len(orange_data)
                matched_names = set(green_data['nama_restoran'].str.upper().dropna().unique())
                orange_data = orange_data[~orange_data['nama_restoran'].str.upper().isin(matched_names)]
                st.info(f"🔍 ESB: Filtered {original_count - len(orange_data):,} matched records")
            
        except Exception as e:
            st.error(f"❌ Error processing ESB data: {e}")
    
    # =========================================================================
    # PROSES DATA SCRAPING DENGAN VALIDASI SANGAT KETAT
    # =========================================================================
    df_scraping_clean = clean_and_validate_coordinates_strict(
        df_scraping, 'latitude', 'longitude', "Scraping Data"
    )
    
    blue_data = pd.DataFrame()
    if not df_scraping_clean.empty:
        try:
            blue_data = df_scraping_clean[[
                'title', 'address', 'longitude', 'latitude'
            ]].copy()
            
            blue_data = blue_data.rename(columns={
                'title': 'nama_restoran',
                'latitude': 'lat',
                'longitude': 'lon',
                'address': 'alamat'
            })
            
            blue_data['kategori'] = 'Hanya Scraping'
            blue_data['color'] = 'blue'
            blue_data['source'] = 'scraping'
            
            # Filter out data yang sudah ada di matched
            if not green_data.empty:
                original_count = len(blue_data)
                matched_names = set(green_data['nama_restoran'].str.upper().dropna().unique())
                blue_data = blue_data[~blue_data['nama_restoran'].str.upper().isin(matched_names)]
                st.info(f"🔍 Scraping: Filtered {original_count - len(blue_data):,} matched records")
            
        except Exception as e:
            st.error(f"❌ Error processing scraping data: {e}")
    
    # Final summary
    total_records = len(green_data) + len(orange_data) + len(blue_data)
    st.success(f"🎯 **PROSES SELESAI** - Match: {len(green_data):,}, ESB-only: {len(orange_data):,}, Scraping-only: {len(blue_data):,}, Total: {total_records:,}")
    
    return green_data, orange_data, blue_data

# =============================================================================
# FUNGSI CHECK FILE
# =============================================================================
def check_default_files():
    """Cek apakah file default ada di repository"""
    available_files = {}
    for file_type, file_path in Config.PARQUET_FILES.items():
        if os.path.exists(file_path):
            available_files[file_type] = file_path
            st.sidebar.success(f"✅ {file_type.upper()} file found: {file_path}")
        else:
            st.sidebar.warning(f"⚠️ {file_type.upper()} file not found: {file_path}")
    return available_files

# =============================================================================
# VISUALISASI DENGAN PLOTLY - DIPERBAIKI
# =============================================================================
def create_plotly_map(green_data, orange_data, blue_data, show_layers):
    """Buat peta interaktif dengan Plotly"""
    
    layers_data = []
    
    if show_layers['match'] and not green_data.empty:
        green_temp = green_data.copy()
        green_temp['size'] = 8
        green_temp['opacity'] = 0.8
        layers_data.append(green_temp)
        
    if show_layers['esb'] and not orange_data.empty:
        orange_temp = orange_data.copy()
        orange_temp['size'] = 6
        orange_temp['opacity'] = 0.7
        layers_data.append(orange_temp)
        
    if show_layers['scraping'] and not blue_data.empty:
        blue_temp = blue_data.copy()
        blue_temp['size'] = 6
        blue_temp['opacity'] = 0.7
        layers_data.append(blue_temp)
    
    if not layers_data:
        st.warning("⚠️ Tidak ada data yang ditampilkan. Silakan pilih layer di sidebar.")
        return None
    
    try:
        combined_data = pd.concat(layers_data, ignore_index=True)
        
        if combined_data.empty:
            st.warning("⚠️ Tidak ada data yang valid untuk ditampilkan.")
            return None
        
        # Buat hover text
        def create_hover_text(row):
            text = f"<b>{row['nama_restoran']}</b><br>"
            text += f"Kategori: {row['kategori']}<br>"
            text += f"Koordinat: {row['lat']:.4f}, {row['lon']:.4f}<br>"
            
            if row['kategori'] == 'Match' and 'similarity_score' in row:
                text += f"Similarity: {row['similarity_score']:.3f}<br>"
            if 'alamat' in row and pd.notna(row['alamat']):
                text += f"Alamat: {row['alamat'][:100]}..."
            
            return text
        
        combined_data['hover_text'] = combined_data.apply(create_hover_text, axis=1)
        
        # Buat peta dengan center di Indonesia dan bounds terbatas
        fig = px.scatter_mapbox(
            combined_data,
            lat="lat",
            lon="lon",
            color="kategori",
            color_discrete_map={
                'Match': 'green',
                'Hanya ESB': 'orange', 
                'Hanya Scraping': 'blue'
            },
            hover_name="hover_text",
            zoom=5,
            height=700,
            title="Peta Sebaran Restoran 3 Pulau Indonesia (Validasi Ketat)"
        )
        
        # Set bounds untuk membatasi view hanya Indonesia
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(lat=-2.5, lon=118.0),
                zoom=4,
                bounds=dict(
                    west=95.0,  # Batas barat Indonesia
                    east=141.0, # Batas timur Indonesia  
                    south=-11.0, # Batas selatan Indonesia
                    north=7.0   # Batas utara Indonesia
                )
            ),
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )
        
        # Update marker
        fig.update_traces(
            marker=dict(
                size=combined_data['size'],
                opacity=combined_data['opacity'],
                sizemode='diameter'
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error creating map: {e}")
        return None

# =============================================================================
# ANALISIS STATISTIK
# =============================================================================
def create_comprehensive_statistics(green_data, orange_data, blue_data):
    total_green = len(green_data) if not green_data.empty else 0
    total_orange = len(orange_data) if not orange_data.empty else 0
    total_blue = len(blue_data) if not blue_data.empty else 0
    total_all = total_green + total_orange + total_blue
    
    if total_all == 0:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Tidak ada data untuk ditampilkan", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=20)
        empty_fig.update_layout(width=400, height=300)
        return empty_fig, empty_fig, empty_fig, empty_fig, 0, 0, 0, 0
    
    # Pie chart
    sizes = [total_green, total_orange, total_blue]
    labels = [f'Match ({total_green:,})', f'Hanya ESB ({total_orange:,})', f'Hanya Scraping ({total_blue:,})']
    fig_pie = px.pie(values=sizes, names=labels, title='Distribusi Data Restoran 3 Pulau',
                     color=labels, color_discrete_map={labels[0]: '#00ff00', labels[1]: '#ffa500', labels[2]: '#0000ff'})
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    
    # Bar chart
    fig_bar = go.Figure(data=[go.Bar(x=['Match', 'Hanya ESB', 'Hanya Scraping'], y=[total_green, total_orange, total_blue],
                                     marker_color=['#00ff00', '#ffa500', '#0000ff'], text=[f'{x:,}' for x in [total_green, total_orange, total_blue]], textposition='auto')])
    fig_bar.update_layout(title='Perbandingan Kategori Data 3 Pulau', xaxis_title='Kategori', yaxis_title='Jumlah Restoran')
    
    # Similarity
    fig_similarity = go.Figure()
    if not green_data.empty and 'similarity_score' in green_data.columns:
        sim = green_data['similarity_score'].dropna()
        if len(sim) > 0:
            fig_similarity.add_trace(go.Histogram(x=sim, nbinsx=20, name='Similarity Score', marker_color='#00ff00', opacity=0.7))
            fig_similarity.update_layout(title='Distribusi Similarity Score (Data Match)', xaxis_title='Similarity Score', yaxis_title='Frequency')
        else:
            fig_similarity.add_annotation(text="Tidak ada data similarity", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    else:
        fig_similarity.add_annotation(text="Tidak ada data match", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig_similarity.update_layout(title='Distribusi Similarity Score')
    
    # Top Restoran
    fig_top = go.Figure()
    all_rest = pd.concat([green_data, orange_data, blue_data], ignore_index=True)
    if not all_rest.empty and 'nama_restoran' in all_rest.columns:
        top = all_rest['nama_restoran'].value_counts().head(15)
        if len(top) > 0:
            fig_top.add_trace(go.Bar(y=top.index, x=top.values, orientation='h', marker_color='lightblue', text=top.values, textposition='auto'))
            fig_top.update_layout(title='Top 15 Restoran Berdasarkan Jumlah Lokasi', xaxis_title='Jumlah Lokasi', yaxis_title='Nama Restoran')
        else:
            fig_top.add_annotation(text="Tidak ada data restoran", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    else:
        fig_top.add_annotation(text="Tidak ada data restoran", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig_top.update_layout(title='Top Restoran')
    
    return fig_pie, fig_bar, fig_similarity, fig_top, total_green, total_orange, total_blue, total_all

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.title("🗺️ Pinpoint Map 3 Pulau - Full Data Analysis")
    st.markdown("Visualisasi interaktif data restoran **3 Pulau** Indonesia dengan **VALIDASI KOORDINAT SANGAT KETAT**")
    
    # Warning tentang validasi ketat
    st.warning("""
    **⚠️ MODE VALIDASI SANGAT KETAT AKTIF**
    - Hanya menampilkan koordinat dalam wilayah Indonesia
    - Memblokir koordinat di Malaysia, Singapura, Brunei, Timor Leste
    - Filter agresif terhadap koordinat di laut atau negara tetangga
    - Mungkin ada pengurangan jumlah data yang signifikan
    """)
    
    # Session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'green_data' not in st.session_state:
        st.session_state.green_data = pd.DataFrame()
    if 'orange_data' not in st.session_state:
        st.session_state.orange_data = pd.DataFrame()
    if 'blue_data' not in st.session_state:
        st.session_state.blue_data = pd.DataFrame()

    # Sidebar
    st.sidebar.header("🎛️ Kontrol Visualisasi")
    
    # Informasi file
    st.sidebar.subheader("📁 Dataset 3 Pulau")
    available_files = check_default_files()
    
    if len(available_files) < 3:
        st.sidebar.error("❌ Dataset 3 Pulau tidak lengkap")
        st.sidebar.info("Pastikan file berikut ada di folder `src/`:")
        for file_type, file_path in Config.PARQUET_FILES.items():
            st.sidebar.write(f"- {os.path.basename(file_path)}")
        return
    
    st.sidebar.success("✅ Dataset 3 Pulau tersedia!")
    
    # Informasi validasi
    st.sidebar.subheader("⚙️ Validasi Koordinat")
    st.sidebar.info("""
    **Wilayah yang Diizinkan:**
    - Sumatera, Jawa, Kalimantan
    - Bali, Nusa Tenggara
    - Sulawesi
    
    **Wilayah Diblokir:**
    - Malaysia, Singapura, Brunei
    - Timor Leste, Filipina
    - Papua Nugini
    """)
    
    # Load data
    if not st.session_state.data_loaded:
        if st.sidebar.button("🚀 Muat Data dengan Validasi Ketat", type="primary"):
            with st.spinner("🔄 Memuat dataset dengan validasi SANGAT KETAT..."):
                try:
                    green_data, orange_data, blue_data = load_and_process_data_strict()
                    
                    st.session_state.green_data = green_data
                    st.session_state.orange_data = orange_data
                    st.session_state.blue_data = blue_data
                    st.session_state.data_loaded = True
                    
                    total_loaded = len(green_data) + len(orange_data) + len(blue_data)
                    st.success(f"✅ Data berhasil dimuat! Total: {total_loaded:,} records")
                    
                    # Tampilkan peringatan jika data banyak yang terfilter
                    initial_estimate = 14355  # Dari statistik sebelumnya
                    if total_loaded < initial_estimate * 0.5:
                        st.warning(f"⚠️ Banyak data terfilter! Hanya {total_loaded:,} dari perkiraan {initial_estimate:,} records")
                    
                except Exception as e:
                    st.error(f"❌ Gagal memuat data: {str(e)}")
                    return
        else:
            st.info("👆 Klik 'Muat Data dengan Validasi Ketat' untuk memulai")
            return
    
    # Kontrol layer
    st.sidebar.subheader("👁️ Kontrol Layer")
    show_layers = {
        'match': st.sidebar.checkbox("✅ Data Match (Hijau)", True),
        'esb': st.sidebar.checkbox("🟠 Hanya ESB (Orange)", True),
        'scraping': st.sidebar.checkbox("🔵 Hanya Scraping (Biru)", True)
    }
    
    # Statistik
    st.header("📊 Analisis Statistik Dataset 3 Pulau")
    
    fig_pie, fig_bar, fig_similarity, fig_top, tg, to, tb, ta = create_comprehensive_statistics(
        st.session_state.green_data, 
        st.session_state.orange_data, 
        st.session_state.blue_data
    )
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("✅ Data Match", f"{tg:,}")
    with col2: st.metric("🟠 Hanya ESB", f"{to:,}")
    with col3: st.metric("🔵 Hanya Scraping", f"{tb:,}")
    with col4: st.metric("📊 Total Semua Data", f"{ta:,}")
    
    # Charts
    st.subheader("📈 Visualisasi Distribusi")
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(fig_pie, use_container_width=True)
    with col2: st.plotly_chart(fig_bar, use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3: st.plotly_chart(fig_similarity, use_container_width=True)
    with col4: st.plotly_chart(fig_top, use_container_width=True)
    
    # Peta
    st.header("🗺️ Peta Interaktif 3 Pulau Indonesia")
    
    total_points = len(st.session_state.green_data) + len(st.session_state.orange_data) + len(st.session_state.blue_data)
    total_displayed = sum([
        len(st.session_state.green_data) if show_layers['match'] else 0,
        len(st.session_state.orange_data) if show_layers['esb'] else 0,
        len(st.session_state.blue_data) if show_layers['scraping'] else 0
    ])
    
    st.info(f"🎯 **Menampilkan {total_displayed:,} dari {total_points:,} titik data** (setelah validasi ketat)")
    
    # Buat peta
    plotly_map = create_plotly_map(
        st.session_state.green_data, 
        st.session_state.orange_data, 
        st.session_state.blue_data, 
        show_layers
    )
    
    if plotly_map:
        st.plotly_chart(plotly_map, use_container_width=True)
        
        # Legenda
        st.markdown(f"""
        ### 🎯 Legenda & Informasi
        
        **Kategori Data:**
        - ✅ **Hijau**: Data Match ({len(st.session_state.green_data):,}) - Restoran yang ada di kedua dataset
        - 🟠 **Orange**: Hanya ESB ({len(st.session_state.orange_data):,}) - Hanya ada di dataset ESB  
        - 🔵 **Biru**: Hanya Scraping ({len(st.session_state.blue_data):,}) - Hanya ada di dataset Google Maps
        
        **Validasi Koordinat:**
        - 🇮🇩 **HANYA INDONESIA**: Koordinat di negara tetangga otomatis difilter
        - 🔒 **BLOKIR**: Malaysia, Singapura, Brunei, Timor Leste
        - 🗺️ **BOUNDS TERBATAS**: Peta hanya menampilkan wilayah Indonesia
        """)
    else:
        st.error("❌ Gagal membuat peta. Tidak ada data yang valid setelah validasi ketat.")
    
    # Data tables
    st.header("📋 Detail Data")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "✅ Match", "🟠 ESB", "🔵 Scraping"])
    
    with tab1:
        summary_data = {
            'Kategori': ['Match', 'Hanya ESB', 'Hanya Scraping', 'Total'],
            'Jumlah': [tg, to, tb, ta],
            'Persentase': [
                f"{(tg/ta)*100:.2f}%" if ta > 0 else "0%",
                f"{(to/ta)*100:.2f}%" if ta > 0 else "0%", 
                f"{(tb/ta)*100:.2f}%" if ta > 0 else "0%",
                '100%'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    with tab2:
        if not st.session_state.green_data.empty:
            display_cols = ['nama_restoran', 'lat', 'lon', 'similarity_score', 'confidence']
            available_cols = [col for col in display_cols if col in st.session_state.green_data.columns]
            st.dataframe(st.session_state.green_data[available_cols], use_container_width=True)
        else:
            st.info("❌ Tidak ada data Match yang valid")
    
    with tab3:
        if not st.session_state.orange_data.empty:
            display_cols = ['nama_restoran', 'lat', 'lon', 'alamat']
            available_cols = [col for col in display_cols if col in st.session_state.orange_data.columns]
            st.dataframe(st.session_state.orange_data[available_cols], use_container_width=True)
        else:
            st.info("❌ Tidak ada data ESB-only yang valid")
    
    with tab4:
        if not st.session_state.blue_data.empty:
            display_cols = ['nama_restoran', 'lat', 'lon', 'alamat']
            available_cols = [col for col in display_cols if col in st.session_state.blue_data.columns]
            st.dataframe(st.session_state.blue_data[available_cols], use_container_width=True)
        else:
            st.info("❌ Tidak ada data Scraping-only yang valid")
    
    # Informasi sistem
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Informasi Sistem")
    
    st.sidebar.info(f"""
    **Status Data:**
    - ✅ Match: {tg:,}
    - 🟠 ESB: {to:,}  
    - 🔵 Scraping: {tb:,}
    - 🎯 Total: {ta:,}
    
    **Validasi:**
    - 🔒 Mode: SANGAT KETAT
    - 👁️ Ditampilkan: {total_displayed:,}
    """)

if __name__ == "__main__":
    main()