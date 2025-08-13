# Wholesale Customers — UAS Report
**Nama/NIM:** _isi di sini_

**Tanggal:** {datetime}

## Tujuan
Melakukan segmentasi pelanggan (clustering) pada dataset **Wholesale Customers** untuk menemukan pola belanja yang dapat digunakan sebagai dasar strategi pemasaran/operasional.

## Dataset
- Sumber: UCI Machine Learning Repository — Wholesale Customers
- Jumlah data: {n_rows} baris, {n_cols} kolom
- Fitur yang digunakan: {features}

## Metode & Alur
1. **Preprocessing**
   - Transformasi `log1p` untuk mengurangi skew pada pengeluaran.
   - **RobustScaler** (lebih tahan outlier) untuk standardisasi fitur.
2. **Reduksi Dimensi**: **PCA 2D** untuk visualisasi.
   - Varian yang dijelaskan oleh komponen: PC1 ≈ {pc1}, PC2 ≈ {pc2}
3. **Clustering**: **K-Means**
   - Penentuan jumlah cluster via **Elbow** & **Silhouette** → _k_ terbaik ≈ **{best_k}**.
   - {elbow_hint}

## Hasil Utama
- Lihat:
  - `outputs/clustered_customers.csv` — data + label cluster
  - `outputs/cluster_profiles.csv` — ringkasan profil (rata-rata per fitur & ukuran cluster)
  - `outputs/pca_clusters.png` — visualisasi 2D
- **Interpretasi Singkat (isi sendiri sesuai hasil):**
  - Contoh: *Cluster 0* = pelanggan dengan belanja tinggi di `Grocery` dan `Detergents_Paper` → kemungkinan **ritel**.
  - Contoh: *Cluster 1* = pengeluaran besar di `Fresh` & `Frozen` → kemungkinan **hotel/restoran**.
  - Tambahkan insight spesifik dari `cluster_profiles.csv`.

## Kesimpulan & Rekomendasi
- Jelaskan insight yang paling berguna (segmentasi, strategi promosi, stok, dsb.).
- Saran tindak lanjut: coba **DBSCAN** / **Gaussian Mixture**, feature engineering tambahan, gabungkan data `Channel/Region` bila tersedia.

## Lampiran
- `eda_summary.json`, `corr_matrix.csv`, `scaled_features.csv`, `pca_2d.csv`
- Gambar: `elbow_plot.png`, `silhouette_plot.png`, `pca_clusters.png`