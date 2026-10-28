NFL FANTASY SCOUT 2026
Mochamad Rizky Maulana Aviansyah

MODEL AKTIF V6 - GARIS TARUHAN (SPREAD DAN IMPLIED TEAM TOTAL)

V6 menambahkan 7 fitur pasar dari jadwal nflverse: spread dari sudut tim, total line, implied team total,
implied total lawan, dome, laga divisi, dan form terbaru x implied team total.
Dipilih dari validasi 2023-2024 dengan aturan: MAE <= V5 dan RMSE <= 99.75% V5. Data uji dinilai sekali setelah pemilihan.
Validasi: MAE 4.3112 -> 4.2688, RMSE 6.0990 -> 6.0323.
Uji 2025-2026 (8,288 observasi): MAE 4.3184 -> 4.2820, RMSE 6.0970 -> 6.0253, R2 0.4121 -> 0.4259.
Penurunan MAE terhadap baseline: 10.48%.
Reproduksi: python src/upgrade_v6.py lalu Perbarui_Prediksi.cmd. Hasil: reports/v6_validation.json, v6_comparison.json, v6_feature_importance.json.
Riwayat historis menggunakan closing line; forecast memakai garis yang tersedia saat pembaruan. Garis yang belum terbit diimputasi median.
Ringkasan V5 di bawah dipertahankan sebagai riwayat.

MODEL V5 - RIWAYAT SNAP DAN CEDERA PEKAN SEBELUMNYA

TUJUAN
Prediksi poin fantasy PPR dan rekomendasi tujuh slot ofensif untuk QB, RB, WR, dan TE.
Dashboard mencakup Scout board, Start or Sit, Lineup lab, dan laporan evaluasi.

RIWAYAT VERSI 1 SAMPAI 3
Versi 1: MAE 4.4506, RMSE 6.2220.
Versi 2: MAE 4.1503, RMSE 6.3883.
Versi 3: MAE 4.3349, RMSE 6.1826.
Model/bobot dipilih pada validasi 2023–2024 dengan RMSE minimum dan batas kenaikan MAE 5% dari versi 2.
Data 2025–2026 sudah dilihat sebelumnya. Ini pengujian ulang retrospektif.

HASIL EKSPERIMEN
Model: v3+75pct_v5_role_hurdle / Ensemble LightGBM
Data latih evaluasi: 2019–2024, 36,290 observasi.
Validasi berdasarkan waktu: musim 2023 dan 2024, dengan data latih hanya dari musim sebelumnya.
Data uji: musim 2025 dan 2026 yang tersedia, 8,288 observasi.
MAE train: 4.1819 poin PPR.
MAE test: 4.3184 poin PPR.
MAE baseline: 4.7835 poin PPR.
Penurunan MAE: 9.72%.
RMSE test: 6.0970. R² test: 0.4121.
Ini adalah regresi. Persentase penurunan galat bukan akurasi klasifikasi atau peningkatan poin lineup.

FILE OFFLINE
Klik dua kali NFL_Fantasy_Scout_2026.html. Dashboard dan unduhan data di dalamnya dapat digunakan tanpa server atau internet.

JALANKAN DASHBOARD
1. Ekstrak ZIP proyek.
2. Buka terminal di folder nfl_fantasy_scout_2026.
3. Jalankan: python -m http.server 8766 --directory dist
   Windows dengan Python Launcher: py -3.11 -m http.server 8766 --directory dist
4. Buka http://localhost:8766
   Jangan membuka index.html langsung karena browser perlu membaca data.json lewat HTTP.
Alternatif Windows: jalankan Buka_Dashboard.cmd.

JALANKAN NOTEBOOK
Buka NFL_Fantasy_Scout_2026_Mochamad_Rizky_Maulana_Aviansyah.ipynb dari folder yang sama.
Hasil sel sudah tersimpan. Untuk menjalankan ulang, pasang requirements.txt jika diperlukan.
RETRAIN=False membaca hasil eksperimen. RETRAIN=True mengulang pelatihan dari cache.
Google Colab: unggah ZIP proyek, ekstrak, pindah ke folder proyek, lalu buka/jalankan notebook.
Data dan src harus ikut diekstrak. Notebook tunggal tidak memuat semua dataset.

REPRODUKSI DARI AWAL
python -m pip install -r requirements.txt
python src/pipeline.py
python src/upgrade.py
python src/rebalance.py
python src/refresh_forecasts.py --cached
python src/test_forecast_dates.py
python src/availability.py
python src/prospective.py --freeze
python src/verify_project.py
node src/test_lineup.cjs
python src/build_notebook.py
python src/execute_notebook.py
python src/package_project.py
Perintah pertama yang melatih model mengambil sumber apabila cache tidak ada.
Pipeline memakai satu thread untuk kompatibilitas Windows yang membatasi multiprocessing.
Untuk pembaruan forecast gunakan Perbarui_Prediksi.cmd. Pertahankan data/raw untuk benchmark yang sama.
Lakukan pembaruan setelah akhir pekan pertandingan dan periksa kelengkapan sumber.
Nama forecasts_week8.csv pada laporan mengacu pada snapshot awal proyek ini.
Tanggal, kandidat, metrik, dan prediksi dapat berubah jika data sumber diperbarui.

ISI PAKET
dist/                   Dashboard dan prediksi yang dapat ditampilkan tanpa server model.
src/pipeline.py         Pengambilan data, fitur, seleksi, evaluasi, dan ekspor.
src/verify_project.py   Verifikasi integritas dan kebocoran waktu.
src/test_lineup.cjs     Pemeriksaan optimizer terhadap enumerasi menyeluruh.
data/raw/               Snapshot CSV sumber 2019–2026 dan jadwal.
models/                 Model evaluasi dan forecast yang terpisah.
reports/                Metrik, provenance, data fitur, prediksi uji, dan grafik.
Notebook .ipynb         Penjelasan dan eksplorasi yang dapat dijalankan ulang.
LINKEDIN_COPY.txt       Draf deskripsi proyek. Belum dipublikasikan ke LinkedIn.

BATAS PENGGUNAAN
Snapshot benchmark tetap dibekukan pada W6 dan sebagian W7. Data forecast terbaru berada di data/live, lengkap sampai W7 termasuk DAL–PHI.
Pertandingan yang belum tersedia tidak dibuat menjadi nol. Prediksi awal ditujukan untuk W8.
Kandidat pernah mendapat offensive opportunity musim yang sama dan terakhir terlihat <=3 pekan.
Week 1, pemain sebelum penampilan pertama, K, DST, dan IDP tidak dicakup. Tim bye menargetkan laga terjadwal berikutnya dan tidak masuk lineup pekan bye.
Nol pada evaluasi berarti kandidat tidak tercatat pada pertandingan yang berkasnya tersedia.
Snapshot roster dan cedera dari nflverse tersedia. ACT bukan konfirmasi game-day active. Tim yang berbeda dari tim model disisihkan. Sumber yang gagal atau kedaluwarsa ditandai.
Rentang empiris berasal dari galat validasi, bukan jaminan skor. Skor PPR aktual dapat negatif.
Forecast bukan live feed dan tidak terhubung ke akun fantasy pengguna.
Model forecast sudah dilatih ulang pada semua hasil yang tersedia setelah evaluasi dibekukan.
Hasil uji berasal dari model evaluasi yang terpisah, bukan model forecast yang sudah refit.

SUMBER DAN ATRIBUSI
Statistik: https://github.com/nflverse/nflverse-data/releases/tag/stats_player
Jadwal: https://github.com/nflverse/nfldata
Dokumentasi: https://nflreadr.nflverse.com/reference/load_player_stats.html
Sumber statistik nflverse-data memuat lisensi CC BY 4.0. Lisensi disertakan di THIRD_PARTY.
CSV digunakan sebagai sumber, lalu difilter, diubah menjadi fitur historis, dan dimodelkan.
URL dan SHA-256 tiap sumber tercatat di reports/provenance.json.
Analisis independen. Tidak berafiliasi dengan NFL atau platform fantasy.

LINGKUNGAN EKSPERIMEN
Python 3.11.5; pandas 2.2.1; NumPy 1.26.4; scikit-learn 1.6.1.
Versi paket persis ada di requirements-lock.txt.
Model joblib dibuat oleh proyek ini. Gunakan environment yang sama atau latih ulang untuk versi lain.


PREDIKSI MUSIM BERJALAN
Diperbarui: 27 Oct 2026 22:26 WIB. Lengkap sampai pekan 7.
Laga terdekat: 30 Oct 2026 07:15 WIB.
Target adalah poin fantasy pada laga berikutnya per pemain/tim, bukan skor akhir atau pemenang pertandingan.
Musim reguler NFL 2026 berlanjut ke Januari 2027. Playoff dan musim 2027 belum didukung.
Tim bye dapat menargetkan pekan berbeda. Lineup lab mengoptimalkan hanya pekan terdekat.
Statistik dibatasi pada pertandingan selesai sebelum waktu snapshot. Laga yang sudah dimulai tidak diprediksi.
Tanggal WIB dihitung dari waktu Eastern dan memperhitungkan DST. Jadwal tetap dapat berubah.
Pembaruan manual: jalankan Perbarui_Prediksi.cmd, atau Perbarui_Prediksi.cmd.
Pembaruan berjalan manual. Roster dan cedera mengikuti snapshot nflverse. Laporan pekan target mungkin belum terbit. Konfirmasi aktif saat laga tetap harus diperiksa.
Source dan hasil validasi: reports/rebalance_plan.json dan rebalance_validation.json.
Forecast: reports/forecasts_next_game.csv. Berkas forecasts_week8.csv dipertahankan sebagai alias kompatibilitas dan dapat memuat pekan 8/9.

VERSI PAKET TAMBAHAN
LightGBM 4.7.0. Wheel Windows disertakan di vendor dan lisensi MIT terdapat di metadata paket. Untuk OS lain, pasang requirements.txt.

PEMERIKSAAN KETERSEDIAAN DAN BUKTI
Perbarui_Prediksi.cmd mengambil forecast, roster, dan cedera, mengunci snapshot baru, lalu menilai snapshot lama setelah hasil tersedia.
Penguncian memakai waktu aktual. reports/frozen tidak ditimpa. Hash mendeteksi perubahan lokal, bukan bukti waktu independen dari pihak ketiga.
Jalankan python src/prospective.py untuk menilai snapshot yang ada, atau tambahkan --freeze untuk menyimpan snapshot baru sebelum kickoff.
Gunakan --replay untuk mengulang simulasi historis. Protokol tersimpan di reports/lineup_protocol.json.
Simulasi menggunakan roster buatan. Metrik poin lineup berbeda dari MAE model dan tidak menjamin keuntungan pada liga pengguna.
ACT bukan konfirmasi aktif saat laga. Laporan cedera yang belum terbit tidak dianggap sehat. Bukti roster berlaku maksimal 48 jam.
Jangan memakai status pemain terbaru untuk memodifikasi kembali keputusan snapshot lama.
Dashboard lokal belum menjadi demo publik. Panduan hosting tanpa model/raw data ada di PANDUAN_PUBLIKASI.txt.

EKSPERIMEN TERDAHULU SEBELUM DATA SNAP
61 konfigurasi termasuk pembanding dinilai memakai validasi 2023-2024.
Status: No candidate met both validation requirements. V3 predictions and locked snapshots retained.
MAE, RMSE, sumber dan pilihan lineup lama dipertahankan jika tidak ada kandidat yang memenuhi aturan validasi.
Reproduksi: python src/improve_v4.py lalu python src/improve_v4.py --stage2 dan python src/summarize_optimization.py.
Pengujian fitur: python src/test_v4.py. Laporan: reports/v4_validation_phase2.json dan reports/optimization_summary.json.
Ketepatan dengan toleransi maksimal 5 poin: 69.87% dibanding baseline 64.60%.
Pada aktual bukan nol: 62.94%. Ini bukan akurasi klasifikasi.
Tidak ada kandidat gagal yang dipilih berdasarkan hasil replay 2025-2026.

DATA PERAN SEBELUM LAGA
Model aktif: v5. Data tambahan memakai snap dan cedera pekan sebelumnya.
MAE seluruh kandidat: V3 4.3349, model 4.3184.
RMSE seluruh kandidat: V3 6.1826, model 6.0970.
Subset fantasy 6193 baris dinilai terpisah. Populasi lebih sempit tidak boleh dipakai untuk mengklaim perbaikan terhadap metrik populasi lama.
Reproduksi: python src/pregame_features.py lalu python src/upgrade_v5.py. Jalankan Perbarui_Prediksi.cmd sesudahnya.
Pemeriksaan: python src/test_pregame.py. Sumber dan hash: reports/pregame_sources.json.
Untuk membaca ulang fitur dari CSV prediksi gunakan pd.read_csv(path, float_precision='round_trip') agar input model pohon tetap persis. Arsip tidak memiliki waktu publikasi pertama seluruh snap. Semua hasil 2025-2026 adalah replay historis.
