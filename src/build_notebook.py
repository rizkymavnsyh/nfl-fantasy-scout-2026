"""Build the explorable notebook and readable portfolio notes from verified results."""
from pathlib import Path
import json
import platform
import nbformat as nbf
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / 'reports/metrics.json').read_text(encoding='utf-8'))
nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md('''# NFL Fantasy Scout 2026
## Prediksi poin PPR dan rekomendasi lineup

**Mochamad Rizky Maulana Aviansyah**  
Proyek portofolio machine learning untuk analisis NFL fantasy.

Notebook ini menjelaskan data, fitur sebelum pertandingan, pemilihan model, hasil uji, dan prediksi pekan 8 musim 2026. Dashboard menyajikan hasil yang sama dalam Scout board, Start or Sit, dan Lineup lab.

**Cara menjalankan:** ekstrak seluruh paket proyek, lalu buka notebook dari folder proyek. Hasil eksperimen yang sudah dijalankan tersedia di `reports`. Aktifkan `RETRAIN = True` pada bagian pemodelan jika ingin mengulang pelatihan dari data yang tersimpan. Tidak ada pemasangan paket atau unggahan ke layanan lain secara otomatis.''')
md('''## 1. Tujuan dan batas proyek

Tujuannya adalah memperkirakan poin fantasy PPR untuk pertandingan pada pekan berikutnya. Pengguna dapat membandingkan dua pemain dan menyusun tujuh slot ofensif dari roster pilihannya, yaitu satu QB, dua RB, dua WR, satu TE, dan satu FLEX.

Masalah utamanya adalah **regresi**, karena target berupa jumlah poin. Rekomendasi lineup memakai hasil prediksi regresi dan optimasi kombinasi posisi. Proyek ini tidak memprediksi pemenang pertandingan atau keuntungan taruhan.

Penilaian memakai kolom `fantasy_points_ppr` dari nflverse. Aturan khusus liga, seperti bonus yard atau nilai passing touchdown yang berbeda, belum diterapkan. K, DST, dan IDP tidak termasuk.''')
code('''import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / 'src/pipeline.py').exists() and (ROOT / 'nfl_fantasy_scout_2026/src/pipeline.py').exists():
    ROOT = ROOT / 'nfl_fantasy_scout_2026'
assert (ROOT / 'src/pipeline.py').exists(), 'Ekstrak seluruh ZIP proyek dan buka notebook dari folder tersebut.'
sys.path.insert(0, str(ROOT / 'src'))
import pipeline
pd.set_option('display.max_columns', 16)
plt.rcParams.update({'figure.dpi': 115, 'axes.spines.top': False, 'axes.spines.right': False})
print('Folder proyek:', ROOT.name)
''')
md('''## 2. Sumber dan pemeriksaan data

Statistik mingguan diperoleh dari [nflverse player statistics](https://github.com/nflverse/nflverse-data/releases/tag/stats_player). Jadwal pertandingan berasal dari [nflverse/nfldata](https://github.com/nflverse/nfldata). Data disimpan dengan URL sumber dan SHA-256 agar snapshot dapat ditelusuri.

Data 2026 pada paket ini lengkap sampai pekan 6, dengan sebagian pekan 7. Statistik pemain untuk DAL–PHI pekan 7 belum tersedia pada snapshot sumber. Pertandingan tersebut tidak diberi nilai nol dan tidak dipakai sebagai label uji. Prediksi pekan 8 untuk kedua tim menggunakan riwayat terakhir yang tersedia.

Satu baris statistik mewakili satu pemain pada satu pekan. Duplikasi diperiksa pada gabungan ID pemain, musim, dan pekan.''')
code('''stats, games = pipeline.load_data()
provenance = json.loads((ROOT / 'reports/provenance.json').read_text(encoding='utf-8'))
print('Statistik QB/RB/WR/TE:', stats.shape)
print('Jadwal musim reguler:', games.shape)
print('Duplikat pemain-musim-pekan:', stats.duplicated(['player_id','season','week']).sum())
display(stats.groupby(['season','position']).size().unstack(fill_value=0))
display(pd.DataFrame(provenance['sources'])[['file','bytes','sha256']])
''')
code('''display(stats[['player_display_name','position','season','week','team','opponent_team',
               'passing_yards','carries','targets','receptions','fantasy_points_ppr']].head(10))
print('Fitur yang tidak tersedia di snapshot bukan otomatis nol poin.')
''')
md('''## 3. Menentukan kandidat dan membentuk fitur

Sebelum setiap pekan, kandidat ditentukan dari pemain yang sudah memiliki passing attempt, carry, atau target pada musim yang sama dan terakhir terlihat maksimal tiga pekan lalu. Tim mengikuti penampilan terakhir yang diketahui. Tim juga harus memiliki pertandingan terjadwal.

Untuk pertandingan yang datanya tersedia, kandidat yang tidak tercatat tampil diberi target nol. Kebijakan ini membuat evaluasi mencakup pemain yang kemudian tidak bermain, sehingga tidak hanya memilih pemain aktif setelah hasil pertandingan diketahui. Ketidakhadiran baris pada pertandingan yang berkas statistiknya belum tersedia diperlakukan berbeda dan tidak diisi nol.

Setiap fitur dihitung sebelum hasil pekan target ditambahkan ke riwayat. Fitur mencakup nilai sebelumnya, rerata tiga dan lima hasil terakhir, rerata musim berjalan, variasi poin, posisi, kandang/tandang, jeda, dan rerata poin yang diberikan lawan kepada posisi terkait pada pertandingan sebelumnya.

Pemain sebelum penampilan pertamanya dan pekan pembuka berada di luar cakupan. Rekomendasi masih memerlukan pengecekan cedera, roster, serta starter resmi.''')
code('''data = pd.read_csv(ROOT / 'reports/feature_dataset.csv')
features = json.loads((ROOT / 'reports/metrics.json').read_text(encoding='utf-8'))['features']
assert set(features) <= set(data.columns)
print('Observasi pemodelan:', len(data))
print('Jumlah fitur sebelum encoding:', len(features))
print('Fitur:', features)
display(data.groupby('season').agg(observasi=('target','size'), rerata_poin=('target','mean')))
display(data[['name','position','season','week','fantasy_points_ppr_lag1',
              'fantasy_points_ppr_mean5','opponent_allowed_mean5','target']].head(10))
''')
code('''fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
data.target.hist(bins=50, ax=axes[0], color='#1c493c', edgecolor='white')
axes[0].set(title='Distribusi target pada populasi kandidat', xlabel='Poin PPR', ylabel='Observasi')
data.groupby('position').target.mean().reindex(['QB','RB','WR','TE']).plot.bar(ax=axes[1], color='#e85c32', rot=0)
axes[1].set(title='Rerata poin menurut posisi', xlabel='Posisi', ylabel='Poin PPR')
plt.tight_layout()
plt.show()
''')
md('''## 4. Pemilihan model berdasarkan waktu

Model dibandingkan dengan baseline rata-rata lima hasil terakhir. Kandidatnya adalah Ridge Regression dan dua konfigurasi HistGradientBoostingRegressor. Parameter model ditentukan sebelum melihat hasil data uji.

| Tahap | Data latih | Data validasi atau uji |
|---|---|---|
| Fold validasi 1 | 2019–2022 | 2023 |
| Fold validasi 2 | 2019–2023 | 2024 |
| Evaluasi akhir | 2019–2024 | 2025 dan 2026 yang tersedia |

Pemilihan menggunakan rerata MAE kedua fold validasi. Setelah dipilih, model evaluasi dilatih ulang pada 2019–2024 dan dievaluasi pada data uji. Data uji tidak dipakai untuk mencoba konfigurasi tambahan. Model forecast kemudian dilatih ulang pada seluruh pertandingan lengkap yang tersedia, termasuk pertandingan yang sebelumnya menjadi data uji. Model evaluasi dan model forecast disimpan terpisah.

Fitur statistik pertandingan target tidak digunakan. Standardisasi, imputasi, dan encoding dipelajari hanya dari bagian latih pada setiap fit.''')
code('''RETRAIN = False
if RETRAIN:
    # Memakai sumber cache agar hasil dapat direproduksi.
    # Jika sumber diperbarui, hasil dan tanggal snapshot juga dapat berubah.
    pipeline.run(refresh=False)
metrics = json.loads((ROOT / 'reports/metrics.json').read_text(encoding='utf-8'))
cv = pd.DataFrame([{'model': c['model'], 'MAE_validasi': c['cv_mae'],
                   'MAE_2023': c['folds'][0]['mae'], 'MAE_2024': c['folds'][1]['mae']}
                  for c in metrics['cv_candidates']]).sort_values('MAE_validasi')
display(cv.round(4))
print('Model terpilih:', metrics['selected_model'])
''')
md('''## 5. Hasil pengujian

MAE menunjukkan besar selisih absolut rata-rata dalam satuan poin PPR. Misalnya, prediksi 15 poin dan hasil 11 poin menghasilkan galat absolut 4 poin. RMSE memberi bobot lebih besar pada kesalahan yang besar. R² menunjukkan perbaikan terhadap prediksi rata-rata target pada sampel evaluasi.

Ketiganya bukan persentase ketepatan kelas. Karena itu, proyek ini tidak menyebut MAE atau R² sebagai akurasi klasifikasi.''')
code('''test = pd.read_csv(ROOT / 'reports/predictions_test.csv', float_precision='round_trip')
summary = pd.DataFrame({'Train': metrics['train'], 'Test': metrics['test'], 'Baseline test': metrics['baseline']}).T
display(summary.round(4))
print(f"Penurunan MAE terhadap baseline: {metrics['mae_improvement_pct']:.2f}%")
by_position = pd.DataFrame([{'posisi': p, 'n': r['model']['n'], 'MAE_model': r['model']['mae'],
                             'MAE_baseline': r['baseline']['mae'], 'RMSE_model': r['model']['rmse']}
                            for p,r in metrics['by_position'].items()]).set_index('posisi')
display(by_position.round(4))
display(pd.DataFrame([{'musim': y, 'n': r['model']['n'], 'MAE_model': r['model']['mae'],
                       'MAE_baseline': r['baseline']['mae']}
                      for y,r in metrics['by_season'].items()]).round(4))
''')
code('''fig, ax = plt.subplots(figsize=(8, 3.6))
by_position[['MAE_model','MAE_baseline']].plot.bar(ax=ax, color=['#1c493c','#bbc4af'], rot=0)
ax.set(title='Galat uji menurut posisi', ylabel='MAE dalam poin PPR', xlabel='Posisi')
ax.legend(['Model','Baseline'])
plt.tight_layout()
plt.savefig(ROOT / 'reports/mae_by_position.png', dpi=160)
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
sample = test.sample(min(1000,len(test)), random_state=42)
ax.scatter(sample.target, sample.prediction, s=10, alpha=.25, color='#1c493c')
ax.plot([0,50],[0,50], '--', color='#e85c32', linewidth=1)
ax.set(title='Prediksi dan aktual pada sampel data uji', xlabel='Aktual PPR', ylabel='Prediksi PPR')
plt.tight_layout()
plt.savefig(ROOT / 'reports/predicted_vs_actual.png', dpi=160)
plt.show()
''')
md(f'''Model mencapai **MAE {m['test']['mae']:.2f} poin PPR**, dibandingkan **{m['baseline']['mae']:.2f} poin** pada baseline, pada **{m['test']['n']:,} observasi uji**. Penurunan galatnya sekitar **{m['mae_improvement_pct']:.2f}%**. R² uji sebesar **{m['test']['r2']:.3f}** menunjukkan bahwa prediksi per pemain masih memiliki ketidakpastian yang berarti.

Galat train sebesar **{m['train']['mae']:.2f}** dan galat test sebesar **{m['test']['mae']:.2f}** tidak menunjukkan selisih besar pada protokol ini. Hal tersebut tidak membuktikan model bebas overfitting atau selalu berlaku pada musim mendatang. Seluruh posisi dalam data uji memiliki MAE lebih rendah daripada baseline, tetapi QB memiliki galat absolut paling besar.''')
md('''## 6. Rentang galat dan pemeriksaan contoh

Rentang memakai persentil ke-80 galat absolut validasi per posisi, lalu diletakkan secara simetris di sekitar prediksi. Batas bawah dipotong pada nol untuk tampilan. Poin fantasy aktual dapat bernilai negatif. Ini merupakan rentang galat empiris, bukan interval yang dijamin valid untuk setiap pemain.

Cakupan diuji kembali pada data uji. Perhitungan rentang berbagi fold pengembangan dengan proses pemilihan model, sehingga keterbatasan tersebut perlu dipertimbangkan.''')
code('''print(f"Cakupan rentang pada data uji: {metrics['interval']['coverage_test']*100:.2f}%")
display(pd.Series(metrics['interval']['half_width_by_position'], name='setengah_lebar_rentang'))
audit = test[['name','position','season','week','target','prediction','baseline']].copy()
audit['galat_absolut'] = abs(audit.target-audit.prediction)
display(audit.sample(12, random_state=42).round(2))
display(audit.nlargest(10,'galat_absolut').round(2))
''')
md('''## 7. Prediksi pekan 8 musim 2026

Prediksi ini menggunakan model forecast yang dilatih ulang dengan semua hasil yang tersedia setelah eksperimen evaluasi selesai. Nilai aktual pekan 8 belum menjadi masukan pada snapshot ini.

Pemain yang terakhir terlihat lebih dari satu pekan lalu ditandai untuk pemeriksaan. Kondisi tersebut dapat disebabkan bye week, cedera, perubahan peran, atau keterlambatan statistik. Tanda ini bukan diagnosis cedera.''')
code('''forecast = pd.read_csv(ROOT / 'reports/forecasts_week8.csv')
display(forecast[['name','position','team','opponent','week','prediction','low','high','baseline','risk']]
        .sort_values('prediction', ascending=False).head(20).round(2))
top = forecast.nlargest(12,'prediction').sort_values('prediction')
fig, ax = plt.subplots(figsize=(8,4.5))
ax.barh(top.name, top.prediction, color='#1c493c')
ax.set(title='Prediksi PPR pekan 8 · snapshot 27 Oktober 2026', xlabel='Prediksi poin PPR')
plt.tight_layout()
plt.savefig(ROOT / 'reports/top_forecasts.png', dpi=160)
plt.show()
''')
md('''## 8. Start or Sit dan penyusunan lineup

Start or Sit membandingkan prediksi dua pemain. Selisih kecil dengan rentang galat yang tumpang tindih perlu dibaca sebagai keputusan yang belum pasti. Statistik pada kartu adalah konteks masukan, bukan penjelasan kausal atau nilai SHAP.

Lineup lab hanya menggunakan pemain yang dimasukkan pengguna ke roster. Algoritma mencoba setiap RB, WR, dan TE sebagai FLEX. Untuk setiap kandidat FLEX, algoritma mengambil pemain dengan prediksi tertinggi pada slot wajib yang tersisa. Kombinasi dengan jumlah prediksi tertinggi dipilih. Pemain tidak boleh muncul dua kali.

Implementasi JavaScript pada `dist/app.js` diperiksa dengan enumerasi semua kombinasi pada 20 roster acak kecil. Roster contoh pada dashboard hanya untuk demonstrasi.''')
code('''# Contoh perbandingan dua WR dari daftar prediksi.
pair = forecast[forecast.position=='WR'].nlargest(2,'prediction')
display(pair[['name','prediction','low','high','baseline']].round(2))
gap = pair.prediction.iloc[0]-pair.prediction.iloc[1]
print(f"Selisih prediksi: {gap:.2f} poin PPR. Periksa kelayakan bermain sebelum memilih.")
''')
md('''## 9. Verifikasi, keterbatasan, dan langkah berikutnya

Pemeriksaan integritas memastikan ID pemain-musim-pekan unik, semua prediksi bernilai hingga, dan MAE yang dilaporkan sesuai CSV. Pemeriksaan kebocoran data mengubah hasil pekan 3 dan memastikan fitur untuk pekan 3 tetap sama, sedangkan fitur untuk pekan sesudahnya berubah.

Keterbatasan utama adalah belum adanya laporan cedera dan starter langsung, ketergantungan tim pada penampilan terakhir, ketidaklengkapan snapshot DAL–PHI pekan 7, dan cakupan kandidat yang mengecualikan pemain sebelum penampilan pertama. Touchdown dan perubahan peran juga sulit diprediksi.

Pembaruan berikutnya sebaiknya memprioritaskan snapshot roster dan cedera sebelum laga, validasi prospektif pada pekan baru yang belum pernah diperiksa, dan evaluasi lineup terhadap baseline rekomendasi. Metrik pengurangan MAE tidak boleh disebut sebagai peningkatan total poin lineup, karena keduanya berbeda.''')
code('''verification = json.loads((ROOT / 'reports/verification.json').read_text(encoding='utf-8'))
display(pd.Series(verification['checks'], name='lulus'))
assert all(verification['checks'].values())
print('Dashboard lokal: jalankan python -m http.server 8766 --directory dist dari folder proyek.')
print('Buka http://localhost:8766 di browser.')
''')
if m.get('version') in ('v2','v3','v5'):
    for cell in cells:
        if cell.cell_type=='code':
            cell.source=cell.source.replace('pipeline.run(refresh=False)', 'from upgrade import run as run_upgrade\n    run_upgrade()')
        else:
            cell.source=cell.source.replace(
              'Kandidatnya adalah Ridge Regression dan dua konfigurasi HistGradientBoostingRegressor. Parameter model ditentukan sebelum melihat hasil data uji.',
              'Versi 2 membandingkan enam konfigurasi gradient boosting dan campuran prediksinya. Fitur baru mencakup kestabilan poin, peluang bermain, produksi tanpa touchdown, dan riwayat lintas musim. Rerata serta median dibandingkan karena touchdown dan beberapa skor ekstrem dapat memperbesar galat. Kandidat dan bobot campuran dicatat dalam upgrade_plan.json sebelum pemodelan.')
            cell.source=cell.source.replace(
              'Data uji tidak dipakai untuk mencoba konfigurasi tambahan.',
              'Data 2025–2026 sudah pernah diperiksa pada versi 1. Pada versi 2, data ini menjadi pengujian ulang retrospektif. Seleksi tetap hanya memakai validasi 2023–2024, dengan kenaikan RMSE validasi dibatasi maksimal 3% terhadap versi 1. Hasil pengujian ulang belum merupakan konfirmasi independen pada data baru.')
    comparison=m['version_comparison']
    if comparison['new']['rmse']>comparison['old']['rmse']:
        cells.append(nbf.v4.new_markdown_cell('''### Kompromi hasil versi 2

Versi 2 memakai galat absolut sebagai sasaran model. MAE lebih rendah, tetapi RMSE meningkat dan R² menurun dibandingkan versi 1. Artinya, rata-rata selisih absolut lebih kecil, sementara kesalahan yang besar belum membaik. Versi ini lebih baik pada sasaran MAE dan tidak diklaim unggul pada semua ukuran. Versi 1 disimpan untuk perbandingan.'''))
    cells.insert(2,nbf.v4.new_markdown_cell(f'''## Perubahan versi 2

| Ukuran | Versi 1 | Versi 2 |
|---|---:|---:|
| MAE validasi historis | {comparison['cv_mae_old']:.4f} | {comparison['cv_mae_new']:.4f} |
| MAE pengujian ulang | {comparison['old']['mae']:.4f} | {comparison['new']['mae']:.4f} |
| RMSE pengujian ulang | {comparison['old']['rmse']:.4f} | {comparison['new']['rmse']:.4f} |
| R² pengujian ulang | {comparison['old']['r2']:.4f} | {comparison['new']['r2']:.4f} |

Populasi, label, dan baseline tetap sama dengan versi 1. Data 2025–2026 telah dilihat sebelumnya. Pekan mendatang yang belum pernah diperiksa diperlukan untuk konfirmasi independen.'''))
if m.get('version') in ('v3','v5'):
    comp=m['version_comparison']; snap=m['forecast_snapshot']
    for cell in cells:
        if cell.cell_type=='code':
            cell.source=cell.source.replace('from upgrade import run as run_upgrade', 'from rebalance import run as run_upgrade')
            cell.source=cell.source.replace('    run_upgrade()','    run_upgrade()\n    from refresh_forecasts import refresh\n    refresh(download=False)')
            cell.source=cell.source.replace("reports/forecasts_week8.csv","reports/forecasts_next_game.csv")
            cell.source=cell.source.replace('Prediksi PPR pekan 8 · snapshot 27 Oktober 2026','Prediksi PPR pada laga berikutnya · musim 2026')
            if 'verification = json.loads' in cell.source:
                cell.source+="\ndate_checks = json.loads((ROOT / 'reports/forecast_date_verification.json').read_text(encoding='utf-8'))\ndisplay(pd.Series(date_checks))\nassert all(v for v in date_checks.values() if isinstance(v, bool))"
        else:
            cell.source=cell.source.replace('pemilihan model, hasil uji, dan prediksi pekan 8 musim 2026','pemilihan model, hasil pengujian ulang, dan prediksi laga berikutnya musim reguler 2026 yang berlanjut ke Januari 2027')
            cell.source=cell.source.replace('Versi 2 membandingkan enam konfigurasi gradient boosting dan campuran prediksinya.','Versi 3 membandingkan LightGBM dengan sasaran kuadrat, Huber, Fair, median, serta campuran model.')
            cell.source=cell.source.replace('upgrade_plan.json','rebalance_plan.json')
            cell.source=cell.source.replace('Pada versi 2, data ini menjadi pengujian ulang retrospektif.','Pada versi 3, data ini tetap menjadi pengujian ulang retrospektif.')
            cell.source=cell.source.replace('dengan kenaikan RMSE validasi dibatasi maksimal 3% terhadap versi 1','dengan sasaran RMSE minimum dan batas kenaikan MAE validasi maksimal 5% terhadap versi 2')
            cell.source=cell.source.replace('Data 2026 pada paket ini lengkap sampai pekan 6, dengan sebagian pekan 7.','Snapshot benchmark yang dibekukan lengkap sampai pekan 6, dengan sebagian pekan 7.')
            cell.source=cell.source.replace('Prediksi pekan 8 untuk kedua tim menggunakan riwayat terakhir yang tersedia.','Forecast terbaru memakai data/live yang terpisah dan sudah memuat DAL–PHI. Snapshot forecast lengkap sampai pekan 7.')
            cell.source=cell.source.replace('## 7. Prediksi pekan 8 musim 2026','## 7. Prediksi laga berikutnya musim 2026–2027')
            cell.source=cell.source.replace('Nilai aktual pekan 8 belum menjadi masukan pada snapshot ini.','Hanya pertandingan yang belum dimulai pada waktu snapshot yang boleh menjadi target. Setiap pemain diarahkan ke pertandingan berikutnya dari tim terakhir yang diketahui. Tim yang bye dapat memiliki target pekan 9, sementara tim lain menargetkan pekan 8. Optimasi lineup dibatasi pada satu pekan.')
            cell.source=cell.source.replace('ketidaklengkapan snapshot DAL–PHI pekan 7','perbedaan antara snapshot benchmark lama dan data forecast terbaru')
    cells[2]=nbf.v4.new_markdown_cell(f'''## Perubahan versi 3 dan penggunaan saat ini

| Ukuran pengujian ulang | Versi 1 | Versi 2 | Versi 3 |
|---|---:|---:|---:|
| MAE | {comp['old']['mae']:.4f} | {comp['v2']['mae']:.4f} | {comp['new']['mae']:.4f} |
| RMSE | {comp['old']['rmse']:.4f} | {comp['v2']['rmse']:.4f} | {comp['new']['rmse']:.4f} |
| R² | {comp['old']['r2']:.4f} | {comp['v2']['r2']:.4f} | {comp['new']['r2']:.4f} |

Versi 3 menurunkan RMSE dibandingkan versi 2, dengan MAE yang sedikit lebih tinggi. Dibandingkan versi 1, MAE dan RMSE sama-sama lebih rendah. Data evaluasi sudah pernah dilihat dan tetap merupakan pengujian ulang retrospektif.

Forecast diperbarui pada **{snap['as_of_wib']}**, lengkap sampai pekan **{snap['latest_complete_week']}**. Laga berikutnya dimulai pada **{snap['next_kickoff_wib']}**. Waktu kickoff memperhitungkan perubahan waktu musim panas Amerika. Musim 2026 mencakup pertandingan reguler pada Januari 2027. Playoff dan musim NFL 2027 belum dicakup. Pembaruan berikutnya dijalankan melalui `Perbarui_Prediksi.cmd`.
''')
nb.cells = cells
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name':'python','version':platform.python_version()}}
# Evidence sections use the exact same saved reports as the dashboard.
for cell in nb.cells:
    if cell.cell_type=='markdown':
        cell.source=cell.source.replace('Keterbatasan utama adalah belum adanya laporan cedera dan starter langsung, ketergantungan tim pada penampilan terakhir, ketidaklengkapan snapshot DAL–PHI pekan 7, dan cakupan kandidat yang mengecualikan pemain sebelum penampilan pertama.', 'Ketersediaan pemain memakai snapshot roster dan cedera dari nflverse. ACT bukan konfirmasi aktif saat laga. Laporan pekan target yang belum terbit tetap ditandai. Snapshot benchmark tetap lama, sedangkan forecast menggunakan data/live. Pemain sebelum penampilan pertama belum dicakup.')
        cell.source=cell.source.replace('Pembaruan berikutnya sebaiknya memprioritaskan snapshot roster dan cedera sebelum laga, validasi prospektif pada pekan baru yang belum pernah diperiksa, dan evaluasi lineup terhadap baseline rekomendasi.', 'Bagian tambahan mendokumentasikan pemeriksaan roster, prediksi terkunci, dan evaluasi lineup simulasi terhadap baseline. Konfirmasi prospektif harus menunggu hasil pertandingan baru.')
nb.cells.append(nbf.v4.new_markdown_cell("""## 10. Ketersediaan dan evaluasi prospektif

Status roster dan cedera dicocokkan menggunakan ID pemain, tim, dan pekan target. ACT berarti berada pada roster aktif, bukan jaminan bermain. Pemain nonaktif, tim berbeda, Out/Doubtful, atau bukti roster melebihi 48 jam tidak dipilih oleh lineup otomatis. Laporan yang belum tersedia bukan bukti bahwa pemain sehat.

Snapshot pada `reports/frozen` disimpan sebelum kickoff menggunakan waktu penguncian sesungguhnya. CSV prediksi, pilihan lineup, hash model, dan sumber disimpan tanpa menimpa snapshot lama. Hasil dinilai terpisah setelah skor akhir dan statistik kedua tim tersedia. Belum ada angka prospektif jika laga belum selesai.

Evaluasi lineup historis menggunakan 30 roster simulasi per pekan dengan seed tetap. Kedua kebijakan memilih dari roster yang sama. Pemilihan hanya memakai prediksi atau rerata lima hasil sebelumnya. Nilai aktual dipakai setelah lineup dipilih. Roster simulasi bukan data kepemilikan liga dan tidak menerapkan filter status aktif dan cedera pada laga target. Banyak roster pada pekan sama bukan pengamatan independen."""))
nb.cells.append(nbf.v4.new_code_cell("""availability = json.loads((ROOT / 'reports/availability.json').read_text(encoding='utf-8'))
lineup_report = json.loads((ROOT / 'reports/lineup_replay.json').read_text(encoding='utf-8'))
prospective = json.loads((ROOT / 'reports/prospective_evaluation.json').read_text(encoding='utf-8'))
display(pd.DataFrame([{'Kandidat': availability['total'], 'Lolos pemeriksaan roster': availability['eligible'], 'Laporan cedera terbaru': availability['injury_latest_week']}]))
display(pd.DataFrame([{k: lineup_report[k] for k in ['n_rosters','n_weeks','model_mean_points','baseline_mean_points','mean_difference','win_rate']}]))
display(pd.DataFrame([{k: row[k] for k in ['snapshot_id','n_frozen','n_scored','n_pending','status']} for row in prospective]))"""))

for cell in nb.cells:
    if cell.cell_type=='markdown': cell.source=cell.source.replace('Keterbatasan utama adalah belum adanya laporan cedera dan starter langsung, ketergantungan tim pada penampilan terakhir, perbedaan antara snapshot benchmark lama dan data forecast terbaru, dan cakupan kandidat yang mengecualikan pemain sebelum penampilan pertama.', 'Roster dan cedera memakai snapshot nflverse. Status aktif saat laga tetap perlu diperiksa. Tim yang berbeda dari tim model disisihkan dari lineup. Snapshot benchmark berbeda dari data forecast terbaru. Pemain sebelum penampilan pertama belum diprediksi.')

if (ROOT / 'reports/optimization_summary.json').exists():
    nb.cells.append(nbf.v4.new_markdown_cell("""## 11. Percobaan optimasi dan toleransi poin

Percobaan lanjutan membandingkan model per posisi, prediksi dua tahap, konteks peluang tim dari pekan sebelumnya, pembobotan musim terbaru, dan campuran model. Seluruh pemilihan memakai validasi 2023–2024. Model pengganti harus memiliki MAE validasi yang tidak meningkat dan RMSE validasi turun minimal 0,25%. Model yang hanya memperbaiki satu ukuran tidak otomatis menggantikan versi sekarang.

Toleransi ditetapkan sebesar 3 dan 5 poin PPR sebelum perhitungan. Persentase di bawah berarti bagian prediksi dengan galat absolut yang tidak melebihi batas tersebut. Angka ini bukan akurasi klasifikasi. Data pengujian ulang mencakup kandidat dengan hasil nol, sehingga hasil untuk observasi bukan nol juga ditampilkan. Evaluasi tetap bersifat retrospektif."""))
    nb.cells.append(nbf.v4.new_code_cell("""optimization = json.loads((ROOT / 'reports/optimization_summary.json').read_text(encoding='utf-8'))
print('Jumlah konfigurasi termasuk pembanding:', optimization['candidate_count'])
print('Status:', optimization['status'])
display(pd.DataFrame([{'model': 'Versi 3', **optimization['reference_cv']}, optimization['lowest_rmse_candidate']]).round(4))
tolerance = metrics['tolerance']
display(pd.DataFrame([{'Ukuran': label, 'Model (%)': tolerance['model'][key] * 100, 'Baseline (%)': tolerance['baseline'][key] * 100} for label, key in [('Galat maksimal 5 poin', 'within_5'), ('Galat maksimal 3 poin', 'within_3'), ('Galat maksimal 5 poin, aktual bukan nol', 'nonzero_within_5')]]).round(2))
print('Jumlah observasi aktual bukan nol:', tolerance['model']['nonzero_n'])"""))

if (ROOT / 'reports/v5_comparison.json').exists():
    comparison=json.loads((ROOT / 'reports/v5_comparison.json').read_text(encoding='utf-8'))
    for cell in nb.cells:
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('## Perubahan versi 3 dan penggunaan saat ini','## Riwayat versi 1 sampai 3')
            cell.source=cell.source.replace('## 11. Percobaan optimasi dan toleransi poin','## 11. Riwayat percobaan sebelum data snap')
            cell.source=cell.source.replace('belum adanya laporan cedera dan starter langsung','belum adanya konfirmasi starter langsung')
            if m.get('version')=='v5':
                cell.source=cell.source.replace('Versi 3 membandingkan LightGBM dengan sasaran kuadrat, Huber, Fair, median, serta campuran model.', 'Versi 5 menambahkan snap ofensif dan indikator cedera pekan sebelumnya. Kandidat meliputi regresi, model per posisi, model dua tahap, serta campuran dengan V3.')
                cell.source=cell.source.replace('Pemilihan menggunakan rerata MAE kedua fold validasi.', 'Pemilihan V5 meminimalkan rerata RMSE validasi, dengan MAE tidak boleh lebih tinggi dari V3 dan RMSE harus turun sedikitnya 0,25%.')
                cell.source=cell.source.replace('rebalance_plan.json','v5_plan.json')
                cell.source=cell.source.replace('dengan sasaran RMSE minimum dan batas kenaikan MAE validasi maksimal 5% terhadap versi 2','dengan MAE validasi tidak meningkat dan RMSE turun minimal 0,25% terhadap V3')
        elif m.get('version')=='v5':
            cell.source=cell.source.replace('from rebalance import run as run_upgrade','from upgrade_v5 import run as run_upgrade')
            cell.source=cell.source.replace('    refresh(download=False)', '    refresh(download=False)\n    from pregame_forecasts import update as update_role\n    update_role()\n    from availability import update as update_availability\n    update_availability(download=False)\n    from prospective import replay, score_all, export_panels\n    replay()\n    score_all()\n    export_panels()')
            cell.source=cell.source.replace("data = pd.read_csv(ROOT / 'reports/feature_dataset.csv')", "data = pd.read_csv(ROOT / 'reports/feature_dataset.csv')\nfrom pregame_features import enrich as enrich_role, load_sources as load_role\nsnap_history, injury_history = load_role()\ndata, _ = enrich_role(data, snap_history, injury_history)")
    if comparison['promoted']:
        nb.cells.insert(2,nbf.v4.new_markdown_cell(f"## Model aktif V5\n\nPada populasi pengujian ulang yang sama, MAE V3 {comparison['full_cohort']['v3']['mae']:.4f} menjadi {comparison['full_cohort']['model']['mae']:.4f}. RMSE {comparison['full_cohort']['v3']['rmse']:.4f} menjadi {comparison['full_cohort']['model']['rmse']:.4f}. Pemilihan memakai validasi 2023–2024. Data 2025–2026 sudah diperiksa sebelumnya dan bukan holdout independen."))
    nb.cells.append(nbf.v4.new_markdown_cell("""## 12. Peran sebelum laga dan kandidat fantasy

Data tambahan berasal dari snap counts dan laporan cedera nflverse. Hanya pekan sebelum target yang dipakai. Pembaruan cedera bertanggal setelah tanggal target dikeluarkan. Identitas PFR dan GSIS dicocokkan secara unik tanpa pencocokan nama yang samar. Ketiadaan sumber tidak dianggap nol. Kandidat tanpa snap pada laga yang datanya tersedia dapat memiliki nol setelah identitasnya dikenal dari pekan sebelumnya.

Kandidat fantasy ditentukan sebelum melihat target. QB memerlukan rerata tiga laga minimal 15 passing attempts atau persentase snap terakhir minimal 50%. RB, WR, dan TE memerlukan rerata kesempatan minimal lima atau persentase snap tiga laga minimal 25%. Bukti snap yang diketahui tidak boleh lebih lama dari dua pekan. Kriteria ini menunjukkan relevansi dari riwayat peran dan bukan konfirmasi aktif saat laga.

Metrik utama tetap memakai seluruh populasi agar perbandingan model adil. Subset fantasy dinilai terpisah menggunakan baris yang sama untuk model baru, V3, dan baseline. Waktu publikasi pertama seluruh arsip snap tidak tersedia, sehingga hasil historis merupakan replay. Prediksi baru dikunci dengan sumber dan cutoff yang tercatat."""))
    nb.cells.append(nbf.v4.new_code_cell("""pregame = json.loads((ROOT / 'reports/v5_comparison.json').read_text(encoding='utf-8'))
display(pd.DataFrame([{'Populasi': label, 'N': pregame[key]['n'], 'MAE V3': pregame[key]['v3']['mae'], 'MAE model': pregame[key]['model']['mae'], 'RMSE V3': pregame[key]['v3']['rmse'], 'RMSE model': pregame[key]['model']['rmse'], 'Maksimal 5 poin model (%)': pregame[key]['tolerance']['model']['within_5']*100} for label, key in [('Seluruh kandidat', 'full_cohort'), ('Subset fantasy', 'fantasy_pool')]]).round(4))
display(pd.Series(pregame['source_coverage']))
plan = json.loads((ROOT / 'reports/v5_plan.json').read_text(encoding='utf-8'))
display(pd.Series(plan['snap_history_coverage_by_season'], name='cakupan riwayat snap'))
print(pregame['replay_status'])"""))

nbf.write(nb, ROOT / 'NFL_Fantasy_Scout_2026_Mochamad_Rizky_Maulana_Aviansyah.ipynb')

readme = f'''NFL FANTASY SCOUT 2026
Mochamad Rizky Maulana Aviansyah

TUJUAN
Prediksi poin fantasy PPR dan rekomendasi tujuh slot ofensif untuk QB, RB, WR, dan TE.
Dashboard mencakup Scout board, Start or Sit, Lineup lab, dan laporan evaluasi.

HASIL EKSPERIMEN
Model: {m['selected_model']} / HistGradientBoostingRegressor
Data latih evaluasi: 2019–2024, {m['train']['n']:,} observasi.
Validasi berdasarkan waktu: musim 2023 dan 2024, dengan data latih hanya dari musim sebelumnya.
Data uji: musim 2025 dan 2026 yang tersedia, {m['test']['n']:,} observasi.
MAE train: {m['train']['mae']:.4f} poin PPR.
MAE test: {m['test']['mae']:.4f} poin PPR.
MAE baseline: {m['baseline']['mae']:.4f} poin PPR.
Penurunan MAE: {m['mae_improvement_pct']:.2f}%.
RMSE test: {m['test']['rmse']:.4f}. R² test: {m['test']['r2']:.4f}.
Ini adalah regresi. Persentase penurunan galat bukan akurasi klasifikasi atau peningkatan poin lineup.

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
python src/verify_project.py
node src/test_lineup.cjs
python src/build_notebook.py
Perintah pertama yang melatih model mengambil sumber apabila cache tidak ada.
Pipeline memakai satu thread untuk kompatibilitas Windows yang membatasi multiprocessing.
Untuk mengambil snapshot baru gunakan python src/pipeline.py --refresh.
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
Data 2026 lengkap sampai W6 dan sebagian W7. DAL–PHI W7 belum ada di statistik sumber.
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
Python {platform.python_version()}; pandas {pd.__version__}; NumPy {np.__version__}; scikit-learn {sklearn.__version__}.
Versi paket persis ada di requirements-lock.txt.
Model joblib dibuat oleh proyek ini. Gunakan environment yang sama atau latih ulang untuk versi lain.
'''
if m.get('version') in ('v2','v3','v5'):
    comp=m['version_comparison']
    readme=readme.replace('python src/pipeline.py\n','python src/pipeline.py\npython src/upgrade.py\n')
    readme=readme.replace('python src/build_notebook.py\n','python src/build_notebook.py\npython src/execute_notebook.py\npython src/package_project.py\n')
    readme=readme.replace('HASIL EKSPERIMEN\n',f'''PERUBAHAN VERSI 2
MAE versi 1: {comp['old']['mae']:.4f}. MAE versi 2: {comp['new']['mae']:.4f}.
RMSE versi 1: {comp['old']['rmse']:.4f}. RMSE versi 2: {comp['new']['rmse']:.4f}.
Penurunan MAE dari versi 1: {comp['mae_reduction_from_v1_pct']:.2f}%.
Seleksi hanya memakai validasi 2023–2024. Kandidat/bobot dicatat sebelum eksperimen.
Data 2025–2026 sudah diperiksa pada versi 1 dan kini menjadi pengujian ulang retrospektif.
Belum ada evaluasi independen pada pekan masa depan yang belum dilihat.
Fitur baru: kestabilan poin, peluang, produksi tanpa touchdown, dan riwayat lintas musim.
Rencana dan seluruh hasil validasi tersedia di reports/upgrade_plan.json dan upgrade_validation.json.

HASIL EKSPERIMEN
''')
    readme+=f'''\nKOMPROMI METRIK
MAE ditetapkan sebagai sasaran utama. RMSE versi 2 {comp['new']['rmse']:.4f} dibanding versi 1 {comp['old']['rmse']:.4f}.
Jika RMSE meningkat, versi 2 tidak berarti lebih baik dalam menekan semua kesalahan besar.
Model terpilih adalah {m['selected_model']}. Campuran diuji dalam validasi, tetapi belum tentu dipilih.
'''
if m.get('version') in ('v3','v5'):
    comp=m['version_comparison'];snap=m['forecast_snapshot']
    start=readme.index('PERUBAHAN VERSI 2')
    end=readme.index('HASIL EKSPERIMEN',start)
    readme=readme[:start]+f"VERSI 3 / SEIMBANG MAE DAN RMSE\nVersi 1: MAE {comp['old']['mae']:.4f}, RMSE {comp['old']['rmse']:.4f}.\nVersi 2: MAE {comp['v2']['mae']:.4f}, RMSE {comp['v2']['rmse']:.4f}.\nVersi 3: MAE {comp['new']['mae']:.4f}, RMSE {comp['new']['rmse']:.4f}.\nModel/bobot dipilih pada validasi 2023–2024 dengan RMSE minimum dan batas kenaikan MAE 5% dari versi 2.\nData 2025–2026 sudah dilihat sebelumnya. Ini pengujian ulang retrospektif.\n\n"+readme[end:]
    readme=readme[:readme.index('KOMPROMI METRIK')]
    readme=readme.replace('/ HistGradientBoostingRegressor','/ Ensemble LightGBM')
    readme=readme.replace('python src/upgrade.py\n','python src/upgrade.py\npython src/rebalance.py\npython src/refresh_forecasts.py --cached\npython src/test_forecast_dates.py\npython src/availability.py\npython src/prospective.py --freeze\n')
    readme=readme.replace('Data 2026 lengkap sampai W6 dan sebagian W7. DAL–PHI W7 belum ada di statistik sumber.','Snapshot benchmark tetap dibekukan pada W6 dan sebagian W7. Data forecast terbaru berada di data/live, lengkap sampai W7 termasuk DAL–PHI.')
    readme=readme.replace('Untuk mengambil snapshot baru gunakan python src/pipeline.py --refresh.','Untuk pembaruan forecast gunakan Perbarui_Prediksi.cmd. Pertahankan data/raw untuk benchmark yang sama.')
    readme+=f"\nPREDIKSI MUSIM BERJALAN\nDiperbarui: {snap['as_of_wib']}. Lengkap sampai pekan {snap['latest_complete_week']}.\nLaga terdekat: {snap['next_kickoff_wib']}.\nTarget adalah poin fantasy pada laga berikutnya per pemain/tim, bukan skor akhir atau pemenang pertandingan.\nMusim reguler NFL 2026 berlanjut ke Januari 2027. Playoff dan musim 2027 belum didukung.\nTim bye dapat menargetkan pekan berbeda. Lineup lab mengoptimalkan hanya pekan terdekat.\nStatistik dibatasi pada pertandingan selesai sebelum waktu snapshot. Laga yang sudah dimulai tidak diprediksi.\nTanggal WIB dihitung dari waktu Eastern dan memperhitungkan DST. Jadwal tetap dapat berubah.\nPembaruan manual: jalankan Perbarui_Prediksi.cmd, atau Perbarui_Prediksi.cmd.\nPembaruan berjalan manual. Roster dan cedera mengikuti snapshot nflverse. Laporan pekan target mungkin belum terbit. Konfirmasi aktif saat laga tetap harus diperiksa.\nSource dan hasil validasi: reports/rebalance_plan.json dan rebalance_validation.json.\nForecast: reports/forecasts_next_game.csv. Berkas forecasts_week8.csv dipertahankan sebagai alias kompatibilitas dan dapat memuat pekan 8/9.\n\nVERSI PAKET TAMBAHAN\nLightGBM 4.7.0. Wheel Windows disertakan di vendor dan lisensi MIT terdapat di metadata paket. Untuk OS lain, pasang requirements.txt.\n"

readme += """
PEMERIKSAAN KETERSEDIAAN DAN BUKTI
Perbarui_Prediksi.cmd mengambil forecast, roster, dan cedera, mengunci snapshot baru, lalu menilai snapshot lama setelah hasil tersedia.
Penguncian memakai waktu aktual. reports/frozen tidak ditimpa. Hash mendeteksi perubahan lokal, bukan bukti waktu independen dari pihak ketiga.
Jalankan python src/prospective.py untuk menilai snapshot yang ada, atau tambahkan --freeze untuk menyimpan snapshot baru sebelum kickoff.
Gunakan --replay untuk mengulang simulasi historis. Protokol tersimpan di reports/lineup_protocol.json.
Simulasi menggunakan roster buatan. Metrik poin lineup berbeda dari MAE model dan tidak menjamin keuntungan pada liga pengguna.
ACT bukan konfirmasi aktif saat laga. Laporan cedera yang belum terbit tidak dianggap sehat. Bukti roster berlaku maksimal 48 jam.
Jangan memakai status pemain terbaru untuk memodifikasi kembali keputusan snapshot lama.
Dashboard lokal belum menjadi demo publik. Panduan hosting tanpa model/raw data ada di PANDUAN_PUBLIKASI.txt.
"""

if (ROOT / 'reports/optimization_summary.json').exists():
    optimization=json.loads((ROOT / 'reports/optimization_summary.json').read_text(encoding='utf-8'))
    readme += f"\nEKSPERIMEN LANJUTAN\n{optimization['candidate_count']} konfigurasi termasuk pembanding dinilai memakai validasi 2023-2024.\nStatus: {optimization['status']}\nMAE, RMSE, sumber dan pilihan lineup lama dipertahankan jika tidak ada kandidat yang memenuhi aturan validasi.\nReproduksi: python src/improve_v4.py lalu python src/improve_v4.py --stage2 dan python src/summarize_optimization.py.\nPengujian fitur: python src/test_v4.py. Laporan: reports/v4_validation_phase2.json dan reports/optimization_summary.json.\nKetepatan dengan toleransi maksimal 5 poin: {m['tolerance']['model']['within_5']*100:.2f}% dibanding baseline {m['tolerance']['baseline']['within_5']*100:.2f}%.\nPada aktual bukan nol: {m['tolerance']['model']['nonzero_within_5']*100:.2f}%. Ini bukan akurasi klasifikasi.\nTidak ada kandidat gagal yang dipilih berdasarkan hasil replay 2025-2026.\n"

if (ROOT / 'reports/v5_comparison.json').exists():
    pg=json.loads((ROOT / 'reports/v5_comparison.json').read_text(encoding='utf-8'))
    readme=readme.replace('VERSI 3 / SEIMBANG MAE DAN RMSE','RIWAYAT VERSI 1 SAMPAI 3')
    readme=readme.replace('EKSPERIMEN LANJUTAN','EKSPERIMEN TERDAHULU SEBELUM DATA SNAP')
    readme += f"\nDATA PERAN SEBELUM LAGA\nModel aktif: {m['version']}. Data tambahan memakai snap dan cedera pekan sebelumnya.\nMAE seluruh kandidat: V3 {pg['full_cohort']['v3']['mae']:.4f}, model {pg['full_cohort']['model']['mae']:.4f}.\nRMSE seluruh kandidat: V3 {pg['full_cohort']['v3']['rmse']:.4f}, model {pg['full_cohort']['model']['rmse']:.4f}.\nSubset fantasy {pg['fantasy_pool']['n']} baris dinilai terpisah. Populasi lebih sempit tidak boleh dipakai untuk mengklaim perbaikan terhadap metrik populasi lama.\nReproduksi: python src/pregame_features.py lalu python src/upgrade_v5.py. Jalankan Perbarui_Prediksi.cmd sesudahnya.\nPemeriksaan: python src/test_pregame.py. Sumber dan hash: reports/pregame_sources.json.\nUntuk membaca ulang fitur dari CSV prediksi gunakan pd.read_csv(path, float_precision='round_trip') agar input model pohon tetap persis. Arsip tidak memiliki waktu publikasi pertama seluruh snap. Semua hasil 2025-2026 adalah replay historis.\n"
    if m['version']=='v5': readme=readme.replace('TUJUAN\n','MODEL AKTIF V5 - RIWAYAT SNAP DAN CEDERA PEKAN SEBELUMNYA\n\nTUJUAN\n',1)

(ROOT / 'README.txt').write_text(readme, encoding='utf-8')
(ROOT / 'requirements-lock.txt').write_text(f'pandas=={pd.__version__}\nnumpy=={np.__version__}\nscikit-learn=={sklearn.__version__}\nlightgbm==4.7.0\n', encoding='utf-8')
linkedin_text=f'''Project title
NFL Fantasy Scout 2026 | PPR Forecasting & Lineup Optimization

Description
Built an NFL fantasy analytics dashboard using nflverse player statistics from 2019–2026. The project forecasts next-week PPR points for QB, RB, WR, and TE, compares Start or Sit options, and optimizes seven offensive lineup slots from a user-selected roster.

Used expanding season validation for model selection and a separate 2025–2026 test period. The selected gradient boosting model achieved {m['test']['mae']:.2f} PPR-point MAE across {m['test']['n']:,} test observations, reducing MAE by {m['mae_improvement_pct']:.2f}% compared with a five-game rolling-average baseline. Forecasts include empirical error ranges and transparent data-availability notes.

Initial snapshot: October 27, 2026, forecasting Week 8. Timestamped nflverse roster and injury snapshots are integrated. Target-week injury reports may be unpublished and game-day active status is not guaranteed. This is an independent portfolio project.

Skills
Python, Machine Learning, Regression, Feature Engineering, Time Series Validation, Data Visualization, JavaScript, Optimization

Catatan
Draf ini belum dipublikasikan. Pahami alur dan hasilnya sebelum menjelaskan proyek dalam wawancara.
Proyek disimpan di komputer. Demo belum diterbitkan.
'''
if m.get('version') in ('v2','v3','v5'):
    linkedin_text=linkedin_text.replace('a separate 2025–2026 test period','a retrospective replay of the previously inspected 2025–2026 test period')
    linkedin_text=linkedin_text.replace('The selected gradient boosting model','The selected gradient boosting model optimized for absolute error')
    linkedin_text=linkedin_text.replace('Initial snapshot:', 'The replay uses the same candidate rows and labels as version 1. Independent confirmation on new future weeks is still needed.\n\nInitial snapshot:')
if m.get('version') in ('v3','v5'):
    linkedin_text=linkedin_text.replace('optimized for absolute error','balanced for MAE and RMSE using historical validation')
    linkedin_text=linkedin_text.replace('next-week PPR points','next-game PPR points')
    linkedin_text=linkedin_text.replace('The selected gradient boosting model','The selected LightGBM ensemble')
    linkedin_text=linkedin_text.replace('Initial snapshot: October 27, 2026, forecasting Week 8.','Latest forecast uses completed Week 7 data, timezone-aware next-game scheduling, and WIB kickoff times. The 2026 regular season extends into January 2027. Playoffs are outside scope.')
(ROOT / 'LINKEDIN_COPY.txt').write_text(linkedin_text,encoding='utf-8')
print('Notebook and portfolio notes created.')
