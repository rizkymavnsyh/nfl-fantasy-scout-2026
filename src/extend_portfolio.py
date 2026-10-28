"""Maintain the dashboard integration and documentation hooks for evidence reports."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

js=ROOT/'dist/app.js'
text=js.read_text(encoding='utf-8')
if 'const availabilityOK=' not in text:
    text=text.replace('const PAGE_SIZE=12;', '''const PAGE_SIZE=12;
const availabilityOK=(p,now=Date.now())=>p.lineup_eligible===true&&!!p.availability_as_of_utc&&now-Date.parse(p.availability_as_of_utc)>=0&&now-Date.parse(p.availability_as_of_utc)<=48*3600000;
const availabilityText=p=>(p.availability_note||'Roster belum diperiksa')+' · Cedera W'+p.week+': '+(p.injury_status||'Belum tersedia')+(p.availability_as_of_utc?' · Diambil '+new Date(p.availability_as_of_utc).toLocaleString('id-ID',{timeZone:'Asia/Jakarta'})+' WIB':'')+(!availabilityOK(p)?' · Disisihkan dari lineup otomatis.':' · Periksa konfirmasi aktif sebelum laga.');''')
    text=text.replace("${p.gap>1?' · CEK STATUS':''}","${!availabilityOK(p)?' · CEK / DISISIHKAN':p.injury_status==='Questionable'?' · QUESTIONABLE':''}")
    start=text.index('<p class="availability">')
    end=text.index('</p>',start)
    text=text[:start]+'<p class="availability">${esc(availabilityText(p))}'+text[end:]
    text=text.replace('a.week===b.week&&p.id===best.id', 'a.week===b.week&&availabilityOK(a)&&availabilityOK(b)&&p.id===best.id')
    text=text.replace("a.week!==b.week?'Target pekan berbeda.':diff", "a.week!==b.week?'Target pekan berbeda.':!availabilityOK(a)||!availabilityOK(b)?'Periksa ketersediaan sebelum memilih.':diff")
    text=text.replace('${spark(p)}${context(p)}</article>', '${spark(p)}${context(p)}<p class="availability">${esc(availabilityText(p))}</p></article>')
    text=text.replace('isUpcoming(p)&&p.week===state.data.metrics.prediction_week','isUpcoming(p)&&availabilityOK(p)&&p.week===state.data.metrics.prediction_week')
    text=text.replace('Kandidat dengan laga pada pekan lain disisihkan.', 'Kandidat pada pekan lain, roster tidak aktif, laporan Out/Doubtful, tim berbeda, atau bukti kedaluwarsa disisihkan.')
    text=text.replace('window.FantasyScout={optimizeLineup,isUpcoming};', 'window.FantasyScout={optimizeLineup,isUpcoming,availabilityOK};')
    text=text.replace('renderReport();renderUpgrade();renderCompare();', 'renderReport();renderUpgrade();renderEvidence();renderCompare();')
    text+='''
function renderEvidence(){
 const data=state.data,a=data.availability;
 if(a){const panel=document.createElement('div');panel.className='note';panel.id='availability-summary';panel.textContent='Roster diperiksa: '+a.eligible+' dari '+a.total+' kandidat lolos pemeriksaan saat snapshot. Laporan cedera terbaru: W'+a.injury_latest_week+'. Bukti berlaku maksimal 48 jam. ACT berarti berada pada roster aktif, bukan konfirmasi bermain saat laga.';$('tab-scout').querySelector('.section-heading').insertAdjacentElement('afterend',panel);}
 const wrap=document.createElement('article');wrap.className='report-card';wrap.id='evidence-report';
 const r=data.lineup_evaluation,ps=data.prospective_evaluation||[],latest=ps.at(-1);
 let html='<p class="eyebrow">RECOMMENDATION / FROZEN FORECASTS</p><h3>Apakah pilihan lineup menghasilkan lebih banyak poin?</h3>';
 if(r&&r.n_rosters)html+='<div class="table-wrap"><table><thead><tr><th>Simulasi historis</th><th>Hasil</th></tr></thead><tbody><tr><td>Rerata poin aktual pilihan model</td><td>'+fmt(r.model_mean_points)+'</td></tr><tr><td>Rerata poin aktual pilihan baseline</td><td>'+fmt(r.baseline_mean_points)+'</td></tr><tr><td>Selisih model terhadap baseline</td><td>'+fmt(r.mean_difference)+'</td></tr><tr><td>Roster model lebih unggul</td><td>'+fmt(r.win_rate*100,1)+'%</td></tr></tbody></table></div><p>'+r.n_rosters+' roster simulasi pada '+r.n_weeks+' pekan. Kedua metode memilih dari roster sama: 2 QB, 4 RB, 5 WR, 2 TE. Baseline memilih berdasarkan rata-rata lima hasil sebelumnya. Tidak ada pemilihan menggunakan hasil laga target.</p><div class="note">Ini simulasi retrospektif dari kandidat historis, bukan roster liga sungguhan. Data cedera historis belum dipakai. Simulasi dalam pekan sama saling berkaitan. Hasil ini tidak menjamin keuntungan pada roster Anda.</div>';
 html+='<h3>Prediksi sebelum kickoff</h3>';
 if(latest)html+='<p>Snapshot terkunci '+esc(latest.snapshot_id)+'. '+latest.n_frozen+' prediksi disimpan sebelum kickoff, '+latest.n_scored+' sudah dapat dinilai dan '+latest.n_pending+' menunggu hasil/statistik lengkap.</p><p>'+esc(latest.status)+'</p>'+(latest.n_scored?'<p>MAE '+fmt(latest.mae)+' · RMSE '+fmt(latest.rmse)+'</p>':'<p>Metrik prospektif belum tersedia karena pertandingan target belum selesai.</p>');
 html+='<p>Prediksi dan pilihan lineup tersimpan bersama hash. Pembaruan data tidak menimpa snapshot lama. Penilaian dilakukan setelah skor akhir dan statistik kedua tim tersedia. Hash lokal membantu mendeteksi perubahan, tetapi bukan sertifikasi waktu dari pihak ketiga.</p>';
 wrap.innerHTML=html;$('tab-backtest').querySelector('.section-heading').insertAdjacentElement('afterend',wrap);
}
'''
    js.write_text(text,encoding='utf-8')

html=ROOT/'dist/index.html';text=html.read_text(encoding='utf-8')
text=text.replace('Roster, cedera, trade, dan starter resmi perlu diperiksa. Pemain baru, bye week, K, dan DST berada di luar cakupan.', 'Status roster dan cedera diperiksa dari snapshot nflverse. Laporan yang belum terbit tidak dianggap sehat. Tim bye diarahkan ke laga berikutnya. Pemain sebelum penampilan pertama, K, dan DST berada di luar cakupan.')
text=text.replace('Belum mengintegrasikan laporan cedera langsung atau daftar pemain aktif resmi. Identitas tim mengikuti penampilan terakhir.', 'Roster dan laporan cedera dari nflverse ditampilkan bersama waktu pengambilan. Roster ACT tidak menjamin pemain aktif saat pertandingan. Kandidat dengan tim roster berbeda disisihkan dari lineup. Laporan cedera pekan target yang belum tersedia ditandai.')
html.write_text(text,encoding='utf-8')

builder=ROOT/'src/build_notebook.py';text=builder.read_text(encoding='utf-8')
hook="nbf.write(nb, ROOT / 'NFL_Fantasy_Scout_2026_Mochamad_Rizky_Maulana_Aviansyah.ipynb')"
if '## 10. Ketersediaan dan evaluasi prospektif' not in text:
    addition='''# Evidence sections use the exact same saved reports as the dashboard.
for cell in nb.cells:
    if cell.cell_type=='markdown':
        cell.source=cell.source.replace('Keterbatasan utama adalah belum adanya laporan cedera dan starter langsung, ketergantungan tim pada penampilan terakhir, ketidaklengkapan snapshot DAL–PHI pekan 7, dan cakupan kandidat yang mengecualikan pemain sebelum penampilan pertama.', 'Ketersediaan pemain memakai snapshot roster dan cedera dari nflverse. ACT bukan konfirmasi aktif saat laga. Laporan pekan target yang belum terbit tetap ditandai. Snapshot benchmark tetap lama, sedangkan forecast menggunakan data/live. Pemain sebelum penampilan pertama belum dicakup.')
        cell.source=cell.source.replace('Pembaruan berikutnya sebaiknya memprioritaskan snapshot roster dan cedera sebelum laga, validasi prospektif pada pekan baru yang belum pernah diperiksa, dan evaluasi lineup terhadap baseline rekomendasi.', 'Bagian tambahan mendokumentasikan pemeriksaan roster, prediksi terkunci, dan evaluasi lineup simulasi terhadap baseline. Konfirmasi prospektif harus menunggu hasil pertandingan baru.')
nb.cells.append(nbf.v4.new_markdown_cell("""## 10. Ketersediaan dan evaluasi prospektif

Status roster dan cedera dicocokkan menggunakan ID pemain, tim, dan pekan target. ACT berarti berada pada roster aktif, bukan jaminan bermain. Pemain nonaktif, tim berbeda, Out/Doubtful, atau bukti roster melebihi 48 jam tidak dipilih oleh lineup otomatis. Laporan yang belum tersedia bukan bukti bahwa pemain sehat.

Snapshot pada `reports/frozen` disimpan sebelum kickoff menggunakan waktu penguncian sesungguhnya. CSV prediksi, pilihan lineup, hash model, dan sumber disimpan tanpa menimpa snapshot lama. Hasil dinilai terpisah setelah skor akhir dan statistik kedua tim tersedia. Belum ada angka prospektif jika laga belum selesai.

Evaluasi lineup historis menggunakan 30 roster simulasi per pekan dengan seed tetap. Kedua kebijakan memilih dari roster yang sama. Pemilihan hanya memakai prediksi atau rerata lima hasil sebelumnya. Nilai aktual dipakai setelah lineup dipilih. Roster simulasi bukan data kepemilikan liga dan tidak menerapkan status cedera historis. Banyak roster pada pekan sama bukan pengamatan independen."""))
nb.cells.append(nbf.v4.new_code_cell("""availability = json.loads((ROOT / 'reports/availability.json').read_text(encoding='utf-8'))
lineup_report = json.loads((ROOT / 'reports/lineup_replay.json').read_text(encoding='utf-8'))
prospective = json.loads((ROOT / 'reports/prospective_evaluation.json').read_text(encoding='utf-8'))
display(pd.DataFrame([{'Kandidat': availability['total'], 'Lolos pemeriksaan roster': availability['eligible'], 'Laporan cedera terbaru': availability['injury_latest_week']}]))
display(pd.DataFrame([{k: lineup_report[k] for k in ['n_rosters','n_weeks','model_mean_points','baseline_mean_points','mean_difference','win_rate']}]))
display(pd.DataFrame([{k: row[k] for k in ['snapshot_id','n_frozen','n_scored','n_pending','status']} for row in prospective]))"""))
'''
    text=text.replace(hook,addition+'\n'+hook)
text=text.replace('Week 1, pemain sebelum penampilan pertama, bye, K, DST, dan IDP tidak dicakup.', 'Week 1, pemain sebelum penampilan pertama, K, DST, dan IDP tidak dicakup. Tim bye menargetkan laga terjadwal berikutnya dan tidak masuk lineup pekan bye.')
text=text.replace('Belum ada laporan cedera langsung atau active roster resmi. Tim mengikuti penampilan terakhir.', 'Snapshot roster dan cedera dari nflverse tersedia. ACT bukan konfirmasi game-day active. Tim yang berbeda dari tim model disisihkan. Sumber yang gagal atau kedaluwarsa ditandai.')
text=text.replace('Tidak ada pembaruan otomatis atau laporan cedera langsung.', 'Pembaruan berjalan manual. Roster dan cedera mengikuti snapshot nflverse. Laporan pekan target mungkin belum terbit. Konfirmasi aktif saat laga tetap harus diperiksa.')
text=text.replace("scikit-learn=={sklearn.__version__}\\n'", "scikit-learn=={sklearn.__version__}\\nlightgbm==4.7.0\\n'")
text=text.replace('Live injury and official active-roster feeds are not integrated.', 'Timestamped nflverse roster and injury snapshots are integrated. Target-week injury reports may be unpublished and game-day active status is not guaranteed.')
extra='''
readme += """\nPEMERIKSAAN KETERSEDIAAN DAN BUKTI
Perbarui_Prediksi.cmd mengambil forecast, roster, dan cedera, mengunci snapshot baru, lalu menilai snapshot lama setelah hasil tersedia.
Penguncian memakai waktu aktual. reports/frozen tidak ditimpa. Hash mendeteksi perubahan lokal, bukan bukti waktu independen dari pihak ketiga.
Jalankan python src/prospective.py untuk menilai snapshot yang ada, atau tambahkan --freeze untuk menyimpan snapshot baru sebelum kickoff.
Gunakan --replay untuk mengulang simulasi historis. Protokol tersimpan di reports/lineup_protocol.json.
Simulasi menggunakan roster buatan. Metrik poin lineup berbeda dari MAE model dan tidak menjamin keuntungan pada liga pengguna.
ACT bukan konfirmasi aktif saat laga. Laporan cedera yang belum terbit tidak dianggap sehat. Bukti roster berlaku maksimal 48 jam.
Jangan memakai status pemain terbaru untuk memodifikasi kembali keputusan snapshot lama.
Dashboard lokal belum menjadi demo publik. Panduan hosting tanpa model/raw data ada di PANDUAN_PUBLIKASI.txt.
"""
'''
marker="(ROOT / 'README.txt').write_text(readme, encoding='utf-8')"
if 'PEMERIKSAAN KETERSEDIAAN DAN BUKTI' not in text:text=text.replace(marker,extra+'\n'+marker)
builder.write_text(text,encoding='utf-8')

batch=ROOT/'Perbarui_Prediksi.cmd';text=batch.read_text()
if 'src\\availability.py' not in text:
    text=text.replace('py -3.11 src\\build_notebook.py','py -3.11 src\\availability.py\nif errorlevel 1 goto gagal\npy -3.11 src\\prospective.py --freeze\nif errorlevel 1 goto gagal\npy -3.11 src\\build_notebook.py')
    batch.write_text(text)

(ROOT/'PANDUAN_PUBLIKASI.txt').write_text('''DEMO PUBLIK DAN REPRODUKSI

Folder dist berisi dashboard statis siap hosting. Tidak memerlukan backend atau model joblib pada server.
Di GitHub Pages, unggah isi dist ke repository yang Anda pilih, lalu aktifkan Pages untuk folder root.
Jangan unggah folder .openai atau token. Notebook, src, requirements, dan laporan metodologi dapat dimasukkan ke repository reproduksi terpisah.
Raw data/model besar dapat diperoleh melalui perintah reproduksi dan snapshot ZIP lokal. Pertahankan atribusi nflverse CC BY 4.0.
Demo statis menampilkan waktu snapshot. Data tidak otomatis berubah setelah diterbitkan. Jalankan pembaruan dan publikasikan ulang untuk mengganti data.
Alamat 127.0.0.1 hanya dapat dibuka pada komputer sendiri. Belum ada deployment publik pada tahap ini.
''',encoding='utf-8')
print('Dashboard, notebook builder, refresh workflow and docs extended.')
