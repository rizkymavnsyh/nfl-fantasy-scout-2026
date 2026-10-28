"""Integrate pre-game evidence with the existing local notebook and dashboard."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

p=ROOT/'dist/app.js';s=p.read_text(encoding='utf-8')
if 'function renderPregame()' not in s:
    s=s.replace("fixture:'all'}","fixture:'all',pool:'fantasy'}")
    s=s.replace("let ps=state.data.players.filter(p=>isUpcoming(p)&&", "let ps=state.data.players.filter(p=>(state.pool!=='fantasy'||!state.data.pregame_context||p.fantasy_pool===true)&&isUpcoming(p)&&")
    s=s.replace('renderOptimization();renderCompare();','renderOptimization();renderPregame();renderCompare();')
    s=s.replace("const v3=m.version==='v3';","const v3=['v3','v5'].includes(m.version);")
    s=s.replace("function renderUpgrade(){const m=state.data.metrics;const c=", "function renderUpgrade(){const m=state.data.metrics;const c=")
    s=s.replace('<p class="availability">${esc(availabilityText(p))}</p>', '<p class="availability">${esc(availabilityText(p))}</p><p class="availability">${p.prior_snap_pct==null?\'Riwayat snap belum tersedia.\':\'Snap ofensif terakhir: \'+fmt(p.prior_snap_pct*100,0)+\'% · \'+p.prior_snap_weeks_ago+\' pekan sebelum laga target.\'} ${esc(p.prior_injury_note||\'\')} ${p.fantasy_pool===false?\'Di luar kriteria kandidat fantasy berdasarkan riwayat peran.\':\'\'}</p>')
    s=s.replace("Versi 3 dipertahankan karena belum ada kandidat yang memenuhi kedua syarat perbaikan validasi.","Percobaan terdahulu tanpa sumber snap mempertahankan versi 3. Hasil tambahan data peran bermain dijelaskan pada panel tersendiri.")
    s+='''
function renderPregame(){
 const data=state.data,c=data.pregame_comparison,context=data.pregame_context;
 if(context){
  const label=document.createElement('label');label.innerHTML='Cakupan <select id="pool-filter"><option value="fantasy">Kandidat fantasy</option><option value="all">Semua kandidat</option></select>';
  $('tab-scout').querySelector('.board-summary').appendChild(label);
  $('pool-filter').addEventListener('change',e=>{state.pool=e.target.value;state.page=0;renderBoard();});
  const note=document.createElement('p');note.className='fine-print';note.id='pregame-pool-note';note.textContent=context.fantasy_pool_n+' dari '+context.n+' kandidat memenuhi kriteria fantasy dari kesempatan dan snap sebelumnya. Filter ini bukan konfirmasi starter atau aktif saat laga. Pilih Semua kandidat untuk melihat pemain lain.';$('availability-summary').insertAdjacentElement('afterend',note);
 }
 if(!c)return;
 const panel=document.createElement('article');panel.className='report-card';panel.id='pregame-report';
 const full=c.full_cohort,pool=c.fantasy_pool,cv=c.validation;
 panel.innerHTML='<p class="eyebrow">PRE-GAME ROLE / '+(c.promoted?'ACTIVE MODEL V5':'ADDITIONAL EVIDENCE')+'</p><h3>Informasi peran sebelum pertandingan.</h3><p>Model memakai riwayat jumlah dan persentase snap ofensif, perubahan keterlibatan, serta indikator laporan cedera pekan sebelumnya. Snap dan laporan akhir dari laga target tidak dimasukkan. Snap terakhir bukan jaminan starter pada laga berikutnya.</p><div class="table-wrap"><table><thead><tr><th>Populasi sama · '+fmt(full.n,0)+' observasi</th><th>V3</th><th>Model aktif</th><th>Baseline</th></tr></thead><tbody>'+[['MAE','mae'],['RMSE','rmse'],['R²','r2']].map(([label,key])=>'<tr><td>'+label+'</td><td>'+fmt(full.v3[key],3)+'</td><td>'+fmt(full.model[key],3)+'</td><td>'+fmt(full.baseline[key],3)+'</td></tr>').join('')+'</tbody></table></div><p>'+(c.promoted?'Model V5 dipilih dari validasi historis.':'Model lama dipertahankan setelah validasi.')+' MAE validasi '+fmt(cv.v3.cv_mae,4)+' → '+fmt(cv.selected.cv_mae,4)+'. RMSE validasi '+fmt(cv.v3.cv_rmse,4)+' → '+fmt(cv.selected.cv_rmse,4)+'.</p><h3>Kandidat yang relevan untuk fantasy</h3><p>Subset ditetapkan dari riwayat kesempatan bermain dan snap, tanpa melihat hasil laga target. '+fmt(pool.n,0)+' observasi. Pada subset yang sama: MAE model '+fmt(pool.model.mae)+', V3 '+fmt(pool.v3.mae)+', baseline '+fmt(pool.baseline.mae)+'. Persentase selisih maksimal 5 poin: model '+fmt(pool.tolerance.model.within_5*100,2)+'% dan baseline '+fmt(pool.tolerance.baseline.within_5*100,2)+'%.</p><div class="note">Hasil subset tidak dibandingkan langsung dengan persentase seluruh kandidat karena populasinya berbeda. Semua hasil 2025–2026 merupakan replay historis. Waktu publikasi pertama setiap snap historis tidak tersedia. Konfirmasi pada prediksi baru yang terkunci tetap diperlukan.</div>';
 $('tab-backtest').querySelector('.section-heading').insertAdjacentElement('afterend',panel);
}
'''
    p.write_text(s,encoding='utf-8')

p=ROOT/'dist/style.css';s=p.read_text(encoding='utf-8')
if '/* pre-game pool */' not in s:s+='\n/* pre-game pool */\n.board-summary{flex-wrap:wrap;gap:12px}.source-links{flex-wrap:wrap}#pregame-pool-note{margin:12px 0 18px}\n';p.write_text(s,encoding='utf-8')

p=ROOT/'src/build_notebook.py';s=p.read_text(encoding='utf-8')
s=s.replace("in ('v2','v3')", "in ('v2','v3','v5')").replace("if m.get('version')=='v3':", "if m.get('version') in ('v3','v5'):")
if '## 12. Peran sebelum laga dan kandidat fantasy' not in s:
    marker="nbf.write(nb, ROOT / 'NFL_Fantasy_Scout_2026_Mochamad_Rizky_Maulana_Aviansyah.ipynb')"
    hook='''if (ROOT / 'reports/v5_comparison.json').exists():
    comparison=json.loads((ROOT / 'reports/v5_comparison.json').read_text(encoding='utf-8'))
    for cell in nb.cells:
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('## Perubahan versi 3 dan penggunaan saat ini','## Riwayat versi 1 sampai 3')
            cell.source=cell.source.replace('belum adanya laporan cedera dan starter langsung','belum adanya konfirmasi starter langsung')
            if m.get('version')=='v5':
                cell.source=cell.source.replace('Versi 3 membandingkan LightGBM dengan sasaran kuadrat, Huber, Fair, median, serta campuran model.', 'Versi 5 menambahkan snap ofensif dan indikator cedera pekan sebelumnya. Kandidat meliputi regresi, model per posisi, model dua tahap, serta campuran dengan V3.')
                cell.source=cell.source.replace('Pemilihan menggunakan rerata MAE kedua fold validasi.', 'Pemilihan V5 meminimalkan rerata RMSE validasi, dengan MAE tidak boleh lebih tinggi dari V3 dan RMSE harus turun sedikitnya 0,25%.')
                cell.source=cell.source.replace('rebalance_plan.json','v5_plan.json')
                cell.source=cell.source.replace('dengan sasaran RMSE minimum dan batas kenaikan MAE validasi maksimal 5% terhadap versi 2','dengan MAE validasi tidak meningkat dan RMSE turun minimal 0,25% terhadap V3')
        elif m.get('version')=='v5':
            cell.source=cell.source.replace('from rebalance import run as run_upgrade','from upgrade_v5 import run as run_upgrade')
    if comparison['promoted']:
        nb.cells.insert(2,nbf.v4.new_markdown_cell(f"## Model aktif V5\\n\\nPada populasi pengujian ulang yang sama, MAE V3 {comparison['full_cohort']['v3']['mae']:.4f} menjadi {comparison['full_cohort']['model']['mae']:.4f}. RMSE {comparison['full_cohort']['v3']['rmse']:.4f} menjadi {comparison['full_cohort']['model']['rmse']:.4f}. Pemilihan memakai validasi 2023–2024. Data 2025–2026 sudah diperiksa sebelumnya dan bukan holdout independen."))
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
'''
    s=s.replace(marker,hook+'\n'+marker)
    marker="(ROOT / 'README.txt').write_text(readme, encoding='utf-8')"
    hook='''if (ROOT / 'reports/v5_comparison.json').exists():
    pg=json.loads((ROOT / 'reports/v5_comparison.json').read_text(encoding='utf-8'))
    readme=readme.replace('VERSI 3 / SEIMBANG MAE DAN RMSE','RIWAYAT VERSI 1 SAMPAI 3')
    readme=readme.replace('EKSPERIMEN LANJUTAN','EKSPERIMEN TERDAHULU SEBELUM DATA SNAP')
    readme += f"\\nDATA PERAN SEBELUM LAGA\\nModel aktif: {m['version']}. Data tambahan memakai snap dan cedera pekan sebelumnya.\\nMAE seluruh kandidat: V3 {pg['full_cohort']['v3']['mae']:.4f}, model {pg['full_cohort']['model']['mae']:.4f}.\\nRMSE seluruh kandidat: V3 {pg['full_cohort']['v3']['rmse']:.4f}, model {pg['full_cohort']['model']['rmse']:.4f}.\\nSubset fantasy {pg['fantasy_pool']['n']} baris dinilai terpisah. Populasi lebih sempit tidak boleh dipakai untuk mengklaim perbaikan terhadap metrik populasi lama.\\nReproduksi: python src/pregame_features.py lalu python src/upgrade_v5.py. Jalankan Perbarui_Prediksi.cmd sesudahnya.\\nPemeriksaan: python src/test_pregame.py. Sumber dan hash: reports/pregame_sources.json.\\nArsip tidak memiliki waktu publikasi pertama seluruh snap. Semua hasil 2025-2026 adalah replay historis.\\n"
    if m['version']=='v5': readme=readme.replace('TUJUAN\\n','MODEL AKTIF V5 - RIWAYAT SNAP DAN CEDERA PEKAN SEBELUMNYA\\n\\nTUJUAN\\n',1)
'''
    s=s.replace(marker,hook+'\n'+marker)
    p.write_text(s,encoding='utf-8')

p=ROOT/'Perbarui_Prediksi.cmd';s=p.read_text()
if 'src\\pregame_features.py' not in s:
    s=s.replace('py -3.11 src\\refresh_forecasts.py','py -3.11 src\\pregame_features.py --refresh-current\nif errorlevel 1 goto gagal\npy -3.11 src\\refresh_forecasts.py')
    s=s.replace('py -3.11 src\\availability.py','py -3.11 src\\pregame_forecasts.py\nif errorlevel 1 goto gagal\npy -3.11 src\\availability.py')
    p.write_text(s)
print('Pre-game evidence integrated into local dashboard, notebook builder and manual refresh.')
