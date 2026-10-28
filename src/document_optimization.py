"""Add transparent experiment and tolerance results to the local deliverables."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'dist/app.js';s=p.read_text(encoding='utf-8')
if 'function renderOptimization()' not in s:
    s=s.replace('renderUpgrade();renderEvidence();renderCompare();','renderUpgrade();renderEvidence();renderOptimization();renderCompare();')
    s+='''
function renderOptimization(){
 const r=state.data.optimization_summary,t=state.data.metrics.tolerance;if(!r||!t)return;
 const panel=document.createElement('article');panel.className='report-card';panel.id='optimization-report';
 const best=r.lowest_rmse_candidate,ref=r.reference_cv;
 panel.innerHTML='<p class="eyebrow">TOLERANCE / CONTROLLED EXPERIMENT</p><h3>Seberapa sering prediksi mendekati hasil?</h3><div class="table-wrap"><table><thead><tr><th>Ukuran</th><th>Model</th><th>Baseline</th></tr></thead><tbody>'+[['Selisih maksimal 5 poin · seluruh kandidat','within_5'],['Selisih maksimal 3 poin · seluruh kandidat','within_3'],['Selisih maksimal 5 poin · hasil aktual bukan nol','nonzero_within_5']].map(([label,key])=>'<tr><td>'+label+'</td><td>'+fmt(t.model[key]*100,2)+'%</td><td>'+fmt(t.baseline[key]*100,2)+'%</td></tr>').join('')+'</tbody></table></div><p>Perhitungan dari '+fmt(state.data.metrics.test.n,0)+' observasi pengujian ulang historis. Sebanyak '+fmt(t.model.nonzero_n,0)+' memiliki poin aktual bukan nol. Persentase ini memakai toleransi galat dalam satuan PPR dan bukan akurasi klasifikasi atau ketepatan skor persis.</p><h3>Hasil percobaan optimasi lanjutan</h3><p>'+r.candidate_count+' konfigurasi termasuk pembanding diuji pada validasi 2023–2024. Model per posisi, model dua tahap, konteks tim, pembobotan musim terbaru, dan campuran model diperiksa. '+(r.promoted?'Kandidat dipilih menggunakan hasil validasi historis.':'Versi 3 dipertahankan karena belum ada kandidat yang memenuhi kedua syarat perbaikan validasi.')+'</p><div class="table-wrap"><table><thead><tr><th>Validasi historis</th><th>MAE</th><th>RMSE</th></tr></thead><tbody><tr><td>Versi 3</td><td>'+fmt(ref.cv_mae,4)+'</td><td>'+fmt(ref.cv_rmse,4)+'</td></tr><tr><td>Kandidat dengan RMSE terendah</td><td>'+fmt(best.cv_mae,4)+'</td><td>'+fmt(best.cv_rmse,4)+'</td></tr></tbody></table></div><p>Kandidat dengan RMSE terendah memiliki MAE lebih tinggi. Syarat penggantian adalah MAE tidak meningkat dan RMSE validasi turun sedikitnya 0,25%. Prediksi terkunci sebelumnya tetap disimpan. Data 2025–2026 yang sudah pernah diperiksa tidak dipakai untuk memilih kandidat tambahan ini.</p>';
 $('tab-backtest').querySelector('.section-heading').insertAdjacentElement('afterend',panel);
}
'''
    p.write_text(s,encoding='utf-8')

p=ROOT/'src/build_notebook.py';s=p.read_text(encoding='utf-8')
if '## 11. Percobaan optimasi dan toleransi poin' not in s:
    marker="nbf.write(nb, ROOT / 'NFL_Fantasy_Scout_2026_Mochamad_Rizky_Maulana_Aviansyah.ipynb')"
    hook='''if (ROOT / 'reports/optimization_summary.json').exists():
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
'''
    s=s.replace(marker,hook+'\n'+marker)
    marker="(ROOT / 'README.txt').write_text(readme, encoding='utf-8')"
    hook='''if (ROOT / 'reports/optimization_summary.json').exists():
    optimization=json.loads((ROOT / 'reports/optimization_summary.json').read_text(encoding='utf-8'))
    readme += f"\\nEKSPERIMEN LANJUTAN\\n{optimization['candidate_count']} konfigurasi termasuk pembanding dinilai memakai validasi 2023-2024.\\nStatus: {optimization['status']}\\nMAE, RMSE, sumber dan pilihan lineup lama dipertahankan jika tidak ada kandidat yang memenuhi aturan validasi.\\nReproduksi: python src/improve_v4.py lalu python src/improve_v4.py --stage2 dan python src/summarize_optimization.py.\\nPengujian fitur: python src/test_v4.py. Laporan: reports/v4_validation_phase2.json dan reports/optimization_summary.json.\\nKetepatan dengan toleransi maksimal 5 poin: {m['tolerance']['model']['within_5']*100:.2f}% dibanding baseline {m['tolerance']['baseline']['within_5']*100:.2f}%.\\nPada aktual bukan nol: {m['tolerance']['model']['nonzero_within_5']*100:.2f}%. Ini bukan akurasi klasifikasi.\\nTidak ada kandidat gagal yang dipilih berdasarkan hasil replay 2025-2026.\\n"
'''
    s=s.replace(marker,hook+'\n'+marker)
    p.write_text(s,encoding='utf-8')
print('Local dashboard and notebook now explain the optimization outcome and tolerance metric.')
