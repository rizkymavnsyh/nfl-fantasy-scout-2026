"""Create a self-contained offline dashboard and a source/data portfolio ZIP."""
from pathlib import Path
import base64
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[1]
html=(ROOT/'dist/index.html').read_text(encoding='utf-8')
css=(ROOT/'dist/style.css').read_text(encoding='utf-8')
js=(ROOT/'dist/app.js').read_text(encoding='utf-8')
data=json.loads((ROOT/'dist/data.json').read_text(encoding='utf-8'))
if not data.get('availability') or not data.get('prospective_evaluation'):
    raise ValueError('Lengkapi data dengan Perbarui_Prediksi.cmd sebelum membuat paket. Bukti roster/prediksi terkunci belum ada.')
embedded=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
html=html.replace('<link rel="stylesheet" href="style.css">',f'<style>{css}</style>')
html=html.replace('<script src="app.js"></script>',f'<script type="application/json" id="portfolio-data">{embedded}</script><script>{js}</script>')
# Keep CSV/JSON downloads usable even when the dashboard is opened from disk.
for name,mime in [('forecasts.csv','text/csv'),('test_predictions.csv','text/csv'),('metrics.json','application/json')]:
    encoded=base64.b64encode((ROOT/'dist'/name).read_bytes()).decode('ascii')
    html=html.replace(f'href="{name}" download',f'href="data:{mime};base64,{encoded}" download="{name}"')
offline=ROOT/'NFL_Fantasy_Scout_2026.html'
offline.write_text(html,encoding='utf-8')
readme=ROOT/'README.txt'
text=readme.read_text(encoding='utf-8')
marker='JALANKAN DASHBOARD\n'
if 'FILE OFFLINE' not in text:
    text=text.replace(marker,'FILE OFFLINE\nKlik dua kali NFL_Fantasy_Scout_2026.html. Dashboard dan unduhan data di dalamnya dapat digunakan tanpa server atau internet.\n\n'+marker)
readme.write_text(text,encoding='utf-8')
archive=ROOT.parent/'NFL_Fantasy_Scout_2026_Mochamad_Rizky_Maulana_Aviansyah.zip'
excluded={'.git','__pycache__','.venv','.ipynb_checkpoints','.sites-checkout'}
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for file in ROOT.rglob('*'):
        if not file.is_file() or excluded.intersection(file.relative_to(ROOT).parts):continue
        if file.suffix=='.log' or '.openai' in file.relative_to(ROOT).parts:continue
        z.write(file,ROOT.name+'/'+file.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(json.dumps({'offline_html':str(offline),'offline_bytes':offline.stat().st_size,
                  'zip':str(archive),'zip_bytes':archive.stat().st_size,
                  'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))
