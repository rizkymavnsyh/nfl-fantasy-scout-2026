"""Execute every cell in one clean Python/IPython process, without kernel sockets.

This avoids Windows kernel connection-file ACL limitations. Outputs are captured
from actual cell execution, including rich tables and plots, not fabricated.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import base64
import io
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.display import Image, display
from traitlets.config import Config

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
path=ROOT/'NFL_Fantasy_Scout_2026_Mochamad_Rizky_Maulana_Aviansyah.ipynb'
nb=nbformat.read(path,as_version=4)
config=Config()
config.HistoryManager.hist_file=':memory:'
shell=InteractiveShell.instance(config=config)
def show_figures(*args,**kwargs):
    for n in plt.get_fignums():
        fig=plt.figure(n)
        buffer=io.BytesIO()
        fig.savefig(buffer,format='png',bbox_inches='tight')
        display(Image(data=buffer.getvalue()))
        plt.close(fig)
plt.show=show_figures
count=0
for cell in nb.cells:
    if cell.cell_type!='code': continue
    count+=1
    with capture_output() as captured:
        result=shell.run_cell(cell.source,store_history=False)
    if not result.success:
        raise RuntimeError(f'Notebook cell {count} failed: {result.error_before_exec or result.error_in_exec}')
    cell.execution_count=count
    outputs=[]
    if captured.stdout: outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr: outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for rich in captured.outputs:
        outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
    cell.outputs=outputs
    print(f'Executed cell {count}: {len(outputs)} outputs',flush=True)
nb.metadata['execution_engine']='IPython in-process, clean Python process; no external kernel connection file'
nbformat.validate(nb)
nbformat.write(nb,path)
print('Completed:',count,'code cells, no cell errors.')
