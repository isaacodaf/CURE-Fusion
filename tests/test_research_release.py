from pathlib import Path
import hashlib,json,sys,unittest,subprocess,tempfile,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis import recording_math_v1 as m
from analysis.utility_metrics import descriptive_metrics_v1
from analysis.constant_recording_math import pointwise_interval

class ResearchRelease(unittest.TestCase):
 def test_manifest_and_no_dataset_or_weights(self):
  inventory=json.loads((ROOT/'MANIFEST_SHA256.json').read_text())
  for name,row in inventory.items():
   p=ROOT/name;self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['sha256']);self.assertEqual(p.stat().st_size,row['bytes'])
   self.assertNotIn('__pycache__',p.parts);self.assertNotIn(p.suffix.lower(),['.pt','.pth','.ckpt','.docx','.tex','.zip'])
   if p.suffix.lower()=='.pdf':self.assertEqual(p.parent,ROOT/'figures')
 def test_finished_figures_and_tables(self):
  import csv
  from analysis.publication_tables import table_rows
  figures=json.loads((ROOT/'figures/index.json').read_text());self.assertEqual([x['number'] for x in figures],list(range(1,9)))
  self.assertEqual(len([p for p in (ROOT/'figures').iterdir() if p.suffix in ('.png','.pdf','.svg')]),24)
  for row in figures:
   for item in row['files'].values():self.assertEqual(hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest(),item['sha256'])
  definitions=json.loads((ROOT/'results/table_views.json').read_text())['tables'];values=table_rows(ROOT,definitions)
  index=json.loads((ROOT/'tables/index.json').read_text())['tables'];self.assertEqual(len(index),13)
  for t in index:
   with (ROOT/'tables'/t['files']['csv']).open(newline='') as f:rows=list(csv.reader(f))
   self.assertEqual(rows,[definitions[t['number']-1]['headers'],*values[t['number']]])
   for cell in t['cell_sources']:
    source=json.loads((ROOT/cell['source']).read_text())
    for pointer in cell['json_pointers']:
     current=source
     for key in pointer.strip('/').split('/'):current=current[int(key)] if isinstance(current,list) else current[key]
 def test_readme_local_asset_links(self):
  import re
  text=(ROOT/'README.md').read_text()
  links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)+re.findall(r'(?:src|href)="([^"]+)"',text)
  for link in links:
   if not link.startswith(('http:','https:','#')):self.assertTrue((ROOT/link.split('#')[0]).is_file(),link)
 def test_source_import_closure_is_local(self):
  import ast
  for p in [*(ROOT/'code').rglob('*.py'),*(ROOT/'analysis').rglob('*.py')]:
   for n in ast.walk(ast.parse(p.read_text())):
    if isinstance(n,ast.ImportFrom):
     module=n.module or ''
     if module.startswith('cure_fusion.'):
      self.assertTrue((ROOT/'code'/Path(*module.split('.')).with_suffix('.py')).is_file(),module)
     elif n.level and p.parent.name=='cure_fusion':
      self.assertTrue((p.parent/Path(*module.split('.')).with_suffix('.py')).is_file(),module)
 def test_recording_draws_not_independent_frames(self):
  indices,weights=m.draws();self.assertEqual(indices.shape,(2000,8));self.assertTrue((weights.sum(1)==8).all())
  errors=np.ones((9,3,2));errors[0]=3;groups=np.array([0,0,1,2,3,4,5,6,7])
  counts,sums=m.statistics(errors,groups);n,pooled=m.pool(counts,sums,np.ones((1,8),np.int64))
  self.assertEqual(n.tolist(),[9]);self.assertAlmostEqual(pooled[0,0,0,0],np.sqrt(17/9))
 def test_undefined_draw_is_not_dropped(self):
  x=np.ones(2000);x[0]=np.nan;r=pointwise_interval(x)
  self.assertIsNone(r['percentile_95']);self.assertEqual(r['undefined_draws'],1)
 def test_mean_target_variance_and_additive_constant(self):
  raw=np.array([[1,1,0,0]],float);target=np.zeros((1,2));variance=np.ones((1,2))
  r=descriptive_metrics_v1(raw,target,variance)
  self.assertEqual(r['RMSE'],[1.,1.]);np.testing.assert_allclose(r['Gaussian_mean_NLL_without_additive_constant'],[.5*(.5+np.log(2))]*2)
 def test_complete_null_and_comparison_rows_retained(self):
  e=json.loads((ROOT/'results/detection/endpoints.json').read_text());self.assertEqual(len(e['endpoints']),16)
  by={(r['arm'],r['condition']):r for r in e['endpoints']};self.assertLess(by['cure','clean']['AP_percent'],by['task_only','clean']['AP_percent'])
  for name in ['action','queryset']:
   self.assertEqual(len(json.loads((ROOT/f'results/{name}/results.json').read_text())),116)
 def test_public_protocol_preserves_science_and_excludes_notes(self):
  index=json.loads((ROOT/'configs/execution/index.json').read_text())
  for name,r in index['experiments'].items():
   p=json.loads((ROOT/r['public_protocol']).read_text());old=json.loads((ROOT/r['original_protocol']).read_text())
   guards=('files','public_reproduction','admission_shared_source_paths')
   self.assertEqual({k:v for k,v in p.items() if k not in guards},{k:v for k,v in old.items() if k not in guards})
   if 'admission_shared_source_paths' in old:
    self.assertEqual(p['admission_shared_source_paths'],[x for x in old['admission_shared_source_paths'] if x in p['files']])
   self.assertTrue(all(Path(k).suffix in ('.py','.json') for k in p['files']))
   r=subprocess.run([sys.executable,str(ROOT/'run_experiment.py'),'--experiment',name,'--check-only'],text=True,capture_output=True)
   self.assertEqual(r.returncode,0,r.stdout+r.stderr)
 def test_manifest_tamper_is_rejected_before_execution(self):
  with tempfile.TemporaryDirectory() as d:
   dest=Path(d)/'release';shutil.copytree(ROOT,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
   target=dest/'code/cure_fusion/dfine_cure_v2.py';target.write_text(target.read_text()+'\n# tamper fixture\n')
   r=subprocess.run([sys.executable,str(dest/'run_experiment.py'),'--experiment','detector','--check-only'],text=True,capture_output=True)
   self.assertNotEqual(r.returncode,0);self.assertIn('Included source or operand changed',r.stderr)

if __name__=='__main__':unittest.main()
