"""Launch exact research code with a portable source-integrity declaration.

Training arithmetic and scientific protocol fields remain unchanged. Historical
guard-only prose/log files are replaced by the included research/metadata hashes.
External dataset/cache/checkpoint paths remain explicit original CLI arguments.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,tempfile,shutil
ROOT=Path(__file__).resolve().parent
ENTRIES={
 'detector':'train_dfine_cure_development_cuda_v2.py',
 'action':'run_csu_action_development_cuda_v1.py',
 'queryset':'run_csu_action_queryset_development_v2.py',
 'cure-feature-head':'run_csu_sensor_head_control_cuda_v1.py',
 'task-feature-head':'run_csu_sensor_head_task_features_cuda_v2.py',
 'native-fitting':'run_rtdetr_native_overfit_v1.py'}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--experiment',choices=ENTRIES,required=True)
 p.add_argument('--python',default=sys.executable,help='Python in the matching CUDA environment')
 p.add_argument('--check-only',action='store_true')
 p.add_argument('arguments',nargs=argparse.REMAINDER)
 a=p.parse_args();manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text())
 for name,v in manifest.items():
  f=ROOT/name
  if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=v['sha256']:
   raise ValueError('Included source or operand changed: '+name)
 index=json.loads((ROOT/'configs/execution/index.json').read_text());spec=index['experiments'][a.experiment]
 public=json.loads((ROOT/spec['public_protocol']).read_text());original=json.loads((ROOT/spec['original_protocol']).read_text())
 guards=('files','public_reproduction','admission_shared_source_paths')
 assert {k:v for k,v in public.items() if k not in guards}=={k:v for k,v in original.items() if k not in guards}
 expected={**index['research_sources'],**{k:v['sha256'] for k,v in index['metadata'].items()}}
 assert public['files']==expected
 if 'admission_shared_source_paths' in original:
  assert public['admission_shared_source_paths']==[x for x in original['admission_shared_source_paths'] if x in expected]
 if a.check_only:
  print(json.dumps({'status':'public_source_and_scientific_protocol_identity_passed','training_executed':False,'historical_notes_required':False,'remaining':'Supply external data/cache/checkpoints and actual admission operands through the original CLI.'}));return 0
 args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
 if not args:raise ValueError('Supply original entry point arguments after --; use -- --help for its documented inputs')
 if '--protocol' in args or '--protocol-sha256' in args:raise ValueError('Public protocol is supplied automatically; do not substitute it')
 with tempfile.TemporaryDirectory(prefix='csu_execution_') as temporary:
  base=Path(temporary);shutil.copytree(ROOT/'code',base/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  for original_path,v in index['metadata'].items():
   dest=base/original_path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/v['file'],dest)
  for name,digest in public['files'].items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest
  protocol=base/'PUBLIC_REPRODUCTION_PROTOCOL.json';shutil.copyfile(ROOT/spec['public_protocol'],protocol)
  if a.experiment=='native-fitting' and '--admission' not in args:
   original_report=index['bundled_native_admission']['REPORT.json']
   args=['--admission',str((base/original_report).parent),*args]
  return subprocess.call([a.python,str(base/'code/scripts'/ENTRIES[a.experiment]),'--protocol',str(protocol),'--protocol-sha256',spec['public_protocol_sha256'],*args],cwd=base)

if __name__=='__main__':raise SystemExit(main())
