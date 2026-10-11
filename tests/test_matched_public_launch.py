from pathlib import Path
import hashlib,json,subprocess,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]

class MatchedLaunch(unittest.TestCase):
 def test_complete_original_source_bindings(self):
  spec=json.loads((ROOT/'configs/execution/matched_controller.json').read_text())
  protocol=ROOT/spec['protocol']
  self.assertEqual(hashlib.sha256(protocol.read_bytes()).hexdigest(),'b5284194979bfe0e40cc06a562b455797b16f31582e90311b76190f443c40968')
  expected=json.loads(protocol.read_text())['files'];actual={}
  for original,item in spec['included_sources'].items():
   actual[original]=hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest()
   self.assertEqual(actual[original],item['sha256'])
  external=spec['external_teacher'];actual[external['source_path']]=external['sha256']
  self.assertEqual(actual,expected)
  self.assertEqual(len(actual),57)

 def test_wrong_teacher_is_rejected_before_execution(self):
  with tempfile.TemporaryDirectory() as temp:
   bad=Path(temp)/'teacher.pt';bad.write_bytes(b'not the declared teacher')
   run=subprocess.run([sys.executable,str(ROOT/'run_experiment.py'),'--experiment','matched-controller','--teacher-checkpoint',str(bad),'--','--cache',str(Path(temp)/'cache'),'--output',str(Path(temp)/'study')],text=True,capture_output=True)
   self.assertNotEqual(run.returncode,0)
   self.assertIn('Archived teacher checkpoint hash differs',run.stderr)
   self.assertFalse((Path(temp)/'study').exists())

if __name__=='__main__':unittest.main()
