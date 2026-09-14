"""Run the unchanged official four-class COCO evaluator on supplied saved JSON."""
import argparse,hashlib,importlib.metadata,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'analysis'))
from analysis_common_v1 import official_coco

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['ground-truth','predictions','output']:p.add_argument('--'+name,type=Path,required=True)
 for name in ['ground-truth-sha256','predictions-sha256']:p.add_argument('--'+name,required=True)
 a=p.parse_args();assert importlib.metadata.version('pycocotools')=='2.0.11'
 for path,digest in [(a.ground_truth,a.ground_truth_sha256),(a.predictions,a.predictions_sha256)]:assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
 assert not a.output.exists()
 gt=json.loads(a.ground_truth.read_text());pred=json.loads(a.predictions.read_text())
 assert [x['id'] for x in gt['categories']]==[1,2,3,4]
 metric,_=official_coco(gt,pred)
 metric['fixed_four_class_AP']=metric['AP'] if all(x is not None for x in metric['per_class']) else None
 metric['scope']='Official COCO box evaluation of supplied saved detections; fixed-four AP is unavailable if any class lacks support.'
 a.output.write_text(json.dumps(metric,indent=2,allow_nan=False)+'\n');print(json.dumps(metric))

if __name__=='__main__':main()
