"""Export the 13 scientific tables from included result and protocol operands."""
from pathlib import Path
import argparse,csv,hashlib,io,json
from analysis.publication_tables import table_rows
ROOT=Path(__file__).resolve().parent

def export_tables(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 registry=json.loads((ROOT/'results/table_views.json').read_text());definitions=registry['tables']
 assert len(definitions)==13
 derived=table_rows(ROOT,definitions);index=[]
 for table in definitions:
  n=table['number'];rows=derived[n];assert rows==table['rows']
  for source in table['source_files']:
   assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256']
  stem=f"table_{n:02d}_{table['slug']}"
  stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n');writer.writerows([table['headers'],*rows])
  (output/(stem+'.csv')).write_text(stream.getvalue())
  md='# '+table['title']+'\n\n'+table['paper_location']+'. '+table['metric_and_units']+'.\n\n'
  md+='| '+' | '.join(table['headers'])+' |\n| '+' | '.join(['---']*len(table['headers']))+' |\n'
  md+=''.join('| '+' | '.join(row)+' |\n' for row in rows)
  (output/(stem+'.md')).write_text(md)
  index.append({k:table[k] for k in ('number','paper_location','title','metric_and_units','source_files','source_mapping','cell_sources')}|{'files':{ext:stem+'.'+ext for ext in ('csv','md')},'rows':len(rows)})
 (output/'index.json').write_text(json.dumps({'table_count':13,'numbering':'Arabic Tables 1–13: main Tables 1–3 and appendix Tables 4–13.','tables':index},indent=2,sort_keys=True)+'\n')
 return index

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();export_tables(args.output)
 print('Exported 13 scientific tables; no manuscript source or model execution required.')

if __name__=='__main__':main()
