import hashlib,json,os
from pathlib import Path
from pams.data import load_dev_target_manifest
from pams.metrics import compute_count_metrics

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); data=(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n').encode(); fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    return hashlib.sha256(data).hexdigest(),len(data)
pred_path=Path('/pams/frozen/predictions.json'); receipt_path=Path('/pams/frozen/prediction.receipt.json'); targets_path=Path('/pams/dev.targets.json'); script_path=Path('/pams-run/score-script.py')
pred_sha=sha(pred_path); receipt_sha=sha(receipt_path); pred=json.loads(pred_path.read_text()); receipt=json.loads(receipt_path.read_text()); assert receipt['prediction_sha256']==pred_sha and receipt['prediction_bytes']==pred_path.stat().st_size and pred['record_total']==84 and len(pred['records'])==84 and receipt['method_key']==pred['method_key']
# First permitted target-bearing operation follows complete frozen-artifact validation.
targets_sha=sha(targets_path); assert targets_sha=='1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6'; targets=load_dev_target_manifest(targets_path); assert [r.video_id for r in targets.records]==[r['video_id'] for r in pred['records']]
report=compute_count_metrics([r['raw_count'] for r in pred['records']],[r.count for r in targets.records],video_ids=[r.video_id for r in targets.records],actions=[r.action for r in targets.records],bootstrap_samples=10000,bootstrap_seed=2026,confidence_level=.95)
assert sha(pred_path)==pred_sha and sha(receipt_path)==receipt_sha and sha(targets_path)==targets_sha
evaluation={'schema_version':1,'artifact_type':'teacher_period_direct_dev_evaluation','classification':pred['classification'],'paper_table_eligible':False,'protocol':'ucfrep_526','split':'dev','method_key':pred['method_key'],'prediction_sha256':pred_sha,'prediction_receipt_sha256':receipt_sha,'dev_targets_sha256':targets_sha,'prediction_script_sha256':pred['prediction_script_sha256'],'scoring_script_sha256':sha(script_path),'report':report.to_dict()}; eval_sha,eval_bytes=write_new('/pams/output/evaluation.json',evaluation); write_new('/pams/output/evaluation.receipt.json',{'schema_version':1,'artifact_type':'teacher_period_direct_evaluation_receipt','method_key':pred['method_key'],'evaluation_file':'evaluation.json','evaluation_sha256':eval_sha,'evaluation_bytes':eval_bytes,'prediction_sha256':pred_sha,'prediction_receipt_sha256':receipt_sha,'dev_targets_sha256':targets_sha,'scoring_script_sha256':evaluation['scoring_script_sha256']})
