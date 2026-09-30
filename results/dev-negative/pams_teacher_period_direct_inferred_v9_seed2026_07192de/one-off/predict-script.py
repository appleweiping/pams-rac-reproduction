import hashlib, json, math, os
from pathlib import Path
import torch
from pams.config import load_config
from pams.data import load_pose_cache_set, load_pose_input_commitment, load_pose_input_manifest, pose_input_identity_sha256, validate_pose_input_binding
from pams.metrics import round_count
from pams.period import estimate_period_from_projected_pose
from pams.training import CheckpointProvenance, collate_pose_sequences, load_model_checkpoint

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    return hashlib.sha256(data).hexdigest(),len(data)
config_path=Path('/pams/config.yaml'); checkpoint=Path('/pams/encoder/encoder.pt'); progress=Path('/pams/encoder/logs/encoder.jsonl'); completion=Path('/pams/encoder/manifests/20260730T025521Z-e6e6a2f657aa.completed.json'); inputs_path=Path('/pams/dev.inputs.json'); commitment_path=Path('/pams/dev.inputs.commitment.json'); script_path=Path('/pams-run/predict-script.py')
expected={'config':'eb4072a195608757cd2542855b33fb000c987f96c94b83a1e448ab78f97e9374','checkpoint':'6817fe468d76c3fa2182dd831695d6fe3d6da208a2b7f8dfa90cffa6872e8053','progress':'aadc8f9bf067b3489efcd80db43c42cf09dba3477c2976bbc16571ff31684622','completion':'f4815e1961ba883c23ef479ddcdef44aea5cad8a50d54e17b8e6a4cccc613f3d','inputs':'f21c49569805e4aea326977f6d7ce2964f503fad129a17e95214594364ca4ae7','commitment':'a2bb2db86abef64e16a8fe1ac196857ac8e77cc9cd461c568b2193401a8cd169'}
for key,path in [('config',config_path),('checkpoint',checkpoint),('progress',progress),('completion',completion),('inputs',inputs_path),('commitment',commitment_path)]: assert sha(path)==expected[key],key
config=load_config(config_path); manifest=load_pose_input_manifest(inputs_path); commitment=load_pose_input_commitment(commitment_path); validate_pose_input_binding(manifest,commitment,sidecar_sha256=expected['inputs'])
assert manifest.protocol=='ucfrep_526' and manifest.split=='dev' and len(manifest.records)==84
sequences,snapshot=load_pose_cache_set(manifest.records,cache_dir='/pams/pose-cache',pose_fingerprint=config.pose_fingerprint)
checkpoint_payload=torch.load(checkpoint,map_location='cpu',weights_only=False); provenance=CheckpointProvenance.from_mapping(checkpoint_payload['provenance'])
device=torch.device('cuda:0'); model=load_model_checkpoint(checkpoint,config,device=device,expected_stage='encoder',expected_provenance=provenance).eval(); rows=[]
with torch.inference_mode():
    for start in range(0,len(sequences),32):
        batch=collate_pose_sequences(sequences[start:start+32]).to(device); _,projected=model.encoder.forward_with_pre_pe(batch.poses,batch.valid_mask); periods,conf=estimate_period_from_projected_pose(projected,minimum=config.period.minimum,maximum=config.period.maximum,valid_mask=batch.valid_mask); valid=batch.valid_mask.sum(1)
        for i,vid in enumerate(batch.video_ids):
            p=float(periods[i]); c=float(conf[i]); v=int(valid[i]); raw=(v-1)/p if c>0 and v>=2 else 0.0
            assert all(math.isfinite(x) for x in (p,c,raw)) and raw>=0
            record=manifest.records[start+i]; assert record.video_id==vid and record.video_sha256 is not None
            rows.append({'video_id':vid,'video_sha256':record.video_sha256,'raw_count':raw,'rounded_count':round_count(raw),'period_frames':p,'confidence':c,'valid_frames':v})
assert len(rows)==84 and len({r['video_id'] for r in rows})==84
payload={'schema_version':1,'artifact_type':'teacher_period_direct_dev_predictions','classification':'pams-teacher-period-direct-inferred-posthoc-diagnostic','paper_table_eligible':False,'protocol':'ucfrep_526','split':'dev','method_key':'pams-teacher-period-direct-inferred-v1','record_total':84,'source_revision':'07192debb7f5748c53a1c6d4f0c0d3f16228e229','algorithm':'forward_with_pre_pe -> estimate_period_from_projected_pose -> confidence/valid guard -> half_up((valid_frames-1)/period)','config_sha256':expected['config'],'config_fingerprint':config.fingerprint,'pose_fingerprint':config.pose_fingerprint,'encoder_checkpoint_sha256':expected['checkpoint'],'encoder_progress_sha256':expected['progress'],'encoder_completion_receipt_sha256':expected['completion'],'dev_inputs_sha256':expected['inputs'],'dev_commitment_sha256':expected['commitment'],'dev_identity_sha256':pose_input_identity_sha256(manifest.records),'pose_cache_set_sha256':snapshot.fingerprint,'prediction_script_sha256':sha(script_path),'records':rows}
Path('/pams/output/inputs').mkdir(parents=True,exist_ok=True); write_new('/pams/output/inputs/dev-pose-cache-snapshot.json',snapshot.to_dict()); pred_sha,pred_bytes=write_new('/pams/output/predictions.json',payload); receipt={'schema_version':1,'artifact_type':'teacher_period_direct_prediction_receipt','method_key':payload['method_key'],'prediction_file':'predictions.json','prediction_sha256':pred_sha,'prediction_bytes':pred_bytes,'record_total':84,'source_revision':payload['source_revision'],'config_fingerprint':config.fingerprint,'pose_fingerprint':config.pose_fingerprint,'encoder_checkpoint_sha256':expected['checkpoint'],'dev_identity_sha256':payload['dev_identity_sha256'],'pose_cache_set_sha256':snapshot.fingerprint,'prediction_script_sha256':payload['prediction_script_sha256']}; write_new('/pams/output/prediction.receipt.json',receipt)
for key,path in [('config',config_path),('checkpoint',checkpoint),('progress',progress),('completion',completion),('inputs',inputs_path),('commitment',commitment_path)]: assert sha(path)==expected[key],key
