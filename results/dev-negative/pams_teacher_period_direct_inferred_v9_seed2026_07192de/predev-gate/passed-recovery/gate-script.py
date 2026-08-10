import hashlib, json, math
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from pams.config import load_config
from pams.data import load_pose_cache_set, load_pose_input_manifest
from pams.metrics import round_count
from pams.period import estimate_period_from_projected_pose
from pams.synthetic import generate_count_sweep, synthetic_stress_suite
from pams.training import CheckpointProvenance, collate_pose_sequences, load_model_checkpoint

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
config_path=Path('/pams/config.yaml'); checkpoint_path=Path('/pams/encoder/encoder.pt')
assert sha(config_path)=='eb4072a195608757cd2542855b33fb000c987f96c94b83a1e448ab78f97e9374'
assert sha(checkpoint_path)=='6817fe468d76c3fa2182dd831695d6fe3d6da208a2b7f8dfa90cffa6872e8053'
config=load_config(config_path)
device=torch.device('cuda:0')
checkpoint_payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False); expected_provenance=CheckpointProvenance.from_mapping(checkpoint_payload['provenance']); model=load_model_checkpoint(checkpoint_path,config,device=device,expected_stage='encoder',expected_provenance=expected_provenance).eval()
manifest=load_pose_input_manifest('/pams/train.inputs.json')
assert manifest.split=='train' and len(manifest.records)==337
sequences,snapshot=load_pose_cache_set(manifest.records,cache_dir='/pams/pose-cache',pose_fingerprint=config.pose_fingerprint)

def infer(items,batch_size=32):
    output=[]
    with torch.inference_mode():
        for start in range(0,len(items),batch_size):
            batch=collate_pose_sequences(items[start:start+batch_size]).to(device)
            _,projected=model.encoder.forward_with_pre_pe(batch.poses,batch.valid_mask)
            periods,conf=estimate_period_from_projected_pose(projected,minimum=config.period.minimum,maximum=config.period.maximum,valid_mask=batch.valid_mask)
            valid=batch.valid_mask.sum(1)
            for i,vid in enumerate(batch.video_ids):
                p=float(periods[i]); c=float(conf[i]); v=int(valid[i])
                raw=(v-1)/p if c>0 and v>=2 else 0.0
                output.append({'video_id':vid,'valid_frames':v,'period':p,'confidence':c,'raw_count':raw,'rounded_count':round_count(raw)})
    return output
train=infer(sequences)
assert len(train)==337 and all(math.isfinite(r[k]) for r in train for k in ('period','confidence','raw_count'))
assert all(r['rounded_count']==0 for r in train if r['confidence']<=0 or r['valid_frames']<2)
positive=[r for r in train if r['confidence']>0 and r['valid_frames']>=2]
rounded_periods=[round_count(r['period']) for r in positive]
mode_count=max(Counter(rounded_periods).values()) if rounded_periods else 0
train_summary={
 'records':len(train),'positive_confidence_records':len(positive),'zero_confidence_records':len(train)-len(positive),
 'period_4_fraction':sum(r['period']<=4.000001 for r in positive)/len(positive),
 'period_128_fraction':sum(r['period']>=127.999999 for r in positive)/len(positive),
 'rounded_period_mode_fraction':mode_count/len(positive),
 'period_min':min(r['period'] for r in positive),'period_median':float(np.median([r['period'] for r in positive])),'period_max':max(r['period'] for r in positive),
 'confidence_mean':float(np.mean([r['confidence'] for r in train])),'raw_count_median':float(np.median([r['raw_count'] for r in train])),'raw_count_max':max(r['raw_count'] for r in train),
 'pose_cache_set_sha256':snapshot.fingerprint,
}
sweep_samples=generate_count_sweep(minimum=2,maximum=40,frames=256,seed=2026)
sweep_rows=infer(tuple(s.sequence for s in sweep_samples))
for row,sample in zip(sweep_rows,sweep_samples): row['target_count']=int(sample.target_count)
stress_samples=synthetic_stress_suite(count=8,frames=256,seed=2026)
stress_names=list(stress_samples)
stress_rows=infer(tuple(stress_samples[n].sequence for n in stress_names))
for row,name in zip(stress_rows,stress_names): row['case']=name; row['target_count']=8
sweep_exact=sum(r['rounded_count']==r['target_count'] for r in sweep_rows)/len(sweep_rows)
sweep_obo=sum(abs(r['rounded_count']-r['target_count'])<=1 for r in sweep_rows)/len(sweep_rows)
stress_exact=sum(r['rounded_count']==8 for r in stress_rows)/len(stress_rows)
stress_obo=sum(abs(r['rounded_count']-8)<=1 for r in stress_rows)/len(stress_rows)
gates={
 'synthetic_count_2_40_exact_fraction_gte_1':sweep_exact>=1.0,
 'synthetic_stress_obo_fraction_gte_1':stress_obo>=1.0,
 'train_period_4_fraction_lte_0_10':train_summary['period_4_fraction']<=0.10,
 'train_period_128_fraction_lte_0_10':train_summary['period_128_fraction']<=0.10,
 'train_rounded_period_mode_fraction_lte_0_25':train_summary['rounded_period_mode_fraction']<=0.25,
 'zero_confidence_nonzero_count_absent':all(r['rounded_count']==0 for r in train if r['confidence']<=0 or r['valid_frames']<2),
}
payload={
 'schema_version':1,'status':'passed' if all(gates.values()) else 'failed','classification':'pams-teacher-period-direct-inferred-label-free-predev-gate','eligible_for_paper_table':False,
 'source_revision':'07192debb7f5748c53a1c6d4f0c0d3f16228e229','encoder_checkpoint_sha256':sha(checkpoint_path),'config_sha256':sha(config_path),'pose_fingerprint':config.pose_fingerprint,
 'algorithm':'forward_with_pre_pe -> estimate_period_from_projected_pose -> confidence/valid guard -> half_up((valid_frames-1)/period)',
 'train337':train_summary,
 'synthetic_count_2_40':{'records':len(sweep_rows),'exact_fraction':sweep_exact,'obo_fraction':sweep_obo,'rows':sweep_rows},
 'synthetic_stress_count8':{'records':len(stress_rows),'exact_fraction':stress_exact,'obo_fraction':stress_obo,'rows':stress_rows},
 'gates':gates,'dev_inputs_mounted':False,'targets_mounted':False,'test_inputs_mounted':False,
}
print(json.dumps(payload,indent=2,sort_keys=True))
