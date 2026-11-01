"""Real official processors + tiny random Qwen CPU generation; no pretrained weights."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import transformers
from transformers import AutoProcessor,Qwen2_5_VLConfig,Qwen2_5_VLForConditionalGeneration
from rosa_repro.dummy import observation
from rosa_repro.vlm_contract import default_config,verify_snapshot
from rosa_repro.vlm_backends import HfGenerator,rendered_request,parsed_result,check_versions
from rosa_repro.execution_plan import runtime_source_manifest
from rosa_repro.regression import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot-3b',type=Path,required=True)
    p.add_argument('--snapshot-7b',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(42)
    versions=check_versions('transformers');cases=[];processor=None
    for role in ('s2','safety','monitor'):
        cfg=default_config(role);path=a.snapshot_3b if role=='safety' else a.snapshot_7b
        verification=verify_snapshot(path,cfg['model_id'],weights=False)
        proc=AutoProcessor.from_pretrained(str(path),local_files_only=True,use_fast=cfg['processor_use_fast'],
            min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels']);proc.tokenizer.padding_side='left'
        obs=observation(1);obs['video.image']=np.zeros((1,2,224,224,3),np.uint8)
        obs['video.image'][0,-1]=50;obs['video.wrist_image']=np.full((1,1,224,224,3),150,np.uint8)
        payload=dict(observation=obs,context={'execution_history':'Two virtual action chunks completed.'})
        prompt,images,trace=rendered_request(payload,cfg,proc)
        inputs=proc(text=[prompt],images=images,padding=True,return_tensors='pt')
        assert inputs['image_grid_thw'].tolist()==[[1,16,16],[1,16,16]]
        assert inputs['pixel_values'].shape==(512,1176)
        assert (inputs['input_ids']==151655).sum().item()==128
        assert trace['images'][0]['selected_frame']==1 and len({x['sha256'] for x in trace['images']})==2
        cases.append(dict(role=role,config=cfg,checkpoint=verification,input_shapes={k:list(v.shape) for k,v in inputs.items()},
                          trace=trace,passed=True))
        if role=='s2':processor=proc;tiny_payload=payload;tiny_config=cfg
    config=Qwen2_5_VLConfig(text_config=dict(vocab_size=151936,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
      num_attention_heads=4,num_key_value_heads=2,rope_scaling={'type':'mrope','mrope_section':[1,1,2]},
      max_position_embeddings=4096,bos_token_id=151643,eos_token_id=151645,tie_word_embeddings=True),
      vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,in_channels=3,patch_size=14,
      spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=32,fullatt_block_indexes=[0],window_size=28),tie_word_embeddings=True)
    model=Qwen2_5_VLForConditionalGeneration(config).eval();tiny_config['max_new_tokens']=2
    prompt,images,trace=rendered_request(tiny_payload,tiny_config,processor)
    began=time.monotonic();raw=HfGenerator(model,processor,tiny_config,'cpu').generate(prompt,images)
    result=parsed_result(tiny_config,raw,trace)
    assert raw['generated_tokens']<=2 and result.output is None and result.error_type=='InvalidModelGeneration'
    report=dict(status='passed',versions=versions,gpu_execution=False,pretrained_model_weights_loaded=False,
        source_manifest=runtime_source_manifest(),processor_cases=cases,
        tiny_random_model=dict(parameters=sum(p.numel() for p in model.parameters()),generated=raw,
            elapsed_s=time.monotonic()-began,parse_status=result.evidence['parse_status'],
            semantic_accuracy_claimed=False,passed=True),
        actual_vllm_gpu_engine_executed=False)
    write_json(a.output/'verification.json',report)
    print(json.dumps(dict(status='passed',processor_roles=3,tiny_parameters=report['tiny_random_model']['parameters'],gpu_execution=False)))


if __name__=='__main__':main()
