"""Validate pinned model distributions against the qualified llama.cpp reference."""
import argparse,array,gzip,hashlib,json,math,os,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--fixtures',type=Path,default=os.environ.get('LLAMA_QUALITY_FIXTURES'));p.add_argument('--gpu',action='store_true');p.add_argument('--reference',type=Path,default=Path(__file__).with_name('quality'));args=p.parse_args()
if args.fixtures is None:
    print('SKIP: set LLAMA_QUALITY_FIXTURES to the SHA-pinned server test fixtures directory');raise SystemExit(77)
manifest=json.loads((args.reference/'reference.json').read_text());raw=gzip.decompress((args.reference/'logits.f32.gz').read_bytes());assert hashlib.sha256(raw).hexdigest()==manifest['raw_logits_sha256']
for name,expected in manifest['provenance']['models'].items():
    content=(args.fixtures/name).read_bytes();assert len(content)==expected['bytes'] and hashlib.sha256(content).hexdigest()==expected['sha256'],f'Wrong pinned model bytes: {name}'
def floats(data):
    result=array.array('f');result.frombytes(data)
    if sys.byteorder!='little':result.byteswap()
    return result

def distribution(row):
    maximum=max(row);values=[math.exp(x-maximum) for x in row];total=sum(values);return [x/total for x in values]

def compare(case,actual,expected):
    n=case['n_vocab'];assert len(actual)==len(expected)==n*case['rows'];error=prob_error=0.0;clear_top_checks=0
    for offset in range(0,len(actual),n):
        a,b=actual[offset:offset+n],expected[offset:offset+n];assert all(math.isfinite(x) for x in a),'Non-finite logit'
        error=max(error,max(abs(x-y) for x,y in zip(a,b)))
        pa,pb=distribution(a),distribution(b);prob_error=max(prob_error,max(abs(x-y) for x,y in zip(pa,pb)))
        first,second=sorted(range(n),key=b.__getitem__,reverse=True)[:2]
        if b[first]-b[second]>2*case['max_abs_logit_error']:
            assert max(range(n),key=a.__getitem__)==first,f'{case["name"]}: clear top-1 changed at row {offset//n}'
            clear_top_checks+=1
    assert error<=case['max_abs_logit_error'],f'{case["name"]}: max logit error {error} > {case["max_abs_logit_error"]}'
    assert prob_error<=case['max_abs_probability_error'],f'{case["name"]}: probability error {prob_error} > {case["max_abs_probability_error"]}'
    print(f'{case["name"]}: {case["rows"]} full-vocabulary rows, max logit error={error:.6g}, probability error={prob_error:.6g}, clear top-1 checks={clear_top_checks}',flush=True)

env=os.environ.copy()
if not args.gpu:env['CUDA_VISIBLE_DEVICES']=''
passed=0
with tempfile.TemporaryDirectory(prefix='ik-model-quality-') as temporary:
    root=Path(temporary)
    for case in manifest['cases']:
        count=case['rows']*case['n_vocab']*4;expected=floats(raw[case['offset']:case['offset']+count]);config=dict(case['config']);config['model']=str(args.fixtures/config['model']);config['n_gpu_layers']=99 if args.gpu else 0
        if 'adapter' in config:config['adapter']=str(args.fixtures/config['adapter'])
        variants=[('cold',{})]
        if 'adapter' in config:variants.append(('hot',{'hot_from_scale':1-config['adapter_scale']}))
        for label,changes in variants:
            cfg=root/f'{case["name"]}-{label}.json';cfg.write_text(json.dumps(dict(config,**changes)));out=cfg.with_suffix('');log=cfg.with_suffix('.log')
            with log.open('w') as stream:run=subprocess.run([str(args.probe),str(cfg),str(out)],env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=90)
            assert run.returncode==0,log.read_text(errors='replace')[-3000:]
            meta=json.loads((out/'result.json').read_text());assert meta['prompt_tokens']==config['prompt_tokens'];assert meta['n_logit_rows']==case['rows'];actual=floats((out/'logits.f32').read_bytes());compare(case,actual,expected);passed+=1
print(f'{passed} pinned quality checks passed ({"GPU" if args.gpu else "CPU"})')
