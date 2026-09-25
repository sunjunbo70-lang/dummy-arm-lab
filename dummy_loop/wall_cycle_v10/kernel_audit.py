"""Numerical and microbenchmark audit; never labels kernel speed as campaign speed."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
import torch
from .tensor_kernels import solve_gap,extrude
from ..wall_cycle.mortar import solve_gap as reference_gap,extrude as reference_extrude,MortarParams

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    rng=np.random.default_rng(271000);n=256;t=rng.uniform(0,.009,(n,6,24));t[rng.random(t.shape)<.3]=0;pitch=rng.uniform(0,.6,n);force=rng.uniform(0,15,n)
    force[:4]=[0,-1,1e6,.5];pitch[:4]=0;t[4:8]=0
    params=MortarParams();expected=np.array([reference_gap(t[i],pitch[i],force[i],.005,params) for i in range(n)])
    report={'scope':'squeeze/extrusion kernel microbenchmark only, not full GPU environment or end-to-end training','torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'equivalence':[],'benchmarks':[]}
    for dtype in [torch.float64,torch.float32]:
        tt=torch.tensor(t,device='cuda',dtype=dtype);pp=torch.tensor(pitch,device='cuda',dtype=dtype);ff=torch.tensor(force,device='cuda',dtype=dtype)
        g,s=solve_gap(tt,pp,ff);ga=g.cpu().numpy();ss=s.cpu().numpy()
        rise=(np.arange(6)+.5)[None,:]*.005*np.sin(pitch)[:,None];gaps=expected[:,0,None]+rise;gaps=np.maximum(gaps,1e-8)
        et,trail,lead=extrude(tt,torch.tensor(gaps,device='cuda',dtype=dtype));ref=[reference_extrude(t[i],gaps[i]) for i in range(n)]
        err=max(float(np.max(abs(et.cpu().numpy()-np.array([x[0] for x in ref])))),float(np.max(abs(trail.cpu().numpy()-np.array([x[1] for x in ref])))),float(np.max(abs(lead.cpu().numpy()-np.array([x[2] for x in ref])))))
        conservation=(tt.sum((1,2))-et.sum((1,2))-trail.sum(1)-lead.sum(1)).abs().max().item()*.005**2*1e6
        gap_mm=float(np.max(abs(ga-expected[:,0]))*1000);support_error=float(np.max(abs(ss-expected[:,1])))
        passed=gap_mm<=.01 and err*1000<=.01 and conservation<=.01 and bool(torch.isfinite(et).all())
        report['equivalence'].append({'dtype':str(dtype),'max_gap_error_mm':gap_mm,'max_support_error_N':support_error,'max_extrusion_error_mm':err*1000,'max_extrusion_conservation_error_ml':conservation,'pass':passed,'cases':n})
    for batch in [1,16,64,256]:
        tt=torch.tensor(t[:batch],device='cuda',dtype=torch.float32);pp=torch.tensor(pitch[:batch],device='cuda',dtype=torch.float32);ff=torch.tensor(force[:batch],device='cuda',dtype=torch.float32)
        for _ in range(3):solve_gap(tt,pp,ff)
        torch.cuda.synchronize();cpu=[];gpu=[]
        for rep in range(3):
            start=time.perf_counter()
            for i in range(batch):reference_gap(t[i],pitch[i],force[i],.005,params)
            cpu.append(time.perf_counter()-start)
            torch.cuda.synchronize();start=time.perf_counter();solve_gap(tt,pp,ff);torch.cuda.synchronize();gpu.append(time.perf_counter()-start)
        report['benchmarks'].append({'batch':batch,'numpy_one_cpu_median_ms':float(np.median(cpu)*1000),'cuda_median_ms':float(np.median(gpu)*1000),'kernel_speedup_vs_one_cpu':float(np.median(cpu)/np.median(gpu)),'repetitions':3})
    report['pass']=all(x['pass'] for x in report['equivalence']);(a.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
