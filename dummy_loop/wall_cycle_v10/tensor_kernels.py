"""Batched CUDA/CPU tensor versions of frozen squeeze and extrusion kernels.
Not a full environment: geometry, transfer, sensor and rollout still unported.
"""
import torch

def squeeze_support(t,gap,cell,tau_y):
    wet=t>=gap.unsqueeze(-1)-1e-12
    ix=torch.arange(1,t.shape[1]+1,device=t.device,dtype=t.dtype)[None,:,None]
    def runs(w):
        zeros=torch.where(w,torch.zeros_like(t),ix)
        return (ix-torch.cummax(zeros,dim=1).values)*w
    up=runs(wet);dn=runs(wet.flip(1)).flip(1)
    d=(torch.minimum(up,dn)-.5)*cell
    pressure=torch.where(wet,2*tau_y*torch.clamp(d,min=0)/gap.unsqueeze(-1),0.)
    return pressure.sum((1,2))*cell*cell

def solve_gap(t,pitch,force,cell=.005,tau_y=250.,min_gap=1e-5,max_gap=.012):
    rise=(torch.arange(t.shape[1],device=t.device,dtype=t.dtype)[None,:]+.5)*cell*torch.sin(pitch.clamp_min(0))[:,None]
    def support(g):return squeeze_support(t,g[:,None]+rise,cell,tau_y)
    lo=torch.full_like(force,min_gap);hi=torch.full_like(force,max_gap)
    smin=support(lo);smax=support(hi)
    for _ in range(20):
        mid=(lo+hi)*.5;mask=support(mid)>=force
        lo=torch.where(mask,mid,lo);hi=torch.where(mask,hi,mid)
    gap=torch.where(smin<force,torch.zeros_like(lo),torch.where(smax>=force,torch.full_like(lo,max_gap),lo))
    gap=torch.where(force<=0,torch.full_like(lo,max_gap),gap)
    actual=torch.where(force<=0,torch.zeros_like(force),torch.where(smin<force,smin,support(gap.clamp_min(min_gap))))
    return gap,actual

def extrude(t,gap):
    excess=(t-gap.unsqueeze(-1)).clamp_min(0);out=t-excess;remaining=excess.sum(1);room=(gap.unsqueeze(-1)-out).clamp_min(0)
    rows=[]
    for r in range(t.shape[1]):
        take=torch.minimum(room[:,r],remaining);rows.append(out[:,r]+take);remaining=remaining-take
    gt=gap[:,0].clamp_min(1e-9);gl=gap[:,-1].clamp_min(1e-9);fraction=gt**3/(gt**3+gl**3)
    return torch.stack(rows,1),remaining*fraction[:,None],remaining*(1-fraction[:,None])

class CapturedGap:
    """Fixed-shape CUDA graph, mutable inputs; avoids repeated Python kernel launches.
    Output buffers are reused. Callers must consume/copy before the next invocation.
    """
    def __init__(self,t,pitch,force):
        if t.device.type!='cuda':raise ValueError('CUDA inputs required')
        self.inputs=[x.clone() for x in (t,pitch,force)]
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.outputs=solve_gap(*self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.outputs=solve_gap(*self.inputs)
    def __call__(self,t,pitch,force):
        for dst,src in zip(self.inputs,(t,pitch,force)):
            if dst.shape!=src.shape or dst.dtype!=src.dtype or dst.device!=src.device:raise ValueError('Graph input schema mismatch')
            dst.copy_(src)
        self.graph.replay()
        return self.outputs
