"""Evidence-only campaign summary. Missing groups stay missing; no fabricated wins."""
import argparse,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def records(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf8').splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);a=p.parse_args();root=a.campaign;manifest=read(root/'manifest.json');lines=['# v0.9 r1.2 完整实验执行报告','',f'执行状态：**{manifest["status"]}**。证据等级：L1，全部是仿真。完成训练不等于达到施工质量目标。','', '观测为带噪声的高度图代理和指令历史，不是渲染的D435图像，也不是实测力反馈。','', '## 实验结果','', '| 评估组 | 回合数 | 合格覆盖 | 边缘覆盖 | RMSE/mm | J | 最终成功回合 |','|---|---:|---:|---:|---:|---:|---:|'];tables={}
 for d in sorted((root/'evaluation').glob('test_*')):
  f=d/'summary.json'
  if not f.exists():continue
  s=read(f);q=s['by_task'].get('all',{});tables[d.name]=s
  if q:lines.append(f'| {d.name} | {s["completed"]}/{s["requested"]} | {q["coverage"]:.1%} | {q["edge_coverage"]:.1%} | {q["rmse_mm"]:.3f} | {q["J"]:.3f} | {q["final_success"]} |')
 if not tables:lines+=['','独立测试尚未完成，不能用开发集结果替代。']
 lines+=['','目标：覆盖及边缘覆盖≥95%，RMSE≤0.5 mm，P95≤1 mm。阶段目标为70%/65%/1 mm。','', '## 训练与资源','', '| 运行 | 已记录更新数 | 环境+IPC秒/轮 | 推理秒/轮 | GPU更新秒/轮 | 其他含写盘秒/轮 |','|---|---:|---:|---:|---:|---:|'];timings={}
 jf=root/'all_training_jobs.json';jobs=read(jf) if jf.exists() else read(root/'jobs.json') if (root/'jobs.json').exists() else []
 for job in jobs:
  d=Path(job['out']);f=d/'metrics/updates.jsonl'
  if not f.exists():continue
  rr=records(f);mean=lambda k:float(np.mean([r[k] for r in rr]));row={k:mean(k) for k in ['policy_s','environment_ipc_s','gpu_update_s','checkpoint_s','total_s']};row['other_including_io_s']=row['total_s']-sum(row[k] for k in ['policy_s','environment_ipc_s','gpu_update_s','checkpoint_s']);row['updates']=len(rr);timings[job['id']]=row
  lines.append(f'| {job["id"]} | {len(rr)} | {row["environment_ipc_s"]:.2f} | {row["policy_s"]:.2f} | {row["gpu_update_s"]:.2f} | {row["other_including_io_s"]:.2f} |')
 lines+=['','两组训练可同时运行；吞吐受到共享CPU/GPU竞争影响，这些记录不能直接冒充独占机器的加速比。完整资源日志在每个run的logs/resources.jsonl。','', '## 解释边界','', '- 分开检查裸墙与精修残局；不能用原本接近合格的墙面掩盖裸墙失败。','- 与FINISH不操作基线比较，识别单纯提前结束的收益。','- 三个训练种子与60个场景是两种统计层级。','- A0的DEPOSIT是装料+接触宏动作，实际施工时间/供料共同受限；一个高层决策包含的原子步骤数不同。','- 消融固定100轮，须与RL1_100_reference比较；不能与更长训练直接归因比较。','- 仅出现非直线轨迹不证明发现了更优新技能；没有反事实验证的行为不作此结论。','- 2.5D材料模型、示例伺服参数及深度代理仍有现实差距；不据此宣称实机施工性能。']
 comparisons={};rng=np.random.default_rng(1909)
 for family in ['RL1','RL2']:
  files=[root/'evaluation'/f'test_nominal_{family}_seed{s}'/'episodes.jsonl' for s in [11,22,33]]
  if not all(f.exists() for f in files):continue
  learned=[{r['seed']:r for r in records(f)} for f in files]
  for baseline in ['T','T+','FINISH']:
   f=root/'evaluation'/f'test_nominal_{baseline}'/'episodes.jsonl'
   if not f.exists():continue
   teacher={r['seed']:r for r in records(f)};seeds=sorted(set(teacher).intersection(*[set(x) for x in learned]));group={}
   if len(seeds)!=60:continue
   for metric in ['J','coverage','edge_coverage','rmse_mm','waste_frac']:
    delta=np.array([[r[s]['final'][metric]-teacher[s]['final'][metric] for s in seeds] for r in learned]);boot=[]
    for _ in range(2000):
     train_ix=rng.integers(0,3,3);scene_ix=rng.integers(0,len(seeds),len(seeds));boot.append(delta[train_ix][:,scene_ix].mean())
    group[metric]={'mean_difference':float(delta.mean()),'hierarchical_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'per_training_seed':delta.mean(1).tolist()}
   comparisons[f'{family}-minus-{baseline}']=group
 lines+=['','完整配对差值与分层bootstrap区间见 summary.json；仅3个训练种子，区间仍有较大不确定性。']
 (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');(root/'summary.json').write_text(json.dumps({'status':manifest['status'],'evaluations':tables,'timings':timings,'paired_comparisons':comparisons},indent=2),encoding='utf8')
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  from .campaign import history
  fig,ax=plt.subplots(figsize=(10,5))
  latest={}
  for job in jobs:latest[job['group'],job['seed']]=job
  for key,job in latest.items():
   h=history(job['out'])
   if h:uu=sorted(h);ax.plot(uu,[h[u]['mean_J'] for u in uu],label=f'{key[0]} seed {key[1]}',linewidth=1)
  ax.set(xlabel='Cumulative PPO update',ylabel='Validation J (lower is better)',title='Development / validation only, not final test');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/'validation_curves.png',dpi=150);plt.close(fig)
 except ImportError:(root/'plot_unavailable.txt').write_text('Matplotlib not installed; all numerical data retained.')
 print(root/'REPORT.md')
if __name__=='__main__':main()
