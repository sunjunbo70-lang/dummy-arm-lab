"""Fixed S quality objective. This module is used ONLY for reward/evaluation."""
import numpy as np

def quality(material,edge_term=True):
 c=material.cfg;h=material.wall[material._score_rows,material._score_cols];mask=material._edge_mask;err=h-c.target_m
 rmse=float(np.sqrt(np.mean(err**2))*1000);edge_rmse=float(np.sqrt(np.mean(err[mask]**2))*1000)
 coverage=float(np.mean((h>=c.acceptable_low_m)&(h<=c.acceptable_high_m)))
 edge_excess=float(np.maximum(h[mask]-c.acceptable_high_m,0).sum()*c.cell_m**2)
 rough=float(np.sqrt((np.mean(np.diff(h,axis=0)**2)+np.mean(np.diff(h,axis=1)**2))/2)*1000)
 J=.3*rmse/2+.2*(1-coverage)+.15*rough
 if edge_term:J+=.2*edge_rmse/2+.15*edge_excess/(mask.sum()*c.cell_m**2*.002)
 return {'J':J,'rmse_mm':rmse,'coverage':coverage,'edge_rmse_mm':edge_rmse,'edge_excess_m3':edge_excess,'roughness_mm':rough,'edge_coverage':float(np.mean((h[mask]>=c.acceptable_low_m)&(h[mask]<=c.acceptable_high_m))),'p95_error_mm':float(np.percentile(abs(err),95)*1000)}

def reward(before,after,losses,seconds,invalid=False,terminal=False,success=False):
 carry,other,outside=losses
 return float(30*(before-after)-(4*carry+4*other+outside)/18e-6-.001*seconds-2*invalid+(100*success-10*after if terminal else 0))
