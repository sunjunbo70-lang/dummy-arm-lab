# Bounded second DAgger recovery (L1)

Round1 gate: 229/465=49.2473%, failed, not used for PPO. Distribution: high_low60/62, finish22/183, bare50/51, rough30/30, edge_ridge39/110, corner28/29. Failures concentrate in finish and edge states; aggregate alone obscures this. Round1 supplied 2413 DAgger examples, small relative to teacher400.

Use the approved maximum second 40-episode DAgger round (seeds60820–60859), visited by round1 BC, same straight environment and teacher. Combine exactly80 episodes, sample DAgger5x relative to teacher400, train6000BC from original BC1. Gate remains executable>=80% and improvement; no PPO on failure. Source core and physical definitions unchanged.

This is the last planned DAgger round, not an unlimited gate search. If it fails, report straight ablation as infeasible under the bounded initialization budget and continue evaluation of completed groups with the missing comparison explicitly identified, rather than lowering the threshold or repeatedly fitting the six gate scenes. Further methodology changes require a separately documented experiment.

Adaptation and reweighting confound single-factor curvature claims. Existing campaign001/002 and checkpoints remain untouched. New outputs: campaign_003_straight_round2.
