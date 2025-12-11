# Adaptive Drone Controller using Lyapunov Theory

This repository compares PD and Lyapunov-based adaptive controllers for quadrotor position tracking under parametric uncertainty. A simplified 6-DOF translational model is used with direct force commands; attitude is abstracted out.

## Key Results (20 s slalom batch)
- AC baseline (1.9 kg, no wind): 0.046 m total error
- AC worst-case (4.0 kg, 10 N wind along [-2,4,0]): 0.144 m
- PD baseline: 0.120 m
- PD worst-case: 0.387 m
- Disturbance adaptation remains imperfect under strong winds (limited excitation / mass-disturbance coupling), but AC outperforms PD across the batch.

## Controllers and Gains
- PD (batch-tuned, 10 s optimization): `Kp_xy=60.0`, `Kp_z=60.0`, `Kd_xy=3.82`, `Kd_z=8.84`, `Ki_xy=0.50`, `Ki_z=0.50`
- AC (batch-tuned, 10 s optimization): `lambda_xy=11.99`, `lambda_z=14.36`, `k_xy=24.93`, `k_z=21.74`, `gamma_alpha=0.34`, `gamma_d=0.99`

## Trajectory and Scenarios
- Trajectory: 20-second 3D slalom with variable forward speed (`vy = 1.5 + 0.8 sin(0.5 t)`), x-weave (A=5 m, ω=1.2), z-weave (A=1 m, ω=0.8).
- Batch: 16 runs (baseline; mass sweep 1.9→4.0 kg; wind sweep along [-2,4,0] scaled 0→10 N; combined mass+wind pairs).

## Running Simulations
```bash
cd simulator
python3 slalom.py         # single slalom run (AC vs PD)
python3 slalom_batch.py   # 16-run batch comparison
```

## Structure
```
controller/   # PD and adaptive controllers
environment/  # simplified translational quadrotor model
simulator/    # slalom and batch scripts, optimizers
more_advanced_dynamics/   # cascaded/attitude models (kept separate)
Tex.tex       # full paper
```

## Notes
- Optimization in `simulator/optimize_batch.py` uses a 10 s horizon for speed; evaluation is on 20 s slalom.
- Disturbance adaptation is limited by excitation; consider richer maneuvers or decoupling mass vs disturbance estimation for future work.
