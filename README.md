# Adaptive Drone Controller using Lyapunov Theory

**Lyapunov-based adaptive control for quadrotor position tracking under parametric uncertainty**

Research project for robotics course
*Harvard University*

---

## Quick Start

```bash
# Run the full simulation comparison
cd simulator
python3 simple.py
```

This will execute three test scenarios comparing PD and Adaptive control:
1. Baseline spiral trajectory tracking
2. Disturbance robustness (+3N constant wind)
3. Mass variation (+1kg package with disturbance)

Results are saved to `simple_result.png`

---

## Overview

This project implements and compares two position control strategies for quadrotor trajectory tracking:

1. **PD Controller**: Classical proportional-derivative control with optimized gains
2. **Adaptive Controller**: Lyapunov-based adaptive control with online parameter estimation

### Key Results

Under combined 53% mass increase and 3N wind disturbance:
- **Adaptive Controller**: 0.055m mean error (0% degradation from baseline)
- **PD Controller**: 0.638m mean error (+145% degradation from baseline)

The adaptive controller demonstrates complete disturbance invariance and successful mass estimation with provable stability guarantees.

---

## Project Structure

```
.
├── README.md                           # This file
├── Tex.tex                            # Research paper (LaTeX)
├── simulator/
│   ├── simple.py                      # Main simulation (simplified dynamics)
│   ├── optimize_gains.py              # Gain optimization using Nelder-Mead
│   └── less_simple.py                 # Cascaded control implementation
├── controller/
│   ├── pd_controller.py               # PD position controller
│   ├── ac_controller.py               # Adaptive controller with Lyapunov law
│   └── attitude_controller.py         # Attitude controller for cascaded system
├── environment/
│   ├── quadcopter_env.py              # Simplified 6-DOF translational model
│   ├── cascaded_quadcopter_env.py     # Full dynamics with attitude
│   ├── Quadcopter_Dynamics.py         # Core dynamics implementation
│   └── Drone.py                       # Drone parameter definitions
└── simple_result.png                   # Generated comparison plot
```

---

## Control Approaches

### 1. PD Controller

Classical position control with separate gains for horizontal (xy) and vertical (z) motion:

```
F_des = m * (Kp*(pos_d - pos) + Kd*(vel_d - vel) + [0, 0, g])
```

**Optimized Gains**:
- `Kp_xy = 29.83`, `Kd_xy = 8.78`
- `Kp_z = 11.12`, `Kd_z = 13.81`

### 2. Adaptive Controller

Lyapunov-based adaptive control with online parameter estimation:

```
F_des = m_hat * (λ*e_pos + k*e_vel + [0, 0, g]) + d_hat

Parameter updates:
  α_hat_dot = γ_α * e_vel^T * a_des
  d_hat_dot = -γ_d * e_vel
```

**Adaptive Gains**:
- `λ = 10.0` (position feedback)
- `k = 15.0` (velocity feedback)
- `γ_α = 0.1` (mass adaptation rate)
- `γ_d = 0.1` (disturbance adaptation rate)

---

## Simplified Dynamics Model

The project uses a simplified 6-DOF translational model that assumes perfect attitude control:

```python
# State: [x, y, z, vx, vy, vz]
# Control: [Fx, Fy, Fz] (force vector in world frame)

m * acceleration = F_des + [0, 0, -m*g] + disturbance
```

This abstraction enables focused evaluation of position control strategies without coupling to attitude dynamics.

---

## Test Scenarios

The simulation evaluates controllers under three progressive scenarios:

1. **Baseline**: Spiral trajectory (20s) with nominal mass (1.9kg)
2. **Wind Disturbance**: Same trajectory with +3N constant horizontal force
3. **Mass + Wind**: +1kg package (53% mass increase) with disturbance

Each scenario tracks the same reference trajectory:
- Spiral ascent: radius 5m, climb rate 2 m/s
- Duration: 20 seconds
- Sampling rate: 100 Hz (dt = 0.01s)

---

## Gain Optimization

Controller gains were optimized using Nelder-Mead optimization:

```bash
cd simulator
python3 optimize_gains.py
```

The optimization minimizes mean trajectory tracking error over the baseline scenario.

---

## Key Features

### Provable Stability
- Lyapunov-based design with rigorous stability proof
- Guaranteed parameter convergence under persistence of excitation
- Bounded parameter estimates with projection

### Complete Disturbance Rejection
- Adaptive controller achieves 0% performance degradation under constant disturbances
- Online disturbance estimation without prior knowledge

### Optimized Performance
- Both controllers use Nelder-Mead optimized gains
- Fair comparison under identical test conditions
- Optimization framework implemented using Claude Code

---

## Running Simulations

### Main Comparison

```bash
cd simulator
python3 simple.py
```

Outputs:
- Console: Mean error metrics for each scenario
- File: `simple_result.png` (6-panel comparison plot)

### Gain Optimization

```bash
cd simulator
python3 optimize_gains.py
```

This will run Nelder-Mead optimization for both controllers (50 iterations each).

---

## Requirements

```
numpy
scipy
matplotlib
```

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

---

## References

1. Slotine & Li (1991): *Applied Nonlinear Control*
2. Ioannou & Sun (1996): *Robust Adaptive Control*
3. Dydek, Annaswamy, & Lavretsky (2013): *Adaptive Control of Quadrotor UAVs*

---

## GitHub Repository

[https://github.com/FSchechner/Adaptive-Drone-Controller-using-Lyapunov-Theory](https://github.com/FSchechner/Adaptive-Drone-Controller-using-Lyapunov-Theory)

---

## Contact

For questions or collaboration, please open an issue on GitHub.

---

**Note**: This project focuses on simplified translational dynamics to evaluate position control strategies. Future work includes experimental validation on physical hardware.
