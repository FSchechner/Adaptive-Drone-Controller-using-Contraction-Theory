# Adaptive Quadcopter Trajectory Tracking

**Adaptive control for quadcopter trajectory tracking under parametric uncertainty**

Research project for Prof. Jean-Jacques Slotine's robotics course
*MIT Nonlinear Systems Laboratory*

---

## Project Overview

This project implements adaptive trajectory tracking for a quadcopter system with uncertain parameters (mass, drag). The system consists of three main components:

### 1. Simulator
Complex quadcopter dynamics that takes **forces** as input:
- Full 6-DOF or simplified 2D dynamics
- State: position, velocity, attitude, angular rates
- Input: thrust forces from rotors
- Uncertain parameters: mass, drag coefficients

### 2. Controller
Linear controller that outputs **forces**:
- PD control for position tracking
- Feedforward compensation for gravity
- Outputs desired thrust forces
- **No complex nonlinear inversions**

### 3. Adaptive Layer
Real-time parameter estimation:
- Estimates mass and drag online
- Updates controller compensation
- Guarantees exponential convergence to trajectory

---

## System Architecture

```
┌─────────────────────────────────┐
│   Desired Trajectory            │
│   x_d(t), ẋ_d(t), ẍ_d(t)       │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│   Linear Controller             │
│   • PD gains: K_p, K_d          │
│   • Feedforward: m̂·(ẍ_d + g)   │  → Forces (u)
│   • Drag compensation: d̂·ẋ     │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│   Adaptive Estimator            │
│   • Online mass estimation      │  → m̂, d̂
│   • Online drag estimation      │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│   Quadcopter Simulator          │
│   • Full nonlinear dynamics     │  → x(t), ẋ(t)
│   • Unknown true parameters     │
└─────────────────────────────────┘
```

---

## Key Features

**Simple Controller Design**
- Linear PD controller with adaptive feedforward
- No need for complex backstepping or feedback linearization
- Easy to tune and implement

**Complex Simulator**
- Realistic quadcopter dynamics
- Matches real-world behavior
- Tests controller robustness

**Adaptive Learning**
- Estimates mass changes (payload variations)
- Adapts to drag variations (altitude, air density)
- Converges exponentially to true parameters

---

## Problem Setup

### Dynamics (Vertical 1D Example)

```
True system:  m·z̈ = u - m·g - d·ż

Where:
  z    = vertical position
  u    = total thrust (control input)
  m    = mass (UNKNOWN)
  d    = drag coefficient (UNKNOWN)
  g    = gravity (known)
```

### Controller Structure

```python
# Linear controller
e = z - z_d                           # Position error
e_dot = ż - ż_d                       # Velocity error

# Control law (linear in error!)
u = m̂·(z̈_d + g) + K_p·e + K_d·e_dot + d̂·ż
    └──────┬────┘   └─────┬──────┘   └─┬─┘
    feedforward      PD control     drag comp
```

### Adaptation Law

```python
# Parameter estimates update
m̂̇ = -Γ_m · (z̈_d + g) · e_dot
d̂̇ = -Γ_d · ż · e_dot

Where Γ_m, Γ_d are adaptation gains (tunable)
```

---

## Why This Approach?

### Traditional Approach (Complex)
1. Design feedback linearization controller
2. Derive Lyapunov function
3. Backstepping through position → velocity → attitude
4. 10+ pages of math

### Our Approach (Simple)
1. Use linear PD controller
2. Add adaptive feedforward for uncertain parameters
3. Prove convergence using contraction theory
4. Done!

**Key advantage:** Separation of concerns
- Controller design: Simple linear PD
- Adaptation: Standard gradient descent
- Stability: Verified via contraction

---

## Quick Start

### Installation
```bash
git clone https://github.com/FSchechner/Adaptive-Drone-Controller-using-Contraction-Theory
cd Adaptive-Drone-Controller-using-Contraction-Theory
pip install -r requirements.txt
```

### Run Basic Simulation
```python
from simulator import QuadcopterSimulator
from controller import AdaptiveController
from trajectories import SinusoidTrajectory

# Create simulator with unknown parameters
sim = QuadcopterSimulator(
    true_mass=2.0,      # True mass (unknown to controller)
    true_drag=0.3       # True drag (unknown to controller)
)

# Create adaptive controller with initial guesses
controller = AdaptiveController(
    mass_estimate=1.5,   # Initial guess
    drag_estimate=0.2,   # Initial guess
    K_p=10.0,           # Position gain
    K_d=5.0,            # Velocity gain
    Gamma_m=1.0,        # Mass adaptation rate
    Gamma_d=0.5         # Drag adaptation rate
)

# Define trajectory
traj = SinusoidTrajectory(amplitude=2.0, frequency=0.5)

# Run simulation
results = sim.run(controller, traj, duration=20.0, dt=0.01)

# Plot results
results.plot_tracking()      # Position vs desired
results.plot_parameters()    # Parameter convergence
results.plot_forces()        # Control forces
```

---

## Project Structure

```
.
├── simulator/
│   ├── dynamics.py           # Quadcopter dynamics
│   ├── simulator.py          # Main simulation loop
│   └── visualization.py      # Plotting tools
│
├── controller/
│   ├── adaptive.py           # Adaptive controller
│   ├── pd_controller.py      # Base PD controller
│   └── estimator.py          # Parameter estimation
│
├── trajectories/
│   ├── circle.py             # Circular trajectory
│   ├── lemniscate.py         # Figure-8 trajectory
│   └── waypoints.py          # Waypoint following
│
├── analysis/
│   ├── convergence.py        # Convergence analysis
│   ├── robustness.py         # Robustness tests
│   └── comparison.py         # Compare with fixed controller
│
└── tests/
    ├── test_simulator.py
    ├── test_controller.py
    └── test_adaptation.py
```

---

## Scenarios to Test

1. **Payload Pickup:** Mass changes from 1.5kg → 2.5kg during flight
2. **Altitude Change:** Drag varies as air density changes
3. **Combined Uncertainty:** Both mass and drag unknown
4. **Fast Maneuvers:** Aggressive trajectory with high accelerations
5. **Slow Maneuvers:** Gentle trajectory to test steady-state adaptation

---

## Expected Results

**Without Adaptation (Fixed Controller)**
- Large tracking error when parameters change
- Overshoot/oscillation with wrong mass estimate
- Steady-state error with wrong drag estimate

**With Adaptation**
- Small transient error during adaptation
- Parameters converge to true values
- Exponential convergence to desired trajectory
- Robust to parameter changes

---

## Theory: Contraction-Based Adaptive Control

The controller guarantees exponential convergence using **contraction theory**:

1. **Nominal system is contracting:** With correct parameters, all trajectories converge
2. **Matched uncertainty:** Parameter errors enter through same channel as control
3. **Adaptation law:** Standard gradient descent driven by tracking error
4. **Result:** Exponential convergence of both tracking error and parameter estimates

**Mathematical guarantee:**
```
||e(t)|| ≤ exp(-λt) ||e(0)||   (exponential convergence)
```

---

## References

**Primary Reference:**
- Lopez, B. T., & Slotine, J. J. E. (2020). *Contraction Metrics in Adaptive Nonlinear Control*. arXiv:1912.13138

**Foundational Work:**
- Lohmiller, W., & Slotine, J. J. E. (1998). *On Contraction Analysis for Nonlinear Systems*. Automatica, 34(6), 683-696

**Quadcopter Control:**
- Lee, T., Leok, M., & McClamroch, N. H. (2010). *Geometric Tracking Control of a Quadrotor UAV on SE(3)*. IEEE CDC

---

## Contact

GitHub: [https://github.com/FSchechner/Adaptive-Drone-Controller-using-Contraction-Theory](https://github.com/FSchechner/Adaptive-Drone-Controller-using-Contraction-Theory)
