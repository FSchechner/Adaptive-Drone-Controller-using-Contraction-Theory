# Introduction

## Problem Statement

Autonomous quadrotors have become increasingly important in applications ranging from package delivery and aerial photography to search-and-rescue operations and infrastructure inspection. A critical challenge in these applications is maintaining precise trajectory tracking while the system's physical parameters vary during flight. Unlike ground vehicles or fixed-wing aircraft, quadrotors are inherently unstable systems requiring continuous active control, making them particularly sensitive to parameter variations.

The challenge becomes evident in practical scenarios: a delivery drone's mass changes by ±67% when picking up or dropping packages; drag coefficients vary by ±30% as altitude changes affect air density; and battery depletion gradually shifts the center of mass. Traditional fixed-gain controllers, designed for nominal operating conditions, exhibit severe performance degradation when these parameters deviate from their assumed values. A PD controller tuned for a 1.5 kg quadrotor may fail to maintain stable flight when the mass increases to 2.5 kg, resulting in oscillations, trajectory deviations, or complete loss of control.

## Current Approaches and Limitations

The standard solution to parameter uncertainty in control systems is **gain scheduling**, where multiple controllers are designed for different operating points and gains are interpolated during flight. However, this approach suffers from several limitations:

1. **Scalability**: The number of operating points grows exponentially with the dimension of the parameter space
2. **Conservative design**: Controllers must be tuned for worst-case scenarios, sacrificing performance
3. **Smooth transitions**: Ensuring stability during gain switching requires careful analysis
4. **Unknown parameters**: Gain scheduling requires *a priori* knowledge of which operating regime the system is in

**Model Predictive Control (MPC)** addresses some of these issues by incorporating model predictions, but requires significant computational resources and accurate system models—both challenging for resource-constrained aerial platforms with uncertain dynamics.

**Adaptive control** offers an elegant alternative by estimating unknown parameters online and adjusting control laws accordingly. Classical adaptive control using Lyapunov's direct method has been successfully applied to quadrotors, but comes with significant design complexity. For a quadrotor with uncertain mass and drag, the Lyapunov-based backstepping approach requires:

- Recursive design through position → velocity → attitude → angular rate subsystems
- Construction of an explicit Lyapunov function (often 10+ pages of algebra)
- Strict-feedback structural assumptions
- Complete redesign when adding new uncertain parameters

## Contraction Theory: A Simpler Path

This project explores an alternative framework based on **contraction theory**, which offers a fundamentally different perspective on stability and convergence. Rather than proving that trajectories converge to an equilibrium point (as in Lyapunov theory), contraction theory proves that trajectories converge *to each other*. This shift in perspective leads to several practical advantages:

**Simplified design**: Instead of constructing an explicit Lyapunov function, we verify a matrix inequality numerically. For our quadrotor, this reduces the design complexity from 10-15 pages of recursive algebra to 3-5 pages of direct computation.

**Modularity**: Adding new uncertain parameters (e.g., propeller efficiency, center-of-mass offset) requires only appending columns to a regressor matrix, rather than redesigning the entire control architecture.

**Explicit convergence rates**: The contraction condition directly provides exponential convergence rates, making performance analysis transparent.

**Matched uncertainty**: For systems where uncertainties enter through the same channels as control inputs (as with mass and drag in quadrotor dynamics), contraction-based adaptation naturally handles these perturbations.

## Contributions and Scope

This project implements and validates contraction-based adaptive control for quadrotor trajectory tracking with uncertain mass and drag coefficients. Specifically, we:

1. **Develop** a simplified 3D quadrotor model with matched uncertainties in mass and drag
2. **Design** a contraction-based adaptive controller that estimates parameters online
3. **Verify** the contraction condition and prove exponential convergence to desired trajectories
4. **Simulate** multiple scenarios: nominal operation, heavy/light payloads, and altitude-dependent drag variations
5. **Compare** performance against fixed-gain PD control to quantify improvement

The implementation focuses on vertical dynamics (where mass and drag uncertainties are most pronounced) while maintaining full 3D position and attitude control. This approach balances theoretical rigor with practical applicability, demonstrating that contraction theory can significantly simplify adaptive controller design without sacrificing performance.

## Organization

The remainder of this report is organized as follows:
- **Section 2** reviews contraction theory fundamentals and the Lopez-Slotine framework for adaptive control
- **Section 3** presents the quadrotor dynamic model and identifies matched uncertainties
- **Section 4** derives the contraction-based adaptive controller
- **Section 5** presents simulation results comparing adaptive and fixed controllers
- **Section 6** discusses implications, limitations, and future work
