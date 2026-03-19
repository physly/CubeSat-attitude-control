# CubeSat Orientation Control (Version 2)

## Description

CubeSat Orientation Computation Program Using Quaternions

This program provides a lightweight, real-time algorithm for computing the current and desired orientation of a CubeSat using quaternions. The algorithm is designed for minimal runtime load, while heavy computations are performed once during precomputation.

Program workflow:

1. Input data:

* vectors (vcts in the program): an array of key satellite orientation points, each row contains:
  `[time, e1_x, e1_y, e1_z, e2_x, e2_y, e2_z, e3_x, e3_y, e3_z]`
  where e1, e2, e3 are mutually perpendicular unit vectors of the laboratory coordinate system.

2. Precompute (once / offline):

* Interpolation of the desired orientation over time at a specified frequency.
* For each time point (with some step), rotation quaternions are computed.
* Heavy operations, such as SQUAD interpolation, are performed only once.

3. Runtime (onboard CubeSat):

* At each time step, the satellite receives from cameras two non-collinear vectors in the laboratory frame (its current X axis and a perpendicular vector - Y axis).
* Gram–Schmidt orthogonalization is applied to compute the current orthonormal basis in the satellite frame.
* The basis is converted into the current quaternion (`q_current`).
* The program retrieves the desired quaternion (`q_desired`) from the precomputed trajectory using a stateful time index.

4. Output:

* `q_current` — current satellite orientation (orientation of lab frame in satellite frame)
* `q_desired` — desired orientation at the current moment (lab frame orientation in satellite frame)
* `idx` — current index in the precomputed trajectory

Features:

* Fully quaternion-based — no rotation matrices needed.
* Runtime computation is extremely lightweight: only Gram–Schmidt, a quaternion conversion, and a lookup.
* Precompute handles all heavy interpolation calculations.
* Handles degenerate and boundary cases safely.
* Designed for potential implementation on real CubeSats with minimal hardware resources.

---

## Requirements

* Python 3.x
* Standard library only (`math` module)
* No additional dependencies

---

## Usage

1. Prepare your `vcts` array with key orientation points and time stamps.
2. Call `precompute_trajectory(vcts, frequency)` to precompute the desired quaternions offline.
3. At runtime, provide the satellite’s current two non-collinear vectors and the current time to the `compute` function.
4. The function will return the current and desired quaternions along with the trajectory index.

Example:

```python
vcts = [
    [1, 0, 0, 0.8, 0, 0.6, 0.6, 0, -0.8],
    [0.6, 0.4, 0, -0.77, 0.64, 0, -0.26, -0.31, 0.92],
    [0.4, 0, 0.6, -0.06, 1, 0, 0, 0, 1],
    [0, 0.6, 0.4, 0, 1, 0, -0.06, 0, 1]
]

tms = [0, 0.5, 6, 10]
vct1 = [0, 0.6, 0.4]
vct2 = [0.4, 0, 0.6]
time_needed = 5.426
frequency = 10

precomputed = precompute_trajectory(vcts, frequency)
q_current, q_desired, idx = compute(vct1, vct2, time_needed, tms, precomputed, 2)
print(q_current, q_desired, idx)
```

---

## Notes

* This program is designed to be lightweight and efficient for CubeSat onboard computation.
* All heavy computations (SQUAD interpolation) are done in precompute, reducing runtime load.
* Input vectors do not need to be perfectly perpendicular; Gram–Schmidt ensures orthonormal basis construction.
* Designed to handle edge cases safely, including degenerate or near-zero vectors.
