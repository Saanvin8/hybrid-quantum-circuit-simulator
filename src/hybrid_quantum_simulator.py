"""
Hybrid Quantum Circuit Simulator
--------------------------------
Reference implementation for the Phoenix Association IT Team induction task.

Design constraints:
- Pure state-vector simulation.
- NumPy is used for numerical arrays only.
- No Qiskit/Aer/PennyLane or black-box quantum simulator is used.
- Mandatory gates: X, H, CNOT.
- Optional gates implemented: Z, SWAP.
- Hybrid extension: parameterized Ry gate for a classical parameter-search demo.

Qubit convention:
- Qubit 0 is the least-significant bit of the computational-basis index.
- Basis index i corresponds to |q_(n-1)...q_1 q_0>.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional

import numpy as np


X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)


class QuantumSimulator:
    """Small, readable state-vector simulator.

    The complete n-qubit pure state is stored as 2**n complex128
    amplitudes. Gate kernels update amplitudes directly instead of
    constructing a 2**n x 2**n matrix.
    """

    def __init__(self, n_qubits: int):
        if not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError("n_qubits must be a positive integer")
        # Deliberate safety limit for an educational reference implementation.
        if n_qubits > 24:
            raise ValueError("Reference implementation is limited to 24 qubits")
        self.n = n_qubits
        self.state = np.zeros(2 ** n_qubits, dtype=np.complex128)
        self.state[0] = 1.0 + 0.0j  # |00...0>

    def _check_qubit(self, q: int) -> None:
        if not 0 <= q < self.n:
            raise ValueError(f"qubit {q} outside [0, {self.n - 1}]")

    def _apply_single(self, gate: np.ndarray, q: int) -> None:
        """Apply a 2x2 gate to one qubit.

        Amplitudes are processed in pairs whose indices differ by 2**q.
        The pair is copied to local variables before writing, preventing
        an in-place aliasing error.
        """
        self._check_qubit(q)
        bit = 1 << q
        for base in range(0, len(self.state), 2 * bit):
            for offset in range(bit):
                i = base + offset
                j = i + bit
                a = self.state[i]
                b = self.state[j]

                self.state[i] = gate[0, 0] * a + gate[0, 1] * b
                self.state[j] = gate[1, 0] * a + gate[1, 1] * b

    def x(self, q: int) -> None:
        """Pauli-X / NOT gate."""
        self._apply_single(X, q)

    def z(self, q: int) -> None:
        """Pauli-Z phase-flip gate."""
        self._apply_single(Z, q)

    def h(self, q: int) -> None:
        """Hadamard gate: creates/undoes equal superpositions."""
        self._apply_single(H, q)

    def ry(self, theta: float, q: int) -> None:
        """Parameterized Ry gate for the optional hybrid demo."""
        self._check_qubit(q)
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
        self._apply_single(gate, q)

    def cnot(self, control: int, target: int) -> None:
        """Controlled-NOT.

        If control is |1>, the target bit is flipped. In a state-vector
        representation this is a permutation of amplitude pairs.
        """
        self._check_qubit(control)
        self._check_qubit(target)
        if control == target:
            raise ValueError("control and target must be different")

        control_bit = 1 << control
        target_bit = 1 << target

        for i in range(len(self.state)):
            if (i & control_bit) and not (i & target_bit):
                j = i | target_bit
                self.state[i], self.state[j] = self.state[j], self.state[i]

    def swap(self, q1: int, q2: int) -> None:
        """Swap two qubit positions."""
        self._check_qubit(q1)
        self._check_qubit(q2)
        if q1 == q2:
            return

        b1, b2 = 1 << q1, 1 << q2
        new_state = self.state.copy()

        for i in range(len(self.state)):
            bit1 = (i >> q1) & 1
            bit2 = (i >> q2) & 1
            if bit1 != bit2:
                j = i ^ b1 ^ b2
                if i < j:
                    new_state[i], new_state[j] = self.state[j], self.state[i]

        self.state = new_state

    def probabilities(self) -> np.ndarray:
        """Return computational-basis probabilities |alpha_i|^2."""
        return np.abs(self.state) ** 2

    def norm_error(self) -> float:
        """Return |<psi|psi> - 1|."""
        return float(abs(np.vdot(self.state, self.state).real - 1.0))

    def sample(self, shots: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """Sample measurement outcomes and return a histogram."""
        if shots <= 0:
            raise ValueError("shots must be positive")

        p = self.probabilities()
        # Small floating-point drift is corrected only for sampling.
        p = p / p.sum()

        rng = np.random.default_rng(seed)
        draws = rng.choice(len(p), size=shots, p=p)
        return np.bincount(draws, minlength=len(p))

    @property
    def state_vector_bytes(self) -> int:
        """Raw state-vector storage in bytes."""
        return int(self.state.nbytes)


def bell_state() -> QuantumSimulator:
    """Create |Phi+> = (|00> + |11>)/sqrt(2)."""
    sim = QuantumSimulator(2)
    sim.h(0)
    sim.cnot(0, 1)
    return sim


def verify_bell_state(tolerance: float = 1e-12) -> dict:
    """Verify deterministic simulator properties for the Bell circuit."""
    sim = bell_state()
    expected = np.array([0.5, 0.0, 0.0, 0.5])
    actual = sim.probabilities()

    return {
        "max_probability_error": float(np.max(np.abs(actual - expected))),
        "norm_error": sim.norm_error(),
        "passed": bool(
            np.max(np.abs(actual - expected)) < tolerance
            and sim.norm_error() < tolerance
        ),
    }


def hybrid_parameter_search(target_probability: float = 0.75,
                            steps: int = 361) -> dict:
    """Classical outer loop + quantum circuit inner loop.

    The classical layer searches theta in [0, 2*pi] and asks the quantum
    simulator for P(|1>) after Ry(theta)|0>. This demonstrates the
    quantum-classical feedback pattern used by hybrid algorithms without
    hiding any quantum operation behind a library.
    """
    best = None

    for theta in np.linspace(0.0, 2.0 * np.pi, steps):
        sim = QuantumSimulator(1)
        sim.ry(theta, 0)
        p1 = float(sim.probabilities()[1])
        loss = (p1 - target_probability) ** 2

        candidate = (loss, theta, p1)
        if best is None or candidate[0] < best[0]:
            best = candidate

    loss, theta, p1 = best
    return {
        "theta": float(theta),
        "p1": float(p1),
        "loss": float(loss),
    }


def benchmark(n_qubits: int, repeats: int = 5) -> dict:
    """Benchmark one H and one CNOT gate using median wall-clock time."""
    if n_qubits < 2:
        raise ValueError("Benchmark requires at least 2 qubits")

    sim = QuantumSimulator(n_qubits)

    h_times = []
    cnot_times = []

    for _ in range(repeats):
        sim.state.fill(0.0)
        sim.state[0] = 1.0 + 0.0j
        start = perf_counter()
        sim.h(0)
        h_times.append(perf_counter() - start)

        sim.state.fill(0.0)
        sim.state[0] = 1.0 + 0.0j
        start = perf_counter()
        sim.cnot(0, 1)
        cnot_times.append(perf_counter() - start)

    return {
        "qubits": n_qubits,
        "state_vector_elements": 2 ** n_qubits,
        "state_vector_bytes": sim.state_vector_bytes,
        "h_median_s": float(np.median(h_times)),
        "cnot_median_s": float(np.median(cnot_times)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Educational hybrid quantum circuit simulator")
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.benchmark:
        print(benchmark(args.qubits))
        return

    sim = bell_state() if args.qubits == 2 else QuantumSimulator(args.qubits)
    if args.qubits != 2:
        sim.h(0)
        if args.qubits >= 2:
            sim.cnot(0, 1)

    print("State vector:")
    print(sim.state)
    print("Probabilities:")
    print(sim.probabilities())
    print("Norm error:", sim.norm_error())
    print("Measurement counts:")
    print(sim.sample(args.shots, args.seed))


if __name__ == "__main__":
    main()
