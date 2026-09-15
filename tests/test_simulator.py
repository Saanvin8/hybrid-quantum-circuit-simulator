import numpy as np
from src.hybrid_quantum_simulator import QuantumSimulator, bell_state, verify_bell_state


def test_hadamard_coin_toss():
    sim = QuantumSimulator(1)
    sim.h(0)
    p = sim.probabilities()
    assert np.allclose(p, [0.5, 0.5], atol=1e-12)


def test_x_gate():
    sim = QuantumSimulator(1)
    sim.x(0)
    assert np.allclose(sim.state, [0, 1])


def test_bell_state():
    result = verify_bell_state()
    assert result["passed"]


def test_cnot_control_zero_does_nothing():
    sim = QuantumSimulator(2)
    sim.x(1)  # |10> in the q1q0 display convention
    before = sim.state.copy()
    sim.cnot(0, 1)  # q0 is 0, so no action
    assert np.allclose(sim.state, before)


def test_norm_after_gates():
    sim = QuantumSimulator(3)
    sim.h(0)
    sim.x(1)
    sim.z(2)
    sim.cnot(0, 2)
    assert sim.norm_error() < 1e-12
