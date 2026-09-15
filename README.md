# Hybrid Quantum Circuit Simulator

A small, transparent state-vector quantum circuit simulator for the Phoenix Association IT Team induction task.

## Scope

- 1–3 qubits required; implementation is written to work beyond that for benchmarking.
- Mandatory gates: X, H, CNOT.
- Optional gates: Z, SWAP.
- Hybrid extension: parameterized `Ry` plus a classical parameter search.
- No Qiskit/Aer/PennyLane simulator is used.
- NumPy is used only for numerical arrays and random sampling.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run Bell state

```bash
python examples/bell_state.py
```

Expected probabilities are approximately:

```text
[0.5, 0.0, 0.0, 0.5]
```

## Run simulator CLI

```bash
python src/hybrid_quantum_simulator.py --qubits 2 --shots 10000 --seed 42
```

## Run benchmarks

```bash
python src/hybrid_quantum_simulator.py --qubits 16 --benchmark
```

## Run tests

```bash
pytest -q
```

## Qubit convention

Qubit 0 is the least-significant bit of the basis-state index.

## Engineering note

The state vector contains `2**n` complex128 amplitudes, so the raw state storage is:

`16 * 2**n` bytes.

This exponential scaling is the main limitation of exact full-state-vector simulation.
