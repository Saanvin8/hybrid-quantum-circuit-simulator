from src.hybrid_quantum_simulator import bell_state

sim = bell_state()

print("Bell-state amplitudes:")
print(sim.state)

print("Bell-state probabilities:")
print(sim.probabilities())

print("10,000 measurement shots:")
print(sim.sample(shots=10_000, seed=42))
