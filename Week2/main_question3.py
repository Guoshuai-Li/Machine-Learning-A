import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameters
n_experiments = 1000000
n_samples = 20
p = 0.35  # Bernoulli parameter (mean)
alpha_values = np.arange(0.5, 1.05, 0.05)  # α ∈ {0.5, 0.55, 0.6, ..., 0.95, 1}

# 1. Generate experiments and compute empirical frequencies
# Generate all experiments at once for efficiency
experiments = np.random.binomial(n_samples, p, n_experiments)
sample_means = experiments / n_samples

empirical_frequencies = []
for alpha in alpha_values:
    freq = np.mean(sample_means >= alpha)
    empirical_frequencies.append(freq)


# 3. Markov's bound
markov_bounds = []
mu = p  # E[Xi] = p = 0.35

for alpha in alpha_values:
    if alpha > mu:
        # Markov: P(X ≥ α) ≤ E[X]/α
        bound = mu / alpha
    else:
        # When α ≤ μ, Markov bound is trivial (≥ 1), so we use 1
        bound = 1.0
    markov_bounds.append(bound)

# 4. Chebyshev's bound
chebyshev_bounds = []
var_mean = p * (1 - p) / n_samples  # Var[(1/n)∑Xi] = Var[Xi]/n = p(1-p)/n

for alpha in alpha_values:
    if alpha > mu:
        # Chebyshev: P(|X - μ| ≥ t) ≤ σ²/t²
        # For one-sided: P(X ≥ α) ≤ σ²/(α - μ)²
        t = alpha - mu
        bound = var_mean / (t ** 2)
        bound = min(bound, 1.0)  # Cap at 1
    else:
        bound = 1.0
    chebyshev_bounds.append(bound)

# 5. Hoeffding's bound
hoeffding_bounds = []

for alpha in alpha_values:
    if alpha > mu:
        # Hoeffding: P(Sn - E[Sn] ≥ t) ≤ exp(-2t²/∑(bi-ai)²)
        # For Bernoulli: ai = 0, bi = 1, so (bi-ai)² = 1
        # Sample mean version: P((1/n)∑Xi ≥ α) ≤ exp(-2n(α-μ)²)
        t = alpha - mu
        bound = np.exp(-2 * n_samples * (t ** 2))
    else:
        bound = 1.0
    hoeffding_bounds.append(bound)

# 6. Plot comparison
plt.figure(figsize=(12, 8))

plt.plot(alpha_values, empirical_frequencies, 'ko-', label='Empirical Frequency', linewidth=2, markersize=6)
plt.plot(alpha_values, markov_bounds, 'r^-', label="Markov's Bound", linewidth=2, markersize=6)
plt.plot(alpha_values, chebyshev_bounds, 'bs-', label="Chebyshev's Bound", linewidth=2, markersize=6)
plt.plot(alpha_values, hoeffding_bounds, 'gd-', label="Hoeffding's Bound", linewidth=2, markersize=6)

plt.xlabel('α', fontsize=12)
plt.ylabel('P((1/20)∑Xi ≥ α)', fontsize=12)
plt.title('Comparison of Concentration Inequalities\n(Bernoulli(0.35), n=20, 1M experiments)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to better see the differences
plt.xlim(0.45, 1.05)
plt.ylim(1e-6, 1.1)

plt.tight_layout()
plt.show()

# 7. Exact probabilities for α = 1 and α = 0.95
# P((1/20)∑Xi ≥ 1) = P(∑Xi ≥ 20) = P(∑Xi = 20)
exact_prob_1 = binom.sf(19, n_samples, p)  # P(X ≥ 20) = P(X > 19)
print(f"   Exact P((1/20)∑Xi ≥ 1.0) = {exact_prob_1:.2e}")

# P((1/20)∑Xi ≥ 0.95) = P(∑Xi ≥ 19)
exact_prob_095 = binom.sf(18, n_samples, p)  # P(X ≥ 19) = P(X > 18)
print(f"   Exact P((1/20)∑Xi ≥ 0.95) = {exact_prob_095:.2e}")

