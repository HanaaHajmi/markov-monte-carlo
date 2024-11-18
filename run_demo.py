"""
Démonstration complète : Chaînes de Markov et Monte-Carlo
==========================================================
Script exécutable qui génère toutes les figures et affiche les résultats.
Lancez avec : python run_demo.py
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from markov_chain import MarkovChain
from monte_carlo import (
    estimate_pi, plot_pi_estimation, plot_pi_convergence,
    mc_integrate, illustrate_tcl, bootstrap_ci, plot_bootstrap_ci
)

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("  CHAÎNES DE MARKOV ET MONTE-CARLO")
print("=" * 60)


# -----------------------------------------------------------------------
# PARTIE 1 : Chaîne de Markov simple (3 états)
# -----------------------------------------------------------------------

print("\n--- 1. Chaîne de Markov à 3 états ---")

P_simple = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.1, 0.3, 0.6],
])
states = ["Soleil", "Nuage", "Pluie"]

mc = MarkovChain(P_simple, states)

pi = mc.stationary_distribution()
print(f"Distribution stationnaire : {dict(zip(states, pi.round(4)))}")

mu0 = np.array([1.0, 0.0, 0.0])  # On part de l'état Soleil
mu_10 = mc.distribution_at_step(mu0, 10)
mu_50 = mc.distribution_at_step(mu0, 50)
print(f"μ_0  = {mu0}")
print(f"μ_10 = {mu_10.round(4)}")
print(f"μ_50 = {mu_50.round(4)}")
print(f"π    = {pi.round(4)}")

mc.plot_trajectory(n_steps=80, start_state=0, seed=42,
                   save_path="outputs/trajectory_3states.png")

mc.plot_convergence(mu0, max_steps=60,
                    save_path="outputs/convergence_tvd_3states.png")

mc.plot_stationary_comparison(n_steps=50_000, seed=0,
                               save_path="outputs/stationary_comparison.png")


# -----------------------------------------------------------------------
# PARTIE 2 : Chaîne de Markov — exemple file d'attente (M/M/1 discrète)
# -----------------------------------------------------------------------

print("\n--- 2. Chaîne de Markov — file d'attente (5 états) ---")

# p = proba d'arrivée, q = proba de service
p, q = 0.3, 0.5
n_states = 5

P_queue = np.zeros((n_states, n_states))
for i in range(n_states):
    if i == 0:
        P_queue[i, 0] = 1 - p
        P_queue[i, 1] = p
    elif i == n_states - 1:
        P_queue[i, i - 1] = q
        P_queue[i, i] = 1 - q
    else:
        P_queue[i, i - 1] = q * (1 - p)
        P_queue[i, i] = p * q + (1 - p) * (1 - q)
        P_queue[i, i + 1] = p * (1 - q)

mc_queue = MarkovChain(P_queue, states=[f"N={i}" for i in range(n_states)])
pi_queue = mc_queue.stationary_distribution()
print(f"Distribution stationnaire (file) : {dict(zip(mc_queue.states, pi_queue.round(4)))}")

mu0_queue = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
mc_queue.plot_convergence(mu0_queue, max_steps=80,
                           save_path="outputs/convergence_queue.png")


# -----------------------------------------------------------------------
# PARTIE 3 : Monte-Carlo — estimation de π
# -----------------------------------------------------------------------

print("\n--- 3. Estimation de π par Monte-Carlo ---")

for n in [100, 1_000, 10_000, 100_000]:
    pi_est = estimate_pi(n, seed=0)
    print(f"  n = {n:>8}  →  π ≈ {pi_est:.6f}  (erreur = {abs(pi_est - np.pi):.6f})")

plot_pi_estimation(n_samples=3000, seed=42,
                   save_path="outputs/pi_estimation_circle.png")

plot_pi_convergence(max_samples=100_000, step=500, seed=42,
                    save_path="outputs/pi_convergence.png")


# -----------------------------------------------------------------------
# PARTIE 4 : Intégration de Monte-Carlo
# -----------------------------------------------------------------------

print("\n--- 4. Intégration de Monte-Carlo ---")

# ∫_0^1 e^(-x²) dx  (pas de forme fermée)
f = lambda x: np.exp(-x**2)
est, err = mc_integrate(f, 0, 1, n_samples=200_000, seed=0)
print(f"  ∫_0^1 exp(-x²) dx ≈ {est:.6f}  ±  {err:.6f}")
print(f"  Valeur scipy       : 0.746824  (référence)")

# ∫_0^π sin(x) dx = 2
g = lambda x: np.sin(x)
est2, err2 = mc_integrate(g, 0, np.pi, n_samples=200_000, seed=1)
print(f"  ∫_0^π sin(x) dx ≈ {est2:.6f}  ±  {err2:.6f}  (valeur exacte = 2.0)")


# -----------------------------------------------------------------------
# PARTIE 5 : Théorème Central Limite
# -----------------------------------------------------------------------

print("\n--- 5. Illustration du Théorème Central Limite ---")

# Distribution exponentielle (asymétrique)
def expo_samples(n, rng):
    return rng.exponential(scale=1.0, size=n)

illustrate_tcl(expo_samples, "Exponentielle(1)", ns=[1, 5, 30, 200],
               n_replications=5000, seed=42,
               save_path="outputs/tcl_exponential.png")

# Distribution de Bernoulli
def bernoulli_samples(n, rng):
    return rng.binomial(1, 0.3, size=n).astype(float)

illustrate_tcl(bernoulli_samples, "Bernoulli(0.3)", ns=[1, 10, 50, 300],
               n_replications=5000, seed=42,
               save_path="outputs/tcl_bernoulli.png")


# -----------------------------------------------------------------------
# PARTIE 6 : Bootstrap
# -----------------------------------------------------------------------

print("\n--- 6. Intervalle de confiance par bootstrap ---")

rng = np.random.default_rng(99)
# Données simulées : mélange de deux gaussiennes
data = np.concatenate([rng.normal(2, 0.5, 80), rng.normal(5, 1, 20)])

observed_mean = data.mean()
ci_low, ci_high, boot_stats = bootstrap_ci(data, np.mean, n_bootstrap=10_000,
                                            alpha=0.05, seed=0)

print(f"  Moyenne observée : {observed_mean:.4f}")
print(f"  IC 95% bootstrap : [{ci_low:.4f}, {ci_high:.4f}]")

plot_bootstrap_ci(boot_stats, ci_low, ci_high, observed_mean,
                  save_path="outputs/bootstrap_ci.png")


print("\n" + "=" * 60)
print("  Toutes les figures sauvegardées dans outputs/")
print("=" * 60)
