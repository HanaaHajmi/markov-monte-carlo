# Chaînes de Markov et Monte-Carlo

Projet Python implémentant from scratch les chaînes de Markov à temps discret et les méthodes de Monte-Carlo, dans le cadre de mon M1 Statistique et Science des Données à l'Université Grenoble Alpes.

## Contenu

```
markov-monte-carlo/
├── src/
│   ├── markov_chain.py     # Classe MarkovChain : simulation, distribution stationnaire, convergence
│   └── monte_carlo.py      # Estimation de π, intégration, TCL, bootstrap
├── outputs/                # Figures générées automatiquement
├── run_demo.py             # Script de démonstration complet
├── requirements.txt
└── README.md
```

## Ce que le projet couvre

### Chaînes de Markov
- Implémentation d'une chaîne de Markov homogène à partir d'une matrice de transition
- Calcul de la **distribution stationnaire** par décomposition en valeurs propres
- **Simulation** de trajectoires
- **Convergence** vers la distribution stationnaire : distance en variation totale
- **Ergodicité** : fréquences empiriques de visite vs distribution théorique
- Exemple appliqué : modèle météo 3 états, file d'attente discrète

### Monte-Carlo
- **Estimation de π** par la méthode du quart de cercle, avec visualisation de la convergence en O(1/√n)
- **Intégration numérique** : estimation de ∫f(x)dx avec erreur standard
- **Théorème Central Limite** : illustration empirique sur distributions exponentielles et de Bernoulli
- **Bootstrap** : calcul d'intervalles de confiance à 95% par rééchantillonnage


### Utilisation de la classe MarkovChain

```python
import numpy as np
from src.markov_chain import MarkovChain

P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.1, 0.3, 0.6],
])
mc = MarkovChain(P, states=["Soleil", "Nuage", "Pluie"])

# Distribution stationnaire
pi = mc.stationary_distribution()

# Simulation
traj = mc.simulate(n_steps=100, start_state=0, seed=42)

# Convergence
mu0 = np.array([1., 0., 0.])
mc.plot_convergence(mu0, max_steps=60)
```

### Utilisation Monte-Carlo

```python
from src.monte_carlo import estimate_pi, mc_integrate, bootstrap_ci
import numpy as np

# Estimation de π
pi_est = estimate_pi(n_samples=100_000, seed=0)

# Intégration
f = lambda x: np.exp(-x**2)
val, err = mc_integrate(f, 0, 1, n_samples=200_000)

# Bootstrap
data = np.random.normal(3, 1, 100)
ci_low, ci_high, _ = bootstrap_ci(data, np.mean, n_bootstrap=10_000)
```

## Résultats

| Expérience | Résultat |
|---|---|
| Estimation de π (n=100 000) | ≈ 3.14152 |
| ∫₀¹ exp(−x²) dx (n=200 000) | ≈ 0.7468 ± 0.0002 |
| ∫₀^π sin(x) dx (n=200 000) | ≈ 2.0001 ± 0.0009 |
| Convergence TCL visible dès | n = 30 |

## Notions mathématiques

- **Matrice stochastique** : P ≥ 0, lignes somment à 1
- **Distribution stationnaire** : πP = π, solution du système linéaire via valeurs propres de Pᵀ
- **Distance en variation totale** : ||μ - π||_TV = ½ Σ|μᵢ - πᵢ|
- **Ergodicité** : convergence des moyennes empiriques vers la moyenne stationnaire
- **Loi des Grands Nombres** : X̄ₙ → μ p.s. quand n → ∞
- **TCL** : √n(X̄ₙ − μ)/σ → N(0,1) en loi
- **Erreur Monte-Carlo** : O(1/√n) indépendamment de la dimension

## Références

- Norris, J.R. (1997). *Markov Chains*. Cambridge University Press.
- Robert, C.P. & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer.
- Notes de cours — Probabilités M1 SSD, Université Grenoble Alpes.
