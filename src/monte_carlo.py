"""
Simulations de Monte-Carlo
---------------------------
Estimation de π, intégration, convergence en loi, intervalles de confiance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# -----------------------------------------------------------------------
# 1. Estimation de π par la méthode du cercle
# -----------------------------------------------------------------------

def estimate_pi(n_samples: int, seed: int = None) -> float:
    """
    Estime π par la méthode de Monte-Carlo (quart de cercle dans [0,1]²).
    """
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 1, size=(n_samples, 2))
    inside = (points[:, 0]**2 + points[:, 1]**2) <= 1.0
    return 4 * inside.mean()


def pi_convergence(max_samples: int = 100_000, step: int = 500, seed: int = 42) -> tuple:
    """
    Trace la convergence de l'estimation de π en fonction du nombre de tirages.

    Retourne
    --------
    ns : np.ndarray
        Nombre de tirages à chaque étape.
    estimates : np.ndarray
        Estimation de π à chaque étape.
    """
    rng = np.random.default_rng(seed)
    ns = np.arange(step, max_samples + 1, step)
    estimates = np.empty(len(ns))

    points = rng.uniform(0, 1, size=(max_samples, 2))
    inside = (points[:, 0]**2 + points[:, 1]**2) <= 1.0

    for i, n in enumerate(ns):
        estimates[i] = 4 * inside[:n].mean()

    return ns, estimates


def plot_pi_estimation(n_samples: int = 2000, seed: int = 0, save_path: str = None):
    """
    Visualise les points tirés et le quart de cercle.
    """
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 1, size=(n_samples, 2))
    inside = (points[:, 0]**2 + points[:, 1]**2) <= 1.0
    pi_est = 4 * inside.mean()

    theta = np.linspace(0, np.pi / 2, 300)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(points[inside, 0], points[inside, 1], s=1.5, color="#2c7bb6", alpha=0.6, label="Dans le cercle")
    ax.scatter(points[~inside, 0], points[~inside, 1], s=1.5, color="#d7191c", alpha=0.6, label="Hors du cercle")
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)
    ax.set_aspect("equal")
    ax.set_title(f"Estimation de π = {pi_est:.5f}  (n = {n_samples})", fontsize=12)
    ax.legend(markerscale=5, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_pi_convergence(max_samples: int = 100_000, step: int = 500, seed: int = 42, save_path: str = None):
    """
    Trace l'erreur |π_est - π| en fonction de n, avec la borne 1/√n.
    """
    ns, estimates = pi_convergence(max_samples, step, seed)
    errors = np.abs(estimates - np.pi)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.loglog(ns, errors, color="#2c7bb6", linewidth=1.5, label=r"$|\hat{\pi}_n - \pi|$")
    ax.loglog(ns, 1 / np.sqrt(ns), "k--", linewidth=1.2, label=r"$1/\sqrt{n}$ (borne théorique)")
    ax.set_xlabel("Nombre de tirages (log)", fontsize=12)
    ax.set_ylabel("Erreur (log)", fontsize=12)
    ax.set_title(r"Convergence de l'estimateur Monte-Carlo de $\pi$", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# -----------------------------------------------------------------------
# 2. Intégration de Monte-Carlo
# -----------------------------------------------------------------------

def mc_integrate(f, a: float, b: float, n_samples: int = 100_000, seed: int = None) -> tuple:
    """
    Estime ∫_a^b f(x) dx par Monte-Carlo.

    Retourne
    --------
    estimate : float
        Estimation de l'intégrale.
    std_error : float
        Erreur standard de l'estimation.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=n_samples)
    fx = f(x)
    estimate = (b - a) * fx.mean()
    std_error = (b - a) * fx.std() / np.sqrt(n_samples)
    return estimate, std_error


# -----------------------------------------------------------------------
# 3. Théorème Central Limite — illustration empirique
# -----------------------------------------------------------------------

def illustrate_tcl(dist_func, dist_name: str, ns: list = None, n_replications: int = 5000,
                   seed: int = 42, save_path: str = None):
    """
    Illustre le TCL : pour différents n, trace la distribution de √n (X̄_n - μ) / σ
    et la compare à N(0,1).

    Paramètres
    ----------
    dist_func : callable
        Fonction prenant (size, rng) et retournant un tableau de tirages.
    dist_name : str
        Nom de la distribution pour le titre.
    ns : list
        Valeurs de n à tester.
    n_replications : int
        Nombre de répétitions pour chaque n.
    """
    if ns is None:
        ns = [1, 5, 30, 200]

    rng = np.random.default_rng(seed)
    x_grid = np.linspace(-4, 4, 300)

    fig, axes = plt.subplots(1, len(ns), figsize=(4 * len(ns), 4), sharey=True)

    for ax, n in zip(axes, ns):
        samples = np.array([dist_func(n, rng).mean() for _ in range(n_replications)])
        mu = samples.mean() if n == 1 else samples.mean()
        sigma = samples.std()
        standardized = (samples - samples.mean()) / (samples.std() + 1e-12)

        ax.hist(standardized, bins=50, density=True, color="#2c7bb6", alpha=0.7, label="Empirique")
        ax.plot(x_grid, norm.pdf(x_grid), "r-", linewidth=2, label="N(0,1)")
        ax.set_title(f"n = {n}", fontsize=12)
        ax.set_xlabel("Valeur standardisée", fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Densité", fontsize=10)
            ax.legend(fontsize=9)

    fig.suptitle(f"Théorème Central Limite — distribution {dist_name}", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# -----------------------------------------------------------------------
# 4. Intervalles de confiance par bootstrap
# -----------------------------------------------------------------------

def bootstrap_ci(data: np.ndarray, statistic, n_bootstrap: int = 10_000,
                 alpha: float = 0.05, seed: int = None) -> tuple:
    """
    Calcule un intervalle de confiance bootstrap pour une statistique donnée.

    Paramètres
    ----------
    data : np.ndarray
        Données observées.
    statistic : callable
        Fonction appliquée à chaque échantillon bootstrap (ex: np.mean).
    n_bootstrap : int
        Nombre de réplications bootstrap.
    alpha : float
        Niveau de risque (IC à 1-alpha).

    Retourne
    --------
    ci_low, ci_high : float
        Bornes de l'intervalle de confiance.
    boot_stats : np.ndarray
        Distribution bootstrap de la statistique.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.array([
        statistic(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    ci_low = np.percentile(boot_stats, 100 * alpha / 2)
    ci_high = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return ci_low, ci_high, boot_stats


def plot_bootstrap_ci(boot_stats: np.ndarray, ci_low: float, ci_high: float,
                      observed: float, save_path: str = None):
    """
    Visualise la distribution bootstrap et l'intervalle de confiance.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(boot_stats, bins=60, density=True, color="#2c7bb6", alpha=0.7, label="Distribution bootstrap")
    ax.axvline(ci_low, color="#d7191c", linewidth=2, linestyle="--", label=f"IC 95% : [{ci_low:.3f}, {ci_high:.3f}]")
    ax.axvline(ci_high, color="#d7191c", linewidth=2, linestyle="--")
    ax.axvline(observed, color="k", linewidth=2, label=f"Valeur observée : {observed:.3f}")
    ax.set_xlabel("Valeur de la statistique", fontsize=12)
    ax.set_ylabel("Densité", fontsize=12)
    ax.set_title("Intervalle de confiance par bootstrap", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
