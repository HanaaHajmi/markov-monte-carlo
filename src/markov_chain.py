"""
Chaînes de Markov à temps discret
----------------------------------
Implémentation from scratch : matrice de transition, distribution stationnaire,
ergodicité, vitesse de mélange.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class MarkovChain:
    """
    Chaîne de Markov à temps discret homogène.

    Paramètres
    ----------
    P : np.ndarray, shape (n, n)
        Matrice de transition (lignes somment à 1).
    states : list, optional
        Noms des états.
    """

    def __init__(self, P: np.ndarray, states: list = None):
        P = np.array(P, dtype=float)
        assert P.ndim == 2 and P.shape[0] == P.shape[1], "P doit être carrée"
        assert np.allclose(P.sum(axis=1), 1), "Les lignes de P doivent sommer à 1"
        assert (P >= 0).all(), "Les probabilités doivent être positives"

        self.P = P
        self.n = P.shape[0]
        self.states = states if states else [str(i) for i in range(self.n)]

    # ------------------------------------------------------------------
    # Distribution stationnaire
    # ------------------------------------------------------------------

    def stationary_distribution(self) -> np.ndarray:
        """
        Calcule la distribution stationnaire π telle que πP = π, Σπ = 1.
        Méthode : valeurs propres de P^T.
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        # La valeur propre 1 correspond à la distribution stationnaire
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = eigenvectors[:, idx].real
        pi = pi / pi.sum()
        return pi

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, n_steps: int, start_state: int = 0, seed: int = None) -> np.ndarray:
        """
        Simule une trajectoire de longueur n_steps.

        Retourne
        --------
        trajectory : np.ndarray, shape (n_steps + 1,)
            Indices des états visités.
        """
        rng = np.random.default_rng(seed)
        trajectory = np.empty(n_steps + 1, dtype=int)
        trajectory[0] = start_state

        for t in range(n_steps):
            trajectory[t + 1] = rng.choice(self.n, p=self.P[trajectory[t]])

        return trajectory

    # ------------------------------------------------------------------
    # Convergence vers la distribution stationnaire
    # ------------------------------------------------------------------

    def distribution_at_step(self, mu0: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Calcule la distribution μ_n = μ_0 P^n par itération matricielle.

        Paramètres
        ----------
        mu0 : np.ndarray
            Distribution initiale.
        n_steps : int
            Nombre de pas.
        """
        mu = np.array(mu0, dtype=float)
        for _ in range(n_steps):
            mu = mu @ self.P
        return mu

    def convergence_to_stationary(self, mu0: np.ndarray, max_steps: int = 100) -> np.ndarray:
        """
        Calcule la distance en variation totale entre μ_n et π pour chaque pas.

        Retourne
        --------
        tvd : np.ndarray, shape (max_steps,)
            Distance en variation totale à chaque étape.
        """
        pi = self.stationary_distribution()
        mu = np.array(mu0, dtype=float)
        tvd = np.empty(max_steps)

        for t in range(max_steps):
            tvd[t] = 0.5 * np.sum(np.abs(mu - pi))
            mu = mu @ self.P

        return tvd

    # ------------------------------------------------------------------
    # Ergodicité : fréquences empiriques
    # ------------------------------------------------------------------

    def empirical_frequencies(self, n_steps: int, start_state: int = 0, seed: int = None) -> np.ndarray:
        """
        Calcule les fréquences empiriques de visite de chaque état
        sur une trajectoire simulée.
        """
        traj = self.simulate(n_steps, start_state, seed)
        counts = np.bincount(traj, minlength=self.n)
        return counts / counts.sum()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_convergence(self, mu0: np.ndarray, max_steps: int = 80, save_path: str = None):
        """
        Trace la distance en variation totale ||μ_n - π||_TV en fonction de n.
        """
        tvd = self.convergence_to_stationary(mu0, max_steps)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(tvd, color="#2c7bb6", linewidth=2)
        ax.set_xlabel("Nombre de pas", fontsize=12)
        ax.set_ylabel(r"$\|\mu_n - \pi\|_{TV}$  (échelle log)", fontsize=12)
        ax.set_title("Convergence vers la distribution stationnaire", fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_trajectory(self, n_steps: int = 100, start_state: int = 0, seed: int = 42, save_path: str = None):
        """
        Trace une trajectoire simulée.
        """
        traj = self.simulate(n_steps, start_state, seed)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.step(range(len(traj)), traj, where="post", color="#d7191c", linewidth=1.2)
        ax.set_yticks(range(self.n))
        ax.set_yticklabels(self.states)
        ax.set_xlabel("Pas de temps", fontsize=12)
        ax.set_ylabel("État", fontsize=12)
        ax.set_title(f"Trajectoire simulée ({n_steps} pas)", fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_stationary_comparison(self, n_steps: int = 10000, seed: int = 0, save_path: str = None):
        """
        Compare la distribution stationnaire théorique et les fréquences empiriques.
        """
        pi = self.stationary_distribution()
        freq = self.empirical_frequencies(n_steps, seed=seed)

        x = np.arange(self.n)
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - width/2, pi, width, label="π théorique", color="#2c7bb6", alpha=0.85)
        ax.bar(x + width/2, freq, width, label=f"Fréquences empiriques (n={n_steps})", color="#d7191c", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(self.states)
        ax.set_ylabel("Probabilité", fontsize=12)
        ax.set_title("Distribution stationnaire : théorie vs simulation", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
