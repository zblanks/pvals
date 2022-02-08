from enum import Enum
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import khatri_rao


class Initializer(Enum):
    """Enum to restrict algorithm inialization distribution types"""

    UNIFORM: str = "uniform"
    GAUSSIAN: str = "gaussian"


class PVALS:
    def __init__(
        self,
        X: np.ndarray,
        r: int,
        eta: float = 1e-7,
        max_iter: int = 100,
        n_restart: int = 10,
        seed: int = 17,
        initializer: Initializer = Initializer.GAUSSIAN,
    ) -> None:
        """Class implementing 'Plain-Vanilla' Alternating Least Squares for
        tensor factorization

        Args:
            X (np.ndarray): Tensor containing relevant data
            r (int): Desired rank to decompose the tensor
            eta (float): Convergence sensitivity parameter
            max_iter (int): Maximum number of algorithm steps
            n_restart (int): Number of random restarts for the ALS algorithm
            seed (int): Random seed for matrix initialization
        """
        self.X = X
        self.i, self.j, self.k = self.X.shape

        self.r = r
        self.eta = eta
        self.max_iter = max_iter
        self.n_restart = n_restart
        self._rng = np.random.default_rng(seed)
        self.initializer = initializer

    def pvals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = np.zeros(shape=(self.X.shape[0], self.r))
        B = np.zeros(shape=(self.X.shape[1], self.r))
        C = np.zeros(shape=(self.X.shape[2], self.r))

        best_obj_val: float = np.inf
        self.best_losses: List[float] = []

        for _ in range(self.n_restart):
            # Random initialization of tensor factorization
            if self.initializer is Initializer.GAUSSIAN:
                tmp_A = self._rng.normal(size=A.shape)
                tmp_B = self._rng.normal(size=B.shape)
                tmp_C = self._rng.normal(size=C.shape)
            elif self.initializer is Initializer.UNIFORM:
                tmp_A = self._rng.random(size=A.shape)
                tmp_B = self._rng.random(size=B.shape)
                tmp_C = self._rng.random(size=C.shape)
            else:
                raise ValueError(
                    "Only considering Gaussian and & Uniform starting points"
                )

            tmp_A, tmp_B, tmp_C, losses = self._run_als(tmp_A, tmp_B, tmp_C)

            if losses[-1] < best_obj_val:
                best_obj_val = losses[-1]
                self.best_losses = losses.copy()
                A = np.copy(tmp_A)
                B = np.copy(tmp_B)
                C = np.copy(tmp_C)

        return A, B, C

    def _run_als(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:

        losses: List[float] = []
        X0 = self._matricize_X(0)
        X1 = self._matricize_X(1)
        X2 = self._matricize_X(2)
        start_obj_val = self._compute_objective(A, B, C)
        losses.append(start_obj_val)

        for _ in range(self.max_iter):
            A = self._solve_least_squares(X0, A, B, C, axis=0)
            B = self._solve_least_squares(X1, A, B, C, axis=1)
            C = self._solve_least_squares(X2, A, B, C, axis=2)
            losses.append(self._compute_objective(A, B, C))

            # Check for ALS convergence
            if abs(losses[-1] - losses[-2]) <= self.eta:
                break

        return A, B, C, losses

    def _compute_objective(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
        X0 = self._matricize_X(axis=0)
        M = khatri_rao(C, B)
        return np.linalg.norm(X0 - M @ A.T) ** 2

    def plot_losses(self) -> None:
        _, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter(np.arange(len(self.best_losses)), self.best_losses, color="black")
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_xlabel("Iteration", fontsize=14)
        ax.set_yscale("log")
        ax.set_title("ALS Best Objective Function Progression", fontsize=20)
        plt.show()

    def _matricize_X(self, axis: int) -> np.ndarray:
        matrix_shape = {
            0: (self.j * self.k, self.i),
            1: (self.i * self.k, self.j),
            2: (self.i * self.j, self.k),
        }

        Y = np.zeros(shape=matrix_shape[axis])

        for f in range(Y.shape[1]):
            Y[:, f] = self.X.take(f, axis=axis).flatten(order="F")

        return Y

    def _solve_least_squares(
        self, X: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, axis: int
    ) -> np.ndarray:

        if axis == 0:
            M = khatri_rao(C, B)
        elif axis == 1:
            M = khatri_rao(C, A)
        else:
            M = khatri_rao(B, A)

        return X.T @ M @ np.linalg.inv(M.T @ M)
