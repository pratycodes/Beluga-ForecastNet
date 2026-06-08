"""Beluga Whale Optimization for hyperparameter search."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class BWO:
    """Beluga Whale Optimizer over continuous bounded positions."""

    def __init__(
        self,
        population_size: int,
        max_iter: int,
        dimension: int,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        seed: int | None = None,
        verbose: bool = True,
        no_improvement_patience: int | None = None,
        min_delta: float = 0.0,
    ) -> None:
        if population_size < 1:
            raise ValueError("population_size must be at least 1")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if dimension < 1:
            raise ValueError("dimension must be at least 1")

        self.population_size = population_size
        self.max_iter = max_iter
        self.dimension = dimension
        self.lb = np.array(lower_bound, dtype=float)
        self.ub = np.array(upper_bound, dtype=float)
        self.verbose = verbose
        self.no_improvement_patience = no_improvement_patience
        self.min_delta = min_delta
        self.rng = np.random.default_rng(seed)
        self.best_history: list[float] = []

        if self.lb.shape != (dimension,) or self.ub.shape != (dimension,):
            raise ValueError("Bounds must match optimizer dimension")
        if np.any(self.lb > self.ub):
            raise ValueError("Each lower bound must be <= upper bound")

        self.population = self.rng.uniform(
            self.lb,
            self.ub,
            (population_size, dimension),
        )
        logger.info(
            "Initialized BWO population_size=%s max_iter=%s dimension=%s",
            population_size,
            max_iter,
            dimension,
        )

    def exploration(
        self,
        xi: np.ndarray,
        xr: np.ndarray,
        iteration: int,
    ) -> np.ndarray:
        """Exploration update from the BWO search phase."""

        _ = iteration
        new_x = xi.copy()
        for index in range(self.dimension):
            r1 = self.rng.random()
            r2 = self.rng.random()
            phase = 2 * np.pi * r2
            movement = (xr[index] - xi[index]) * (1 + r1)
            if index % 2 == 0:
                new_x[index] = xi[index] + movement * np.sin(phase)
            else:
                new_x[index] = xi[index] + movement * np.cos(phase)
        return new_x

    def exploitation(
        self,
        xi: np.ndarray,
        xr: np.ndarray,
        xbest: np.ndarray,
        iteration: int,
    ) -> np.ndarray:
        """Exploitation update around the current best candidate."""

        r3 = self.rng.random()
        r4 = self.rng.random()
        c1 = 2 * r4 * (1 - iteration / self.max_iter)
        lf = self.levy_flight()
        return r3 * xbest - r4 * xi + c1 * lf * (xr - xi)

    def whale_fall(
        self,
        xi: np.ndarray,
        xr: np.ndarray,
        iteration: int,
    ) -> np.ndarray:
        """Whale-fall update used to diversify the population."""

        r5 = self.rng.random()
        r6 = self.rng.random()
        r7 = self.rng.random()
        wf = 0.1 - (0.05 * iteration / self.max_iter)
        c2 = 2 * wf * self.population_size
        step = (self.ub - self.lb) * np.exp(-c2 * iteration / self.max_iter)
        return r5 * xi - r6 * xr + r7 * step

    def levy_flight(self, beta: float = 1.5) -> float:
        """Sample a Levy-flight step."""

        sigma = (
            (
                math.gamma(1 + beta)
                * math.sin(math.pi * beta / 2)
                / (
                    math.gamma((1 + beta) / 2)
                    * beta
                    * 2 ** ((beta - 1) / 2)
                )
            )
            ** (1 / beta)
        )
        u = self.rng.normal(0, sigma)
        v = self.rng.normal(0, 1)
        return 0.05 * u / abs(v) ** (1 / beta)

    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
    ) -> tuple[np.ndarray, float]:
        """Optimize a fitness function and return best position and score."""

        fitness_values = np.array(
            [self._safe_fitness(fitness_fn, position) for position in self.population],
            dtype=float,
        )
        best_index = int(np.argmin(fitness_values))
        best_position = self.population[best_index].copy()
        best_fitness = float(fitness_values[best_index])
        logger.info("Initial BWO best fitness=%.6f", best_fitness)
        stagnant_iterations = 0

        for iteration in range(self.max_iter):
            previous_best = best_fitness
            balance_factor = self.rng.random() * (
                1 - iteration / (2 * self.max_iter)
            )
            for index in range(self.population_size):
                xi = self.population[index]
                random_index = int(self.rng.integers(self.population_size))
                xr = self.population[random_index]

                if balance_factor > 0.5:
                    candidate = self.exploration(xi, xr, iteration)
                else:
                    candidate = self.exploitation(xi, xr, best_position, iteration)

                fall_probability = 0.1 - 0.05 * iteration / self.max_iter
                if self.rng.random() < fall_probability:
                    candidate = self.whale_fall(candidate, xr, iteration)

                candidate = np.clip(candidate, self.lb, self.ub)
                candidate_fitness = self._safe_fitness(fitness_fn, candidate)

                if candidate_fitness < fitness_values[index]:
                    self.population[index] = candidate
                    fitness_values[index] = candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_fitness = float(candidate_fitness)
                        best_position = candidate.copy()

            self.best_history.append(best_fitness)
            if self.verbose:
                logger.info(
                    f"Iter {iteration + 1}/{self.max_iter} "
                    f"Best fitness={best_fitness:.6f}"
                )
            if best_fitness < previous_best - self.min_delta:
                stagnant_iterations = 0
            else:
                stagnant_iterations += 1
            if (
                self.no_improvement_patience is not None
                and stagnant_iterations >= self.no_improvement_patience
            ):
                logger.info(
                    "Stopping BWO early after %s stagnant iterations",
                    stagnant_iterations,
                )
                break

        logger.info("Finished BWO best_fitness=%.6f best_position=%s", best_fitness, best_position)
        return best_position, best_fitness

    @staticmethod
    def _safe_fitness(
        fitness_fn: Callable[[np.ndarray], float],
        position: np.ndarray,
    ) -> float:
        value = float(fitness_fn(position))
        if not np.isfinite(value):
            return float("inf")
        return value
