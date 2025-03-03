from stable_baselines3 import PPO
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import shutil


@dataclass
class PerformanceMetrics:
    """Class to hold performance metrics for a single day"""

    grammar: float
    vocabulary: float
    fluency: float
    objectives: float

    def validate(self):
        """Validate that all metrics are between 0 and 1"""
        for field, value in self.__dict__.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{field} must be between 0 and 1, got {value}")


class LevelManager:
    def __init__(self, model_path: str):
        self.model = PPO.load(model_path)

    def predict(
        self, metrics_history: List[PerformanceMetrics], current_level: int
    ) -> Dict:
        """Predict whether to adjust the level based on performance history

        Args:
            metrics_history (List[PerformanceMetrics]): List of daily performance metrics
            current_level (int): Current level (1-5)

        Returns:
            Dict containing prediction details
        """
        # Validate inputs
        if not 1 <= current_level <= 5:
            raise ValueError(
                f"Current level must be between 1 and 5, got {current_level}"
            )
        if len(metrics_history) != 5:
            raise ValueError(
                f"Metrics history must contain exactly 5 days, got {len(metrics_history)}"
            )

        # Validate each day's metrics
        for day_metrics in metrics_history:
            day_metrics.validate()

        # Prepare observation for model
        flat_metrics = []
        for metrics in metrics_history:
            flat_metrics.extend(
                [
                    metrics.grammar,
                    metrics.vocabulary,
                    metrics.fluency,
                    metrics.objectives,
                ]
            )

        # Add normalized level
        normalized_level = (current_level - 1) / 4
        observation = np.array(flat_metrics + [normalized_level], dtype=np.float32)

        # Get model prediction
        action, _ = self.model.predict(observation, deterministic=True)
        action_scalar = action.item() if isinstance(action, np.ndarray) else action
        decision = ["decrease", "maintain", "increase"][action_scalar]

        # Calculate average performance metrics
        recent_avg = np.mean(
            [
                np.mean(
                    [
                        metrics.grammar,
                        metrics.vocabulary,
                        metrics.fluency,
                        metrics.objectives,
                    ]
                )
                for metrics in metrics_history[-2:]  # Last 2 days
            ]
        )

        # Calculate trend
        avg_start = np.mean(
            [
                np.mean(
                    [
                        metrics.grammar,
                        metrics.vocabulary,
                        metrics.fluency,
                        metrics.objectives,
                    ]
                )
                for metrics in metrics_history[:2]  # First 2 days
            ]
        )
        avg_end = np.mean(
            [
                np.mean(
                    [
                        metrics.grammar,
                        metrics.vocabulary,
                        metrics.fluency,
                        metrics.objectives,
                    ]
                )
                for metrics in metrics_history[-2:]  # Last 2 days
            ]
        )
        trend = avg_end - avg_start

        return {
            "decision": decision,
            "current_level": current_level,
            "metrics_summary": {
                "recent_average": float(recent_avg),
                "performance_trend": float(trend),
                "days_analyzed": len(metrics_history),
            },
        }
