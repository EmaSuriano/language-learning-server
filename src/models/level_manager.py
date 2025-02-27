from stable_baselines3 import PPO
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import shutil
import os

# Get configuration from environment
MODEL_PATH = os.getenv("LEVEL_MANAGER_PATH")


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


def create_sparkline(values: List[float], width: int = 20) -> str:
    """Create a sparkline visualization for a series of values"""
    # Characters for different levels
    chars = "▁▂▃▄▅▆▇█"

    # Normalize values to 0-7 range (length of chars - 1)
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        normalized = [0] * len(values)
    else:
        normalized = [
            (x - min_val) / (max_val - min_val) * (len(chars) - 1) for x in values
        ]

    # Convert to sparkline
    return "".join(chars[round(x)] for x in normalized)


def create_bar(value: float, width: int = 20) -> str:
    """Create a horizontal bar visualization"""
    filled = int(value * width)
    return f"[{'█' * filled}{' ' * (width - filled)}]"


def create_trend_arrow(trend: float) -> str:
    """Create a trend indicator arrow"""
    if trend > 0.1:
        return "↗️ "
    elif trend < -0.1:
        return "↘️ "
    else:
        return "➡️ "


def visualize_metrics(metrics_history: List[PerformanceMetrics], result: dict) -> None:
    """Create a terminal visualization of the metrics"""
    # Get terminal width
    term_width, _ = shutil.get_terminal_size()
    bar_width = min(30, term_width - 40)

    # Calculate daily averages
    daily_avgs = [
        np.mean(
            [metrics.grammar, metrics.vocabulary, metrics.fluency, metrics.objectives]
        )
        for metrics in metrics_history
    ]

    # Create header
    print("\n" + "═" * term_width)
    print(f"Performance Analysis (Current Level: {result['current_level']})")
    print("═" * term_width)

    # Show trend and recommendation
    trend = result["metrics_summary"]["performance_trend"]
    trend_arrow = create_trend_arrow(trend)
    recommendation = result["decision"].upper()
    rec_color = {
        "increase": "\033[92m",  # green
        "maintain": "\033[93m",  # yellow
        "decrease": "\033[91m",  # red
    }.get(result["decision"], "")
    reset_color = "\033[0m"

    print(f"\nTrend: {trend_arrow} ({trend:+.2f})")
    print(f"Recommendation: {rec_color}{recommendation}{reset_color}")

    # Show sparkline of daily averages
    print("\nPerformance Timeline:")
    print(
        f"  {create_sparkline(daily_avgs, width=20)}  ({min(daily_avgs):.2f} - {max(daily_avgs):.2f})"
    )

    # Show individual metrics
    print("\nDetailed Metrics (Latest Day):")
    latest = metrics_history[-1]
    metrics = [
        ("Grammar", latest.grammar),
        ("Vocabulary", latest.vocabulary),
        ("Fluency", latest.fluency),
        ("Objectives", latest.objectives),
    ]

    max_label_len = max(len(label) for label, _ in metrics)
    for label, value in metrics:
        bar = create_bar(value, width=bar_width)
        print(f"  {label.rjust(max_label_len)}: {bar} {value:.2f}")

    print("\nRecent Average Performance:")
    recent_avg = result["metrics_summary"]["recent_average"]
    avg_bar = create_bar(recent_avg, width=bar_width)
    print(f"  {avg_bar} {recent_avg:.2f}")

    print("═" * term_width + "\n")


if __name__ == "__main__":
    # Initialize predictor
    predictor = LevelManager(model_path=MODEL_PATH)

    # Create some example scenarios
    scenarios = [
        {
            "name": "High Performance",
            "metrics": [PerformanceMetrics(0.95, 0.92, 0.98, 0.90) for _ in range(5)],
            "level": 2,
            "result": {},
        },
        {
            "name": "Struggling Performance",
            "metrics": [PerformanceMetrics(0.25, 0.22, 0.28, 0.20) for _ in range(5)],
            "level": 4,
            "result": {},
        },
        {
            "name": "Improving Performance",
            "metrics": [
                PerformanceMetrics(0.65, 0.68, 0.72, 0.70),
                PerformanceMetrics(0.75, 0.78, 0.82, 0.80),
                PerformanceMetrics(0.85, 0.88, 0.92, 0.90),
                PerformanceMetrics(0.90, 0.92, 0.94, 0.93),
                PerformanceMetrics(0.95, 0.96, 0.98, 0.97),
            ],
            "level": 3,
            "result": {},
        },
        {
            "name": "Decreasing Performance",
            "metrics": [
                PerformanceMetrics(0.75, 0.78, 0.82, 0.80),
                PerformanceMetrics(0.65, 0.68, 0.72, 0.70),
                PerformanceMetrics(0.55, 0.58, 0.62, 0.60),
                PerformanceMetrics(0.45, 0.48, 0.52, 0.50),
                PerformanceMetrics(0.35, 0.38, 0.42, 0.40),
            ],
            "level": 3,
            "result": {},
        },
    ]

    # Run predictions for each scenario
    print("\nTesting Level Predictor with Different Scenarios")
    print("=" * shutil.get_terminal_size().columns)

    for scenario in scenarios:
        result = predictor.predict(scenario["metrics"], scenario["level"])
        scenario["result"] = result

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        visualize_metrics(
            metrics_history=scenario["metrics"], result=scenario["result"]
        )
