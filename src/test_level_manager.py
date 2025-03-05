import shutil
from typing import List

import numpy as np
from models.level_manager import LevelManager, PerformanceMetrics

from config import Config


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
    predictor = LevelManager(model_path=Config.level_manager_path())

    # Create some example scenarios
    scenarios = [
        {
            "name": "High Performance",
            "metrics": [PerformanceMetrics(0.95, 0.92, 0.98, 0.90) for _ in range(5)],
            "level": 2,
        },
        {
            "name": "Struggling Performance",
            "metrics": [PerformanceMetrics(0.25, 0.22, 0.28, 0.20) for _ in range(5)],
            "level": 4,
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
        },
    ]

    # Run predictions for each scenario
    print("\nTesting Level Predictor with Different Scenarios")
    print("=" * shutil.get_terminal_size().columns)

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")

        result = predictor.predict(scenario["metrics"], scenario["level"])

        visualize_metrics(metrics_history=scenario["metrics"], result=result)
