import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict

from stable_baselines3 import PPO
import os

from config import Config


def create_test_scenarios() -> List[Dict]:
    """Create a comprehensive set of test scenarios"""
    return [
        {
            "name": "very_high_performance",
            "metrics": [(0.95, 0.92, 0.98, 0.90)] * 5,
            "level": 2,
            "expected": "increase",
            "explanation": "Consistently very high performance should trigger promotion",
            "category": "clear_promotion",
        },
        {
            "name": "very_low_performance",
            "metrics": [(0.25, 0.22, 0.28, 0.20)] * 5,
            "level": 4,
            "expected": "decrease",
            "explanation": "Consistently very low performance should trigger demotion",
            "category": "clear_demotion",
        },
        {
            "name": "steady_medium_performance",
            "metrics": [(0.65, 0.63, 0.67, 0.64)] * 5,
            "level": 3,
            "expected": "maintain",
            "explanation": "Stable medium performance should maintain level",
            "category": "maintenance",
        },
        {
            "name": "rapid_improvement",
            "metrics": [
                (0.65, 0.68, 0.72, 0.70),
                (0.75, 0.78, 0.82, 0.80),
                (0.85, 0.88, 0.92, 0.90),
                (0.95, 0.98, 1.00, 1.00),
                (1.00, 1.00, 1.00, 1.00),
            ],
            "level": 3,
            "expected": "increase",
            "explanation": "Quick improvement should lead to promotion",
            "category": "improvement",
        },
        {
            "name": "gradual_improvement",
            "metrics": [
                (0.60, 0.62, 0.58, 0.61),
                (0.65, 0.67, 0.63, 0.66),
                (0.70, 0.72, 0.68, 0.71),
                (0.75, 0.77, 0.73, 0.76),
                (0.80, 0.82, 0.78, 0.81),
            ],
            "level": 2,
            "expected": "increase",
            "explanation": "Steady improvement should lead to promotion",
            "category": "improvement",
        },
        {
            "name": "rapid_decline",
            "metrics": [
                (0.65, 0.68, 0.72, 0.70),
                (0.55, 0.58, 0.62, 0.60),
                (0.45, 0.48, 0.52, 0.50),
                (0.35, 0.38, 0.42, 0.40),
                (0.25, 0.28, 0.32, 0.30),
            ],
            "level": 3,
            "expected": "decrease",
            "explanation": "Quick decline should lead to demotion",
            "category": "decline",
        },
        {
            "name": "recovery_after_drop",
            "metrics": [
                (0.75, 0.73, 0.77, 0.74),
                (0.45, 0.43, 0.47, 0.44),
                (0.55, 0.53, 0.57, 0.54),
                (0.65, 0.63, 0.67, 0.64),
                (0.75, 0.73, 0.77, 0.74),
            ],
            "level": 3,
            "expected": "maintain",
            "explanation": "Recovery to original level should maintain",
            "category": "maintenance",
        },
        {
            "name": "inconsistent_high",
            "metrics": [
                (0.85, 0.82, 0.88, 0.84),
                (0.65, 0.68, 0.62, 0.64),
                (0.95, 0.92, 0.98, 0.94),
                (0.75, 0.72, 0.78, 0.74),
                (0.85, 0.82, 0.88, 0.84),
            ],
            "level": 4,
            "expected": "maintain",
            "explanation": "High but inconsistent performance should maintain level",
            "category": "maintenance",
        },
        {
            "name": "mixed_metrics",
            "metrics": [
                (0.85, 0.55, 0.75, 0.65),
                (0.87, 0.57, 0.77, 0.67),
                (0.89, 0.59, 0.79, 0.69),
                (0.91, 0.61, 0.81, 0.71),
                (0.93, 0.63, 0.83, 0.73),
            ],
            "level": 3,
            "expected": "maintain",
            "explanation": "Mixed performance across metrics should maintain level",
            "category": "maintenance",
        },
        {
            "name": "plateau_with_breakthrough",
            "metrics": [
                (0.65, 0.63, 0.67, 0.64),
                (0.67, 0.65, 0.69, 0.66),
                (0.85, 0.83, 0.87, 0.84),
                (0.87, 0.85, 0.89, 0.86),
                (0.89, 0.87, 0.91, 0.88),
            ],
            "level": 2,
            "expected": "increase",
            "explanation": "Breakthrough after plateau should lead to promotion",
            "category": "improvement",
        },
    ]


def evaluate_scenarios(model, scenarios: List[Dict]) -> Dict:
    """Evaluate model performance on predefined scenarios"""
    results = {
        "scenario_results": [],
        "category_accuracy": {},
        "confusion_matrix": {
            "decrease": {"decrease": 0, "maintain": 0, "increase": 0},
            "maintain": {"decrease": 0, "maintain": 0, "increase": 0},
            "increase": {"decrease": 0, "maintain": 0, "increase": 0},
        },
        "category_counts": {},
    }

    for scenario in scenarios:
        # Prepare observation
        flat_metrics = [m for day in scenario["metrics"] for m in day]
        normalized_level = (scenario["level"] - 1) / 4
        obs = np.array(flat_metrics + [normalized_level], dtype=np.float32)

        # Get model prediction
        action, _ = model.predict(obs)
        action_scalar = action.item() if isinstance(action, np.ndarray) else action
        prediction = ["decrease", "maintain", "increase"][action_scalar]

        # Record results
        correct = prediction == scenario["expected"]
        results["scenario_results"].append(
            {
                "name": scenario["name"],
                "category": scenario["category"],
                "expected": scenario["expected"],
                "predicted": prediction,
                "correct": correct,
            }
        )

        # Update confusion matrix
        results["confusion_matrix"][scenario["expected"]][prediction] += 1

        # Update category statistics
        if scenario["category"] not in results["category_accuracy"]:
            results["category_accuracy"][scenario["category"]] = {
                "correct": 0,
                "total": 0,
            }
        results["category_accuracy"][scenario["category"]]["total"] += 1
        if correct:
            results["category_accuracy"][scenario["category"]]["correct"] += 1

        # Update category counts
        if scenario["category"] not in results["category_counts"]:
            results["category_counts"][scenario["category"]] = 0
        results["category_counts"][scenario["category"]] += 1

    return results


def plot_scenario_results(results: Dict, save_path: str):
    """Create visualizations for scenario test results"""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Model Scenario Test Results", fontsize=16, y=0.95)

    # 1. Category Accuracy Bar Plot
    ax1 = plt.subplot(2, 2, 1)
    categories = []
    accuracies = []
    for category, stats in results["category_accuracy"].items():
        categories.append(category.replace("_", " ").title())
        accuracies.append(stats["correct"] / stats["total"] * 100)

    bars = ax1.bar(categories, accuracies)
    ax1.set_title("Accuracy by Scenario Category")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim([0, 100])
    plt.xticks(rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    # 2. Confusion Matrix Heatmap
    ax2 = plt.subplot(2, 2, 2)
    conf_matrix = np.zeros((3, 3))
    labels = ["decrease", "maintain", "increase"]
    for i, expected in enumerate(labels):
        for j, predicted in enumerate(labels):
            conf_matrix[i, j] = results["confusion_matrix"][expected][predicted]

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax2,
    )
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Expected")

    # 3. Individual Scenario Results
    ax3 = plt.subplot(2, 1, 2)
    scenario_names = [
        r["name"].replace("_", " ").title() for r in results["scenario_results"]
    ]
    correct = [1 if r["correct"] else 0 for r in results["scenario_results"]]

    colors = ["g" if c else "r" for c in correct]
    bars = ax3.bar(scenario_names, [1] * len(scenario_names), color=colors, alpha=0.6)
    ax3.set_title("Individual Scenario Results")
    ax3.set_ylabel("Result")
    plt.xticks(rotation=45, ha="right")

    # Add predicted/expected labels
    for i, r in enumerate(results["scenario_results"]):
        ax3.text(
            i, 0.5, f"E: {r['expected']}\nP: {r['predicted']}", ha="center", va="center"
        )

    # Adjust layout and save if path provided
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plots saved to {save_path}")

    return fig


def load_model(path="level_adjustment_model_final"):
    """Load a saved model"""
    return PPO.load(path)


def test_model_scenarios(
    model_path: str = "level_adjustment_model_final",
    save_dir: str = "scenario_test_results",
):
    """Run complete scenario testing and generate visualizations"""
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = load_model(model_path)

    # Create and evaluate scenarios
    scenarios = create_test_scenarios()
    results = evaluate_scenarios(model, scenarios)

    # Generate plots
    plot_scenario_results(
        results, save_path=os.path.join(save_dir, "scenario_results.png")
    )

    # Print summary
    print("\nScenario Test Results Summary:")
    print("=" * 50)
    print("\nCategory Performance:")
    for category, stats in results["category_accuracy"].items():
        accuracy = (stats["correct"] / stats["total"]) * 100
        print(
            f"{category.replace('_', ' ').title()}: "
            f"{accuracy:.1f}% ({stats['correct']}/{stats['total']})"
        )

    print("\nIndividual Scenario Results:")
    for result in results["scenario_results"]:
        status = "✓" if result["correct"] else "✗"
        print(
            f"{status} {result['name']}: Expected {result['expected']}, "
            f"Got {result['predicted']}"
        )

    return results, scenarios


if __name__ == "__main__":
    model_path = Config.level_manager_path()
    save_dir = os.path.join(os.path.dirname(__file__), "scenario_test_results")

    results, scenarios = test_model_scenarios(model_path=model_path, save_dir=save_dir)
