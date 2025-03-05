import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import random
from stable_baselines3.common.callbacks import EvalCallback

from config import Config


class LevelAdjustmentEnv(gym.Env):
    """Custom Environment for level adjustment based on user performance"""

    def __init__(self):
        super().__init__()

        # Define action and observation spaces
        # Actions: decrease (0), maintain (1), increase (2)
        self.action_space = spaces.Discrete(3)

        # Observation space: 20 metrics (4 metrics × 5 days) + current level
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(21,), dtype=np.float32
        )

        self.current_episode = 0
        self.max_episodes = 1000

    def reset(self, seed=None):
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        self.current_episode += 1

        # More sophisticated scenario selection based on training phase
        training_progress = min(1.0, self.current_episode / self.max_episodes)

        if training_progress < 0.2:
            # Early training: focus on clear patterns
            scenario_types = [
                "high_performance",
                "low_performance",
                "clear_improvement",
                "clear_decline",
            ]
        elif training_progress < 0.5:
            # Mid training: introduce more complex patterns
            scenario_types = [
                "high_performance",
                "low_performance",
                "gradual_improvement",
                "inconsistent_decline",
                "plateau_with_breakthrough",
                "recovery_after_setback",
                "cyclical",
            ]
        else:
            # Late training: focus on edge cases and complex patterns
            scenario_types = [
                "mixed_metrics",
                "volatile_improvement",
                "slow_decline",
                "plateau_with_minor_changes",
                "inconsistent_with_trend",
                "level_inappropriate",
                "stress_pattern",
                "fatigue_pattern",
            ]

        scenario_type = random.choice(scenario_types)
        metrics, level = self._generate_scenario(scenario_type)
        self.current_metrics = metrics
        self.current_level = level
        self.expected_action = self._determine_expected_action(metrics, level)

        # Flatten metrics and combine with normalized level
        flat_metrics = [m for day in metrics for m in day]
        normalized_level = (level - 1) / 4  # Normalize level to [0,1]
        observation = np.array(flat_metrics + [normalized_level], dtype=np.float32)

        return observation, {}

    def step(self, action):
        """Execute one time step within the environment"""
        # Convert numpy array to scalar if necessary
        if isinstance(action, np.ndarray):
            action = action.item()

        # Map actions to decisions
        action_map = {0: "decrease", 1: "maintain", 2: "increase"}
        decision = action_map[action]

        # Calculate reward
        reward = self._calculate_reward(decision)

        # Episode ends after each decision
        done = True

        # Return current observation
        flat_metrics = [m for day in self.current_metrics for m in day]
        normalized_level = (self.current_level - 1) / 4
        observation = np.array(flat_metrics + [normalized_level], dtype=np.float32)

        return observation, reward, done, False, {}

    def _generate_scenario(self, scenario_type):
        """Generate a scenario based on type with realistic learning patterns"""

        def apply_noise(value, noise_level=0.05, noise_type="uniform"):
            if noise_type == "uniform":
                noise = random.uniform(-noise_level, noise_level)
            elif noise_type == "gaussian":
                noise = random.gauss(0, noise_level / 2)
            elif noise_type == "biased":
                # Biased towards negative noise (simulating fatigue)
                noise = random.uniform(-noise_level * 1.5, noise_level * 0.5)
            return max(0.0, min(1.0, value + noise))

        def generate_learning_curve(
            start, end, days, curve_type="linear", modifiers=None
        ):
            modifiers = modifiers or {}
            base_curve = []

            if curve_type == "linear":
                base_curve = [
                    start + (end - start) * (i / (days - 1)) for i in range(days)
                ]
            elif curve_type == "exponential":
                base_curve = [
                    start + (end - start) * (1 - np.exp(-2 * i / days))
                    for i in range(days)
                ]
            elif curve_type == "logarithmic":
                base_curve = [
                    start + (end - start) * (np.log(1 + 7 * i / days))
                    for i in range(days)
                ]
            elif curve_type == "plateau":
                mid_point = modifiers.get("mid_point", days // 2)
                steepness = modifiers.get("steepness", 0.5)
                base_curve = [
                    start
                    + (end - start) * (1 / (1 + np.exp(-steepness * (i - mid_point))))
                    for i in range(days)
                ]
            elif curve_type == "cyclical":
                frequency = modifiers.get("frequency", 2)
                amplitude = modifiers.get("amplitude", 0.1)
                base_curve = [
                    start
                    + (end - start) * (i / (days - 1))
                    + amplitude * np.sin(2 * np.pi * frequency * i / days)
                    for i in range(days)
                ]
            elif curve_type == "stress":
                # Simulates performance under stress - good starts, declining middle, potential recovery
                peak_day = modifiers.get("peak_day", days // 3)
                recovery_strength = modifiers.get("recovery_strength", 0.7)
                base_curve = []
                for i in range(days):
                    if i < peak_day:
                        val = start + (end - start) * (i / peak_day)
                    else:
                        decline = (
                            1 - (i - peak_day) / (days - peak_day)
                        ) * recovery_strength
                        val = end - (end - start) * (1 - decline)
                    base_curve.append(val)
            elif curve_type == "breakthrough":
                # Simulates sudden improvement after plateau
                breakthrough_day = modifiers.get("breakthrough_day", days * 2 // 3)
                base_curve = []
                for i in range(days):
                    if i < breakthrough_day:
                        val = start + (end - start) * 0.3
                    else:
                        val = start + (end - start) * (
                            0.3
                            + 0.7 * (i - breakthrough_day) / (days - breakthrough_day)
                        )
                    base_curve.append(val)

            # Apply any time-based modifiers
            if modifiers.get("fatigue_effect"):
                base_curve = [
                    v * (1 - 0.1 * (i / (days - 1))) for i, v in enumerate(base_curve)
                ]
            if modifiers.get("warmup_effect"):
                base_curve = [
                    v * (0.8 + 0.2 * (i / (days - 1))) for i, v in enumerate(base_curve)
                ]

            return base_curve

        level = random.randint(1, 5)

        # Initialize level based on scenario type
        level = random.randint(1, 5)

        # Correlation matrix for metrics (grammar, vocabulary, fluency, objectives)
        correlation_matrix = {
            "grammar": {"vocabulary": 0.7, "fluency": 0.5, "objectives": 0.3},
            "vocabulary": {"fluency": 0.4, "objectives": 0.3},
            "fluency": {"objectives": 0.6},
        }

        if scenario_type == "mixed_metrics":
            # Different patterns for different metrics
            patterns = [
                ("exponential", 0.4, 0.8),  # grammar
                ("plateau", 0.5, 0.7),  # vocabulary
                ("cyclical", 0.6, 0.75),  # fluency
                ("linear", 0.45, 0.85),  # objectives
            ]
            metrics = []
            for day in range(5):
                day_metrics = []
                for idx, (pattern, start, end) in enumerate(patterns):
                    modifiers = {
                        "fatigue_effect": random.random() < 0.3,
                        "warmup_effect": random.random() < 0.2,
                    }
                    curve = generate_learning_curve(start, end, 5, pattern, modifiers)
                    metric_value = apply_noise(curve[day], 0.05, "gaussian")
                    day_metrics.append(metric_value)
                metrics.append(tuple(day_metrics))

        elif scenario_type == "stress_pattern":
            # Simulates performance under stress conditions
            base_curves = []
            for _ in range(4):
                curve = generate_learning_curve(
                    0.7,
                    0.4,
                    5,
                    "stress",
                    {
                        "peak_day": random.randint(1, 2),
                        "recovery_strength": random.uniform(0.5, 0.9),
                    },
                )
                base_curves.append(curve)

            metrics = []
            for day in range(5):
                day_metrics = []
                for metric_idx in range(4):
                    value = apply_noise(base_curves[metric_idx][day], 0.07, "biased")
                    day_metrics.append(value)
                metrics.append(tuple(day_metrics))

        elif scenario_type == "fatigue_pattern":
            # Shows decline in later sessions
            start_values = [0.8, 0.75, 0.85, 0.7]
            metrics = []
            for day in range(5):
                day_metrics = []
                fatigue_factor = 1.0 - (day / 10)  # Progressive fatigue
                for metric_idx, base in enumerate(start_values):
                    value = base * fatigue_factor
                    value = apply_noise(value, 0.06, "biased")
                    day_metrics.append(value)
                metrics.append(tuple(day_metrics))

        elif scenario_type == "plateau_with_breakthrough":
            # Plateau followed by sudden improvement
            metrics = []
            breakthrough_day = random.randint(2, 3)
            for day in range(5):
                day_metrics = []
                for metric_idx in range(4):
                    if day < breakthrough_day:
                        base = 0.6
                    else:
                        improvement = (day - breakthrough_day + 1) * 0.15
                        base = min(0.95, 0.6 + improvement)
                    value = apply_noise(base, 0.04, "gaussian")
                    day_metrics.append(value)
                metrics.append(tuple(day_metrics))

        elif scenario_type == "volatile_improvement":
            # Improvement with high volatility
            base_curve = generate_learning_curve(0.4, 0.8, 5, "exponential")
            metrics = []
            for day in range(5):
                day_metrics = []
                volatility = (
                    0.15 if day > 0 else 0.05
                )  # Higher volatility after first day
                for metric_idx in range(4):
                    value = apply_noise(base_curve[day], volatility, "gaussian")
                    day_metrics.append(value)
                metrics.append(tuple(day_metrics))

        elif scenario_type == "level_inappropriate":
            # Performance suggesting wrong level assignment
            if random.random() < 0.5:
                # Too easy
                level = max(2, level)  # Ensure we can decrease
                metrics = []
                for _ in range(5):
                    day_metrics = [apply_noise(0.95, 0.03) for _ in range(4)]
                    metrics.append(tuple(day_metrics))
            else:
                # Too difficult
                level = min(4, level)  # Ensure we can increase
                metrics = []
                for _ in range(5):
                    day_metrics = [apply_noise(0.25, 0.05) for _ in range(4)]
                    metrics.append(tuple(day_metrics))

        elif scenario_type == "high_performance":
            # Consistently high performance with slight variations
            base_curve = generate_learning_curve(0.85, 0.95, 5, "plateau")
            metrics = []
            for base in base_curve:
                day_metrics = []
                for _ in range(4):  # 4 metrics per day
                    # Different metrics might have slightly different patterns
                    metric_value = apply_noise(base, 0.05)
                    day_metrics.append(metric_value)
                metrics.append(tuple(day_metrics))

        elif scenario_type == "low_performance":
            # Struggling pattern with occasional small improvements
            base_curve = generate_learning_curve(0.3, 0.25, 5, "linear")
            metrics = []
            for base in base_curve:
                day_metrics = []
                for metric_idx in range(4):
                    # Some metrics might be slightly better than others
                    metric_base = base + (
                        0.05 if metric_idx == 2 else 0
                    )  # e.g., fluency might be better
                    metric_value = apply_noise(metric_base, 0.07)
                    day_metrics.append(metric_value)
                metrics.append(tuple(day_metrics))

        elif scenario_type == "improving":
            # Different improvement patterns for different metrics
            metrics = []
            improvement_patterns = [
                generate_learning_curve(
                    0.4, 0.8, 5, "exponential"
                ),  # rapid improvement
                generate_learning_curve(0.45, 0.75, 5, "linear"),  # steady improvement
                generate_learning_curve(0.5, 0.85, 5, "logarithmic"),  # early gains
                generate_learning_curve(
                    0.35, 0.7, 5, "plateau"
                ),  # improvement with plateau
            ]

            for day in range(5):
                day_metrics = []
                for metric_idx in range(4):
                    base = improvement_patterns[metric_idx][day]
                    metric_value = apply_noise(base, 0.03)
                    day_metrics.append(metric_value)
                metrics.append(tuple(day_metrics))

        elif scenario_type == "declining":
            # Different decline patterns with potential recovery attempts
            base_curve = generate_learning_curve(0.8, 0.4, 5, "exponential")
            recovery_day = random.randint(2, 4)
            metrics = []

            for day in range(5):
                day_metrics = []
                for metric_idx in range(4):
                    base = base_curve[day]
                    # Simulate recovery attempt
                    if day == recovery_day:
                        base += 0.15
                    metric_value = apply_noise(base, 0.06)
                    day_metrics.append(metric_value)
                metrics.append(tuple(day_metrics))

        else:  # inconsistent
            # Generate realistic inconsistent patterns
            metrics = []
            base_values = [
                random.uniform(0.4, 0.6) for _ in range(4)
            ]  # base for each metric
            volatilities = [
                random.uniform(0.1, 0.2) for _ in range(4)
            ]  # different volatility per metric

            for _ in range(5):
                day_metrics = []
                for metric_idx in range(4):
                    # Each metric varies around its base with its own volatility
                    base = base_values[metric_idx]
                    volatility = volatilities[metric_idx]
                    metric_value = apply_noise(base, volatility)
                    # Occasionally add significant jumps
                    if random.random() < 0.2:
                        metric_value += random.choice([-0.2, 0.2])
                    metric_value = max(0.0, min(1.0, metric_value))
                    day_metrics.append(metric_value)
                metrics.append(tuple(day_metrics))

        return metrics, level

    def _determine_expected_action(self, metrics, level):
        """Determine the expected action based on metrics and current level"""
        # Calculate average performance
        recent_avg = np.mean([np.mean(day) for day in metrics[-2:]])  # Last 2 days
        trend = np.mean([np.mean(day) for day in metrics[-2:]]) - np.mean(
            [np.mean(day) for day in metrics[:2]]
        )  # Trend

        if recent_avg > 0.85 and level < 5:
            return "increase"
        elif recent_avg < 0.3 and level > 1:
            return "decrease"
        elif trend > 0.2 and level < 5:
            return "increase"
        elif trend < -0.2 and level > 1:
            return "decrease"
        else:
            return "maintain"

    def _calculate_reward(self, decision):
        """Calculate reward based on the action taken"""
        if decision == self.expected_action:
            base_reward = 1.0
        else:
            # Penalize wrong decisions more severely
            if self.expected_action == "maintain" and decision in [
                "increase",
                "decrease",
            ]:
                base_reward = -0.5
            elif self.expected_action != "maintain" and decision == "maintain":
                base_reward = -0.3
            else:
                base_reward = -1.0  # Completely wrong decision

        # Add performance-based component
        recent_performance = np.mean(
            [np.mean(day) for day in self.current_metrics[-2:]]
        )
        performance_modifier = (recent_performance - 0.5) * 0.2

        return base_reward + performance_modifier


def train_and_save_model(total_timesteps=500000, save_path="level_adjustment_model"):
    """Train the PPO model for level adjustment and save it"""
    # Create environment
    env = LevelAdjustmentEnv()

    # Initialize the model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    )

    # Create eval environment
    eval_env = LevelAdjustmentEnv()

    # Create the callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}_best/",
        log_path=f"{save_path}_logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Train with callback
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    print("Model saved", save_path)
    model.save(save_path)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=100, deterministic=True
    )
    print(f"\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return model


if __name__ == "__main__":
    # Train and save the model
    train_and_save_model(save_path=Config.level_manager_path())
