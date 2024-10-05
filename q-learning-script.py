import pandas as pd
import numpy as np
import json
from collections import deque
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Check if data file exists, if not, create sample data
try:
    data = pd.read_excel('course_data.xlsx')
    print("Loaded existing course data.")
except FileNotFoundError:
    print("Generating sample course data...")
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame({
        'x_chapters': np.random.randint(1, 11, n_samples),
        'y_quizzes': np.random.uniform(0, 1, n_samples),
        'z_interactivity': np.random.uniform(0, 1, n_samples),
        'p1_breaks': np.random.randint(1, 11, n_samples),
        'p2_response_speed': np.random.uniform(0, 1, n_samples),
        'p3_quiz_scores': np.random.uniform(40, 100, n_samples)
    })

    # Add some correlations to make the data more realistic
    data['p3_quiz_scores'] += 10 * (data['y_quizzes'] - 0.5)  # More quizzes → better scores
    data['p2_response_speed'] -= 0.1 * data['x_chapters']  # More chapters → slower response
    data['p3_quiz_scores'] = data['p3_quiz_scores'].clip(0, 100)
    data['p2_response_speed'] = data['p2_response_speed'].clip(0, 1)

# Define states with optimized granularity
chapters = np.linspace(min(data['x_chapters']), max(data['x_chapters']), 15)
quizzes = np.linspace(min(data['y_quizzes']), max(data['y_quizzes']), 10)
interactivity = np.linspace(min(data['z_interactivity']), max(data['z_interactivity']), 8)


def get_discretized_state(x, y, z):
    x_idx = np.digitize(x, chapters) - 1
    y_idx = np.digitize(y, quizzes) - 1
    z_idx = np.digitize(z, interactivity) - 1
    return x_idx, y_idx, z_idx


states = [(x, y, z) for x in range(len(chapters)) for y in range(len(quizzes)) for z in range(len(interactivity))]
actions = ['shorten_video', 'segment_video', 'add_quiz', 'increase_interactivity', 'decrease_interactivity']

# Initialize Q-table with optimistic initial values
Q_table = pd.DataFrame(np.ones((len(states), len(actions))) * 10,
                       index=[str(s) for s in states],
                       columns=actions)


class HyperParameters:
    def __init__(self):
        self.learning_rate = 0.03
        self.discount_factor = 0.95
        self.initial_exploration_rate = 1.0
        self.min_exploration_rate = 0.1
        self.exploration_decay = 0.999
        self.batch_size = 64
        self.memory_size = 20000
        self.target_update_frequency = 500


params = HyperParameters()


class ExperienceReplay:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))


memory = ExperienceReplay(params.memory_size)


def get_state_data(state):
    x, y, z = chapters[state[0]], quizzes[state[1]], interactivity[state[2]]
    nearby_data = data[
        (abs(data['x_chapters'] - x) <= (chapters[1] - chapters[0]) * 1.5) &
        (abs(data['y_quizzes'] - y) <= (quizzes[1] - quizzes[0]) * 1.5) &
        (abs(data['z_interactivity'] - z) <= (interactivity[1] - interactivity[0]) * 1.5)
        ]
    return nearby_data if not nearby_data.empty else None


def calculate_reward(state_data):
    if state_data is None:
        return -5

    avg_data = state_data.mean()
    p1, p2, p3 = avg_data['p1_breaks'], avg_data['p2_response_speed'], avg_data['p3_quiz_scores']

    reward = 0
    reward += 10 * np.exp(-0.2 * p1)
    reward += 15 * (1 / (1 + np.exp(-10 * (p2 - 0.5))))
    reward += 20 * (p3 / 100) ** 2

    balance_factor = (
            np.exp(-(p1 - 4) ** 2 / 4) *
            np.exp(-(p2 - 0.6) ** 2 / 0.04) *
            np.exp(-(p3 - 75) ** 2 / 100)
    )
    reward += 10 * balance_factor

    return reward


def choose_action(state, exploration_rate):
    if np.random.rand() < exploration_rate:
        return np.random.choice(actions)
    else:
        return Q_table.loc[str(state)].idxmax()


def apply_action(state, action):
    x, y, z = state
    if action == 'shorten_video':
        x = max(0, x - 1)
    elif action == 'segment_video':
        y = min(len(quizzes) - 1, y + 1)
    elif action == 'add_quiz':
        y = min(len(quizzes) - 1, y + 1)
    elif action == 'increase_interactivity':
        z = min(len(interactivity) - 1, z + 1)
    elif action == 'decrease_interactivity':
        z = max(0, z - 1)
    return x, y, z


def update_q_table(experiences):
    for state, action, reward, next_state in experiences:
        state_str, next_state_str = str(state), str(next_state)
        best_next_action = Q_table.loc[next_state_str].idxmax()

        current_q = Q_table.loc[state_str, action]
        next_q = Q_table.loc[next_state_str, best_next_action]

        error = reward + params.discount_factor * next_q - current_q
        if abs(error) <= 1:
            update = 0.5 * error ** 2
        else:
            update = abs(error) - 0.5

        Q_table.loc[state_str, action] += params.learning_rate * update * np.sign(error)


# Training loop
num_episodes = 2500
results = []
exploration_rate = params.initial_exploration_rate
moving_avg_reward = deque(maxlen=100)

print("Starting training...")
for episode in range(num_episodes):
    state = states[np.random.choice(len(states))]
    episode_rewards = []

    for step in range(25):
        action = choose_action(state, exploration_rate)
        state_data = get_state_data(state)
        reward = calculate_reward(state_data)
        next_state = apply_action(state, action)

        memory.add(state, action, reward, next_state)
        episode_rewards.append(reward)

        if len(memory.memory) >= params.batch_size:
            experiences = memory.sample(params.batch_size)
            update_q_table(experiences)

        state = next_state

    avg_episode_reward = np.mean(episode_rewards)
    moving_avg_reward.append(avg_episode_reward)

    if len(moving_avg_reward) == moving_avg_reward.maxlen:
        if np.mean(moving_avg_reward) < 5:
            exploration_rate = min(0.5, exploration_rate * 1.05)
        else:
            exploration_rate = max(params.min_exploration_rate,
                                   exploration_rate * params.exploration_decay)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Average Reward: {avg_episode_reward:.2f}")
        print(f"Moving Average Reward: {np.mean(moving_avg_reward):.2f}")
        print(f"Exploration Rate: {exploration_rate:.4f}")
        print()

    results.append({
        'episode': episode + 1,
        'average_reward': avg_episode_reward,
        'moving_average_reward': np.mean(moving_avg_reward) if moving_avg_reward else avg_episode_reward,
        'exploration_rate': exploration_rate
    })

print("Training completed. Running validation...")

# Validation phase
validation_episodes = 200
validation_rewards = []

exploration_rate = 0.05
for episode in range(validation_episodes):
    state = states[np.random.choice(len(states))]
    episode_reward = 0

    for _ in range(15):
        action = choose_action(state, exploration_rate)
        state_data = get_state_data(state)
        reward = calculate_reward(state_data)
        episode_reward += reward
        state = apply_action(state, action)

    validation_rewards.append(episode_reward / 15)

# Prepare and save results
output_data = {
    'training_results': results,
    'q_table': Q_table.to_dict(),
    'best_actions': Q_table.idxmax(axis=1).to_dict(),
    'hyperparameters': vars(params),
    'validation_results': {
        'mean': float(np.mean(validation_rewards)),
        'std': float(np.std(validation_rewards)),
        'median': float(np.median(validation_rewards)),
        '95th_percentile': float(np.percentile(validation_rewards, 95)),
        '5th_percentile': float(np.percentile(validation_rewards, 5))
    }
}

# Save results
with open('rl_results_with_viz.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("\nGenerating visualizations...")

# Convert results to DataFrame for plotting
df = pd.DataFrame(results)

# Set up the plotting style
plt.style.use('seaborn')
sns.set_palette("husl")

# Create training plots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Q-Learning Training Analysis', fontsize=16)

# Plot 1: Moving Average Reward
axs[0, 0].plot(df['episode'], df['moving_average_reward'])
axs[0, 0].set_title('Moving Average Reward')
axs[0, 0].set_xlabel('Episode')
axs[0, 0].set_ylabel('Reward')

# Plot 2: Exploration Rate
axs[0, 1].plot(df['episode'], df['exploration_rate'])
axs[0, 1].set_title('Exploration Rate')
axs[0, 1].set_xlabel('Episode')
axs[0, 1].set_ylabel('Exploration Rate')

# Plot 3: Average Reward Distribution
sns.histplot(df['average_reward'], kde=True, ax=axs[1, 0])
axs[1, 0].set_title('Average Reward Distribution')
axs[1, 0].set_xlabel('Reward')
axs[1, 0].set_ylabel('Count')

# Plot 4: Moving Average vs Exploration Rate
axs[1, 1].scatter(df['exploration_rate'], df['moving_average_reward'], alpha=0.5)
axs[1, 1].set_title('Moving Average Reward vs Exploration Rate')
axs[1, 1].set_xlabel('Exploration Rate')
axs[1, 1].set_ylabel('Moving Average Reward')

plt.tight_layout()
plt.savefig('training_analysis.png')
plt.close()

# Create validation plot
plt.figure(figsize=(10, 6))
sns.boxplot(y=validation_rewards)
plt.title('Validation Rewards Distribution')
plt.savefig('validation_results.png')
plt.close()

# Print summary
print("\nTraining Summary:")
print(f"Total episodes: {num_episodes}")
print(f"Final exploration rate: {exploration_rate:.4f}")
print(f"Final moving average reward: {df['moving_average_reward'].iloc[-1]:.2f}")

print("\nValidation Summary:")
print(f"Mean reward: {np.mean(validation_rewards):.2f}")
print(f"Standard deviation: {np.std(validation_rewards):.2f}")
print(f"Median reward: {np.median(validation_rewards):.2f}")
print(f"95th percentile: {np.percentile(validation_rewards, 95):.2f}")
print(f"5th percentile: {np.percentile(validation_rewards, 5):.2f}")

best_actions = pd.Series(Q_table.idxmax(axis=1).values).value_counts()
print("\nBest Actions Distribution:")
for action, count in best_actions.items():
    print(f"{action}: {count} states ({count / len(states) * 100:.1f}%)")

print("\nVisualization completed. Please check:")
print("1. training_analysis.png - Overall training performance")
print("2. validation_results.png - Validation results distribution")
