import matplotlib.pyplot as plt
import numpy as np

GRID_SIZE = 10
TRAINING_ATTEMPTS = 10


def random_cell(excluded=None):
    if excluded is None:
        excluded = set()

    while True:
        cell = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
        if cell not in excluded:
            return cell


START = (0, 0)
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)

EPISODES = 500
MAX_STEPS = 150
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05


def get_state_index(cell):
    return cell[0] * GRID_SIZE + cell[1]


def is_in_bounds(cell):
    row, col = cell
    return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE


def manhattan_distance(cell, target):
    return abs(cell[0] - target[0]) + abs(cell[1] - target[1])


def choose_best_action(q_table, position, valid_actions):
    q_values = q_table[get_state_index(position), valid_actions]
    best_value = np.max(q_values)
    best_actions = [
        action
        for action, q_value in zip(valid_actions, q_values)
        if np.isclose(q_value, best_value)
    ]
    return np.random.choice(best_actions)


HOLES = set()
while len(HOLES) < GRID_SIZE:
    HOLES.add(random_cell(excluded={START, GOAL} | HOLES))


def train_q_learning():
    # Q-tabulka ma pro kazdy stav 4 mozne akce: nahoru, dolu, doleva, doprava.
    q_table = np.zeros((GRID_SIZE * GRID_SIZE, 4))
    episode_rewards = []
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    epsilon = EPSILON

    # Pro kazdou epochu zacneme na startu a pokusime se dosahnout cil.
    for _ in range(EPISODES):
        position = START
        total_reward = 0

        # V kazde epizode mame limit kroku, aby se agent nenachazel nekonecne
        # dlouho v bludisti.
        for _ in range(MAX_STEPS):
            # Vybereme jen akce, ktere vedou do mrizky.
            valid_actions = [
                index
                for index, move in enumerate(moves)
                if is_in_bounds((position[0] + move[0], position[1] + move[1]))
            ]

            # Bud zvollime nahodnou akci nebo tu s nejvyssi Q-hodnotou
            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                action = choose_best_action(q_table, position, valid_actions)

            current_distance = manhattan_distance(position, GOAL)
            next_position = (
                position[0] + moves[action][0],
                position[1] + moves[action][1],
            )

            # jsme v dire? penalizace a navrat na start
            if next_position in HOLES:
                transition_reward = -100
                done = False
                next_position = START
            # jsme v cili?
            elif next_position == GOAL:
                transition_reward = 100
                done = True
            # posouvame se
            else:
                next_distance = manhattan_distance(next_position, GOAL)
                transition_reward = -1 + 3 * (current_distance - next_distance)
                done = False

            # Aktualizace Q-hodnoty podle Bellmanovy rovnice
            state = get_state_index(position)
            next_state = get_state_index(next_position)
            current_q_value = q_table[state, action]
            max_next_q_value = 0 if done else np.max(q_table[next_state])
            target_q_value = transition_reward + DISCOUNT_FACTOR * max_next_q_value

            q_table[state, action] = current_q_value + LEARNING_RATE * (
                target_q_value - current_q_value
            )

            total_reward += transition_reward
            position = next_position

            if done:
                break

        episode_rewards.append(total_reward)
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

    return q_table, episode_rewards, moves


def get_greedy_path(q_table, moves):
    # Po natrenovani vzdy vybirame akci s nejvyssi Q-hodnotou.
    position = START
    path = [position]

    # Vybudujeme cestu, dokud nedojdeme do cil nebo neprekrocime limit kroku.
    for _ in range(MAX_STEPS):
        if position == GOAL:
            break

        valid_actions = [
            index
            for index, move in enumerate(moves)
            if is_in_bounds((position[0] + move[0], position[1] + move[1]))
        ]

        # print(f"Position: {position}, Valid actions: {valid_actions}")

        action = choose_best_action(q_table, position, valid_actions)
        position = (position[0] + moves[action][0], position[1] + moves[action][1])

        if position in HOLES:
            position = START

        path.append(position)

        if position == GOAL:
            break

    return path


def train_until_goal():
    for _ in range(TRAINING_ATTEMPTS):
        q_table, rewards, moves = train_q_learning()
        path = get_greedy_path(q_table=q_table, moves=moves)

        # print(f"Trained path: {path}")

        if path[-1] == GOAL:
            return rewards, path

    return rewards, path


def save_result_figure(rewards, path, grid_size, start, goal, holes):
    fig, (learning_ax, path_ax) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]}
    )
    learning_ax.plot(rewards, color="steelblue", alpha=0.4, label="reward per episode")

    window = 50
    if len(rewards) > window:
        # Klouzavy prumer hezky ukaze, jestli se agent stabilne zlepsuje.
        moving_average = np.convolve(rewards, np.ones(window) / window, mode="valid")
        learning_ax.plot(
            range(window - 1, len(rewards)),
            moving_average,
            color="darkred",
            linewidth=2,
            label="moving average",
        )

    learning_ax.set_xlabel("Episode")
    learning_ax.set_ylabel("Reward")
    learning_ax.set_title("Q-learning training progress")
    learning_ax.grid(True, alpha=0.3)
    learning_ax.legend()

    path_ax.set_xlim(-0.5, grid_size - 0.5)
    path_ax.set_ylim(-0.5, grid_size - 0.5)
    path_ax.set_xticks(range(grid_size))
    path_ax.set_yticks(range(grid_size))
    path_ax.grid(True, color="black", linewidth=1)
    path_ax.set_aspect("equal")

    for row in range(grid_size):
        for col in range(grid_size):
            cell = (row, col)
            color = "white"
            if cell == start:
                color = "lightblue"
            elif cell == goal:
                color = "gold"
            elif cell in holes:
                color = "black"
            path_ax.add_patch(
                plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color=color, alpha=0.7)
            )

    x_coords = [position[1] for position in path]
    y_coords = [position[0] for position in path]
    path_ax.plot(x_coords, y_coords, color="crimson", linewidth=2, marker="o")

    path_ax.text(
        start[1],
        start[0],
        "START",
        color="blue",
        fontsize=14,
        ha="center",
        va="center",
        fontweight="bold",
    )
    path_ax.text(
        goal[1],
        goal[0],
        "GOAL",
        color="green",
        fontsize=14,
        ha="center",
        va="center",
        fontweight="bold",
    )
    for hole in holes:
        path_ax.text(
            hole[1],
            hole[0],
            "HOLE",
            color="red",
            fontsize=14,
            ha="center",
            va="center",
            fontweight="bold",
        )

    path_ax.set_title("Learned path")
    plt.tight_layout()
    plt.savefig("images/qlearning_result.png", dpi=150)
    plt.close(fig)


def main():
    # np.random.seed(42)

    rewards, path = train_until_goal()

    save_result_figure(rewards, path, GRID_SIZE, START, GOAL, HOLES)


if __name__ == "__main__":
    main()
