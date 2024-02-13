import numpy as np

def team_rennes_policy(agent):
    """
    Stochastic policy for the agent.
    """
    initial_left_prob = 0.3
    initial_right_prob = 0.3
    initial_stay_prob = 1 - initial_left_prob - initial_right_prob

    left_prob = initial_left_prob
    right_prob = initial_right_prob
    stay_prob = initial_stay_prob

    if agent.position > 0 and agent.known_rewards[agent.position - 1] > 0:
        left_prob = 1

    if agent.position < len(agent.known_rewards) - 1 and agent.known_rewards[agent.position + 1] > 0:
        right_prob = 1

    total_prob = left_prob + right_prob + stay_prob
    left_prob /= total_prob
    right_prob /= total_prob
    stay_prob = 1 - left_prob - right_prob

    action = np.random.choice(["left", "right", "none"], p=[left_prob, right_prob, stay_prob])

    return action