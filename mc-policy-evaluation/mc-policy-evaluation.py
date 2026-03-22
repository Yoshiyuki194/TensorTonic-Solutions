import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    returns_sum = np.zeros(n_states)
    returns_count = np.zeros(n_states)
    V = np.zeros(n_states)

    for episode in episodes:
        states_visited_in_episode = set()

        G = 0
        for i in range(len(episode) - 1, -1, -1):
            state, reward = episode[i]
            G = reward + gamma * G
            
        episode_states = [step[0] for step in episode]
        for i, (state, reward) in enumerate(episode):
            if state not in episode_states[:i]:
                g_t = sum(step[1] * (gamma ** j) for j, step in enumerate(episode[i:]))
                returns_sum[state] += g_t
                returns_count[state] += 1

    for s in range(n_states):
        if returns_count[s] > 0:
            V[s] = returns_sum[s] / returns_count[s]
            
    return V