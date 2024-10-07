import random
import numpy as np

# Total number of querys in an experiment
TOTAL_QUERIES = 10000
# Number of arms to be pulled in an experiment
TOTAL_ARMS = 10
# Number of Servers that can be used for parallel query
TOTAL_SERVERS = 8


class Arm:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

        self._mean_hat = float("inf")
        self._ucb = float("inf")
        self._total_pulls = 0

    def _update_arm(self, result):
        self._total_pulls += 1
        self._mean_hat = self._mean_hat + (result - self._mean_hat) / self._total_pulls
        self._ucb = self._mean_hat + np.sqrt(
            2 * np.log(TOTAL_QUERIES) / self._total_pulls
        )

    def _get_query_result(self):
        return np.random.normal(self.mean, self.std)

    def update_arm(self):
        result = self._get_query_result()
        self._update_arm(result)
        return result


def get_arm_to_pull(arms):
    _arm = None
    for i in range(TOTAL_ARMS):
        if not _arm or arms[i]._ucb > _arm._ucb:
            _arm = arms[i]
    return _arm


class Trial:
    def __init__(self, start_time, arm):
        self._start_time = start_time
        self._end_time = self._start_time + random.randint(1, 10)
        self._arm = arm

    def run_trial(self, current_time):
        if current_time < self._end_time:
            return False, None
        return True, self._arm.update_arm()


if __name__ == "__main__":
    arms = [Arm(np.random.normal(0, 1), 1) for _ in range(TOTAL_ARMS)]

    current_time = 0
    total_reward = 0
    query_processed = 0

    trials = [None] * TOTAL_SERVERS

    while True:
        print("Current Time: ", current_time, "Query Processed: ", query_processed)
        if query_processed >= TOTAL_QUERIES:
            break

        for i in range(TOTAL_SERVERS):
            if trials[i] is None:
                trials[i] = Trial(current_time, get_arm_to_pull(arms))
            done, reward = trials[i].run_trial(current_time)
            if done:
                total_reward += reward
                query_processed += 1
                trials[i] = None
        current_time += 1

    print("Total Reward: ", total_reward)
    oracle_reward = max([arm.mean for arm in arms]) * TOTAL_QUERIES
    print("Total Regets: ", oracle_reward - total_reward)
