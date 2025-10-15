### agent.py

import math
import random
from environment import Environment
from config import CATEGORIES

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len([cat for cat, attempts in self.state[3].items() if attempts > 0])

    def best_child(self, c_param=1.4):
        choices_weights = [(child.value / (child.visits + 1e-4) + 
                             c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-4)), child)
                            for child in self.children.values()]
        return max(choices_weights, key=lambda x: x[0])[1]

class MCTSAgent:
    def __init__(self, iterations=1000):
        self.iterations = iterations

    def select_action(self, environment):
        root = Node(environment.get_state())

        for _ in range(self.iterations):
            node = root
            env_copy = Environment()
            env_copy.success_prob, env_copy.successes, env_copy.failures, env_copy.attempts_left = node.state

            # Selection
            while node.is_fully_expanded() and not env_copy.check_termination():
                node = node.best_child()
                env_copy.step(list(node.state[3].keys())[0])

            # Expansion
            if not node.is_fully_expanded():
                for category in CATEGORIES:
                    if env_copy.attempts_left[category] > 0 and category not in node.children:
                        new_state, _, done = env_copy.step(category)
                        node.children[category] = Node(new_state, parent=node)
                        break

            # Simulation
            current_node = node.children.get(category, node)
            total_reward = 0
            for _ in range(10):
                sim_env = Environment()
                sim_env.success_prob, sim_env.successes, sim_env.failures, sim_env.attempts_left = current_node.state
                while not sim_env.check_termination():
                    sim_action = random.choice([cat for cat, attempts in sim_env.attempts_left.items() if attempts > 0])
                    _, reward, done = sim_env.step(sim_action)
                    if done:
                        total_reward += reward
                        break

            # Backpropagation
            while current_node is not None:
                current_node.visits += 1
                current_node.value += total_reward
                current_node = current_node.parent

        return root.best_child(c_param=0).parent