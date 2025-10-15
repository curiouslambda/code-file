# # environment.py
# import numpy as np

# class GameEnvironment:
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.choices = {'A': 10, 'B': 10, 'C': 10}
#         self.success_rates = {'A': 0.75, 'B': 0.75, 'C': 0.75}
#         self.success_counts = {'A': 0, 'B': 0, 'C': 0}
#         self.total_attempts = 0
#         self.done = False
#         return self.get_state()

#     def get_state(self):
#         return (
#             tuple(self.choices.values()), 
#             tuple(self.success_rates.values()), 
#             tuple(self.success_counts.values()),
#             self.total_attempts
#         )

#     def step(self, action):
#         if self.done:
#             raise ValueError("Game is already over.")
        
#         category = action
#         if self.choices[category] == 0:
#             raise ValueError(f"No remaining choices for category {category}.")
        
#         success = np.random.rand() < self.success_rates[category]
#         self.choices[category] -= 1
#         self.total_attempts += 1

#         if success:
#             self.success_counts[category] += 1
#             new_success_rate = max(0.25, self.success_rates[category] - 0.10)
#         else:
#             new_success_rate = min(0.75, self.success_rates[category] + 0.10)
        
#         for cat in self.success_rates:
#             self.success_rates[cat] = new_success_rate

#         if self.total_attempts >= 30:
#             self.done = True
#             reward = self.calculate_reward()
#         else:
#             reward = 0
#             self.done = False
        
#         return self.get_state(), reward, self.done

#     def calculate_reward(self):
#         ab_success = self.success_counts['A'] + self.success_counts['B']
#         c_success = self.success_counts['C']
#         if 14 <= ab_success <= 15 and c_success <= 4:
#             return 5  # Moderate reward
#         elif ab_success >= 16 and c_success <= 4:
#             return 10  # Large reward
#         else:
#             return 0  # No reward

#     def check_win(self):
#         return self.success_counts['A'] + self.success_counts['B'] >= 16 and self.success_counts['C'] <= 4


# environment.py
import numpy as np

class GameEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.choices = {'A': 10, 'B': 10, 'C': 10}
        self.success_rates = {'A': 0.75, 'B': 0.75, 'C': 0.75}
        self.success_counts = {'A': 0, 'B': 0, 'C': 0}
        self.total_attempts = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        return (
            tuple(self.choices.values()), 
            tuple(self.success_rates.values()), 
            tuple(self.success_counts.values()),
            self.total_attempts
        )

    def step(self, action):
        if self.done:
            raise ValueError("Game is already over.")
        
        category = action
        if self.choices[category] == 0:
            raise ValueError(f"No remaining choices for category {category}.")
        
        success = np.random.rand() < self.success_rates[category]
        self.choices[category] -= 1
        self.total_attempts += 1

        if success:
            reward = 20 if category in ['A', 'B'] else -20  # A, B success: +1, C success: -1
            self.success_counts[category] += 1
            new_success_rate = max(0.25, self.success_rates[category] - 0.10)
        else:
            reward = -20 if category in ['A', 'B'] else 20  # A, B failure: -1, C failure: +1
            new_success_rate = min(0.75, self.success_rates[category] + 0.10)
        
        for cat in self.success_rates:
            self.success_rates[cat] = new_success_rate

        if self.total_attempts >= 30:
            self.done = True
            reward += self.calculate_reward()
        else:
            self.done = False
        
        return self.get_state(), reward, self.done

    def calculate_reward(self):
        ab_success = self.success_counts['A'] + self.success_counts['B']
        c_success = self.success_counts['C']
        if 14 <= ab_success <= 15 and c_success <= 4:
            # print(" 최소 8 6 돌!!!!!")
            return 200  # Moderate reward
        elif ab_success >= 16 and c_success <= 4:
            print(" 오오오오오오오!! 대 성 공 돌!!!!!")
            return 500  # Large reward
        else:
            return -200  # No reward

    def check_win(self):
        return self.success_counts['A'] + self.success_counts['B'] >= 16 and self.success_counts['C'] <= 4
    
    def get_ab_success(self):
        return self.success_counts['A'] + self.success_counts['B']

    def get_c_success(self):
        return self.success_counts['C']