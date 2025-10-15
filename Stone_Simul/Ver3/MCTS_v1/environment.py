from config import GAME_CONFIG
import random

class GameEnvironment:
    def __init__(self):
        self.success_prob = GAME_CONFIG["initial_success_prob"]
        self.max_prob = GAME_CONFIG["max_success_prob"]
        self.min_prob = GAME_CONFIG["min_success_prob"]
        self.remaining_steps = {cat: GAME_CONFIG["max_attempts_per_category"] for cat in GAME_CONFIG["categories"]}
        self.success = {cat: 0 for cat in GAME_CONFIG["categories"]}
        self.failure = {cat: 0 for cat in GAME_CONFIG["categories"]}

    def reset(self):
        self.success_prob = GAME_CONFIG["initial_success_prob"]
        self.remaining_steps = {cat: GAME_CONFIG["max_attempts_per_category"] for cat in GAME_CONFIG["categories"]}
        self.success = {cat: 0 for cat in GAME_CONFIG["categories"]}
        self.failure = {cat: 0 for cat in GAME_CONFIG["categories"]}
        return self.get_state()
    
    def clone(self):
        new_env = GameEnvironment()
        new_env.success_prob = self.success_prob
        new_env.remaining_steps = self.remaining_steps.copy()
        new_env.success = self.success.copy()
        new_env.failure = self.failure.copy()
        return new_env


    def step(self, category):
        if self.remaining_steps[category] <= 0:
            raise ValueError(f"No steps remaining for category {category}.")
        
        success = random.random() < self.success_prob
        if success:
            self.success[category] += 1
            self.success_prob = max(self.success_prob - 0.1, self.min_prob)
        else:
            self.failure[category] += 1
            self.success_prob = min(self.success_prob + 0.1, self.max_prob)
        
        self.remaining_steps[category] -= 1
        print(f"Step executed: category={category}, remaining_steps={self.remaining_steps}")  # 로그 추가
        done = self.check_done()
        reward = self.get_reward(category, done)
        return self.get_state(), reward, done

    def get_state(self):
        print(f"""success_prob: {self.success_prob},
            remaining_steps: {self.remaining_steps.copy()},
            success: {self.success.copy()},
            failure: {self.failure.copy()},
            done: {self.check_done()},""")
        return {
            "success_prob": self.success_prob,
            "remaining_steps": self.remaining_steps.copy(),
            "success": self.success.copy(),
            "failure": self.failure.copy(),
            "done": self.check_done(),
        }

    def check_done(self):
        if self.success["C"] >= GAME_CONFIG["failure_threshold"]["C"] or self.failure["A"] >= 4 or self.failure["B"] >= 4:
            return True
        if sum(self.remaining_steps.values()) == 0:
            return True
        if any(v == 0 for v in self.remaining_steps.values()):
            print("하나 이상의 카테고리가 0이므로 종료합니다.")
            return True
        return False

    def get_reward(self, category, done):
        if done:
            if self.success["A"] >= 7 and self.success["B"] >= 7 and self.success["C"] <= 4:
                return 100  # 승리 보상
            else:
                return -100  # 패배 보상
        return 1 if category in ["A", "B"] else -1
