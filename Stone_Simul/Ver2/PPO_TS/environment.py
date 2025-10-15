# environment.py
import numpy as np

class GameEnv:
    def __init__(self, config):
        self.config = config
        self.episode_number = 0
        # self.reset()

    def reset(self,episode_number):
        # 초기 상태 설정
        self.state = [10, 10, 10, 0, 0, 0, 0, 0, 0, 0.75]  # [A남은기회, B남은기회, C남은기회, A성공, B성공, C성공, A실패, B실패, C실패, 성공확률]
        self.steps = 0
        self.episode_number = episode_number

        self.file_name = f"./data/game_state_log_episode_{self.episode_number}.txt"  # 에피소드 번호에 따른 파일 이름 설정

        # 새로운 에피소드 파일을 열어서 처음부터 기록할 준비
        with open(self.file_name, "w") as file:
            file.write(f"Episode {self.episode_number} 시작\n")
        
        return np.array(self.state)

    def step(self, action):
        success_prob = self.state[-1]
        success = np.random.rand() < success_prob  # 성공 여부 결정

        # 각 카테고리의 남은 기회를 체크하여 기회가 없으면 선택을 무효화
        if action == 0 and self.state[0] == 0:  # A 카테고리의 남은 기회가 0인 경우
            reward = -0.5  # 무효화된 선택에 대한 페널티 (원하는 값으로 조정 가능)
            return np.array(self.state), reward, False

        if action == 1 and self.state[1] == 0:  # B 카테고리의 남은 기회가 0인 경우
            reward = -0.5
            return np.array(self.state), reward, False

        if action == 2 and self.state[2] == 0:  # C 카테고리의 남은 기회가 0인 경우
            reward = -0.5
            return np.array(self.state), reward, False

        # 상태 업데이트
        if action == 0:  # A
            self.state[0] -= 1
            if success:
                self.state[3] += 1
                self.state[-1] = max(0.25, self.state[-1] - 0.1)
                reward = 0.1  # A에서 성공했을 때 보상
            else:
                self.state[6] += 1
                self.state[-1] = min(0.75, self.state[-1] + 0.1)
                reward = -0.1  # 실패 시 페널티

        elif action == 1:  # B
            self.state[1] -= 1
            if success:
                self.state[4] += 1
                self.state[-1] = max(0.25, self.state[-1] - 0.1)
                reward = 0.1  # B에서 성공했을 때 보상
            else:
                self.state[7] += 1
                self.state[-1] = min(0.75, self.state[-1] + 0.1)
                reward = -0.1  # 실패 시 페널티

        elif action == 2:  # C
            self.state[2] -= 1
            if success:
                self.state[5] += 1
                self.state[-1] = max(0.25, self.state[-1] - 0.1)
                reward = -0.5  # C에서 성공하면 페널티
            else:
                self.state[8] += 1
                self.state[-1] = min(0.75, self.state[-1] + 0.1)
                reward = 0.1  # C에서 실패하면 작은 보상

        self.steps += 1

        print(f"현재 {self.steps}의 상태 :", self.state)
        self.log_state()  # 상태 기록
        
        done = self.steps >= self.config.max_game_rounds
        print(f"완료 조건 : {done}")

        # 보상 설정 (승리 조건)
        # reward = 0
        if done:
            if self.state[3] >= 7 and self.state[4] >= 7 and self.state[5] <= 4:
                reward = +10  # 성공
            else:
                reward = -10  # 실패

        return np.array(self.state), reward, done

    def render(self):
        print(f"State: {self.state}")

    
    def log_state(self):
        # 상태를 텍스트 파일에 기록
        with open(self.file_name, "a") as file:
            file.write(f"Step {self.steps}: {self.state}\n")
