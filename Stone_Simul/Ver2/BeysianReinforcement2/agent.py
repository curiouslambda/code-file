from environment import GameEnvironment
from config import Config
from scipy.stats import beta
import json
import os
from datetime import datetime


class BayesianAgent:
    def __init__(self):
        self.env = GameEnvironment(Config.initial_alpha, Config.initial_beta)
        self.alpha = Config.initial_alpha  # 알파 초기값 설정
        self.beta = Config.initial_beta   # 베타 초기값 설정
        self.episode_data = []  # 한 에피소드 동안 데이터를 저장할 리스트
        self.episode_count = 0  # 에피소드 카운터

    def reset_environment(self):
        """환경을 리셋하여 새 에피소드 시작, 알파/베타는 그대로 유지"""
        self.env = GameEnvironment(self.env.alpha, self.env.beta)
        # 환경은 리셋하지만, 알파와 베타는 유지

    def select_action(self):
        """Thompson Sampling을 사용하여 사전/사후 확률을 기반으로 행동 선택"""
        samples = {}
        
        for category in self.env.attempts_left:
            if self.env.attempts_left[category] > 0:
                # 베타 분포에서 샘플링, 이때 self.alpha, self.beta 사용
                sampled_prob = beta(self.alpha, self.beta).rvs()
                samples[category] = sampled_prob

                # 데이터를 에피소드 리스트에 추가
                self.episode_data.append({
                    f"sampled_prob_{category}": sampled_prob
                })

        return max(samples, key=samples.get)
    
    # def update_params(self, success, category):
    #     """성공/실패에 따라 알파와 베타 값 업데이트"""
    #     if success:
    #         if category == 'C':
    #             # 카테고리 C에서는 베타 증가
    #             self.beta += 1
    #         else:
    #             # 카테고리 A, B에서는 알파 증가
    #             self.alpha += 1
    #     else:
    #         # 실패하면 베타 증가
    #         self.beta += 1

    def save_episode_data(self, episode_count):
        """에피소드 데이터를 JSON 파일로 저장"""
        current_date = datetime.now().date()
        
        # 폴더 경로 설정
        folder_path = f"./data/{current_date}_1"
        os.makedirs(folder_path, exist_ok=True)  # 폴더가 없으면 생성
        
        # 파일 이름 설정 (에피소드 별로 구분)
        self.episode_count += 1
        file_name = f"episode_{episode_count}.json"
        file_path = os.path.join(folder_path, file_name)
        
        # 데이터를 JSON 파일로 저장
        with open(file_path, 'w') as json_file:
            json.dump(self.episode_data, json_file, indent=4, ensure_ascii=False)

    def play(self, episode_count):
        """게임을 실행하며 학습"""
        self.episode_data = []  # 새로운 에피소드 시작 시 데이터를 초기화
        # self.reset_environment()  # 환경 리셋
        
        while not self.env.is_game_over():
            # 행동 선택 (카테고리 선택)
            category = self.select_action()

            # 게임 환경에서 시도
            success = self.env.attempt(category)
            # print(f"카테고리 {category}에서 {'성공' if success else '실패'}")

            # 데이터를 에피소드 리스트에 추가
            self.episode_data.append({
                "category": category,
                "alpha": self.env.alpha,
                "beta": self.env.beta,
                "success_prob": self.env.success_prob,
                "success_count" : self.env.successes.copy()
                
            })
        # print(self.env.alpha, self.env.beta)

        # 한 에피소드가 끝났을 때 데이터를 저장
        self.save_episode_data(episode_count)

            

        # 게임 종료 후 승리 여부 확인
        if self.env.check_win():
            print(self.env.successes)
            print(self.env.failures)
            print("===== 게임에서 승리했습니다! =====")    
        else:
            # print(self.env.successes)
            # print(self.env.failures)
            # print("===== 게임에서 패배했습니다. =====")
            pass