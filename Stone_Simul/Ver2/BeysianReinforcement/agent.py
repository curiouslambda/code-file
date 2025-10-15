# agent.py

from environment import GameEnvironment
from scipy.stats import beta
import json
import os
from datetime import datetime


class BayesianAgent:
    def __init__(self):
        self.env = GameEnvironment()
        self.episode_data = []  # 한 에피소드 동안 데이터를 저장할 리스트
        self.episode_count = 0  # 에피소드 카운터

    def reset_environment(self):
        """환경을 리셋하여 새 에피소드 시작"""
        self.env = GameEnvironment()  # 새로운 환경 생성

    def select_action(self):
        """Thompson Sampling을 사용하여 사전/사후 확률을 기반으로 행동 선택"""
        samples = {}
        
        for category in self.env.attempts_left:
            if self.env.attempts_left[category] > 0:
                # 현재 각 카테고리의 성공 확률을 베타 분포에서 샘플링
                sampled_prob = beta(self.env.alpha, self.env.beta).rvs()
                samples[category] = sampled_prob
                
                # 데이터를 에피소드 리스트에 추가
                self.episode_data.append({
                    # "category": category,
                    # "alpha": self.env.alpha[category],
                    # "beta": self.env.beta[category],
                    f"sampled_prob_{category}": sampled_prob
                    
                })
        
        # 샘플링된 확률 중 가장 높은 성공 확률을 가진 카테고리 선택
        return max(samples, key=samples.get)

    def save_episode_data(self, episode_count):
        """에피소드 데이터를 JSON 파일로 저장"""
        current_date = datetime.now().date()
        
        # 폴더 경로 설정
        folder_path = f"./data/{current_date}_8"
        os.makedirs(folder_path, exist_ok=True)  # 폴더가 없으면 생성
        
        # 파일 이름 설정 (에피소드 별로 구분)
        self.episode_count += 1
        file_name = f"episode_{episode_count}.json"
        file_path = os.path.join(folder_path, file_name)
        
        # 데이터를 JSON 파일로 저장
        with open(file_path, 'w') as json_file:
            json.dump(self.episode_data, json_file, indent=4, ensure_ascii=False)
        
        # print(f"에피소드 {self.episode_count} 데이터가 {file_name}로 저장되었습니다.")

    def play(self, episode_count):
        """게임을 실행하며 학습"""
        self.episode_data = []  # 새로운 에피소드 시작 시 데이터를 초기화
        self.reset_environment()  # 환경 리셋
        
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
            
            
