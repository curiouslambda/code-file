from environment import GameEnvironment
from config import Config
from scipy.stats import beta
from graphs import *
import numpy as np
import json
import os
from datetime import datetime


class BayesianAgent:
    def __init__(self):
        self.env = GameEnvironment(Config.initial_alpha, Config.initial_beta, Config.initial_alpha_c, Config.initial_beta_c)
        self.alpha = Config.initial_alpha  # 알파 초기값 설정
        self.beta = Config.initial_beta   # 베타 초기값 설정
        self.alpha_c = Config.initial_alpha_c
        self.beta_c = Config.initial_beta_c
        self.episode_data = []  # 한 에피소드 동안 데이터를 저장할 리스트
        self.total_data = []  # 전체 에피소드 동안 데이터를 저장할 리스트
        self.episode_count = 0  # 에피소드 카운터
        self.failure_count = 0
        self.success_count = 0

    def reset_environment(self):
        """환경을 리셋하여 새 에피소드 시작, 알파/베타는 그대로 유지"""
        self.env = GameEnvironment(self.env.alpha, self.env.beta, self.env.alpha_c, self.env.beta_c)
        # 환경은 리셋하지만, 알파와 베타는 유지

    def select_action(self):
        """Thompson Sampling을 사용하여 사전/사후 확률을 기반으로 행동 선택"""
        samples = {}
        
        for category in self.env.attempts_left:
            if self.env.attempts_left[category] > 0:
                if category == 'A' or category == 'B':
                    # 베타 분포에서 샘플링, 이때 self.alpha, self.beta 사용
                    sampled_prob = beta(self.alpha, self.beta).rvs()
                    samples[category] = sampled_prob
                    # 데이터를 에피소드 리스트에 추가
                    self.episode_data.append({
                    f"sampled_prob_{category}": sampled_prob
                    })

                    if sum(self.env.successes.values()) + sum(self.env.failures.values()) == 30:
                        self.memory_data.append({
                        f"sampled_prob_{category}": sampled_prob
                        })
                else:
                    # C의 알파, 베타를 사용
                    sampled_prob_c = beta(self.alpha_c, self.beta_c).rvs()
                    samples['C'] = sampled_prob_c
                    # 데이터를 에피소드 리스트에 추가
                    self.episode_data.append({
                    f"sampled_prob_{category}": sampled_prob_c
                    })
                    
                    if sum(self.env.successes.values()) + sum(self.env.failures.values()) == 30:
                        self.memory_data.append({
                        f"sampled_prob_{category}": sampled_prob_c
                        })

                # # 데이터를 에피소드 리스트에 추가
                # self.episode_data.append({
                #     f"sampled_prob_{category}": sampled_prob
                # })

                # self.memory_data.append({
                #     f"sampled_prob_{category}": sampled_prob
                # })

        return max(samples, key=samples.get)

    def save_episode_data(self, episode_count):
        """에피소드 데이터를 JSON 파일로 저장"""
        current_date = datetime.now().date()
        
        # 폴더 경로 설정
        folder_path = f"./data/{current_date}_1"
        os.makedirs(folder_path, exist_ok=True)  # 폴더가 없으면 생성
        
        # 파일 이름 설정 (에피소드 별로 구분)
        self.episode_count += 1
        file_name = f"episode_{episode_count}.json"
        file_name_total = "total_data.json"
        file_path = os.path.join(folder_path, file_name)
        file_path_total = os.path.join(folder_path, file_name_total)
        
        # 데이터를 JSON 파일로 저장
        with open(file_path, 'w') as json_file:
            json.dump(self.episode_data, json_file, indent=4, ensure_ascii=False)
        
        with open(file_path_total, 'w') as json_file:
            json.dump(self.total_data, json_file, indent=4, ensure_ascii=False)
        
    def play(self, episode_count):
        """게임을 실행하며 학습"""
        self.episode_data = []  # 새로운 에피소드 시작 시 데이터를 초기화
        self.memory_data = []
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
                "alpha_c" : self.env.alpha_c,
                "beta_c" : self.env.beta_c,
                "success_prob": self.env.success_prob,
                "success_count" : self.env.successes.copy()
                
            })

            if sum(self.env.successes.values()) + sum(self.env.failures.values()) == 30:
                self.memory_data.append({
                    "episode": episode_count,
                    "step": sum(self.env.successes.values()) + sum(self.env.failures.values()),  # 현재 스텝 수
                    "category": category,
                    "alpha": self.env.alpha,
                    "beta": self.env.beta,
                    "alpha_c" : self.env.alpha_c,
                    "beta_c" : self.env.beta_c,
                    "success_prob": self.env.success_prob,
                    "success_count": self.env.successes.copy(),
                    "failure_count": self.env.failures.copy(),
                })
        # print(self.env.alpha, self.env.beta)

        # 한 에피소드가 끝났을 때 데이터를 저장
        self.total_data.append(self.memory_data)
        self.save_episode_data(episode_count)
        

        # if self.env.successes['C'] >= 5:
        #     self.failure_count += 1

        # if self.env.failures['C'] > 5:
        #     self.success_count += 1

        # if self.failure_count % 20 == 0:
        #     # print(self.failure_count % 5)
        #     self.env.alpha_c = self.env.alpha_c * 0.99
        
        # # if self.success_count % 20 == 0:
        # #     self.env.alpha_c = self.env.alpha_c * 1.02
            
        # if self.env.successes['A'] >= 7 or self.env.successes['B'] >= 7:
        #     self.env.alpha += 10
        # elif self.env.successes['A'] >= 7 and self.env.successes['B'] >= 7:
        #     self.env.alpha = self.env.alpha * 1.01

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

    def visualize_total_data(self):
        """전체 에피소드 데이터를 시각화"""
        current_date = datetime.now().date()
        data_path = f"./data/{current_date}_1/"
        folder_path = f"./data/{current_date}_1/total"
        os.makedirs(folder_path, exist_ok=True)

        # 시각화 함수 호출
        plot_alpha_beta_changes(self.total_data, folder_path)
        plot_first_step_probabilities(data_path, folder_path)
        plot_success_changes(self.total_data, folder_path)
        plot_beta_distribution_changes(self.total_data, folder_path)
        