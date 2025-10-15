# graphs.py
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from scipy.stats import beta

def plot_alpha_beta_changes(total_data, save_path):
    """모든 에피소드의 alpha와 beta 변화 시각화"""
    alpha_vals = []
    beta_vals = []
    alpha_c_vals = []
    beta_c_vals = []
    
    for episode_data in total_data:
        for step_data in episode_data:
            alpha_vals.append(step_data['alpha'])
            beta_vals.append(step_data['beta'])
            alpha_c_vals.append(step_data['alpha_c'])
            beta_c_vals.append(step_data['beta_c'])

    plt.figure(figsize=(10, 6))
    plt.plot(alpha_vals, label='Alpha')
    plt.plot(beta_vals, label='Beta')
    plt.plot(alpha_c_vals, label = 'Alpha_C')
    plt.plot(beta_c_vals, label = 'Beta_C')
    plt.title(f"Alpha/Beta Changes Across All Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f"total_alpha_beta_changes.png"))
    plt.close()

def plot_first_step_probabilities(data_path, save_path):
    """각 에피소드 파일의 마지막 step의 샘플링 확률 시각화"""
    categories = ['A', 'B', 'C']
    episode_probs = {category: [] for category in categories}

    file_list = os.listdir(data_path)
    
    for file in file_list[:-2]:
        file_path = os.path.join(data_path, file)

        with open(file_path, 'r') as f:
            episode_data = json.load(f)
            first_step_data = episode_data[:3]  # 처음 단계 데이터
            
            episode_probs['A'].append(first_step_data[0]['sampled_prob_A'])
            episode_probs['B'].append(first_step_data[1]['sampled_prob_B'])
            episode_probs['C'].append(first_step_data[2]['sampled_prob_C'])

    # 각 카테고리별로 그래프 생성
    for category in categories:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_probs[category], marker='o', linestyle='-', label=f'Sampled Probability {category}')

        x = np.arange(len(episode_probs[category]))
        y = episode_probs[category]
        m, b = np.polyfit(x, y, 1)  # 1차 방정식 추세선
        plt.plot(x, m * x + b, color='red', linestyle='-', label='Linear Trend Line')


        plt.title(f"First Step Sampled Probability for Category {category} Across Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f"last_step_probabilities_{category}.png"))
        plt.close()


def plot_success_changes(total_data, save_path):
    """모든 에피소드의 성공/실패 히스토그램"""
    categories = ['A', 'B', 'C']
    successes = {category: [] for category in categories}
    
    for episode_data in total_data:
        for step_data in episode_data:
            successes['A'].append(step_data['success_count']['A'])
            successes['B'].append(step_data['success_count']['B'])
            successes['C'].append(step_data['success_count']['C'])

    for category in categories:
        plt.figure(figsize=(10, 6))
        plt.plot(successes[category], linestyle='-', label=f'{category}')

        # 선형 회귀선을 추가
        x = np.arange(len(successes[category]))
        y = successes[category]
        m, b = np.polyfit(x, y, 1)  # 1차 방정식 추세선
        plt.plot(x, m * x + b, color='red', linestyle='-', label='Linear Trend Line')

        plt.title(f"Total Success Count Changes {category}")
        plt.xlabel("Outcome")
        plt.ylabel("Count")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_path, f"success_changes_{category}.png"))
        plt.close()


def plot_beta_distribution_changes(total_data, save_path):
    """모든 에피소드에 걸쳐 베타 분포의 변화를 시각화"""
    x = np.linspace(0, 1, 100)  # 확률 축 (0에서 1 사이)

    # A,B 카테고리에 대한 분포
    plt.figure(figsize=(10, 6))
    for episode_data in total_data:
        for step_data in episode_data:
            if step_data['episode'] % 20 == 0:
                alpha = step_data['alpha']
                beta_value = step_data['beta']

                # A,B 카테고리의 베타 분포 PDF 계산
                pdf = beta(alpha, beta_value).pdf(x)
                plt.plot(x, pdf, label=f'Episode {step_data["episode"]}', alpha=0.2)

    plt.title("Beta Distribution Changes for Category A,B Across All Episodes")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "beta_distribution_changes_A_B.png"))
    plt.close()

    # C 카테고리에 대한 분포
    plt.figure(figsize=(10, 6))
    for episode_data in total_data:
        for step_data in episode_data:
            if step_data['episode'] % 20 == 0:
                alpha_c = step_data['alpha_c']
                beta_c = step_data['beta_c']

                # C 카테고리의 베타 분포 PDF 계산
                pdf_c = beta(alpha_c, beta_c).pdf(x)
                plt.plot(x, pdf_c, label=f'Episode {step_data["episode"]} for C', alpha=0.2)

    plt.title("Beta Distribution Changes for Category C Across All Episodes")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "beta_distribution_changes_C.png"))
    plt.close()