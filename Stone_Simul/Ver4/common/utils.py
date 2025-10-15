import logging
import os
from typing import Dict, Any, List, Tuple
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(log_file: str) -> logging.Logger:
    """로깅 설정"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """YAML 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def plot_value_curve(deltas: List[float], save_path: str):
    """Value Iteration 수렴 곡선 플롯"""
    plt.figure(figsize=(10, 6))
    plt.plot(deltas, 'b-', label='Delta')
    plt.xlabel('Iteration')
    plt.ylabel('Delta')
    plt.title('Value Iteration Convergence')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_value_heatmap(value_matrix: np.ndarray, xlabel: str, ylabel: str, save_path: str):
    """가치 함수 히트맵 플롯"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(value_matrix, cmap='viridis', annot=True, fmt='.2f')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Value Function Heatmap')
    plt.savefig(save_path)
    plt.close()

def plot_policy_heatmap(policy_matrix: np.ndarray, xlabel: str, ylabel: str, save_path: str):
    """정책 히트맵 플롯"""
    plt.figure(figsize=(10, 8))
    # A=0, B=1, C=2를 색상으로 구분
    cmap = plt.cm.get_cmap('Set3', 3)
    sns.heatmap(policy_matrix, cmap=cmap, annot=True, fmt='d')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Policy Heatmap (0:A, 1:B, 2:C)')
    plt.savefig(save_path)
    plt.close()

def plot_value_vs_probability(value_matrix: np.ndarray, probs: np.ndarray, states: List[Tuple[int, int, int, int]], save_path: str):
    """성공 확률에 따른 가치 변화 플롯 (여러 상태)"""
    plt.figure(figsize=(12, 8))
    
    # x축 눈금 설정
    xticks = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    plt.xticks(xticks)
    
    # 각 상태에 대한 가치 함수 플롯
    for a, b, c, slots in states:
        value_vector = value_matrix[a, b, c, :, slots].squeeze()
        plt.plot(probs, value_vector, 
                label=f'a={a}, b={b}, c={c}, slots={slots}',
                marker='o')
    
    plt.xlabel('Success Probability')
    plt.ylabel('Value')
    plt.title('Value Function vs Success Probability')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_value_vs_remaining_slots(value_matrix: np.ndarray, save_path: str):
    """남은 시도 횟수에 따른 가치 변화 3D 플롯"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(range(value_matrix.shape[0]), range(value_matrix.shape[1]))
    ax.plot_surface(X, Y, value_matrix.T, cmap='viridis')
    
    ax.set_xlabel('State Index')
    ax.set_ylabel('Remaining Slots')
    ax.set_zlabel('Value')
    ax.set_title('Value Function vs Remaining Slots')
    
    plt.savefig(save_path)
    plt.close()

def plot_state_transition(env, policy, initial_state, get_state_index, max_steps: int, save_path: str):
    """상태 전이 시각화"""
    plt.figure(figsize=(10, 8))
    
    # 초기 상태에서 시작
    current_state = initial_state
    states = [current_state]
    
    # 최대 스텝만큼 시뮬레이션
    for _ in range(max_steps):
        if env.is_terminal(current_state):
            break
            
        # 정책에 따라 행동 선택
        state_idx = get_state_index(current_state)
        action = policy[state_idx]
        
        # 다음 상태 계산 (성공/실패 모두 고려)
        success_state = env.get_next_state(current_state, action, True)
        fail_state = env.get_next_state(current_state, action, False)
        
        # 상태 전이 시각화
        plt.arrow(current_state.a, current_state.b, 
                 success_state.a - current_state.a, 
                 success_state.b - current_state.b,
                 head_width=0.1, head_length=0.1, fc='g', ec='g', alpha=0.5)
        plt.arrow(current_state.a, current_state.b,
                 fail_state.a - current_state.a,
                 fail_state.b - current_state.b,
                 head_width=0.1, head_length=0.1, fc='r', ec='r', alpha=0.5)
        
        current_state = success_state  # 성공 상태로 이동
        states.append(current_state)
    
    # 상태 점 표시
    states = np.array([(s.a, s.b) for s in states])
    plt.scatter(states[:, 0], states[:, 1], c='b', s=100)
    
    plt.xlabel('A Success Count')
    plt.ylabel('B Success Count')
    plt.title('State Transition Visualization')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_multi_c_value_heatmaps(value_matrix: np.ndarray, c_values: List[int], p_idx: int, slots: int, save_path: str):
    """여러 c 값에 대한 가치 함수 히트맵을 서브플롯으로 그립니다."""
    n_plots = len(c_values)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, c in enumerate(c_values):
        sns.heatmap(value_matrix[:, :, c, p_idx, slots], 
                   cmap='viridis', 
                   annot=True, 
                   fmt='.2f',
                   ax=axes[idx])
        axes[idx].set_xlabel('A Success Count')
        axes[idx].set_ylabel('B Success Count')
        axes[idx].set_title(f'Value Function Heatmap (c={c})')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_mcts_results(episode_results: List[Dict], save_dir: str):
    """MCTS 결과 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 에피소드별 누적 성공률
    episodes = [r['episode'] for r in episode_results]
    cumulative_success = np.cumsum([r['success'] for r in episode_results])
    cumulative_rate = cumulative_success / np.arange(1, len(episodes) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, cumulative_rate, 'b-', label='Cumulative Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Success Rate')
    plt.title('Episode vs. Cumulative Success Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'cumulative_success_rate_1.png'))
    plt.close()
    
    # 2. 구간별 성공률 변화 (10 에피소드 윈도우)
    window_size = 10
    rolling_success = []
    for i in range(len(episodes)):
        start_idx = max(0, i - window_size + 1)
        window_success = np.mean([r['success'] for r in episode_results[start_idx:i+1]])
        rolling_success.append(window_success)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rolling_success, 'r-', label=f'{window_size}-Episode Rolling Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Rolling Success Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'rolling_success_rate_1.png'))
    plt.close()
    
    # 3. 에피소드별 스텝 수 분포
    steps = [r['steps'] for r in episode_results]
    plt.figure(figsize=(10, 6))
    plt.hist(steps, bins=30, density=True, alpha=0.7)
    plt.xlabel('Steps per Episode')
    plt.ylabel('Frequency')
    plt.title('Distribution of Episode Lengths')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'episode_lengths_1.png'))
    plt.close()
    
    # 4. 성공/실패별 A/B/C 성공 횟수 비교
    success_states = [r['final_state'] for r in episode_results if r['success']]
    fail_states = [r['final_state'] for r in episode_results if not r['success']]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    categories = ['A', 'B', 'C']
    data = [
        ([s.a for s in success_states], [s.a for s in fail_states]),
        ([s.b for s in success_states], [s.b for s in fail_states]),
        ([s.c for s in success_states], [s.c for s in fail_states])
    ]
    
    for ax, (success_data, fail_data), category in zip(axes, data, categories):
        ax.boxplot([success_data, fail_data], labels=['Success', 'Fail'])
        ax.set_title(f'{category} Success Count Distribution')
        ax.set_ylabel('Count')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'success_fail_comparison_1.png'))
    plt.close()
    
    # 5. 루트 액션 선택 빈도 변화
    action_counts = {'A': 0, 'B': 0, 'C': 0}
    action_history = []
    
    for result in episode_results:
        state = result['final_state']
        # 첫 번째 액션 추정 (이전 상태와 비교)
        if state.a > 0:
            action = 'A'
        elif state.b > 0:
            action = 'B'
        elif state.c > 0:
            action = 'C'
        else:
            action = np.random.choice(['A', 'B', 'C'])
        
        action_counts[action] += 1
        action_history.append(action_counts.copy())
    
    # 누적 액션 선택 빈도 계산
    episodes = range(1, len(action_history) + 1)
    a_counts = [h['A'] for h in action_history]
    b_counts = [h['B'] for h in action_history]
    c_counts = [h['C'] for h in action_history]
    
    plt.figure(figsize=(10, 6))
    plt.stackplot(episodes, [a_counts, b_counts, c_counts], 
                 labels=['A', 'B', 'C'],
                 alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Action Count')
    plt.title('Root Action Selection Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'action_selection_frequency_1.png'))
    plt.close() 