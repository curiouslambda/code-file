import logging
import os
from typing import Dict, Any, List, Tuple
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from common.env import State

# 한글 폰트 설정
import matplotlib.font_manager as fm

# 시스템에서 사용 가능한 폰트 찾기
available_fonts = [f.name for f in fm.fontManager.ttflist]

# 한글 폰트 우선순위 설정
korean_fonts = ['Malgun Gothic', 'NanumGothic', 'Nanum Gothic', 'Arial Unicode MS', 'DejaVu Sans']
font_family = []

for font in korean_fonts:
    if font in available_fonts:
        font_family.append(font)
        break

# 한글 폰트가 없으면 기본 폰트 사용
if not font_family:
    font_family = ['DejaVu Sans', 'sans-serif']

plt.rcParams['font.family'] = font_family
plt.rcParams['axes.unicode_minus'] = False

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
    plt.savefig(os.path.join(save_dir, 'cumulative_success_rate_5.png'))
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
    plt.savefig(os.path.join(save_dir, 'rolling_success_rate_5.png'))
    plt.close()
    
    # 3. 에피소드별 스텝 수 분포
    steps = [r['steps'] for r in episode_results]
    plt.figure(figsize=(10, 6))
    plt.hist(steps, bins=30, density=True, alpha=0.7)
    plt.xlabel('Steps per Episode')
    plt.ylabel('Frequency')
    plt.title('Distribution of Episode Lengths')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'episode_lengths_5.png'))
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
    plt.savefig(os.path.join(save_dir, 'success_fail_comparison_5.png'))
    plt.close()
    
    # 5. 루트 액션 선택 빈도 변화
    action_counts = {'A': 0, 'B': 0, 'C': 0}
    action_history = []
    
    for result in episode_results:
        # 실제 첫 번째 액션 사용 (추정이 아닌 실제 값)
        if 'first_action' in result:
            action = result['first_action']
        else:
            # 이전 버전 호환성을 위한 추정 로직
            state = result['final_state']
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
    plt.savefig(os.path.join(save_dir, 'action_selection_frequency_5.png'))
    plt.close()

def plot_hybrid_results(results: Dict, save_dir: str):
    """하이브리드 결과 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 행동 빈도 막대 그래프
    plt.figure(figsize=(10, 6))
    actions = list(results['action_frequencies'].keys())
    frequencies = list(results['action_frequencies'].values())
    
    bars = plt.bar(actions, frequencies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.title('하이브리드 솔버 행동 빈도', fontsize=14, fontweight='bold')
    plt.xlabel('행동', fontsize=12)
    plt.ylabel('빈도', fontsize=12)
    plt.ylim(0, max(frequencies) * 1.1)
    
    # 값 표시
    for bar, freq in zip(bars, frequencies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{freq:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'action_frequencies_maxsi500.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 전체 성공률 및 가이던스 일치율
    plt.figure(figsize=(12, 6))
    
    # 서브플롯 1: 전체 성공률
    plt.subplot(1, 2, 1)
    plt.pie([results['overall_success_rate'], 1 - results['overall_success_rate']], 
            labels=['성공', '실패'], autopct='%1.1f%%', 
            colors=['#4ECDC4', '#FF6B6B'])
    plt.title('전체 성공률 (Overall Success Rate)', fontsize=14, fontweight='bold')
    
    # 서브플롯 2: 가이던스 일치율
    plt.subplot(1, 2, 2)
    guidance_agreement_rate = results.get('guidance_agreement_rate', 0.0)
    plt.pie([guidance_agreement_rate, 1 - guidance_agreement_rate], 
            labels=['DP 일치', 'DP 불일치'], autopct='%1.1f%%',
            colors=['#45B7D1', '#96CEB4'])
    plt.title('DP 가이던스 일치율 (DP Guidance Agreement Rate)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'success_and_guidance_rates_maxsi500.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_guidance_statistics(guidance_stats: Dict[str, float], save_dir: str):
    """가이던스 통계 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 시뮬레이션 vs MCTS 가이던스 사용률 비교
    plt.figure(figsize=(12, 6))
    
    # 서브플롯 1: 시뮬레이션 가이던스 사용률
    plt.subplot(1, 2, 1)
    simulation_rate = guidance_stats['simulation_guidance_rate']
    plt.pie([simulation_rate, 1 - simulation_rate], 
            labels=['가이던스 사용', '일반 선택'], autopct='%1.1f%%', 
            colors=['#4ECDC4', '#FF6B6B'])
    plt.title('시뮬레이션 가이던스 사용률', fontsize=14, fontweight='bold')
    
    # 서브플롯 2: MCTS 탐색 가이던스 사용률
    plt.subplot(1, 2, 2)
    mcts_rate = guidance_stats['mcts_guidance_rate']
    plt.pie([mcts_rate, 1 - mcts_rate], 
            labels=['가이던스 사용', '일반 선택'], autopct='%1.1f%%',
            colors=['#45B7D1', '#96CEB4'])
    plt.title('MCTS 탐색 가이던스 사용률', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'guidance_usage_comparison_maxsi500.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 가이던스 사용 통계 상세 분석
    plt.figure(figsize=(15, 8))
    
    # 서브플롯 1: 시뮬레이션 스텝 통계
    plt.subplot(1, 3, 1)
    simulation_metrics = {
        '가이던스 사용 스텝': guidance_stats['simulation_guidance_count'],
        '일반 선택 스텝': guidance_stats['total_simulation_steps'] - guidance_stats['simulation_guidance_count']
    }
    
    colors = ['#4ECDC4', '#FF6B6B']
    metric_names = list(simulation_metrics.keys())
    metric_values = list(simulation_metrics.values())
    bars = plt.bar(metric_names, metric_values, color=colors)
    
    plt.title('시뮬레이션 스텝 통계', fontsize=14, fontweight='bold')
    plt.ylabel('스텝 수', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # 값 표시
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values) * 0.01,
                f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 서브플롯 2: MCTS 시뮬레이션 통계
    plt.subplot(1, 3, 2)
    mcts_metrics = {
        '가이던스 사용 시뮬레이션': guidance_stats['guidance_usage_count'],
        '일반 시뮬레이션': guidance_stats['total_simulations'] - guidance_stats['guidance_usage_count']
    }
    
    colors = ['#45B7D1', '#96CEB4']
    metric_names = list(mcts_metrics.keys())
    metric_values = list(mcts_metrics.values())
    bars = plt.bar(metric_names, metric_values, color=colors)
    
    plt.title('MCTS 시뮬레이션 통계', fontsize=14, fontweight='bold')
    plt.ylabel('시뮬레이션 수', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # 값 표시
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values) * 0.01,
                f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 서브플롯 3: 가이던스 사용률 비교
    plt.subplot(1, 3, 3)
    usage_rates = {
        '시뮬레이션': guidance_stats['simulation_guidance_rate'] * 100,
        'MCTS 탐색': guidance_stats['mcts_guidance_rate'] * 100
    }
    
    colors = ['#4ECDC4', '#45B7D1']
    rate_names = list(usage_rates.keys())
    rate_values = list(usage_rates.values())
    bars = plt.bar(rate_names, rate_values, color=colors)
    
    plt.title('가이던스 사용률 비교', fontsize=14, fontweight='bold')
    plt.ylabel('사용률 (%)', fontsize=12)
    plt.ylim(0, max(rate_values) * 1.1)
    
    # 값 표시
    for bar, value in zip(bars, rate_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'guidance_statistics_detailed_maxsi500.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_episode_trajectories(episode_data: List[Dict], save_dir: str, num_samples: int = 10):
    """에피소드 경로(Trajectory) 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 성공/실패 에피소드 분리
    success_episodes = [ep for ep in episode_data if ep['success']]
    failure_episodes = [ep for ep in episode_data if not ep['success']]
    
    # 샘플링 (최대 num_samples개)
    success_samples = success_episodes[:min(num_samples, len(success_episodes))]
    failure_samples = failure_episodes[:min(num_samples, len(failure_episodes))]
    
    plt.figure(figsize=(15, 10))
    
    # 색상 매핑
    action_colors = {'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#9B59B6'}
    
    # 성공한 에피소드들
    plt.subplot(2, 1, 1)
    for i, episode in enumerate(success_samples):
        trajectory = episode.get('trajectory', [])
        if trajectory:
            # 경로 그리기
            remaining_slots = [step['remaining_slots'] for step in trajectory]
            a_successes = [step['a'] for step in trajectory]
            
            # 각 스텝을 점으로 표시하고 선으로 연결
            for j in range(len(trajectory) - 1):
                action = trajectory[j]['action']
                color = action_colors.get(action, '#666666')
                
                # 현재 점과 다음 점을 연결하는 선
                plt.plot([remaining_slots[j], remaining_slots[j+1]], 
                        [a_successes[j], a_successes[j+1]], 
                        color=color, alpha=0.7, linewidth=2)
                
                # 현재 점 표시
                plt.scatter(remaining_slots[j], a_successes[j], 
                          color=color, s=50, alpha=0.8, edgecolors='black', linewidth=1)
            
            # 마지막 점 표시
            if trajectory:
                last_action = trajectory[-1]['action']
                last_color = action_colors.get(last_action, '#666666')
                plt.scatter(remaining_slots[-1], a_successes[-1], 
                          color=last_color, s=50, alpha=0.8, edgecolors='black', linewidth=1)
    
    plt.title('성공한 에피소드 경로 (Success Trajectories)', fontsize=14, fontweight='bold')
    plt.xlabel('남은 시도 횟수 (Remaining Slots)', fontsize=12)
    plt.ylabel('A 성공 횟수', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(30, 0)  # X축을 역순으로
    
    # 범례 추가
    legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=f'Action {action}') 
                      for action, color in action_colors.items()]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # 실패한 에피소드들
    plt.subplot(2, 1, 2)
    for i, episode in enumerate(failure_samples):
        trajectory = episode.get('trajectory', [])
        if trajectory:
            # 경로 그리기
            remaining_slots = [step['remaining_slots'] for step in trajectory]
            a_successes = [step['a'] for step in trajectory]
            
            # 각 스텝을 점으로 표시하고 선으로 연결
            for j in range(len(trajectory) - 1):
                action = trajectory[j]['action']
                color = action_colors.get(action, '#666666')
                
                # 현재 점과 다음 점을 연결하는 선
                plt.plot([remaining_slots[j], remaining_slots[j+1]], 
                        [a_successes[j], a_successes[j+1]], 
                        color=color, alpha=0.7, linewidth=2)
                
                # 현재 점 표시
                plt.scatter(remaining_slots[j], a_successes[j], 
                          color=color, s=50, alpha=0.8, edgecolors='black', linewidth=1)
            
            # 마지막 점 표시
            if trajectory:
                last_action = trajectory[-1]['action']
                last_color = action_colors.get(last_action, '#666666')
                plt.scatter(remaining_slots[-1], a_successes[-1], 
                          color=last_color, s=50, alpha=0.8, edgecolors='black', linewidth=1)
    
    plt.title('실패한 에피소드 경로 (Failure Trajectories)', fontsize=14, fontweight='bold')
    plt.xlabel('남은 시도 횟수 (Remaining Slots)', fontsize=12)
    plt.ylabel('A 성공 횟수', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(30, 0)  # X축을 역순으로
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'episode_trajectories_maxsi500.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_optimal_action_heatmap(hybrid_solver, save_dir: str):
    """최적 행동 히트맵 - a=5, b=5, c=2 상태에서의 최적 행동 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 상태 공간 정의
    p_values = np.arange(0.25, 0.76, 0.05)  # 성공 확률 범위
    remaining_slots = np.arange(0, 19, 1)    # 남은 시도 횟수 범위 (a=5,b=5,c=2 상태에서 최대 18개)
    
    # 최적 행동 행렬 초기화
    optimal_action_matrix = np.zeros((len(remaining_slots), len(p_values)), dtype=int)
    
    # 각 상태에서 최적 행동 계산
    for i, slots in enumerate(remaining_slots):
        for j, p in enumerate(p_values):
            # 고정된 상태 (a=5, b=5, c=2)로 정책 계산
            state = State(a=5, b=5, c=2, p=p, remaining_slots=slots)
            
            # MCTS 트리 구축하여 정책 계산
            root = hybrid_solver._build_mcts_tree(state)
            action_probs = hybrid_solver._get_action_probabilities(root)
            
            # 가장 확률이 높은 행동의 인덱스 찾기
            optimal_action_idx = np.argmax(action_probs)
            optimal_action_matrix[i, j] = optimal_action_idx
    
    # 히트맵 그리기
    plt.figure(figsize=(12, 8))
    
    # seaborn 히트맵 사용
    import seaborn as sns
    
    # 색상 팔레트 정의 (A=0: 빨강, B=1: 초록, C=2: 보라)
    colors = ['#FF6B6B', '#4ECDC4', '#9B59B6']
    cmap = sns.color_palette(colors, n_colors=3)
    
    # 히트맵 생성
    sns.heatmap(optimal_action_matrix, 
                cmap=cmap,
                xticklabels=np.round(p_values, 2),
                yticklabels=remaining_slots[::1],  # 모든 값 표시 (0~18)
                cbar_kws={'label': '최적 행동'},
                annot=False)  # 숫자 표시 안함
    
    # 축 레이블
    plt.xlabel('현재 성공 확률 (p)', fontsize=12)
    plt.ylabel('남은 시도 횟수 (remaining_slots)', fontsize=12)
    plt.title('최적 행동 히트맵 (a=5, b=5, c=2) - 최대 18개 남음', fontsize=14, fontweight='bold')
    
    # 컬러바에 행동 이름 표시
    cbar = plt.gca().collections[0].colorbar
    cbar.set_ticks([0.33, 1.0, 1.67])  # 각 색상의 중간값
    cbar.set_ticklabels(['A', 'B', 'C'])
    cbar.set_label('최적 행동', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimal_action_heatmap_maxsi500.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_method_comparison(methods: Dict[str, float], save_dir: str):
    """방법별 성공률 비교 그래프"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    method_names = list(methods.keys())
    success_rates = list(methods.values())
    
    bars = plt.bar(method_names, success_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.title('Method Comparison - Success Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.ylim(0, max(success_rates) * 1.1)
    
    # 값 표시
    for bar, rate in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close() 