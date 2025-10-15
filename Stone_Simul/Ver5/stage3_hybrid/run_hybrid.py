"""
하이브리드 DP-MCTS 솔버 실행 스크립트
"""

import sys
import os
# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import logging
import numpy as np
from typing import Dict, List
from common.env import LostArkStoneEnv
from stage3_hybrid.hybrid_solver import HybridSolver
from common.utils import plot_value_curve, plot_value_heatmap, plot_policy_heatmap, plot_hybrid_results, plot_method_comparison, plot_guidance_statistics, plot_episode_trajectories, plot_optimal_action_heatmap

def setup_logging(config: Dict):
    """로깅 설정"""
    log_level = config.get('log_level', 'INFO')
    log_file = config.get('log_file', 'logs/hybrid.log')
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def run_hybrid_experiment(config: Dict) -> Dict:
    """하이브리드 실험 실행"""
    # 환경 초기화
    env = LostArkStoneEnv()
    
    # 하이브리드 솔버 초기화
    hybrid_solver = HybridSolver(env, config)
    
    # 초기 상태
    initial_state = env.get_initial_state()
    
    logging.info("하이브리드 DP-MCTS 실험 시작")
    logging.info(f"설정: {config}")
    
    # 통계 수집
    num_episodes = config.get('num_episodes', 100)
    results = hybrid_solver.get_action_statistics(initial_state, num_episodes)
    
    logging.info(f"실험 완료:")
    logging.info(f"  - 전체 성공률: {results['overall_success_rate']:.3f}")
    logging.info(f"  - 평균 에피소드 길이: {results['avg_episode_length']:.2f}")
    logging.info(f"  - 가이던스 일치율: {results['guidance_agreement_rate']:.3f}")
    logging.info(f"  - 가이던스 적용률: {results['guidance_applicable_rate']:.3f}")
    logging.info(f"  - 행동 빈도: {results['action_frequencies']}")
    
    # 가이던스 분석 결과 출력
    hybrid_solver.print_guidance_analysis()
    
    # 가이던스 통계 그래프 생성
    guidance_stats = hybrid_solver.get_guidance_statistics()
    output_dir = config.get('output_dir', 'outputs/hybrid/figures')
    plot_guidance_statistics(guidance_stats, output_dir)
    
    # 에피소드 경로 시각화
    episode_data = results.get('episode_data', [])
    if episode_data:
        plot_episode_trajectories(episode_data, output_dir, num_samples=15)
    
    # 최적 행동 히트맵 생성
    plot_optimal_action_heatmap(hybrid_solver, output_dir)
    
    return results

def plot_hybrid_results_wrapper(results: Dict, config: Dict):
    """하이브리드 결과 시각화 래퍼 함수"""
    output_dir = config.get('output_dir', 'outputs/hybrid/figures')
    plot_hybrid_results(results, output_dir)
    logging.info(f"그래프 저장 완료: {output_dir}")

def compare_methods(config: Dict):
    """DP, MCTS, 하이브리드 방법 비교"""
    env = LostArkStoneEnv()
    initial_state = env.get_initial_state()
    
    # DP 가이던스 유틸리티 초기화
    from stage3_hybrid.utils import DPGuidanceUtils
    dp_guidance = DPGuidanceUtils(env, config.get('dp_values_path', 'outputs/dp_values.npz'))
    
    # 각 방법별 성공률 측정
    methods = {}
    
    # 1. 실제 DP 정책
    logging.info("실제 DP 정책 테스트 중...")
    dp_success_count = 0
    num_episodes = config.get('num_episodes', 100)
    
    for _ in range(num_episodes):
        state = initial_state
        while not env.is_terminal(state):
            # 실제 DP 정책 사용
            dp_action = dp_guidance.get_dp_action(state)
            if dp_action is None:
                # DP 결과가 없는 경우 기본 휴리스틱 사용
                if state.a < 7:
                    action = 'A'
                elif state.b < 7:
                    action = 'B'
                elif state.c < 4:
                    action = 'C'
                else:
                    action = np.random.choice(['A', 'B', 'C'])
            else:
                action = dp_action
            
            success = np.random.random() < state.p
            state = env.get_next_state(state, action, success)
        
        if env.get_reward(state) > 0:
            dp_success_count += 1
    
    methods['DP'] = dp_success_count / num_episodes
    
    # 2. 랜덤 정책
    logging.info("랜덤 정책 테스트 중...")
    random_success_count = 0
    
    for _ in range(num_episodes):
        state = initial_state
        while not env.is_terminal(state):
            action = np.random.choice(['A', 'B', 'C'])
            success = np.random.random() < state.p
            state = env.get_next_state(state, action, success)
        
        if env.get_reward(state) > 0:
            random_success_count += 1
    
    methods['Random'] = random_success_count / num_episodes
    
    # 3. 하이브리드 (이미 계산됨)
    hybrid_results = run_hybrid_experiment(config)
    methods['Hybrid'] = hybrid_results['overall_success_rate']
    
    # 비교 그래프
    output_dir = config.get('output_dir', 'outputs/hybrid/figures')
    plot_method_comparison(methods, output_dir)
    
    logging.info(f"방법별 성공률: {methods}")
    return methods

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='하이브리드 DP-MCTS 솔버 실행')
    parser.add_argument('--config', type=str, default='config/hybrid_config.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--compare', action='store_true',
                       help='방법 비교 실행')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 재현성을 위한 랜덤 시드 설정
    seed = config.get('seed', 42)
    np.random.seed(seed)
    logging.info(f"랜덤 시드 설정: {seed}")
    
    # 로깅 설정
    setup_logging(config)
    
    if args.compare:
        # 방법 비교
        compare_methods(config)
    else:
        # 하이브리드 실험만 실행
        results = run_hybrid_experiment(config)
        
        # 결과 저장
        save_path = config.get('results_save_path', 'outputs/hybrid/hybrid_results.npz')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        hybrid_solver = HybridSolver(LostArkStoneEnv(), config)
        hybrid_solver.save_results(results, save_path)
        
        # 그래프 생성
        plot_hybrid_results_wrapper(results, config)

if __name__ == "__main__":
    main() 