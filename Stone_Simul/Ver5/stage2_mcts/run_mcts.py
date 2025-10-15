import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import logging
import numpy as np
from common.env import LostArkStoneEnv, State
from common.utils import setup_logging, load_config, plot_mcts_results
from stage2_mcts.mcts import MCTS
from typing import Tuple

def run_episode(env: LostArkStoneEnv, solver: MCTS) -> Tuple[bool, int, State, str]:
    """한 에피소드 실행"""
    state = env.get_initial_state()
    steps = 0
    
    # 첫 번째 액션 저장
    first_action = 'A'  # 기본값 설정
    
    while not env.is_terminal(state):
        # MCTS로 다음 행동 결정
        action = solver.search(state)
        
        # 첫 번째 액션 기록
        if steps == 0:
            first_action = action
        
        # 행동 실행
        success = np.random.random() < state.p
        state = env.get_next_state(state, action, success)
        steps += 1
    
    return env.get_reward(state) > 0, steps, state, first_action

def main():
    # 설정 파일 로드
    config = load_config('config/mcts_config.yaml')
    
    # 재현성을 위한 랜덤 시드 설정
    seed = config.get('seed', 42)
    np.random.seed(seed)
    logging.info(f"랜덤 시드 설정: {seed}")
    
    # 로깅 설정
    logger = setup_logging(config['log_file'])
    
    # 환경 생성
    env = LostArkStoneEnv()
    
    # MCTS 솔버 생성
    solver = MCTS(env, config)
    
    # 여러 에피소드 실행
    n_episodes = int(config.get('n_episodes', 100))
    success_count = 0
    total_steps = 0
    episode_results = []
    
    logger.info(f"MCTS 학습 시작 (총 {n_episodes} 에피소드)")
    
    for episode in range(n_episodes):
        success, steps, final_state, first_action = run_episode(env, solver)
        success_count += int(success)
        total_steps += steps
        episode_results.append({
            'episode': episode + 1,
            'success': success,
            'steps': steps,
            'final_state': final_state,
            'first_action': first_action
        })
        
        if (episode + 1) % 10 == 0:
            logger.info(f"에피소드 {episode + 1}/{n_episodes} 완료")
            logger.info(f"현재 성공률: {success_count/(episode + 1)*100:.2f}%")
    
    # 최종 결과 출력
    logger.info("\n=== 최종 결과 ===")
    logger.info(f"총 에피소드: {n_episodes}")
    logger.info(f"성공 횟수: {success_count}")
    logger.info(f"성공률: {success_count/n_episodes*100:.2f}%")
    logger.info(f"평균 스텝 수: {total_steps/n_episodes:.2f}")
    
    # 성공한 에피소드들의 최종 상태 분석
    success_states = [r['final_state'] for r in episode_results if r['success']]
    if success_states:
        avg_a = np.mean([s.a for s in success_states])
        avg_b = np.mean([s.b for s in success_states])
        avg_c = np.mean([s.c for s in success_states])
        logger.info("\n=== 성공한 에피소드의 평균 상태 ===")
        logger.info(f"평균 A 성공: {avg_a:.2f}")
        logger.info(f"평균 B 성공: {avg_b:.2f}")
        logger.info(f"평균 C 성공: {avg_c:.2f}")
    
    # 결과 시각화
    plot_mcts_results(episode_results, 'outputs/figures/mcts')
    
    logger.info("MCTS 학습 완료")

if __name__ == "__main__":
    main() 