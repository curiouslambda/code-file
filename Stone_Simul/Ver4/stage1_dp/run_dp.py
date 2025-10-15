import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import logging
from common.env import LostArkStoneEnv
from common.utils import setup_logging, load_config
from stage1_dp.dp_solver import DPSolver

def main():
    # 설정 파일 로드
    config = load_config('config/dp_config.yaml')
    
    # 로깅 설정
    logger = setup_logging(config['log_file'])
    
    # 환경 생성
    env = LostArkStoneEnv()
    
    # DP 솔버 생성 및 실행
    solver = DPSolver(env, config)
    deltas = solver.value_iteration()
    
    # 결과 저장
    solver.deltas = deltas  # deltas를 인스턴스 변수로 저장
    solver.save_results()
    
    logger.info("DP 학습 완료")

if __name__ == "__main__":
    main() 