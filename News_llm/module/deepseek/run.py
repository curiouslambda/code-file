import os
import json
import csv
from datetime import datetime
import sys
from utils.deepseek_client import DeepSeekClient
from utils.logger_manager import LoggerManager

# 로거 인스턴스 생성
logger = LoggerManager.get_logger(log_dir=r"data/log")

def load_model_list(file_path="model_list.txt"):
    """모델 리스트를 파일에서 불러옴."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            models = [line.strip() for line in f.readlines() if line.strip()]
        return models
    except FileNotFoundError:
        logger.error(f"모델 리스트 파일({file_path})을 찾을 수 없습니다.")
        print(f"오류: {file_path} 파일이 존재하지 않습니다. 기본 모델을 사용합니다.")
        return ["deepseek-r1:14b"]

def select_model(models):
    """사용자가 사용할 모델을 선택하도록 함."""
    print("사용 가능한 모델 목록:")
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    
    while True:
        choice = input("사용할 모델 번호를 선택하세요: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("올바른 번호를 입력하세요.")

def run(model_name=None):
    """
    모델이 유효한지 확인 후, 일반 대화를 수행하도록 구성.
    """
    if model_name is None:
        models = load_model_list()
        model_name = select_model(models)
    
    try:
        client = DeepSeekClient(model=model_name)  # 모델 유효성 체크
    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        print(f"오류: {model_name} 모델을 로드할 수 없습니다. 올바른 모델 이름인지 확인하세요.")
        return

    logger.info(f"=== {model_name} 모델과 대화 시작 ===")
    print(f"=== {model_name} 모델과 대화 ===")
    print("질문을 입력하세요 (종료하려면 'exit' 입력)\n")
    
    while True:
        user_question = input("질문: ").strip()
        if user_question.lower() in ["exit", "quit", "종료"]:
            logger.info("사용자가 프로그램 종료 요청")
            print("프로그램을 종료합니다.")
            break
        
        try:
            logger.info(f"사용자 질문: {user_question}")
            answer = client.ask(user_question)
            print(f"\n=== {model_name} 응답 ===\n{answer}\n")
            logger.info(f"{model_name} 응답: {answer}")
        except Exception as e:
            error_msg = f"오류 발생: {str(e)}"
            logger.error(error_msg)
            print(error_msg)

if __name__ == "__main__":
    # 실행 시 커맨드라인 인자로 모델 이름을 전달할 수 있음.
    # 예: python script.py my-model-name
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = None
    run(model_name)
