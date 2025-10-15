import logging
import os
from datetime import datetime

class LoggerManager:
    """
    프로젝트 내 중앙 로깅 시스템을 관리하는 클래스.
    - 모든 대화 및 오류를 `.data/yyyymmddhhmmss_log.log` 파일에 저장.
    - 콘솔 출력과 파일 저장을 동시에 수행.
    """
    _loggers = {}  # 로거 인스턴스 저장

    @classmethod
    def get_logger(cls, name="DeepSeekClient", log_dir = ".data"):
        """
        로거를 반환. 이미 생성된 로거가 있으면 재사용.
        :param name: 로거 이름
        :return: 설정된 로거 객체
        """
        if name in cls._loggers:
            return cls._loggers[name]  # 기존 로거 반환

        # 로그 파일 경로 설정
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)  # 로그 디렉토리 생성

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        log_file = os.path.join(log_dir, f"{timestamp}_log.log")

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # 콘솔 핸들러 추가
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)

        # 파일 핸들러 추가
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        cls._loggers[name] = logger  # 생성된 로거 저장
        return logger
