# Common

### 목적
- 프로젝트 공용 구성 요소 제공
- 환경(`LostArkStoneEnv`)과 시각화/로깅 유틸 함수 집합

### 주요 파일
- `env.py`: 상태 `State` dataclass, 승리 조건, 전이(`get_next_state`), 보상(`get_reward`), 종료 판정, 상태공간 크기 설정정
- `utils.py`: 로깅 설정, YAML 설정 로드, DP/MCTS/하이브리드 결과 시각화(수렴곡선, 히트맵, 확률·슬롯 변화, 상태전이, MCTS/하이브리드 통계 등)

### 사용
- 환경 생성: `from common.env import LostArkStoneEnv`
- 로깅/설정: `from common.utils import setup_logging, load_config`

