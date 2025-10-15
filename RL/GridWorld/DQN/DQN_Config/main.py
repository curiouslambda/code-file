import torch
from agent import DQNAgent, dqn_learning
from environment import GRID_SIZE
from utils import visualize_training, save_fig
from config import SAVED_PLOT_PATH, AGENT_PARAMS

if __name__ == "__main__":
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델과 에이전트를 GPU로 이동하여 초기화
    model = DQNAgent(AGENT_PARAMS["STATE_SIZE"], AGENT_PARAMS["ACTION_SIZE"], device)

    # 훈련할 때 데이터를 GPU로 이동
    _, steps_to_goal = dqn_learning(device)

    # 훈련 결과 시각화
    visualize_training(steps_to_goal, SAVED_PLOT_PATH)
