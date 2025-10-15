from agent import BayesianAgent

if __name__ == "__main__":
    num_episodes = 100  # 총 100번의 에피소드를 실행
    agent = BayesianAgent()  # 하나의 에이전트만 생성

    for episode in range(num_episodes):
        print(f"에피소드 {episode + 1} 시작")
        agent.reset_environment()  # 환경만 리셋, 에이전트는 그대로 유지
        agent.play(episode + 1)  # 한 에피소드 실행
