# main.py

from agent import BayesianAgent

if __name__ == "__main__":
    num_episodes = 100  # 총 100번의 에피소드를 실행
    episode_count = 1  # 에피소드 카운터, 1부터 시작
    agent = BayesianAgent()

    for episode in range(num_episodes):
        print(f"에피소드 {episode + 1} 시작")
        # agent = BayesianAgent()  # 매 에피소드마다 새로운 에이전트 생성
        agent.play(episode_count)  # 한 에피소드 실행
        episode_count += 1  # 에피소드 카운트 증가
        # print(f"에피소드 {episode + 1} 종료\n")