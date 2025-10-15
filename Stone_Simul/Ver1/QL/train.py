# # train.py
# from environment import GameEnvironment
# from agent import QLearningAgent
# from utils import plot_training_statistics, save_training_statistics
# from tqdm import tqdm
# import json

# def train_agent(episodes=1000):
#     env = GameEnvironment()
#     agent = QLearningAgent(actions=['A', 'B', 'C'])

#     file_path = './data/'

#     stone_14 = 0
#     stone_16 = 0

#     rewards = []
#     stone_14_counts = []
#     stone_16_counts = []

#     for episode in tqdm(range(episodes)):
#         state = env.reset()
#         total_reward = 0
#         action_list = []
#         past_state = []
#         after_state = []
#         data = {
#             "action" : action_list,
#             "state" : past_state,
#             "state_after" : after_state
#         }
#         while True:
#             action = agent.choose_action(state)
#             # print(f"선택한 행동 : {action}")
#             action_list.append(action)
#             # print(f"액션 리스트 : {action_list}")

#             next_state, reward, done = env.step(action)
#             agent.update(state, action, reward, next_state)
#             # print(f"이전 상태 : {state}")
#             past_state.append(state)

#             state = next_state
#             # print(f"다음 상태 : {state}")
#             after_state.append(state)

#             total_reward += reward

#             if done:
#                 break

#         if total_reward >=780:
#             print("개쩌는 돌 납시오!")
        
#         ab_success = env.get_ab_success()
#         c_success = env.get_c_success()

#         if 14 <= ab_success <= 15 and c_success <= 4:
#             stone_14 += 1
#         elif ab_success >= 16 and c_success <= 4:
#             stone_16 += 1
        
#         stone_14_counts.append(stone_14)
#         stone_16_counts.append(stone_16)
        
#         if (episode+1) % 10000 == 0 and episode !=0:
#             print(f"\n{episode+1} 에피소드의 액션 리스트 : \n{action_list}\n")
#             print(f"에피소드 {episode+1}: 보상 합 = {total_reward}")

#             json_file_path = file_path + f'data_epi{episode + 1}.json'
#             json_file = json.dumps(data, indent = 4)
#             with open(json_file_path, "w") as f:
#                 f.write(json_file)
    
#     print(f"14 돌 : {stone_14}회\n16 돌 : {stone_16}회\n")

#     statistics = {
#         "rewards": rewards,
#         "stone_14_counts": stone_14_counts,
#         "stone_16_counts": stone_16_counts
#     }
#     # save_training_statistics(statistics, file_path + 'training_statistics.json')

#     # plot_training_statistics(rewards, stone_14_counts, stone_16_counts)
    

#     return agent

# train.py
from environment import GameEnvironment
from agent import QLearningAgent
from utils import plot_training_statistics, save_training_statistics
from config import Config
from tqdm import tqdm
import json

def train_agent(episodes=1000):
    env = GameEnvironment()
    agent = QLearningAgent(actions=['A', 'B', 'C'])

    file_path = './data/'

    stone_14 = 0
    stone_16 = 0

    rewards = []
    stone_14_counts = []
    stone_16_counts = []
    performance_history = []

    for episode in tqdm(range(episodes)):
        state = env.reset()
        total_reward = 0
        action_list = []
        past_state = []
        after_state = []
        data = {
            "action" : action_list,
            "state" : past_state,
            "state_after" : after_state
        }

        # Update alpha
        # print(f"알파 값 : {Config.ALPHA}")
        # agent.alpha = Config.ALPHA * (Config.GAMMA ** episode)

        # print(f"업데이트 된 알파값 {agent.alpha}")
        while True:
            action = agent.choose_action(state)
            # print(f"선택한 행동 : {action}")
            action_list.append(action)
            # print(f"액션 리스트 : {action_list}")

            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            # print(f"이전 상태 : {state}")
            past_state.append(state)

            state = next_state
            # print(f"다음 상태 : {state}")
            after_state.append(state)

            total_reward += reward
            

            if done:
                break

        rewards.append(total_reward)
        performance_history.append(total_reward)

        # Check performance and adjust alpha
        if (episode + 1) % Config.PERFORMANCE_CEHCK_INTERVAL == 0:
            average_reward = sum(performance_history[-Config.PERFORMANCE_CEHCK_INTERVAL:]) / Config.PERFORMANCE_CEHCK_INTERVAL
            if len(performance_history) > Config.PERFORMANCE_CEHCK_INTERVAL and average_reward <= sum(performance_history[-2 * Config.PERFORMANCE_CEHCK_INTERVAL: -Config.PERFORMANCE_CEHCK_INTERVAL]) / Config.PERFORMANCE_CEHCK_INTERVAL:
                agent.alpha *= Config.GAMMA
        if total_reward >=780:
            print("개쩌는 돌 납시오!")
        
        ab_success = env.get_ab_success()
        c_success = env.get_c_success()

        if 14 <= ab_success <= 15 and c_success <= 4:
            stone_14 += 1
        elif ab_success >= 16 and c_success <= 4:
            stone_16 += 1
        
        stone_14_counts.append(stone_14)
        stone_16_counts.append(stone_16)
        
        if (episode+1) % 10000 == 0 and episode !=0:
            print(f"\n{episode+1} 에피소드의 액션 리스트 : \n{action_list}\n")
            print(f"에피소드 {episode+1}: 보상 합 = {total_reward}")

            json_file_path = file_path + f'data_epi{episode + 1}.json'
            json_file = json.dumps(data, indent = 4)
            with open(json_file_path, "w") as f:
                f.write(json_file)
    
    print(f"14 돌 : {stone_14}회\n16 돌 : {stone_16}회\n")

    statistics = {
        "rewards": rewards,
        "stone_14_counts": stone_14_counts,
        "stone_16_counts": stone_16_counts
    }
    # print(f"에피: {episode}, 보상 : {rewards}, 14돌 : {stone_14_counts}, 16돌 : {stone_16_counts}")
    save_training_statistics(statistics, file_path + f'/statistics/{Config.EPISODES}_{Config.ALPHA}_{Config.PERFORMANCE_CEHCK_INTERVAL}.json')

    plot_training_statistics(list(range(1, episodes + 1)), rewards, stone_14_counts, stone_16_counts)
    

    return agent
