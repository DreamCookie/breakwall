import os
from breakout_env import BreakoutEnv
from agent import DQNAgent
import pygame

def train(agent, env, episodes):
    total_steps = 0
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            total_steps += 1

            agent.replay()

            # Отрисовка во время обучения // не стоит
            # env.render()
            # pygame.time.delay(10)

            if done:
                print(f"Эпизод {e+1}/{episodes}, Результат: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
                break

        # Обновляем модель каждые 1000 эпизодов
        if (e + 1) % 1000 == 0:
            agent.save('dqn_model.pth')

    agent.save('dqn_model.pth')
    pygame.quit()

def play(agent, env):
    running = True
    while running:
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    agent.save('dqn_model.pth')
                    running = False
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        agent.save('dqn_model.pth')
                        running = False
                        pygame.quit()
                        return

            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # Запоминаем переход и обучаемся
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

            env.render()
            pygame.time.delay(10)

        # agent.update_target_network()
        print(f"Результат игры: {total_reward:.2f}")

if __name__ == "__main__":
    env = BreakoutEnv()
    state_size = env.get_state().shape[0]
    action_size = 3  # действия: стоять, влево, вправо
    agent = DQNAgent(state_size, action_size)

    model_filepath = 'dqn_model.pth'

    if agent.load(model_filepath):
        print("Запуск игры с загруженной моделью.")
        play(agent, env)
    else:
        print("Начинаем обучение, так как модель не найдена.")
        episodes = 1000
        train(agent, env, episodes)
