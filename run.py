import gym
import numpy as np
from src.agent import Agent

if __name__ == '__main__':
  n_games = 500
  scores, epsilons = [], []

  env = gym.make('LunarLander-v2')

  agent = Agent(
    gamma=0.99, 
    epsilon=1.0, 
    batch_size=64, 
    n_actions=4, 
    eps_end=0.01,
    state_size=8, 
    learning_rate=1e-3
  )
  
  for i in range(n_games):
      
      score = 0
      is_terminal = False
      state_we_see = env.reset()

      while not is_terminal:

          # env.render()  # Enable to see the game (slower)

          action = agent.choose_action(state_we_see)
          state_we_got, reward, is_terminal, info = env.step(action)

          agent.remember(state_we_see, action, state_we_got, reward, is_terminal)
          agent.learn()
          
          state_we_see = state_we_got

          score += reward

      scores.append(score)
      epsilons.append(agent.epsilon)
      avg_score = np.mean(scores[-100:])

      print(f'Game {i}: Score {score} (Avg. is {avg_score}), epsilon {agent.epsilon}')

