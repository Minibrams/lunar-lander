import numpy as np
import torch as tt
from src.dqn import DQN


class Agent:
    def __init__(self, gamma, epsilon, learning_rate, state_size, batch_size, n_actions,
                 max_transitions=int(1e5), eps_end=0.05, eps_dec=5e-4):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = learning_rate
        self.action_space = np.arange(n_actions, dtype=np.uint8)
        self.max_transitions = max_transitions
        self.batch_size = batch_size
        self.transition_counter = 0

        self.Q = DQN(
          learning_rate, 
          n_actions=n_actions,
          state_size=state_size,
          hidden_1_dims=256, hidden_2_dims=256
        )

        self.state_we_saw_memory = np.zeros((self.max_transitions, state_size), dtype=np.float32)
        self.action_we_took_memory = np.zeros(self.max_transitions, dtype=np.int32)
        self.state_we_got_memory = np.zeros((self.max_transitions, state_size), dtype=np.float32)
        self.reward_we_got_memory = np.zeros(self.max_transitions, dtype=np.float32)
        self.is_terminal_memory = np.zeros(self.max_transitions, dtype=np.bool)

    def remember(self, state_we_saw, action_we_took, state_we_got, reward_we_got, is_terminal):
        index = self.transition_counter % self.max_transitions
        self.state_we_saw_memory[index] = state_we_saw
        self.state_we_got_memory[index] = state_we_got
        self.reward_we_got_memory[index] = reward_we_got
        self.action_we_took_memory[index] = action_we_took
        self.is_terminal_memory[index] = is_terminal

        self.transition_counter += 1

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = tt.tensor([state]).to(self.Q.device)
            actions = self.Q(state)
            action = tt.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.transition_counter < self.batch_size:
            return

        self.Q.optimizer.zero_grad()

        n_stored_transitions = min(self.transition_counter, self.max_transitions)

        batch = np.random.choice(n_stored_transitions, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_we_saw_batch = tt.tensor(self.state_we_saw_memory[batch]).to(self.Q.device)
        action_we_took_batch = self.action_we_took_memory[batch]
        state_we_got_batch = tt.tensor(self.state_we_got_memory[batch]).to(self.Q.device)
        reward_we_got_batch = tt.tensor(self.reward_we_got_memory[batch]).to(self.Q.device)
        is_terminal_batch = tt.tensor(self.is_terminal_memory[batch]).to(self.Q.device)

        q_eval = self.Q(state_we_saw_batch)[batch_index, action_we_took_batch]
        q_next = self.Q(state_we_got_batch)

        q_next[is_terminal_batch] = 0.0

        q_target = reward_we_got_batch + self.gamma*tt.max(q_next, dim=1)[0]

        loss = self.Q.loss(q_target, q_eval).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
