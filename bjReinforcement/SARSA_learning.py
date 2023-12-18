from classes import Agent
import random

class PlayerSARSA(Agent):
    def __init__(self, epsilon = 0.1, alpha = 0.5, gamma = 1):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def play(self, training_flag):
        while self.hand.value <= 21:
            # Current State
            state = self.state

            # Select action based on ε-greedy policy
            action = self.epsilon_greedy_action(state, training_flag)

            # Perform the action
            if action == 'stand':
                break
            self.deal_card()

            # Update Q-values if training
            if training_flag:
                if self.hand.value > 21:
                    reward = -1  # Penalty for going over 21
                    self.update_q_value(state, action, reward)
                else:
                    # Select next action based on ε-greedy policy for the new state
                    next_action = self.epsilon_greedy_action(self.state, training_flag)
                    reward = 0  # Standard reward for ongoing game
                    self.update_q_value(state, action, reward, next_action)

        return self.hand.value

    def epsilon_greedy_action(self, state, training_flag):
        epsilon = self.epsilon if training_flag else 0
        if random.random() < epsilon:
            return 'hit' if random.randint(0, 1) == 0 else 'stand'
        else:
            return 'hit' if self.values[state + ('hit',)] >= self.values[state + ('stand',)] else 'stand'

    def update_q_value(self, state, action, reward, next_action=None):
        if next_action is None or self.hand.value > 21:
            # No next action or terminal state
            self.values[state + (action,)] += self.alpha * (reward - self.values[state + (action,)])
        else:
            next_state = self.state
            next_q_value = self.values[next_state + (next_action,)]
            self.values[state + (action,)] += self.alpha * (reward + self.gamma * next_q_value - self.values[state + (action,)])