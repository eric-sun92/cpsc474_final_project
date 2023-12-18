from classes import Agent
import random

class PlayerTD(Agent):
    def __init__(self, alpha=0.1, gamma=1.0, epsilon=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, training_flag):
        if random.random() < self.epsilon if training_flag else 0:
            return 'hit' if random.randint(0, 1) == 0 else 'stand'
        else:
            return 'hit' if self.values[self.state + ('hit',)] >= self.values[self.state + ('stand',)] else 'stand'

    def play(self, training_flag):
        while self.hand.value <= 21:
            action = self.choose_action(training_flag)

            state = self.state
            if action == 'stand':
                break
            self.deal_card()

            if training_flag:
                reward = -1 if self.hand.value > 21 else 0
                next_state_action = (self.state + ('hit',)) if self.hand.value <= 21 else None
                self.update_values(state, action, reward, next_state_action)
        return self.hand.value

    def update_values(self, state, action, reward, next_state_action):
        if next_state_action:
            self.values[state + (action,)] += self.alpha * (reward + self.gamma * self.values[next_state_action] - self.values[state + (action,)])
        else:
            self.values[state + (action,)] += self.alpha * (reward - self.values[state + (action,)])
