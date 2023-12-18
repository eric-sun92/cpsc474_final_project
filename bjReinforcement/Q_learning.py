from classes import Agent
import random

class PlayerQL(Agent):
    def __init__(self, epsilon = 0.1, alpha = 0.5, gamma = 1):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def play(self, training_flag):
        while self.hand.value <= 21:
            action = ''

            if self.hand.value <= 11:
                action = 'hit'
            
            epsilon = self.epsilon if training_flag else 0
            # explore
            if random.random() < epsilon:
                action = 'hit' if random.randint(0, 1) == 0 else 'stand'
            
            # exploit
            else:
                if self.values[self.state + ('hit',)] >= self.values[self.state + ('stand',)]:
                    action = 'hit'
                else:
                    action = 'stand'

            state = self.state
            if action == 'stand':
                break
            self.deal_card()
            if training_flag:
                if self.hand.value > 21:
                    self.values[state + ('hit',)] += self.alpha * (-1 - self.values[state + ('hit',)])
                else:
                    max_value = max(self.values[self.state + ('hit',)], self.values[self.state + ('stand',)])
                    self.values[state + (action,)] += self.alpha * (self.gamma * max_value - self.values[state + (action,)])
        return self.hand.value