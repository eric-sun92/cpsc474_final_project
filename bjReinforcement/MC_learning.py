from classes import Agent
import random

class PlayerMC(Agent):
    def __init__(self, epsilon, discount_rate=1):
        super().__init__()
        self.epsilon = epsilon # epsilon value for epsilon-greedy policy
        self.appearances = {} # number of appearances of state-action pairs
        self.episode_steps = [] # states-action pairs that appeared during one hand
        self.discount_rate = discount_rate # discount rate for future rewards

    def play(self, training_flag = True):
        """ 
        Plays a hand of the game.

        Args:
            training_flag (bool): Flag to indicate whether the agent is in training or evaluation mode.
        """
        self.episode_steps.clear()

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
           
            self.episode_steps.append((self.state, action))
            if action == 'stand':
                break
            self.deal_card()
        return self.hand.value

    def propagate_reward(self, reward):
        self.episode_steps.reverse()
        for state, action in self.episode_steps:
            # first -> update number of appearances for state-action pair
            if (state, action) not in self.appearances:
                self.appearances[(state, action)] = 1
            else:
                self.appearances[(state, action)] += 1
            # running average value for state-action pair expected return
            self.values[state + (action,)] += (reward - self.values[state + (action,)])/self.appearances[(state, action)]
            reward = self.discount_rate * reward # discounted reward