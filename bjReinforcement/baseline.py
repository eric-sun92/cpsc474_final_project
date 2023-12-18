from classes import Agent
import random

class PlayerBASE(Agent):
    '''agent learning with on-policy first-visit Monte Carlo method'''
    def __init__(self, epsilon = 0.1, discount_rate = 1):
        super().__init__()

    def play(self, epoch = -1, training_flag = True):
        '''simulation of one episode/hand'''
        while self.hand.value <= 14:
            self.deal_card()
        return self.hand.value
