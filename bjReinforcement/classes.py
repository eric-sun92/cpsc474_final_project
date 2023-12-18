from numpy.random import normal, choice
from numpy import power, floor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

class Agent:
    """
    A base class for implementing a player in the Blackjack game.
    It supports both state-value and action-value based learning.
    """

    def __init__(self):
        self.hand = Hand()
        self.state = (0, 0, 0)  # (hand value, num usable ace, dealer's card)
        self.cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]  # Card values (10 repeated for face cards)
        self.values = self.initialize_values()

    def deal_card(self):
        """
        Adds a new card to the player's hand and updates the state.
        """
        self.hand.add_card(choice(self.cards))
        self.state = (self.hand.value, self.hand.num_usable_aces, self.state[2])


    def set_state(self, state):
        """
        Sets the current state of the agent.
        :param state: A tuple representing the state (hand value, number of usable aces, dealer's card).
        """
        self.state = state
        self.hand.value = state[0]
        self.hand.num_usable_aces = state[1]

    def initialize_values(self):
        """
        Initializes random values for Q-values (state-action pairs).
        :return: A dictionary of initialized values.
        """
        values = {}
        for hand_value in range(2, 22):
            for num_usable_aces in range(3):  # Assuming a maximum of 2 usable aces
                for dealer_card in range(2, 12):
                    for action in ['hit', 'stand']:
                        values[(hand_value, num_usable_aces, dealer_card, action)] = round(normal(0, 0.1), 2)
        return values


class Dealer:
    """
    Represents the dealer in a game of Blackjack. 
    The dealer follows a fixed policy for playing the hand.
    """

    def __init__(self):
        self.hand = Hand()
        self.cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]  # Card values (10 repeated for face cards)

    def play(self, hit_soft_17=False):
        """
        Executes the dealer's play according to the game rules.
        :param hit_soft_17: A boolean indicating whether the dealer hits on a soft 17 (hand value 17 with a usable ace).
        :return: The final value of the dealer's hand after playing.
        """
        while True:
            if self.hand.value < 17 or (hit_soft_17 and self.hand.value == 17 and self.hand.usable_ace):
                self.hand.add_card(choice(self.cards))
            else:
                break
        return self.hand.value

class Hand:
    """
    Represents a hand of cards in Blackjack.
    Keeps track of the hand's value, and the number of usable aces.
    """

    def __init__(self):
        self.value = 0
        self.num_usable_aces = 0

    def add_card(self, card_value):
        """
        Adds a new card to the hand.
        :param card_value: The value of the card to add.
        """
        if card_value != 11:  # For non-Ace cards
            self.value += card_value
        else:  # Handling an Ace
            if self.value + 11 > 21:  # Ace counts as 1 if 11 would cause bust
                self.value += 1
            else:  # Ace counts as 11
                self.value += 11
                self.num_usable_aces += 1

        if self.value > 21 and self.num_usable_aces > 0:  # Adjust if bust with usable Ace
            self.value -= 10
            self.num_usable_aces -= 1

    def clear_hand(self):
        """
        Resets the hand to its initial state.
        """
        self.value = 0
        self.num_usable_aces = 0

class Environment():
    '''game environment for training agent'''
    def __init__(self, player):
        self.dealer = Dealer()
        self.player = player

    def set_game(self):
        '''prepare new hand/episode'''
        cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
        first_card, second_card, dealer_card = choice(cards, 3, replace=True)
        num_usable_aces = 0
        self.player.hand.clear_hand()
        self.dealer.hand.clear_hand()

        # Add first and second card to player's hand and check for usable aces
        self.player.hand.add_card(first_card)
        self.player.hand.add_card(second_card)
        num_usable_aces = self.player.hand.num_usable_aces

        # Update state with the new structure
        state = (self.player.hand.value, num_usable_aces, dealer_card)
        self.player.set_state(state)

        # Add card to dealer's hand
        self.dealer.hand.add_card(dealer_card)

    def evaluate_hand(self, player_result, dealer_result):
        '''evaluate player and dealer hands, propagate reward for the player'''
        reward = 0
        if player_result <= 21:
            if dealer_result > 21:
                reward = 1
            else:
                if player_result > dealer_result:
                    reward = 1
                elif player_result == dealer_result:
                    reward = 0
                else:
                    reward = -1
        else:
            reward = -1
        return reward

    def q_train(self, num_epochs):
        '''training loop, train agent for a fixed number of epochs'''
        for _ in range(num_epochs):
            self.set_game()
            player_result = self.player.play(training_flag = 1)
            dealer_result = self.dealer.play()
            reward = self.evaluate_hand(player_result, dealer_result)
            
            if self.player.hand.value <= 21:
                self.player.values[self.player.state + ('stand',)] += self.player.alpha * (reward - self.player.values[self.player.state + ('stand',)])

    def mc_train(self, epochs = 100):
        '''training loop, train agent for a fixed number of epochs'''
        for _ in range(epochs):
            self.set_game()
            player_result = self.player.play(training_flag = 1)
            dealer_result = self.dealer.play()
            reward = self.evaluate_hand(player_result, dealer_result)
            self.player.propagate_reward(reward)
    # Add this method to the Environment class in classes.py

    def td_train(self, epochs=50000):
        for _ in range(epochs):
            self.set_game()
            player_result = self.player.play(training_flag=True)
            dealer_result = self.dealer.play()
            reward = self.evaluate_hand(player_result, dealer_result)

            if self.player.hand.value <= 21:
                self.player.update_values(self.player.state, 'stand', reward, None)
                
    def sarsa_train(self, num_epochs=50000):
        for _ in range(num_epochs):
            self.set_game()
            player_result = self.player.play(training_flag = 1)
            dealer_result = self.dealer.play()
            reward = self.evaluate_hand(player_result, dealer_result)
            
            if self.player.hand.value <= 21:
                self.player.values[self.player.state + ('stand',)] += self.player.alpha * (reward - self.player.values[self.player.state + ('stand',)])

                        
    def test(self, num_epochs):
        '''test agent's performance'''
        wins = 0
        for _ in range(num_epochs):
            self.set_game()
            player_result = self.player.play(training_flag = 0)
            dealer_result = self.dealer.play()
            if self.evaluate_hand(player_result, dealer_result) == 1:
                wins += 1
        return wins / num_epochs
    
    
