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

    def __init__(self, values_type='state'):
        self.hand = Hand()
        self.state = (0, False, 0)  # (hand value, usable ace, dealer's card)
        self.hands_played = 0
        self.hands_won = 0
        self.hands_drawn = 0
        self.cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]  # Card values (10 repeated for face cards)
        self.values_type = 'state' if values_type == 'state' else 'action'
        self.values = self.initialize_values()

    def deal_card(self):
        """
        Adds a new card to the player's hand and updates the state.
        """
        self.hand.add_card(choice(self.cards))
        self.state = (self.hand.value, self.hand.usable_ace, self.state[2])

    def set_state(self, state):
        """
        Sets the current state of the agent.
        :param state: A tuple representing the state (hand value, usable ace, dealer's card).
        """
        self.state = state
        self.hand.value, self.hand.usable_ace = state[:2]

    def initialize_values(self):
        """
        Initializes random values for state-values or action-values.
        :return: A dictionary of initialized values.
        """
        values = {}
        for hand_value in range(2, 22):
            for dealer_card in range(2, 12):
                if self.values_type == 'state':
                    values[(hand_value, True, dealer_card)] = round(normal(0, 0.1), 2)
                    values[(hand_value, False, dealer_card)] = round(normal(0, 0.1), 2)
                else:
                    action_values = {'hit': round(normal(0, 0.1), 2), 'stand': round(normal(0, 0.1), 2)}
                    values[(hand_value, True, dealer_card)] = action_values
                    values[(hand_value, False, dealer_card)] = action_values.copy()
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
    Keeps track of the hand's value and whether it contains a usable ace.
    """

    def __init__(self):
        self.value = 0
        self.usable_ace = False

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
                self.usable_ace = True

        if self.value > 21 and self.usable_ace:  # Adjust if bust with usable Ace
            self.value -= 10
            self.usable_ace = False

    def clear_hand(self):
        """
        Resets the hand to its initial state.
        """
        self.value = 0
        self.usable_ace = False

class Environment():
    '''game environment for training agent'''
    def __init__(self, player):
        self.dealer = Dealer()
        self.player = player
        self.training_epochs = 1
        self.win_history = [] # list of win averages through training
        self.training_history = [] # list of saved training epochs

    def set_game(self):
        '''prepare new hand/episode'''
        cards = list(range(2,12)) + [10]*3
        first_card, second_card, dealer_card = choice(cards, 3, replace=True)
        usable_ace = False
        self.player.hand.clear_hand()
        self.dealer.hand.clear_hand()
        # first card is Ace
        if first_card == 11:
            if second_card == 11:
                second_card = 1
            usable_ace = True
        else:
            # if first card isn't Ace
            if second_card == 11:
                usable_ace = True
        state = (first_card + second_card, usable_ace, dealer_card)
        self.player.set_state(state)
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

    def train(self, epochs = 100, save_frequency = 10):
        '''training loop, train agent for a fixed number of epochs'''
        for i in range(self.training_epochs, self.training_epochs + epochs):
            self.set_game()
            player_result = self.player.play(epoch = i)
            dealer_result = self.dealer.play()
            reward = self.evaluate_hand(player_result, dealer_result)
            self.player.propagate_reward(reward)
            if i % save_frequency == 0:
                # save current winning average
                self.training_history.append(i)
                self.win_history.append(self.player.hands_won 
                            / self.player.hands_played)
        self.training_epochs += epochs

    @property
    def history(self):
        '''gets training history and wining average throughout training'''
        return self.training_history, self.win_history

    def test(self, epochs = 100):
        '''test agent's performance'''
        wins = 0
        for _ in range(epochs):
            self.set_game()
            player_result = self.player.play(training_phase = False)
            dealer_result = self.dealer.play()
            if self.evaluate_hand(player_result, dealer_result) == 1:
                wins += 1
        return wins / epochs

    def plot_state_values(self, usable_ace = False):
        '''plot current state values'''
        values = []
        # for each state, save max value for one of actions {hit, stand}
        for j in range(11,1,-1):
            ls = [max(self.player.values[(i,usable_ace,j)]['hit'], 
                      self.player.values[(i,usable_ace,j)]['stand'])
                          for i in range(12,22)]
            values.append(ls)
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(values, annot=True, cmap="Blues",
                    cbar_kws={'label': 'state values'})
        ax.figure.axes[-1].yaxis.label.set_size(14)
        ax.set_xticklabels(list(range(12,22)))
        ax.set_yticklabels(['A'] + list(range(10,1,-1)))
        for text in ax.texts:
            text.set_size(13)
        plt.title('State values, usable ace = %s' % usable_ace, fontsize=15)
        plt.ylabel("dealer's hand", fontsize=14)
        plt.xlabel("player's hand", fontsize=14)
        plt.show()

    def plot_strategy(self, usable_ace = False):
        actions = []
        # 1 for hit, 0 for stand
        for j in range(11,1,-1):
            ls = [1 if self.player.values[(i,usable_ace,j)]['hit'] > 
                  self.player.values[(i,usable_ace,j)]['stand'] else 0 
                  for i in range(12,22)]
            actions.append(ls)
        cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=0.9, n_colors=2)
        fig, ax = plt.subplots(figsize=(11,9))
        sns.heatmap(actions, annot=True, cmap=ListedColormap(cmap),
                    linecolor='lightgray', cbar=True, linewidths=0.75,
                    cbar_kws={'orientation': 'vertical'})
        ax.figure.axes[-1].yaxis.label.set_size(14)
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0.25, 0.75])
        colorbar.set_ticklabels(['STAND', 'HIT'])
        ax.set_xticklabels(list(range(12,22)))
        ax.set_yticklabels(['A'] + list(range(10,1,-1)))
        for text in ax.texts:
            if str(text.get_text()) == '1':
                text.set_text('H')
            else:
                text.set_text('S')
            text.set_size(13)
        plt.title('Current strategy, usable ace = %s, H - hit, S - stand' % 
                  usable_ace, fontsize=15)
        plt.xlabel("player's hand", fontsize=14)
        plt.ylabel("dealer's hand", fontsize=14)
        plt.show()

    @staticmethod
    def scheduler(schedule = 'constant', initial_step_size = 0.5, 
                  decay = 0.1, drop = 0.5, epochs_drop = 1000, decay_rate = 0.5):
        '''function returns scheduler-function that controls step-size 
        during training
        schedule: constant, time_based, step_decay, exponential_decay
        decay: value for time-based and exponential scheduler
        drop: value for step-decay scheduler
        epochs_drop: decaying step_size frequency
        decay_rate: exponent for exponential scheduler, float in [0,1]'''

        # step_size parameter is last step-size used during the training
        # epoch parameter is current training epoch
        def constant(epoch):
            return initial_step_size

        def time_based(epoch):
            return initial_step_size / (1 + decay*epoch)

        def step_decay(epoch):
            return initial_step_size*power(drop, floor(epoch/epochs_drop))

        def exponential_decay(epoch):
            return initial_step_size*power(decay_rate, decay*epoch)

        schedulers = {'constant':constant, 'time_based':time_based,
                'step_decay':step_decay, 'exponential_decay':exponential_decay}
        if schedule not in schedulers:
            schedule = 'constant'
        return schedulers[schedule]