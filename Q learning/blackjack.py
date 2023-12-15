from classes import Environment
from Q_learning import PlayerQL

if __name__ == "__main__":
    player = PlayerQL(epsilon = .2, alpha = .1)
    env = Environment(player)
    env.train(30000)
    result = env.test(num_epochs = 20000)
    print('eps = %s, step-size = %s, test result (win average): %s' 
          % (player.epsilon, player.alpha, result))
