from classes import Environment
from Q_learning import PlayerQL
from MC_learning import PlayerMC
from baseline import PlayerBASE
from TD_learning import PlayerTD
from SARSA_learning import PlayerSARSA


if __name__ == "__main__":
      player = PlayerQL(epsilon = .2, alpha = .1)
      env = Environment(player)
      env.q_train(30000)
      result = env.test(num_epochs = 20000)
      print('Q_LEARNING: eps = %s, step-size = %s, test result (win average): %s' 
            % (player.epsilon, player.alpha, result))

      player = PlayerMC(epsilon = 0.2)
      env = Environment(player)
      env.mc_train(50000)
      result = env.test(20000)
      print('FIRST MC: eps = %s, test result (win average): %s' % (env.player.epsilon, result))

      player = PlayerBASE()
      env = Environment(player)
      result = env.test(20000)
      print('BASE: test result (win average): %s' % (result))
      
      player = PlayerTD(alpha = .1, epsilon=.2)
      env = Environment(player)
      env.td_train(100000)
      result = env.test(20000)
      print('TD: test result (win average): %s' % (result))
      
      player = PlayerSARSA(alpha = .05, epsilon=.4)
      env = Environment(player)
      env.sarsa_train(100000)
      result = env.test(20000)
      print('SARSA: test result (win average): %s' % (result))

    
      


    
    
