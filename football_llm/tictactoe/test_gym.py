import sys
import os
sys.path.append('football_llm')
from envs.tictactoe import TicTacToeEnv
from ttt_gym_player import RandomPlayer, MinimaxPlayer, LLM_Agent_Player
import numpy as np

def create_player(player_name, player_mark):
    if player_name == 'Random':
        return RandomPlayer(player_mark)
    elif player_name == 'Minmax':
        return MinimaxPlayer(player_mark)
    elif player_name in 'LLM_Agent':
        return LLM_Agent_Player(player_mark)
    elif player_name == 'LLM_RAG':
        raise NotImplementedError("LLM_RAG player not implemented yet")
    elif player_name == 'URI':
        raise NotImplementedError("URI player not implemented yet")
    elif player_name == 'Human':
        raise NotImplementedError("Human player not implemented yet")
    else:
        raise ValueError(f"Invalid player name for {player_mark} player")

def play_game(x_player_name, o_player_name):
    x_player = create_player(x_player_name, 'X')
    o_player = create_player(o_player_name, 'O')
    env = TicTacToeEnv()
    obs = env.reset()
    done = False
    current_player = x_player

    print("Initial board state:")
    env.render()
    print("--------------------")

    while not done:
        action = current_player.get_action(env)
        player_symbol = 'X' if current_player == x_player else 'O'
        print(f"Player {player_symbol} takes action: {action}")
        
        obs, reward, done, info = env.step(action)
        
        print("Current board state:")
        env.render()
        print("--------------------")
        
        current_player = o_player if current_player == x_player else x_player

    if reward == 1:
        print("Player X wins!")
    elif reward == -1:
        print("Player O wins!")
    else:
        print("It's a draw!")

if __name__ == '__main__':
    player_x_name = 'LLM_Agent'
    player_o_name = 'Minmax'

    print(f"Starting game: {player_x_name} (X) vs {player_o_name} (O)")
    play_game(player_x_name, player_o_name)