import sys
import os
import json
sys.path.append('football_llm')
from envs.tictactoe import TicTacToeEnv
from ttt_gym_player import RandomPlayer, MinimaxPlayer, LLM_Agent_Player, LLM_RAG_Player, CQLPlayer, Minimax_random_Player
import numpy as np

def create_player(player_name, player_mark):
    if player_name == 'Random':
        return RandomPlayer(player_mark)
    elif player_name == 'Minmax':
        return MinimaxPlayer(player_mark)
    elif player_name in 'LLM_Agent':
        return LLM_Agent_Player(player_mark)
    elif player_name == 'LLM_RAG':
        #exp_name_pattern = 'comment=unc-test&alpha=100&keep_rate=1.0&neg_keep_rate=0.0&coef_r=0.25&coef_t=0.25_20240806115620'
        return LLM_RAG_Player(player_mark)
    elif player_name == 'URI':
        return CQLPlayer(player_mark)
    elif player_name == 'Minmax_random':
        return Minimax_random_Player(player_mark)
    elif player_name == 'Human':
        raise NotImplementedError("Human player not implemented yet")
    else:
        raise ValueError(f"Invalid player name for {player_mark} player")

import numpy as np

def play_data_save(x_player_name, o_player_name, num_games=1, save_path=None, render=False):
    x_player = create_player(x_player_name, 'X')
    o_player = create_player(o_player_name, 'O')
    all_game_trajectories = []

    for i in range(num_games):
        env = TicTacToeEnv()
        obs = env.reset()
        done = False
        current_player = x_player
        game_trajectory = []

        while not done:
            if render:
                print(obs)
            action = current_player.get_action(env)
            
            # Record the state before action
            game_trajectory.append({
                'state': env.board.tolist() if isinstance(env.board, np.ndarray) else env.board,
                'player': 1 if current_player == x_player else 2,
                'action': int(action),  # Ensure action is JSON serializable
                'reward': None,  # Will be filled later
                'invalid_move': False if env.board[action] == 0 else True
            })
            
            obs, reward, done, info = env.step(action)
            current_player = o_player if current_player == x_player else x_player

        # Update the last step with the final reward
        game_trajectory[-1]['reward'] = float(reward)  # Ensure reward is JSON serializable

        # Add an extra step to represent the final state
        game_trajectory.append({
            'state': env.board.tolist() if isinstance(env.board, np.ndarray) else env.board,
            'player': None,
            'action': None,
            'reward': None, 
            'invalid_move': None
        })

        print(f"Game {i+1} completed")
        
        # print win side, if all vaild moves are taken
        # print(game_trajectory)
        
        if game_trajectory[-2]['invalid_move'] == True:
            if game_trajectory[-2]['player'] == 1:
                print(f"{o_player_name} wins!, {x_player_name} makes invalid move")
            else:
                print(f"{x_player_name} wins!, {o_player_name} makes invalid move")
        else:
            if reward == 0:
                print("Draw!")
            elif reward == 1:
                print(f"{x_player_name} wins!")
            else:
                print(f"{o_player_name} wins!")
        
        all_game_trajectories.append(game_trajectory)

    if save_path:
        file_name = f"{x_player_name}_vs_{o_player_name}_{num_games}.json"
        with open(os.path.join(save_path,file_name), 'w') as f:
            json.dump(all_game_trajectories, f, indent=2)
        print(f"Game trajectories saved to {save_path}")

    return all_game_trajectories

if __name__ == '__main__':
    player_x_name = 'LLM_RAG'
    player_o_name = 'Minmax_random'
    res_dir_path = 'result'
    trajectories = play_data_save(player_x_name, player_o_name, num_games=50, save_path=res_dir_path, render=False)
    
    # Print the first trajectory
    print("\nFirst game trajectory:")
    for step in trajectories[0]:
        print(step)
    
    
    win_count = 0
    draw_count = 0
    lose_count = 0
    invalid_move_count = 0
    opponent_invalid_move_count = 0
    
    for traj in trajectories:
        if traj[-2]['invalid_move'] == True:
            if traj[-2]['player'] == 1:
                lose_count += 1
                invalid_move_count += 1
            else:
                win_count += 1
                opponent_invalid_move_count += 1
        elif traj[-2]['reward'] == 1:
            win_count += 1
        elif traj[-2]['reward'] == 0:
            draw_count += 1
        else:
            lose_count += 1        
        
    # Print the win rate
    win_rate = win_count / len(trajectories)
    print(f"\nWin rate for {player_x_name} vs {player_o_name}: {win_rate:.2f}")
    draw_rate = draw_count / len(trajectories)
    print(f"Draw rate for {player_x_name} vs {player_o_name}: {draw_rate:.2f}")
    lose_rate = 1 - win_rate - draw_rate
    print(f"Lose rate for {player_x_name} vs {player_o_name}: {lose_rate:.2f}")
    
    # Print the invalid move rate
    invalid_move_rate = invalid_move_count / len(trajectories)
    print(f"\nInvalid move rate for {player_x_name}: {invalid_move_rate:.2f}")
    opponent_invalid_move_rate = opponent_invalid_move_count / len(trajectories)
    print(f"Invalid move rate for {player_o_name}: {opponent_invalid_move_rate:.2f}")