import json 
import os

if __name__ == '__main__':
    
    
    player_x_name = 'LLM_Agent'
    player_o_name = 'Minmax_random'
    
    # player_x_name = 'Random'
    # player_o_name = 'LLM_Agent'
    
    num_games = 50
    
    file_path = 'football_llm/tictactoe/result/LLM_Agent_vs_LLM_RAG_50.json'
    #file_path = 'football_llm/tictactoe/result/Random_vs_LLM_Agent_50.json'
    file = open(file_path, 'r')
    data = json.load(file)
    

    trajectories = data

       
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