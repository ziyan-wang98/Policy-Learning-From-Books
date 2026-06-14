import math
import numpy as np
import os
import time
import json
import random
from copy import deepcopy
from player import HumanPlayer, RandomComputerPlayer, MinmaxPlayer, gpt35Player
from CONFIG import *

data_path = os.path.join(DATASET_PATH, 'tic_tac_toe_data')
if not os.path.exists(data_path):
    os.makedirs(data_path)

class TicTacToe(object):
    def __init__(self):
        self.board = self.make_board()
        self.current_winner = None
        self.ending = None

    @staticmethod
    def make_board():
        return [' ' for _ in range(9)]

    def print_board(self):
        for row in [self.board[i*3:(i+1) * 3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums():
        # 0 | 1 | 2
        number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = deepcopy(letter)
                self.ending = f'Player {letter} wins'
            return True
        else:
            # make invalid moves, oppo player wins.
            self.current_winner = 'O' if letter == 'X' else 'X'
            self.ending = f'Player {self.current_winner} wins because of invalid moving taken by {letter}.'
            return False

    def winner(self, square, letter):
        # check the row
        row_ind = math.floor(square / 3)
        row = self.board[row_ind*3:(row_ind+1)*3]
        # print('row', row)
        if all([s == letter for s in row]):
            return True
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        # print('col', column)
        if all([s == letter for s in column]):
            return True
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            # print('diag1', diagonal1)
            if all([s == letter for s in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            # print('diag2', diagonal2)
            if all([s == letter for s in diagonal2]):
                return True
        return False

    def empty_squares(self):
        return ' ' in self.board

    def num_empty_squares(self):
        return self.board.count(' ')

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == " "]
    
    def get_state_repr(self):
        return [0 if spot == ' ' else (1 if spot == 'X' else 2) for spot in self.board]


def play(game, x_player, o_player, print_game=True):

    if print_game:
        game.print_board_nums()

    letter = 'X'
    while game.empty_squares():
        if letter == 'O':
            square = o_player.get_move(game)
        else:
            square = x_player.get_move(game)
        if game.make_move(square, letter):

            if print_game:
                print(letter + ' makes a move to square {}'.format(square))
                game.print_board()
                print('')

            if game.current_winner:
                if print_game:
                    print(letter + ' wins!')
                return letter  # ends the loop and exits the game
            letter = 'O' if letter == 'X' else 'X'  # switches player

        time.sleep(.8)

    if print_game:
        print('It\'s a tie!')
        
        
def build_minimax_vs_all_tree_all_data(state, player, current_trajectory, minmax_player, noise=0.0):
    trajectories = []
    state = deepcopy(state)
    if state.current_winner or not state.empty_squares():
        # Culculate the reward
        # if state.current_winner == 'X':
        #     reward_x = 1
        # elif state.current_winner == 'O':
        #     reward_x = -1
        # else:
        #     reward_x = 0
   
        # for step in current_trajectory:
        #     if step['player'] == 1:  # X player
        #         step['reward'] = reward_x
        #     elif step['player'] == 2:  # O player
        #         step['reward'] = -reward_x 
        # return [current_trajectory]
        
        if state.current_winner == 'X':
            final_reward_x = 1
        elif state.current_winner == 'O':
            final_reward_x = -1
        else:
            final_reward_x = 0
   
        # Assign rewards: 0 for all steps except the last one
        for i, step in enumerate(current_trajectory):
            if i == len(current_trajectory) - 1:  # Last step
                step['reward'] = final_reward_x
                if final_reward_x == 0:
                    step['ending'] = 'Draw.'
                else:
                    step['ending'] = state.ending
            else:
                step['reward'] = 0

        return [current_trajectory]

    if player == 'X':  # Minimax player
        if len(current_trajectory) == 1:  # First move
            for move in state.available_moves():
                new_state = deepcopy(state)
                new_state.make_move(move, player)
                new_trajectory = current_trajectory + [{
                    'state': new_state.get_state_repr(),
                    'player': 1,
                    'action': move,
                    'reward': None,
                    'ending': None,
                }]
                trajectories.extend(build_minimax_vs_all_tree_all_data(new_state, 'O', new_trajectory, minmax_player, noise))
        else:
            repeat_times = int(min(len(state.available_moves()), np.ceil(1 / (1 - noise))))
            for _ in range(repeat_times):
                if random.random() >= noise:
                    best_move = minmax_player.minimax(deepcopy(state), player)['position']
                    early_stop = True
                else:
                    best_move = random.randint(0, 8)
                    early_stop = False
                new_state = deepcopy(state)
                new_state.make_move(best_move, player)
                new_trajectory = current_trajectory + [{
                    'state': new_state.get_state_repr(),
                    'player': 1,
                    'action': best_move,
                    'reward': None,
                    'ending': None,
                }]
                trajectories.extend(build_minimax_vs_all_tree_all_data(new_state, 'O', new_trajectory, minmax_player, noise))
                if early_stop:
                    break
    else:  # O player - consider all moves
        for move in state.available_moves():
            new_state = deepcopy(state)
            new_state.make_move(move, player)
            new_trajectory = current_trajectory + [{
                'state': new_state.get_state_repr(),
                'player': 2,
                'action': move,
                'reward': None,
                'ending': None,
            }]
            trajectories.extend(build_minimax_vs_all_tree_all_data(new_state, 'X', new_trajectory, minmax_player, noise))

    return trajectories



    
    
    
    

def build_minimax_vs_all_tree(state, player, current_trajectory, minmax_player):
    trajectories = []

    if state.current_winner or not state.empty_squares():
        return [current_trajectory]

    if player == 'X':  # Minimax player
        if len(current_trajectory) == 1:  # First move
            for move in state.available_moves():
                new_state = deepcopy(state)
                new_state.make_move(move, player)
                new_trajectory = current_trajectory + [new_state.get_state_repr()]
                trajectories.extend(build_minimax_vs_all_tree(new_state, 'O', new_trajectory, minmax_player))
        else:
            best_move = minmax_player.minimax(state, player)['position']
            new_state = deepcopy(state)
            new_state.make_move(best_move, player)
            new_trajectory = current_trajectory + [new_state.get_state_repr()]
            trajectories.extend(build_minimax_vs_all_tree(new_state, 'O', new_trajectory, minmax_player))
    else:  # O player - consider all moves
        for move in state.available_moves():
            new_state = deepcopy(state)
            new_state.make_move(move, player)
            new_trajectory = current_trajectory + [new_state.get_state_repr()]
            trajectories.extend(build_minimax_vs_all_tree(new_state, 'X', new_trajectory, minmax_player))

    return trajectories

def collect_minmax_data(collect_data_all_flag = True, noise=0.0):
 
    if collect_data_all_flag:
        initial_state = TicTacToe()
        minmax_player = MinmaxPlayer('X')
        all_trajectories = build_minimax_vs_all_tree_all_data(initial_state, 'X', [{
            'state': initial_state.get_state_repr(),
            'player': None,
            'action': None,
            'reward': None
        }], minmax_player, noise=noise)
    else:
        initial_state = TicTacToe()
        minmax_player = MinmaxPlayer('X')
        all_trajectories = build_minimax_vs_all_tree(initial_state, 'X', [initial_state.get_state_repr()], minmax_player)

    
    data = {
        "total_trajectories": len(all_trajectories),
        "trajectories": all_trajectories,
        "statistics": {
            "lengths": {
                "min": min(len(t) for t in all_trajectories),
                "max": max(len(t) for t in all_trajectories),
                "average": sum(len(t) for t in all_trajectories) / len(all_trajectories)
            },
            "first_moves": {}
        }
    }
    if collect_data_all_flag:
        data["statistics"]["rewards"] = {
            "X": {"win": 0, "loss": 0, "draw": 0},
            "O": {"win": 0, "loss": 0, "draw": 0}
        }

        for trajectory in all_trajectories:
            # import pdb; pdb.set_trace()
            final_reward_x = trajectory[-1]['reward'] if trajectory[-1]['player'] == 1 else -trajectory[-1]['reward']
            if final_reward_x == 1:
                data["statistics"]["rewards"]["X"]["win"] += 1
                data["statistics"]["rewards"]["O"]["loss"] += 1
            elif final_reward_x == -1:
                data["statistics"]["rewards"]["X"]["loss"] += 1
                data["statistics"]["rewards"]["O"]["win"] += 1
            else:
                data["statistics"]["rewards"]["X"]["draw"] += 1
                data["statistics"]["rewards"]["O"]["draw"] += 1
                
    if collect_data_all_flag:
        for trajectory in all_trajectories:
            first_move = trajectory[1]['action']  # Get the first action.
            data["statistics"]["first_moves"][first_move] = data["statistics"]["first_moves"].get(first_move, 0) + 1
        
        with open(os.path.join(data_path, f'all_data_minimax_tic_tac_toe_trajectories-noise-{noise}.json'), 'w') as f:
            json.dump(data, f, indent=2)
    else:
        for trajectory in all_trajectories:
            first_move = trajectory[1].index(1)  # Find the first X position.
            data["statistics"]["first_moves"][first_move] = data["statistics"]["first_moves"].get(first_move, 0) + 1
        with open(os.path.join(data_path, 'minimax_tic_tac_toe_trajectories.json'), 'w') as f:
            json.dump(data, f, indent=2)

    print(f"Total number of trajectories: {len(all_trajectories)}")
    
    # Print trajectory lengths
    lengths = [len(trajectory) for trajectory in all_trajectories]
    print(f"\nTrajectory lengths: min = {min(lengths)}, max = {max(lengths)}, average = {sum(lengths)/len(lengths):.2f}")

    # Count trajectories starting with each possible first move
    if collect_data_all_flag:
        first_moves = {}
        for trajectory in all_trajectories:
            first_move = trajectory[1]['action']  # Get the first action.
            first_moves[first_move] = first_moves.get(first_move, 0) + 1
    else:
        first_moves = {}
        for trajectory in all_trajectories:
            first_move = trajectory[1].index(1)  # Find the position of the first 'X'
            first_moves[first_move] = first_moves.get(first_move, 0) + 1
    
    print("\nTrajectories per first move:")
    for move, count in sorted(first_moves.items()):
        print(f"Move {move}: {count} trajectories")

    if collect_data_all_flag:
        print("\nReward statistics:")
        print("X - Wins: {}, Losses: {}, Draws: {}".format(
            data["statistics"]["rewards"]["X"]["win"],
            data["statistics"]["rewards"]["X"]["loss"],
            data["statistics"]["rewards"]["X"]["draw"]
        ))
        print("O - Wins: {}, Losses: {}, Draws: {}".format(
            data["statistics"]["rewards"]["O"]["win"],
            data["statistics"]["rewards"]["O"]["loss"],
            data["statistics"]["rewards"]["O"]["draw"]
        ))



    # Print the first five games
    print("\nFirst five games:")
    for i, trajectory in enumerate(all_trajectories[:5]):
        print(f"\nGame {i+1}:")
        
        if collect_data_all_flag:
            print_game_all_data(trajectory)
        else:
            print_game(trajectory)
        if i < 4:  # Add a separator between games, except after the last one
            print("=" * 20)
            
            
def print_game_all_data(trajectory):
    for i, step in enumerate(trajectory):
        print(f"Move {i}:")
        print(step)
        print(f"Player: {'X' if step['player'] == 1 else 'O' if step['player'] == 2 else 'Initial'}")
        print(f"Action: {step['action']}")
        print(f"Reward: {step['reward']}")
        print_board(step['state'])
        print()



def print_game(trajectory):
    for i, state in enumerate(trajectory):
        print(f"Move {i}:")
        print_board(state)
        print()

def print_board(state):
    board = ['X' if cell == 1 else ('O' if cell == 2 else ' ') for cell in state]
    print(f" {board[0]} | {board[1]} | {board[2]} ")
    print("-----------")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("-----------")
    print(f" {board[6]} | {board[7]} | {board[8]} ")
    


def create_player(player_name, player_mark):
    if player_name == 'Random':
        return RandomComputerPlayer(player_mark)
    elif player_name == 'Minmax_(Oracle)':
        return MinmaxPlayer(player_mark)
    elif player_name in 'LLM_Agent':
        return gpt35Player(player_mark)
    elif player_name == 'LLM_RAG':
        raise NotImplementedError("LLM_RAG player not implemented yet")
    elif player_name == 'URI':
        raise NotImplementedError("URI player not implemented yet")
    elif player_name == 'Human':
        return HumanPlayer(player_mark)
    else:
        raise ValueError(f"Invalid player name for {player_mark} player")


def play_data_save(x_player_name, o_player_name, num_games=1):
    
    x_player = create_player(x_player_name, 'X')
    o_player = create_player(o_player_name, 'O')
    play_results = []

    
    
    for i in range(num_games):
        t = TicTacToe()
        play(t, x_player, o_player, print_game=False)
        import pdb; pdb.set_trace()
        play_results.append(t.current_winner)
    
    # Save the results
    file_name = f"result/{x_player_name}_vs_{o_player_name}_results_{num_games}.json"
    with open(file_name, 'w') as f:
        json.dump(play_results, f, indent=2)
    
    
    
    
if __name__ == '__main__':
    # x player is the first player and o is the second player
    
    collect_data = True
    play_test = False
    play_save = False
    
    if collect_data:
        collect_minmax_data(noise=0.2)
        
    if play_test:
        x_player = gpt35Player('X')
        # o_player = HumanPlayer('O')
        o_player = MinmaxPlayer('O')
        t = TicTacToe()
        play(t, x_player, o_player, print_game=True)


    if play_save:
        x_player_name = 'LLM_Agent'
        o_player_name = 'Random'
        play_data_save(x_player_name, o_player_name, num_games=1)
