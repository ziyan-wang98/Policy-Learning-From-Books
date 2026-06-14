import gym
import numpy as np
from gym import spaces


class TicTacToeEnv(gym.Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.env_name = 'tictactoe-v0'
        self.action_space = spaces.Discrete(9)
        
        self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=np.int32)
        
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=np.int32)
        self.current_player = 1  
        self.done = False
        return self.board

    def step(self, action):
        if self.board[action] != 0:
            return self.board, -1, True, {'invalid_move': True}


        self.board[action] = self.current_player


        if self._check_win():
            reward = 1 if self.current_player == 1 else -1
            self.done = True
            return self.board, reward, self.done, {}


        if np.all(self.board != 0):
            self.done = True
            return self.board, 0, self.done, {}

        # Switch players.
        self.current_player = 3 - self.current_player  # Toggle between players 1 and 2.

        return self.board, 0, self.done, {}

    def check_win(self, board=None, ret_player=False):
        return self._check_win(board, ret_player)
    
    def _check_win(self, board=None, ret_player=False):
        if board is None:
            board = self.board
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Row
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Column
            [0, 4, 8], [2, 4, 6]  # Diagonal
        ]
        
        for combo in win_combinations:
            if board[combo[0]] != 0 and \
               board[combo[0]] == board[combo[1]] == board[combo[2]]:
                if ret_player:
                    return True, board[combo[0]]
                else:
                    return True
        if ret_player:
            return False, None
        else:
            return False

    def render(self, mode='human'):
        if mode == 'human':
            print("-------------")
            for i in range(3):
                row = self.board[i*3:(i+1)*3]
                print("|", end="")
                for cell in row:
                    if cell == 0:
                        print("   |", end="")
                    elif cell == 1:
                        print(" X |", end="")
                    else:
                        print(" O |", end="")
                print("\n-------------")
        else:
            return self.board

    def available_moves(self, board=None):
        if board is None:
            board = self.board
        return [i for i, x in enumerate(board) if x == 0]
    
    @staticmethod
    def vector_state_to_text(state):
        board = ['1' if cell == 1 else ('2' if cell == 2 else '0') for cell in state]

        text = (f"\n[{board[0]}, {board[1]}, {board[2]}, {board[3]}, {board[4]}, {board[5]}, {board[6]}, {board[7]}, {board[8]}] \n")
        if len(board) > 9:
            text += f"Now Player {'X(1)' if state[9] == 1 else 'O(2)' if state[9] == 2 else 'Initial'} Acts.\n"
        return text
