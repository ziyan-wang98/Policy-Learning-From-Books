import os
import sys
sys.path.append('football_llm')
import math
import random
import openai
import copy
from llm.utils.openai_compat import openai_chat_query
import json

class Player():
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        pass


class HumanPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square = input(self.letter + '\'s turn. Input move (0-9): ')
            try:
                val = int(square)
                if val not in game.available_moves():
                    raise ValueError
                valid_square = True
            except ValueError:
                print('Invalid square. Try again.')
        return val


class RandomComputerPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        square = random.choice(game.available_moves())
        return square


class gpt35Player(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        # if len(game.available_moves()) == 9:
        #     square = random.choice(game.available_moves())
        # else:

        square = self.gpt35(game, self.letter)

        return square

    def gpt35(self, state, player):
        state = copy.deepcopy(state)
        global_prompt = """
            # Tic-Tac-Toe
            ## Introduction
                Tic-Tac-Toe is a classic two-player game, unfolds on a 3-by-3 grid where the
                objective is to align three of one's symbols, Xs for the first player and Os for
                the second, either horizontally, vertically, or diagonally. Strategic
                placement is crucial: besides aiming for three in a row, players must also
                block their opponents' potential alignments to avoid defeat. Players can
                place their next move in an empty cell on the 3-by-3 grid.

            ## Rules
                1. The game is played on a 3x3 grid.
                2. Players take turns placing their symbol (X or O) in empty cells.
                3. The first player to get 3 of their symbols in a row (horizontally, vertically, or diagonally) wins.
                4. If all cells are filled and no player has won, the game is a draw.

            ## Game State Representation
                The game board is represented as a string of 9 characters, where:
                - 'X' represents the first player's moves
                - 'O' represents the second player's moves
                - ' ' (space) represents an empty cell

            ## Your Task:
            - Respond with a JSON object containing your move and reasoning.
            - The move should be a single integer from 0 to 8, representing the cell index.
            - Provide a brief explanation for your choice in the reasoning field.
            """

        action_prompt = f"""
            ## Choose the next move
            You are playing as '{player}'.

            The current state of the board is as follows:

                {state.board[0]} | {state.board[1]} | {state.board[2]}
                ---------
                {state.board[3]} | {state.board[4]} | {state.board[5]}
                ---------
                {state.board[6]} | {state.board[7]} | {state.board[8]}

            The next move should be placed in an empty cell on the 3-by-3 grid.
            The available positions for the next move are: {state.available_moves()}

            Positions are represented by the numbers 0-8, corresponding to the cell's index:

                0 | 1 | 2
                ---------
                3 | 4 | 5
                ---------
                6 | 7 | 8

            Analyze the game state and choose your next move. Respond with a JSON object in the following format:

            {{
            "move": <integer 0-8>,
            "reasoning": "<brief explanation for your choice>"
            }}
        """

        action_response = openai_chat_query(global_prompt, action_prompt, model_name=os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini'), req_json=True)
        action_r = json.loads(action_response)
        action_rj = json.loads(action_r["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
        final_action = action_rj["move"]

        return final_action

class MinmaxPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        if len(game.available_moves()) == 9:
            square = random.choice(game.available_moves())
        else:
            square = self.minimax(game, self.letter)['position']
        return square

    def minimax(self, state, player):
        state = copy.deepcopy(state)
        max_player = self.letter  # yourself
        other_player = 'O' if player == 'X' else 'X'

        # first we want to check if the previous move is a winner
        if state.current_winner == other_player:
            return {'position': None, 'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1 * (
                        state.num_empty_squares() + 1)}
        elif not state.empty_squares():
            return {'position': None, 'score': 0}

        if player == max_player:
            best = {'position': None, 'score': -math.inf}  # each score should maximize
        else:
            best = {'position': None, 'score': math.inf}  # each score should minimize
        for possible_move in state.available_moves():
            state.make_move(possible_move, player)
            sim_score = self.minimax(state, other_player)  # simulate a game after making that move

            # undo move
            state.board[possible_move] = ' '
            state.current_winner = None
            sim_score['position'] = possible_move  # this represents the move optimal next move

            if player == max_player:  # X is max player
                if sim_score['score'] > best['score']:
                    best = sim_score
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score
        return best

