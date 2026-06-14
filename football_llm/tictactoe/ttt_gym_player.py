import random
import math
import numpy as np
import json
import sys
import copy
import os
import os.path as osp
import uuid
from termcolor import colored
sys.path.append('football_llm')
from llm.utils.openai_compat import openai_chat_query
from retrieval.state_filed_retrieval import KnowledgeRetrieval
from llm.utils.llama_index_compat import OpenAIEmbedding
from CONFIG import KnowledgeType
from funcs import load_jsonl_list, npz_extractor, json_str_clean
from d3rlpy.algos import CQL

class RandomPlayer:
    def __init__(self, mark):
        self.mark = mark

    def get_action(self, env):
        return np.random.choice(env.available_moves())

class LLM_RAG_Player:
    def __init__(self, mark):
        self.mark = mark
    def get_action(self, game):
        return self.LLM_RAG(game, self.mark)

    def ret_creator(self, corpus, knowledge_type):
        state_field_saved_path = 'football_llm/tictactoe/rag_state_field'
        os.makedirs(state_field_saved_path, exist_ok=True)
        embed_model =  OpenAIEmbedding()
        return KnowledgeRetrieval(corpus, state_field_saved_path, model_name='gpt-4o-mini',
                                    knowledge_type=knowledge_type, top_k=3, device='cuda:2', embed_model=embed_model, eval_strategy='embedding',env_name='tictactoe')

    def LLM_RAG(self, state, player):
        knowledge_path = 'football_llm/tictactoe/tic_tac_toe_knowledge.jsonl'
        # knowledge_path = os.path.join(os.environ.get('PLFB_ARTIFACT_ROOT', 'plfb_artifacts'), 'book_derived', 'retrieval', 'policy', 'policy.jsonl')
        policy_corpus = self.load_corpus(knowledge_path)
        Pi_retrieval = self.ret_creator(policy_corpus, KnowledgeType.Policy)

        obs_str = f"""
            You are playing as {player}, the {player} player, the {'first' if player == 'X' else 'second'} player. You should put {player} in the empty cell.

            The current state of the board is as follows:

            {' ' if state.board[0] == 0 else 'X' if state.board[0] == 1 else 'O'} | {' ' if state.board[1] == 0 else 'X' if state.board[1] == 1 else 'O'} | {' ' if state.board[2] == 0 else 'X' if state.board[2] == 1 else 'O'}
            ---------
            {' ' if state.board[3] == 0 else 'X' if state.board[3] == 1 else 'O'} | {' ' if state.board[4] == 0 else 'X' if state.board[4] == 1 else 'O'} | {' ' if state.board[5] == 0 else 'X' if state.board[5] == 1 else 'O'}
            ---------
            {' ' if state.board[6] == 0 else 'X' if state.board[6] == 1 else 'O'} | {' ' if state.board[7] == 0 else 'X' if state.board[7] == 1 else 'O'} | {' ' if state.board[8] == 0 else 'X' if state.board[8] == 1 else 'O'}

        """

        pi_code = Pi_retrieval.retrieve_knowledge(obs_str)

        # print(colored(f"Obs {obs_str}", 'yellow'))
        # print(colored(f"Policy knowledge: {pi_code}", 'green'))

        global_prompt = f"""
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
            The game board is represented as a 3x3 grid, which is flattened into a list of 9 elements. Each element represents a cell on the board:

            X |  |
            ---------
            O| X | X
            ---------
            O |  | O

            The value in each cell indicates:
            - ' ' (space): Empty cell
            - 'X': First player's move (X player)
            - 'O': Second player's move (O player)

        ## Your Task:
        - Respond with a JSON object containing your move and reasoning base on the policy knowledge.
        - The move should be a single integer from 0 to 8, representing the cell index.
        - Provide a brief explanation for your choice in the reasoning field.
        - Follow the policy knowledge to make your decision.
        """

        action_prompt = f"""
        ## Choose the next move base on the policy knowledge

        You are playing as {player}, the {player} player, the {'first' if player == 'X' else 'second'} player. You should put {player} in the empty cell.

        The current state of the board is as follows:

            {' ' if state.board[0] == 0 else 'X' if state.board[0] == 1 else 'O'} | {' ' if state.board[1] == 0 else 'X' if state.board[1] == 1 else 'O'} | {' ' if state.board[2] == 0 else 'X' if state.board[2] == 1 else 'O'}
            ---------
            {' ' if state.board[3] == 0 else 'X' if state.board[3] == 1 else 'O'} | {' ' if state.board[4] == 0 else 'X' if state.board[4] == 1 else 'O'} | {' ' if state.board[5] == 0 else 'X' if state.board[5] == 1 else 'O'}
            ---------
            {' ' if state.board[6] == 0 else 'X' if state.board[6] == 1 else 'O'} | {' ' if state.board[7] == 0 else 'X' if state.board[7] == 1 else 'O'} | {' ' if state.board[8] == 0 else 'X' if state.board[8] == 1 else 'O'}

        The next move should be placed in an empty cell on the 3-by-3 grid.


        The policy knowledge of current state is as follows:

        {pi_code}

        Positions are represented by the numbers 0-8, corresponding to the cell's index:

            0 | 1 | 2
            ---------
            3 | 4 | 5
            ---------
            6 | 7 | 8

        # Important!!!!
        You are playing as {player}, the {player} player, the {'first' if player == 'X' else 'second'} player. You should put {player} in the empty cell.

        Choose your next move only on the available positions.

        The available positions are: {state.available_moves()}

        Respond with a JSON object in the following format:

        {{
        "move": <integer 0-8>,
        "reasoning": "<brief explanation for your choice>"
        }}


        """


        for attempt in range(5):
            try:
                action_response = openai_chat_query(global_prompt, action_prompt, model_name='gpt-4o-mini', req_json=True)
                action_r = json.loads(action_response)
                action_rj = json.loads(action_r["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
                final_action = action_rj["move"]
                break
            except Exception as e:
                print(colored(f"Error: {e}, retrying...", "red"))
                if attempt == 5 - 1:
                    raise e
            continue

        # action_response = openai_chat_query(global_prompt, action_prompt, model_name='gpt-4o-mini', req_json=True)

        # action_r = json.loads(action_response)
        # action_rj = json.loads(action_r["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
        # final_action = action_rj["move"]

        return final_action


    def load_corpus(self, knowledge_path):
        dirname = osp.dirname(knowledge_path)
        corpus_path = osp.join(dirname, 'corpus.jsonl')
        if osp.exists(corpus_path):
            with open(corpus_path, 'r') as file:
                return json.load(file)
        else:
            knowledge = load_jsonl_list(knowledge_path)
            corpus = self.knowledge_to_corpus(knowledge)
            with open(corpus_path, 'w') as file:
                json.dump(corpus, file, indent=4)
            return corpus


    def knowledge_to_corpus(self, knowledge):
        corpus = {}
        for i, know_item in enumerate(knowledge):
            new_uuid = str(uuid.uuid4())
            corpus[new_uuid] = ''
            for k, v in know_item.items():
                if isinstance(v, (list, tuple)):
                    corpus[new_uuid] += f"{k}\n" + '\n'.join(str(item) for item in v) + '\n'
                else:
                    corpus[new_uuid] += f"{k}\n{v}\n"
        return corpus




class LLM_Agent_Player:
    def __init__(self, mark):
        self.mark = mark

    def get_action(self, game):
        return self.LLM_Agent(game, self.mark)

    def LLM_Agent(self, state, player):
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
            The game board is represented as a 3x3 grid, which is flattened into a list of 9 elements. Each element represents a cell on the board:

            X |  |
            ---------
            O| X | X
            ---------
            O |  | O

            The value in each cell indicates:
            - ' ' (space): Empty cell
            - 'X': First player's move (X player)
            - 'O': Second player's move (O player)

        ## Your Task:
        - Respond with a JSON object containing your move and reasoning.
        - The move should be a single integer from 0 to 8, representing the cell index.
        - Provide a brief explanation for your choice in the reasoning field.
        """

        action_prompt = f"""
        ## Choose the next move
        You are playing as {player}, the {player} player, the {'first' if player == 'X' else 'second'} player. You should put {player} in the empty cell.

        The current state of the board is as follows:

            {' ' if state.board[0] == 0 else 'X' if state.board[0] == 1 else 'O'} | {' ' if state.board[1] == 0 else 'X' if state.board[1] == 1 else 'O'} | {' ' if state.board[2] == 0 else 'X' if state.board[2] == 1 else 'O'}
            ---------
            {' ' if state.board[3] == 0 else 'X' if state.board[3] == 1 else 'O'} | {' ' if state.board[4] == 0 else 'X' if state.board[4] == 1 else 'O'} | {' ' if state.board[5] == 0 else 'X' if state.board[5] == 1 else 'O'}
            ---------
            {' ' if state.board[6] == 0 else 'X' if state.board[6] == 1 else 'O'} | {' ' if state.board[7] == 0 else 'X' if state.board[7] == 1 else 'O'} | {' ' if state.board[8] == 0 else 'X' if state.board[8] == 1 else 'O'}

        The next move should be placed in an empty cell on the 3-by-3 grid.

        Positions are represented by the numbers 0-8, corresponding to the cell's index:

            0 | 1 | 2
            ---------
            3 | 4 | 5
            ---------
            6 | 7 | 8

        # Important!!!!
        You are playing as {player}, the {player} player, the {'first' if player == 'X' else 'second'} player. You should put {player} in the empty cell.

        Choose your next move only on the available positions.

        The available positions are: {state.available_moves()}

        Respond with a JSON object in the following format:

        {{
        "move": <integer 0-8>,
        "reasoning": "<brief explanation for your choice>"
        }}
        """


        # action_response = openai_chat_query(global_prompt, action_prompt, model_name='gpt-4o-mini', req_json=True)
        # action_r = json.loads(action_response)
        # action_rj = json.loads(action_r["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
        # final_action = action_rj["move"]

        for attempt in range(5):
            try:
                action_response = openai_chat_query(global_prompt, action_prompt, model_name='gpt-4o-mini', req_json=True)
                action_r = json.loads(action_response)
                action_rj = json.loads(action_r["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
                final_action = action_rj["move"]
                break
            except Exception as e:
                print(colored(f"Error: {e}, retrying...", "red"))
                if attempt == 5 - 1:
                    raise e
            continue

        return final_action


class MinimaxPlayer:
    def __init__(self, mark):
        self.mark = mark

    def get_action(self, env, board=None):
        if np.count_nonzero(board == 0) == 9:
            return random.choice(env.available_moves(board))
        else:
            return self.minimax(env, self.mark, board)['position']

    def minimax(self, env, player, board=None):
        max_player = self.mark
        other_player = 'O' if player == 'X' else 'X'
        if board is None:
            board = copy.deepcopy(env.board)
        # Check if the previous move is a winner
        if env._check_win(board):
            return {'position': None,
                    'score': 1 * (self.num_empty_squares(board) + 1) if other_player == max_player
                             else -1 * (self.num_empty_squares(board) + 1)}
        elif len(env.available_moves(board)) == 0:  # No empty squares
            return {'position': None, 'score': 0}

        if player == max_player:
            best = {'position': None, 'score': -math.inf}  # Maximize score
        else:
            best = {'position': None, 'score': math.inf}  # Minimize score

        for possible_move in env.available_moves(board):
            # Make the move
            board[possible_move] = 1 if player == 'X' else 2

            # Simulate a game after making that move
            sim_score = self.minimax(env, other_player, board)

            # Undo move
            board[possible_move] = 0
            sim_score['position'] = possible_move

            if player == max_player:
                if sim_score['score'] > best['score']:
                    best = sim_score
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score

        return best

    def num_empty_squares(self, board):
        return np.count_nonzero(board == 0)



class Minimax_random_Player:
    def __init__(self, mark):
        self.mark = mark

    def get_action(self, env, board=None):
        if np.count_nonzero(board == 0) == 9:
            return random.choice(env.available_moves(board))
        else:
            if random.random() < 0.3:
                print('random')
                return random.choice(env.available_moves(board))
            return self.minimax(env, self.mark, board)['position']

    def minimax(self, env, player, board=None):
        max_player = self.mark
        other_player = 'O' if player == 'X' else 'X'
        if board is None:
            board = copy.deepcopy(env.board)
        # Check if the previous move is a winner
        if env._check_win(board):
            return {'position': None,
                    'score': 1 * (self.num_empty_squares(board) + 1) if other_player == max_player
                             else -1 * (self.num_empty_squares(board) + 1)}
        elif len(env.available_moves(board)) == 0:  # No empty squares
            return {'position': None, 'score': 0}

        if player == max_player:
            best = {'position': None, 'score': -math.inf}  # Maximize score
        else:
            best = {'position': None, 'score': math.inf}  # Minimize score

        for possible_move in env.available_moves(board):
            # Make the move
            board[possible_move] = 1 if player == 'X' else 2

            # Simulate a game after making that move
            sim_score = self.minimax(env, other_player, board)

            # Undo move
            board[possible_move] = 0
            sim_score['position'] = possible_move

            if player == max_player:
                if sim_score['score'] > best['score']:
                    best = sim_score
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score

        return best

    def num_empty_squares(self, board):
        return np.count_nonzero(board == 0)

import os
import d3rlpy
import numpy as np

class CQLPlayer:
    def __init__(self, mark, exp_name_pattern='comment=gamma-0.9&alpha=100&keep_rate=1.0&neg_keep_rate=0.1&target_value=0.02_20240805223556'):
        self.mark = mark
        #exp_name_pattern = 'comment=unc-test&alpha=100&keep_rate=1.0&neg_keep_rate=0.0&coef_r=0.25&coef_t=0.25_20240806115620'
        exp_name_pattern = 'comment=none&alpha=0.1&keep_rate=1.0&neg_keep_rate=1.0&coef_r=0.05&coef_t=0.05_20240806180514'

        self.exp_name_pattern = exp_name_pattern
        self.model = self.load_model()

    def load_model(self):
        model_root_path = 'football_llm/IRL_LOG/tictactoe-v0/d3rlpy_logs/'
        model_path = ''

        for files in os.listdir(model_root_path):
            if files.startswith(self.exp_name_pattern):
                best_perf = -np.inf
                best_perf_model_path = None
                for model_file in os.listdir(os.path.join(model_root_path, files)):
                    if model_file.endswith('.d3'):
                        cur_perf = float(model_file.split('model_rew_')[1].split('&')[0])
                        step = int(model_file.split('step_')[1].split('.')[0])
                        # if step < 2000:
                        #     continue
                        if cur_perf > best_perf:
                            best_perf = cur_perf
                            best_perf_model_path = model_file
                model_path = os.path.join(model_root_path, files, best_perf_model_path)
                print('load path', model_path)

        if not model_path:
            raise ValueError("No suitable model found.")
        else:
            return d3rlpy.load_learnable(model_path)


    def get_action(self, state):
        obs = self.state_to_obs(state)
        action = self.model.predict(obs)
        return self.action_to_move(action, state)

    def state_to_obs(self, state):
        obs = np.array(state.board).reshape(1, -1).astype(np.float32)
        return obs

    def action_to_move(self, action, state):
        valid_moves = state.available_moves()
        return valid_moves[action[0] % len(valid_moves)]
