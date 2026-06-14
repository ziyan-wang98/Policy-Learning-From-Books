
TASK_DESC = "an expert Tic-Tac-Toe player and also an expert in python coding and Reinforcement Learning."


element_type_prompt_dict = {
    "Policy": '"Policy function": The policy in Tic Tac Toe involves strategic placement of X\'s or O\'s on the game board to block the opponent’s moves and create opportunities to complete a line. For example, "Always place your mark in the center if it is your first move, as it maximizes potential winning combinations."',
    "Dynamics": '"Environment Dynamics function": This function describes the rules and changes in the game environment following player actions. For example, "If a player completes a row, column, or diagonal with the same symbol, they win the game. Otherwise, the game continues or ends in a draw if all spaces are filled without a winner."',
    "Reward": '"Rewards function": This function outlines the incentives or penalties based on the game outcomes in Tic Tac Toe. For example, "Completing three consecutive symbols horizontally, vertically, or diagonally results in a win and positive reward, while failing to block an opponent’s winning move results in a loss and a negative reward."',
}

element_type_example_dict = {
    "Policy": "A common opening strategy is to place the first mark in the center of an empty board, which provides the most opportunities for setting up wins in subsequent moves. Another strategy involves placing the second mark in a corner if the center is already occupied by the opponent.",
    "Dynamics": "The board state changes with each player's move, leading to new potential lines for winning. A player wins by placing three of their marks in a horizontal, vertical, or diagonal row. If all nine squares are filled and no player has three in a row, the game is a draw.",
    "Reward": "You can classify the outcomes as follows: 2: Winning move; 1: Blocking an opponent’s winning move; 0: Neutral move; -1: Missing an opportunity to block an opponent’s winning move; -2: Making a move that leads directly to a loss.",
}

element_type_format_dict = {
    "Policy": "Policy function: observation -> action",
    "Dynamics": "Dynamics function: observation, action -> next_observation",
    "Reward": "Reward function: observation, action, next_observation -> reward",
}

OBSERVATION_SPACE_DESC = f"""
    The observation space in Tic Tac Toe consists of a 3x3 grid representing the game board. Each cell can be empty or contain an 'X' or 'O' symbol. 
    We use a 1*9 array with number 0, 1, and 2 to represent empty, 'X', and 'O' respectively.
    The board state is updated after each player's move, reflecting the current game configuration.
    We will use another line to tell you whose turn it is, 1 for X and 2 for O.
"""

ACTION_SPACE_DESC = f"""
    The action space in Tic Tac Toe is discrete from 0 to 8, including 9 possible moves corresponding to the 9 cells on the game board. 
    Players can place their mark ('X (1)' or 'O (2)') in any empty cell to make their move.
"""

element_type_infer_hint = {
    "Policy": """
            For example, if it is X's turn and you observe the following board state:
            [1, 2, 0, 0, 0, 0, 0, 0, 0]

            The strategic thought might be "Placing X in the center (cell 4) could block O's potential diagonal completion and creates multiple opportunities for X to win on subsequent turns. Therefore, the most appropriate action for X in this situation would be to place in the center." 
            Your output action should be 4.
            """,
    "Dynamics": """
            Important keys in the observation:
            - `board`: Current board configuration, should be a list of nine integers where each integer represents a cell as follows: 0 for empty, 1 for X, and 2 for O. For example, [0,1,2,0,0,0,0,0,0].
            - `player_turn`: Indicates whose turn it is to place a mark, either 1 for X or 2 for O.
            
            Based on the current board state and the chosen action, compute the next board state. For instance, if X places a mark in the center, update the board from [0,1,2,0,0,0,0,0,0] to [0,1,2,0,1,0,0,0,0] if cell 4 was empty.
            
            """,
    "Reward": """            
            Based on the current board state, the action taken, and the resulting new board state, compute the reward. Rewards are based on achieving a win, preventing a loss, or progressing neutrally in the game.
        
            The rewards should be 2 for a winning move, 1 for a move that blocks an opponent’s winning move, 0 for a neutral move, -1 for a move that misses an opportunity to block, and -2 for a move that directly leads to a loss.
            """
}


element_type_infer_format = {
    "Policy": """
            {{
                "thought": "Given the current board state [0,1,2,0,0,0,0,0,0] and that it's X's turn. Based on the provided knowledge, to win the game, the strategic move is placing X in the center to block O's potential diagonal and create multiple lines for X, thus the action should be 4.",
                "action": 4,
            }}
        """,
    "Dynamics": """
            {{
                "thought": "Given the board state [0,1,2,0,0,0,0,0,0] and the action to place X in the center, the next state of the board should be [0,1,2,0,1,0,0,0,0] reflecting X's placement.",
                "board": [0,1,2,0,1,0,0,0,0],
                "player_turn": 2,  # It would be O's turn next
            }}
        """,
    "Reward": """
            {{
                "thought": "Given the board state [0,1,2,0,0,0,0,0,0], the action want to place X in the center, and the observed next state [0,1,2,0,1,0,0,0,2].  With X's placement resulting in no immediate win or loss and the game still in play, the reward for this action should be 0.",
                "dense_reward": 0,
            }}
"""
}
