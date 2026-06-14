from book_scripts.prompt_templete import *

def code_gen_prompt(filter_stringency_level, derive_element_type, obs_acs_trajecotry):
    assert derive_element_type == 'Policy', "Currently only support Policy generation."
    filter_stringency_prompt = filter_stringency_level_prompt_dict[filter_stringency_level]
    _code_gen_prompt= """
            I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator. 
            
            I will provide a observation-action trajecotry of the football match, and you should analyze the situation and write a psudo-code of Policy that can reflect how the agent takes action in different observations.
            
                {1}. {2}.
            """.format(derive_element_type, element_type_prompt_dict[derive_element_type], filter_stringency_prompt.format(derive_element_type)) + \
            """            
            
            About the obsevation in the football match:
            
            First, it provides information such as the time and score of the match. 
            Second, which side has control of the ball, and the active player in the Left team you need to propose corresponding policies to him. 
            Next is the position and role information of each player: In this text description, the football grass field is divided into 240 zones. 
            We use zone (x, y) to express the position of the player. "x" is the distance from the left team's penalty area to the right team's penalty area, ranging from 1 to 20, and y is the distance from the lower corner to the upper corner flag, ranging from 1 to 12.
            This means that the center circle position of the field is zone (10, 6), where the game start.
            The lower left corner position of the Left team is (1, 1), and the upper right corner position of the Right team is (20, 12). 
            Venues never interchange or change. Following the position information is direction, meaning the direction the player is currently facing and the direction of future actions.
            
            About the action in the football match:      
            
            action set = (
            0, # action_idle, a no-op action, sticky actions are not affected (player maintains his directional movement etc.).\n
            1, # action_left, sticky action and will change the player's direction, player will continue to move left until another action is taken. Such as, from zone(11,4) to zone(10,4)\n
            2, # action_top_left, sticky action and will change the player's direction, player will continue to move top left until another action is taken. Such as, from zone(11,4) to zone(10,5)\n
            3, # action_top, sticky action and will change the player's direction, player will continue to move top until another action is taken. Such as, from zone(11,4) to zone(11,5)\n
            4, # action_top_right, sticky action and will change the player's direction, player will continue to move top right until another action is taken. Such as, from zone(11,4) to zone(12,5)\n
            5, # action_right, sticky action and will change the player's direction, player will continue to move right until another action is taken. Such as, from zone(11,4) to zone(12,4)\n
            6, # action_bottom_right, sticky action and will change the player's direction, player will continue to move bottom right until another action is taken. Such as, from zone(11,4) to zone(12,3)\n
            7, # action_bottom, sticky action and will change the player's direction, player will continue to move bottom until another action is taken.Such as, from zone(11,4) to zone(11,3)\n
            8, # action_bottom_left, sticky action and will change the player's direction, player will continue to move bottom left until another action is taken. Such as, from zone(11,4) to zone(10,3)\n
            9, # action_long_pass, the player will long pass to their teammate in his current direction. Before you pass, you should relase the sprint if the agent is doing sprint. A long pass covers a large distance on the field. \n
            10, # action_high_pass, the player will high pass to their teammate in his current direction. Before you pass, you should relase the sprint if the agent is doing sprint. A high pass sends the ball into the air, often over obstacles, to reach a teammate.  \n
            11, # action_short_pass, the player will short pass to their teammate in his current direction. Before you pass, you should tune make sure the diresction is fine, using action 0-8 to chenge the direction. A short pass is a quick and close-range exchange between teammates, commonly used to maintain possession and build an attack. \n
            12, # action_shot, players will try to shoot. When near to the oppenent penalty area.\n
            13, # action_sprint, when the player will chose this action, the agent will sprint with sticky action's direction, it will make agent run faster.\n
            14, # action_release_direction, player will stop moving in the current direction. Choose it when you want this agent to change the direction, after this action, you should choose another action from 0-8 to change it.\n
            15, # action_release_sprint, player will stop sprinting, only when the agent's during the sprint. \n
            16, # action_sliding, the player will try to slide the tackle. If your position is on the opponent's path with the ball, you can intercept it. However, if the sliding tackle fails, you will be separated by a large distance, allowing the opponent to ignore the defense.. \n
            17, # action_dribble, players will try to dribble. When they have the ball, dribbling will greatly improve the success rate of dribbling, especially in multi-person double-teams and difficult-to-handle situations..\n
            18, # action_release_dribble, player will stop dribbling.\n
            )
            - You can only control the active player in the Left team, but the active player is dynamically selected by the simulator as the player closest to the ball. You can make shot to the player you would like to active to better achieve your objective.\n
            
            trajecotry of the football match:
            
            {0}
            

            About the anylysis:
                - Based on the provided trajecotry I provided, you should summarize some rules that can reflect how the agent takes action in different observations.
                

            About your psudocode-writing task:
            - Your task is to write a code that can summarize the Policy concept based on the analysis.  
            - Please provide the PYTHON-tyle presudo code.
            - If parts of the rules of the Policy are undefined or cannot be determined from the trajectory, just construct a separate condition and says "UNDEFINED", otherwise, write the presudo code as detailed as you can. 
            - If you would like to use some helper function to implement the Policy function, please implement the helper as detailed as possible too, otherwise, please return a "UNDEFINED" placeholder.
            """.format(obs_acs_trajecotry) + \
            """
            About the format:
                - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.
                
            Response example (you should resposne in the following order):
            {{
                "thought": "summarize the information of the trajecotry",
                "analyze": "your anaylysis results of the trajecotry I provided.",
                "code": "the psudo-code based on the anaylysis to the trajecotry",
            }}
    """ 
    return _code_gen_prompt



def language_gen_prompt(filter_stringency_level, derive_element_type, obs_acs_trajecotry):
    assert derive_element_type == 'Policy', "Currently only support Policy generation."
    filter_stringency_prompt = filter_stringency_level_prompt_dict[filter_stringency_level]
    _code_gen_prompt= """
            I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator. 
            
            I will provide a observation-action trajecotry of the football match, and you should analyze the situation and write a knowledge of Policy that can reflect how the agent takes action in different observations.
            
                {1}. {2}.
            """.format(derive_element_type, element_type_prompt_dict[derive_element_type], filter_stringency_prompt.format(derive_element_type)) + \
            """            
            
            About the obsevation in the football match:
            
            First, it provides information such as the time and score of the match. 
            Second, which side has control of the ball, and the active player in the Left team you need to propose corresponding policies to him. 
            Next is the position and role information of each player: In this text description, the football grass field is divided into 240 zones. 
            We use zone (x, y) to express the position of the player. "x" is the distance from the left team's penalty area to the right team's penalty area, ranging from 1 to 20, and y is the distance from the lower corner to the upper corner flag, ranging from 1 to 12.
            This means that the center circle position of the field is zone (10, 6), where the game start.
            The lower left corner position of the Left team is (1, 1), and the upper right corner position of the Right team is (20, 12). 
            Venues never interchange or change. Following the position information is direction, meaning the direction the player is currently facing and the direction of future actions.
            
            About the action in the football match:      
            
            action set = (
            0, # action_idle, a no-op action, sticky actions are not affected (player maintains his directional movement etc.).\n
            1, # action_left, sticky action and will change the player's direction, player will continue to move left until another action is taken. Such as, from zone(11,4) to zone(10,4)\n
            2, # action_top_left, sticky action and will change the player's direction, player will continue to move top left until another action is taken. Such as, from zone(11,4) to zone(10,5)\n
            3, # action_top, sticky action and will change the player's direction, player will continue to move top until another action is taken. Such as, from zone(11,4) to zone(11,5)\n
            4, # action_top_right, sticky action and will change the player's direction, player will continue to move top right until another action is taken. Such as, from zone(11,4) to zone(12,5)\n
            5, # action_right, sticky action and will change the player's direction, player will continue to move right until another action is taken. Such as, from zone(11,4) to zone(12,4)\n
            6, # action_bottom_right, sticky action and will change the player's direction, player will continue to move bottom right until another action is taken. Such as, from zone(11,4) to zone(12,3)\n
            7, # action_bottom, sticky action and will change the player's direction, player will continue to move bottom until another action is taken.Such as, from zone(11,4) to zone(11,3)\n
            8, # action_bottom_left, sticky action and will change the player's direction, player will continue to move bottom left until another action is taken. Such as, from zone(11,4) to zone(10,3)\n
            9, # action_long_pass, the player will long pass to their teammate in his current direction. Before you pass, you should relase the sprint if the agent is doing sprint. A long pass covers a large distance on the field. \n
            10, # action_high_pass, the player will high pass to their teammate in his current direction. Before you pass, you should relase the sprint if the agent is doing sprint. A high pass sends the ball into the air, often over obstacles, to reach a teammate.  \n
            11, # action_short_pass, the player will short pass to their teammate in his current direction. Before you pass, you should tune make sure the diresction is fine, using action 0-8 to chenge the direction. A short pass is a quick and close-range exchange between teammates, commonly used to maintain possession and build an attack. \n
            12, # action_shot, players will try to shoot. When near to the oppenent penalty area.\n
            13, # action_sprint, when the player will chose this action, the agent will sprint with sticky action's direction, it will make agent run faster.\n
            14, # action_release_direction, player will stop moving in the current direction. Choose it when you want this agent to change the direction, after this action, you should choose another action from 0-8 to change it.\n
            15, # action_release_sprint, player will stop sprinting, only when the agent's during the sprint. \n
            16, # action_sliding, the player will try to slide the tackle. If your position is on the opponent's path with the ball, you can intercept it. However, if the sliding tackle fails, you will be separated by a large distance, allowing the opponent to ignore the defense.. \n
            17, # action_dribble, players will try to dribble. When they have the ball, dribbling will greatly improve the success rate of dribbling, especially in multi-person double-teams and difficult-to-handle situations..\n
            18, # action_release_dribble, player will stop dribbling.\n
            )
            - You can only control the active player in the Left team, but the active player is dynamically selected by the simulator as the player closest to the ball. You can make shot to the player you would like to active to better achieve your objective.\n
            
            trajecotry of the football match:
            
            {0}
            

            About the anylysis:
                - Based on the provided trajecotry I provided, you should summarize some rules that can reflect how the agent takes action in different observations.
                

            About your knowledge-writing task:
            - Your task is to write a knowledge that can summarize the Policy concept based on the analysis.  
            - Please provide the knowledge written by natural language.
            - Write the knowledge as detailed as you can. 
            """.format(obs_acs_trajecotry) + \
            """
            About the format:
                - you should answer in pure JSON format, without any other information or knowledge. For example, you should not add the ```json``` tag in the answer.
                
            Response example (you should resposne in the following order):
            {{
                "thought": "summarize the information of the trajecotry",
                "analyze": "your anaylysis results of the trajecotry I provided.",
                "knowledge": "the knowledge based on the anaylysis to the trajecotry",
            }}
    """ 
    return _code_gen_prompt

def code_state_action_space_gen_prompt(code_list, env_name='football'):
    if env_name == 'football':
        _prompt = """
        I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that wants to learn a football manager policy in a football simulator. 

        I will provide an observation space of the football simulator and some policy functions written by others. Your task is to summarize an observation scope, which is a subspace of the observation space and most suitable to use this policy function to win the game.

        The observation space:
        First, it provides information such as the time and score of the match. 
        Second, which side has control of the ball, and the active player in the Left team you need to propose corresponding policies to him. 
        Next is the position and role information of each player: 
        - In this text description, the football grass field is divided into 240 zones. 
        - We use zone (x, y) to express the position of the player. "x" is the discretized coordinate parallelized with the line from the left team's penalty area to the right team's penalty area, ranging from 1 to 20, and y is the discretized coordinate parallelized with the line from the lower corner to the upper corner flag, ranging from 1 to 12.
        - The left team's penalty area is zone (1, 4)-(1, 8)-(3, 8)-(3,4), and the right team's penalty area is zone (20, 4)-(20, 8)-(18, 8)-(18, 4).
        - This means that the center circle position of the field is zone (10, 6), where the game start.
        - The lower left corner position of the Left team is (1, 1), and the upper right corner position of the Right team is (20, 12). 
        - Venues never interchange or change. Following the position information is direction, meaning the direction the player is currently facing and the direction of future actions.
        
        For each team, the role of players is defined as follows: 
        role_list = ["Goalkeeper", "Forward", "Forward", "Defender", "Defender", "Defender", "Defender", "Midfielder", "Midfielder", "Midfielders", "Forward"]
        In this definition, player'1's role is "Goalkeeper", player'2's role is "Forward", ..., player'11's role is "Forward".
        Smilarly, for the right team, player'12's role is "Goalkeeper", player'13's role is "Forward", ..., player'22's role is "Forward".

        We have N codes for you to analyze: \n {}
        """.format('\n\n'.join([f"code_idx: {i} \n\n " + code for i, code in enumerate(code_list)])) + \
        """
        About the definition of the observation scope of each code: 

        For each code, you should return a description of the best scope of the observation variables to be used as the policy function to win the game, including the following factors:
        - a text summary of the preferred/better scope.
        - score is preferred/better to be in [?] 
        - active_player_role is preferred/better to be in [?] 
        - ball_ownership is preferred/better to be in [?] 
        - ball_ownership_player is preferred/better to be in [?] 
        - ball_zone is preferred/better to be in [?] 
        - ball_direction is preferred/better to be in [?]
        - {the preferred zone of all players from 0 to 21}

        NOTE: 
        1. You should define your observation scope as detailed and tight as possible.
        2. You should define the observation scope for each code by comparing the differences between the codes.


        About the format:
            - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.

        Response example (you should resposne in the following order):
        {
            'code_idx 1': {'preferred scope description': 'a text summary of the suitable situations', 'score': 'preferred scope of the score', 'active_player_role': 'preferred scope of the active_player_role', ...},
            ...
            'code_idx i': {'preferred scope description': 'a text summary of the suitable situations', 'score': 'preferred scope of the score', 'active_player_role': 'preferred scope of the active_player_role', ...},
            ...
            'code_idx N': {'preferred scope description': 'a text summary of the suitable situations', 'score': 'preferred scope of the score', 'active_player_role': 'preferred scope of the active_player_role', ...},
            
        }
        """

    elif env_name == 'tictactoe':
        import ttt_simulator_info as simulator_info
        _prompt = """
            I want you to act like a tic-tac-toe pro player and also an expert in python coding and Reinforcement Learning that wants to learn a tic-tac-toe policy in a tic-tac-toe simulator. 

            I will provide an observation space of the tic-tac-toe simulator and some policy functions written by others. Your task is to summarize an observation scope, which is a subspace of the observation space and most suitable to use this policy function to win the game.

            The observation space:
                {}

            The action space:
                {}

            We have N codes for you to analyze: \n {}
            """.format(simulator_info.OBSERVATION_SPACE_DESC, simulator_info.ACTION_SPACE_DESC, '\n\n'.join([f"code_idx: {i} \n\n " + code for i, code in enumerate(code_list)])) + \
            """
            About the definition of the observation scope of each code: 

            For each code, you should return a description of the best scope of the observation variables to be used as the policy function to win the game, including the following factors:
            - a text summary of the preferred/better scope.
            - preferred board states: [(?), (?), (?), (?), (?), (?), (?), (?), (?), where you should replace "?" with all possible values or "any" if all values are suitable.
            - preferred/better: [?], where you should replace "?" with possible values.

            NOTE: 
            1. You should define your observation scope as detailed and tight as possible.
            2. You should define the observation scope for each code by comparing the differences between the codes.

            About the format:
                - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.

            Response example (you should respond in the following order):
            {{
                'code_idx 1': {{'preferred scope description': 'a text summary of the suitable situations', 'current_player': 'preferred scopes', 'preferred board states': 'preferred scopes'}},
                ...
                'code_idx i': {{'preferred scope description': 'a text summary of the suitable situations', 'current_player': 'preferred scopes', 'preferred board states': 'preferred scopes'}},
                ...
                'code_idx N': {{'preferred scope description': 'a text summary of the suitable situations', 'current_player': 'preferred scopes', 'preferred board states': 'preferred scopes'}}
            }}
            """
    return _prompt

def retrieve_prompt_template(knowledege_type, env_name=None):
    if env_name == 'football':
        return f"The {knowledege_type} function of a football manager which is best fitted for the following obsevation:"
    elif env_name == 'tictactoe':
        return f"The {knowledege_type} knowledge of a tic-tac-toe player to be used to make decisions in the following obsevation:"
    

# def retrieve_prompt_template(knowledege_type):
#     return f"The {knowledege_type} function of a football manager to be used to make decisions in the following obsevation:"