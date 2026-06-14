
from simulator_info import *
from CONFIG import *

def code_instantiation_prompt(knowledge_type, code_string, obs):
    raw_policy_prompt_step1_all_code = """
            I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator. 
            
            I will give you an observation which you are facing in a football simulator, your task is to instantiated a code that serve as a {knowledge_type} function, which is used for {knowledge_function}.
            For example, {knwoledge_function_example}

            Formally, the format of {knowledge_type} function is {knowledge_format}.

            To help you complete the task, I will provide you 
            (1) pieces of relevant knowledge from the tutorial, written in Python-style pseudocode. You can refer to these pre-coded snippets to instantiated a code.
            (2) the observation, action of the football simulator;
            (3) the current observation which you are facing in the football simulator.

            
            Python-style relevant knowledge from the tutorial:
            
            {code_string}
            
            The observation space of the football simulator: 
            {observation_space_desc}

            The action space of the football simulator:
            {action_space_desc}

            The current observation you are facing in the football simulator:

            {obs}

            About your code-instantiation task:\n
            - Please provide the PYTHON-tyle presudo code as detailed as you can.
            - Your task is to rewrite a code that descicribe a {knowledge_type} which is suitable to current observation.  \n
            - You should make the optimal decision based on the analyze current observation. To achive this, you should output the anaysis to "analyze to current observation" \n
            - After analyzing the current observation of the football match, you should rewrite the pseudocode to make it most suitable to derive the {knowledge_type} function for some downstream tasks.  \n        
            - Please keep a main {knowledge_type} function, named "football_manager_{knowledge_type}" in the code, and you can also add some inner functions if necessary. \n

            NOTE: 
            1. The code will be repeatedly used in the next 1 minute of the game in the simulator, so you should make sure that the code can be generalized in the next 1 minute of the game. Thus your code should consider all of the possible situations that might happend in the future, including the chaning of active player, ball positions, and opponent positions. \n
            2. Since I will delete the original code after getting your instantiated code, you cannot call the presudocodes that I provided in the prompt.
            3. The process of variable assignment example usage are allowed to be ignored/simplified, however, you should implment the logic of the code as detailed as possible so that others who without the relevant knowledge can also implment it
            4. You should not implement any function that return with placeholder.

            About the format:
                - you should answer in pure JSON format directly, without any other information. For example, you should not add the ```json``` tag in the answer.
            
            """.format(knowledge_type=knowledge_type, knowledge_function=element_type_prompt_dict[knowledge_type], knowledge_format=element_type_format_dict[knowledge_type],
                       knwoledge_function_example=element_type_example_dict[knowledge_type], code_string=code_string, 
                       observation_space_desc=OBSERVATION_SPACE_DESC, action_space_desc=ACTION_SPACE_DESC, 
                       obs=obs) + \
            """
            Response example (you SHOULD resposne in the following key order and format):
            {
                "analyze": "the analyze to current observation. Focus on how to instantiated a generalizable code based on the current observation.",   
                "code": "the code you rewrite based on the analyze to current observation...",
            }
            """
    return raw_policy_prompt_step1_all_code

# - Please keep a main {knowledge_type} function, named "main_{knowledge_type}_func" in the code, and you can also add some helper functions if necessary. \n
# While the process of variable assignment example usage can be ignored, you should implment the logic of the code as detailed as possible, and you should not implement any function that return with placeholder value unless you indeed have no way to implment it.


def code_str_format(code_str):
    if code_str == 'EMPTY':
        return 'EMPTY'
    else:
        return '=== \n Reason to generate the code:' + '\n'.join(code_str['analyze']) + '\n \n' + '\n'.join(code_str['code']) + '\n ==== \n'

def llm_policy_prompt(code_string, obs):
    knowledge_type = KnowledgeType.Policy
    raw_policy_prompt = """
        I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator. 
        
        I will give you an observation which you are facing in a football simulator, your task is to response a correct results serving as a {knowledge_type} function, which is used for {knowledge_function}.
        For example, {knwoledge_function_example}.

        Formally, the format of {knowledge_type} function is {knowledge_format}.

        To help you complete the task, I will provide you 
        (1) pieces of relevant knowledge from the tutorial, written in Python-style pseudocode. You should output your result based on the logic of the pseudocode.
        (2) the observation, action of the football simulator;
        (3) the current observation which you are facing in the football simulator.
        

        Python-style relevant knowledge which is retrieved from the tutorial books:
        
        {code_string}
        
        The observation space of the football simulator: 
        {observation_space_desc}

        The action space of the football simulator:
        {action_space_desc}

        The current observation you are facing in the football simulator:

        {obs}

        Example:

        For example,if the active player is Player 2 and you want him to close to the ball and control it, in the texted observation, 
      
        - Forward Player 2 is at Zone(9,9).
        - The ball is at Zone(11,8).
      
        your thought should like "Given these coordinates, the ball is diagonally one zone to the right (east) and one zone down (south) from the player's current position. The most direct route to the ball would indeed be diagonally towards the bottom right. Therefore, the most appropriate action for Forward Player 2 in this situation would be: action_bottom_right"        
        Your output action should be 6.

        Requirements:

        About the action choosing:
        - Please choose the action that best fits the code logic.
        
        About the format:
            - you should answer in pure JSON format with the key: 'action': a int number from 0 to 18, 'thought': why you choose this action. without any other information or code. For example, you should not add the ```json``` tag in the answer.
                
        Response format (you should resposne in the following order):        
        {{
            "thought": "give your thought to generate the action the current step's state, based on the policy code above.",
            "action": 0,
        }}

    """.format(knowledge_type=knowledge_type, knowledge_function=element_type_prompt_dict[knowledge_type], knowledge_format=element_type_format_dict[knowledge_type],
                       knwoledge_function_example=element_type_example_dict[knowledge_type], code_string=code_str_format(code_string), 
                       observation_space_desc=OBSERVATION_SPACE_DESC, action_space_desc=ACTION_SPACE_DESC, obs=obs)
    return raw_policy_prompt


def llm_transition_reward_prompt(dynamics_code_string, reward_code_string, obs, action):
    raw_prompt = """
        I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator. 
        
        I will give you an observation which you are facing in a football simulator, your task is to response a correct results serving as a {dynamics} and {reward} functions code from the tutorial, written in Python style. 

        Formally, the format of {dynamics} function is {dynamics_format}. The {dynamics} function describes the mechanism for updating the position of the ball and players in the football game simulator.
        Formally, the format of {reward} function is {reward_format}. The {reward} function describes the mechanism for calculating the reward of the football game simulator.

        To help you complete the task, I will provide you 
        (1) pieces of relevant knowledge from the tutorial, written in Python-style pseudocode. You should output your result based on the logic of the pseudocode.
        (2) the observation, action of the football simulator;
        (3) the current observation which you are facing in the football simulator and the action which made by the active player.
        
        Python-style relevant knowledge which is retrieved from the tutorial books:
        
        {dynamics} function:

                {dynamics_code_string}
        
        {reward} function:

                {reward_code_string}
        
        The observation space of the football simulator: 
        {observation_space_desc}

        The action space of the football simulator:
        {action_space_desc}

        The current observation and action you are facing in the football simulator:

        observation: {obs}

        action: {action}


        ### Task

        Based on the current texted observation and the action set, 
        1. generate the next step text observation. For instance, if the current observation shows the Left team player 3 in zone (12,4) and the chosen action is "Top," the next observation should show the player in zone (12,5).
        2. generate the dense reward of the current step's action and state, depends on the reward code above.

        ### Response Format
        
        Your response should be a json object containing the following keys:
        
        1. `score`: The current socre of the match, should be a list of two integers, such as [0,0], the first integer is the score of the left team, the second integer is the score of the right team.
        2. `step`: The current step of the match, should be an integer that should add 1 to the current text observation step.
        3. `active_left_player`: The active players of the left team, If an active player chooses a pass or shoot (actions 9, 10, 11, 12) or you think the ball is most likely been intercepted by the right team, the active player must be replaced. If the left team has the ball, then the active player should be the ball controller. If the left team does not control the ball, then the active player should be the left team player closest to the ball. It should be an integer, ranging from 0 to 10. Such as 6.
        4. `ball_ownership`: The team that currently has the ball, No matter which team passes the ball, shoots or is intercepted, the ball rights will be overwritten. As long as any player passes the ball, the ball's ownership will be 0 and no one will control the ball. A team is judged to be in possession of the ball only when the ball is at a player's feet. The output should be an integer, 1 means the left team has the ball, 2 means the right team has the ball, 0 means no team has the ball, or the ball is during the passing.
        5. `ball_ownership_player`: The player currently holding the ball should be an integer and match the result generated in the key "ball_ownership" above. If no team controls the ball, it should be -1. If the left team has the ball, this should be an integer, the same as the active player, ranging from 0 to 10. Such as 6. If the right team has the ball, it should be an integer, ranging from 11 to 21. For example, 18. When either team passes the ball and shoots or is likely to be intercepted, pay attention to switching the ball-carrying personnel.
        6. `ball_zone`:  This description elaborates on the mechanism for determining the position of the ball in a football game simulator. In this simulator, the ball's position is represented by a list containing two integers. The first integer indicates the ball's x-coordinate, with a value range from 1 to 20, while the second integer represents the y-coordinate, ranging from 1 to 12. For instance, the coordinates [17,2] signify a specific location of the ball on the field. When an active player from the left team controls the ball, the position of the ball changes according to the player's movements. During each timestep, if the player chooses one of the actions between 0 to 8, or if the sticky action list includes an action that the player continuously performs, the ball will move at least one grid space, depending on the direction of the player's movement. For example, if the current player opts to move left and control the ball (action 1), and the current position of the ball is [10,5], then the position of the ball should approximately update to [9,5] in the next moment. In the case of passing, the ball moves at a faster speed. The player first determines the intended teammate for the pass and then simulates the flight of the ball. In this scenario, the ball can move between 1 to 4 grid spaces per timestep. For example, if a player chooses to make a long pass (action 9), with the current position of the ball at [4,5] and the target teammate at [11,7], then the position of the ball in the next moment is likely to be around [7,6].
        7. `left_active_player_zone`: The zone of the left team active player at the next time step; you should first check the result generated in the key "active_left_player", then check the zone of that player in the current time step observation, and then based on the left team active player provided above action to output. The output should be a list of two integers, the first being the x-coordinate of player 0, ranging from 1 to 20. The second integer is the y-coordinate of player 0, ranging from 1 to 12. For example, [7,6].
        8. `dense_reward`: The dense reward of the current step's action and state, depends on the reward code above.
        9. `thought`: Explanation for the generated observation.


        ### Response example: (you should resposne with the JSON format in the following order, without any other information, explanation or comments):
        
        {{
            "thought": "based on the current texted observation and the action set, the ball and player 3 are in the zone [9,4], and the active player is controlling the ball, and choose the action go top, thus consider the right have very low chance to interape the ball, the ball will be in the zone [9,5] in the next time step."
            "score": [0,0],
            "step": 2,
            "active_left_player": 3,
            "ball_ownership": 1,
            "ball_ownership_player": 3,
            "ball_zone": [9,5],
            "left_active_player_zone": [9,5],
            "dense_reward:": 1,
        }}

    """.format(dynamics=KnowledgeType.Dynamics, dynamics_function=element_type_prompt_dict[KnowledgeType.Dynamics], 
               dynamics_format=element_type_format_dict[KnowledgeType.Dynamics], dynamics_code_string=code_str_format(dynamics_code_string), 
               reward=KnowledgeType.Reward, reward_function=element_type_prompt_dict[KnowledgeType.Reward], 
               reward_format=element_type_format_dict[KnowledgeType.Reward], reward_code_string=code_str_format(reward_code_string), 
               observation_space_desc=OBSERVATION_SPACE_DESC, action_space_desc=ACTION_SPACE_DESC, obs=obs, action=action)
    return raw_prompt




def llm_transition_prompt(dynamics_code_string, obs, action):
    raw_prompt = """
        I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator. 
        
        I will give you an observation which you are facing in a football simulator, your task is to response a correct results serving as a {dynamics} function code from the tutorial, written in Python style. 

        Formally, the format of {dynamics} function is {dynamics_format}. The {dynamics} function describes the mechanism for updating the position of the ball and players in the football game simulator.

        To help you complete the task, I will provide you 
        (1) pieces of relevant knowledge from the tutorial, written in Python-style pseudocode. You should output your result based on the logic of the pseudocode.
        (2) the observation, action of the football simulator;
        (3) the current observation which you are facing in the football simulator and the action which made by the active player.
        
        Python-style relevant knowledge which is retrieved from the tutorial books:
        
        {dynamics} function:

                {dynamics_code_string}
        
        The observation space of the football simulator: 
        {observation_space_desc}

        Important keys in the observation:
        - `score`: The current socre of the match, should be a list of two integers, such as [0,0], the first integer is the score of the left team, the second integer is the score of the right team.
        - `step`: The current step of the match, should be an integer that should add 1 to the current text observation step.
        - `ball_ownership` : The output should be an integer, 1 means the left team has the ball, 2 means the right team has the ball, 0 means no team has the ball, or the ball is during the passing. The team that currently has the ball, No matter which team passes the ball, shoots or is intercepted, the ball rights will be overwritten. As long as any player passes the ball, the ball's ownership will be 0 and no one will control the ball. A team is judged to be in possession of the ball only when the ball is at a player's feet. 
        - `left_active_player_zone`: The zone of the left team active player at the next time step; you should first check the result generated in the key "active_left_player", then check the zone of that player in the current time step observation, and then based on the left team active player provided above action to output. The output should be a list of two integers, the first being the x-coordinate of player 0, ranging from 1 to 20. The second integer is the y-coordinate of player 0, ranging from 1 to 12. For example, [7,6].
        - `ball_zone`:  This description elaborates on the mechanism for determining the position of the ball in a football game simulator. In this simulator, the ball's position is represented by a list containing two integers. The first integer indicates the ball's x-coordinate, with a value range from 1 to 20, while the second integer represents the y-coordinate, ranging from 1 to 12. For instance, the zoon [17,2] signify a specific location of the ball on the field. When an active player from the left team controls the ball, the position of the ball changes according to the player's movements. During each timestep, if the player chooses one of the actions between 0 to 8, or if the sticky action list includes an action that the player continuously performs, depending on the direction of the player's movement. In the case of passing, the ball moves at a faster speed. The player first determines the intended teammate for the pass and then simulates the flight of the ball. In this scenario, the ball can move between 0 to 4 grid spaces per timestep. For example, if a player chooses to make a long pass (action 9), with the current position of the ball at [4,5] and the target teammate at [11,7], then the position of the ball in the next moment is likely to be around [7,6].

        NOTE: if the active player still owner the ball, the player zone and the ball zone should be the same.


        The action space of the football simulator:
        {action_space_desc}

        The current observation and action you are facing in the football simulator:

        observation: {obs}

        action: {action}


        ### Task

        Based on the current texted observation and the action set, generate the next step text observation. 
        
        For instance, if the current observation shows the Left team player 3 in zone (12,4) and the chosen action is "Top," the next observation should show the player in zone (12,5).


        ### Response example: (you should resposne with the JSON format in the following order, without any other information, explanation or comments):
        
        {{
            "thought": "based on the current texted observation and the action set, the ball and player 3 are in the zone [9,4], and the active player is controlling the ball, and choose the action go top, thus consider the right have very low chance to interape the ball, the ball will be in the zone [9,5] in the next time step."
            "score": [0,0],
            "step": 2,
            "ball_ownership": 1,
            "ball_zone": [9,5],
            "left_active_player_zone": [9,5],
        }}

    """.format(dynamics=KnowledgeType.Dynamics, dynamics_function=element_type_prompt_dict[KnowledgeType.Dynamics], 
               dynamics_format=element_type_format_dict[KnowledgeType.Dynamics], dynamics_code_string=code_str_format(dynamics_code_string), 
               observation_space_desc=OBSERVATION_SPACE_DESC, action_space_desc=ACTION_SPACE_DESC, obs=obs, action=action)
    return raw_prompt



def llm_reward_prompt(reward_code_string, obs, action, next_obs):
    raw_prompt = """
        I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator. 
        
        I will give you an observation which you are facing in a football simulator, your task is to response a correct results serving as a  {reward} function code from the tutorial, written in Python style. 

        Formally, the format of {reward} function is {reward_format}. The {reward} function describes the mechanism for calculating the reward of the football game simulator.

        To help you complete the task, I will provide you 
        (1) pieces of relevant knowledge from the tutorial, written in Python-style pseudocode. You should output your result based on the logic of the pseudocode.
        (2) the observation, action of the football simulator;
        (3) the current observation which you are facing in the football simulator and the action which made by the active player.
        
        Python-style relevant knowledge which is retrieved from the tutorial books:
        
        
        {reward} function:

                {reward_code_string}
        
        The observation space of the football simulator: 
        {observation_space_desc}

        The action space of the football simulator:
        {action_space_desc}

        The current observation and action you are facing in the football simulator:

        observation: {obs}

        action: {action}
        
        next_observation: {next_obs}

        ### Task

        Based on the current texted observation and the action set, generate the dense reward of the current step's action and state, based on the reward code above.
        The rewards should be one of -2, -1, 0, 1, or 2.

        ### Response example:  (you should resposne with the JSON format in the following order, without any other information, explanation or comments):
        
        {{
            "thought": "give your thought to generate the dense reward of the current step's action and state, based on the reward code above."
            "dense_reward:": 1,
        }}

    """.format(reward=KnowledgeType.Reward, reward_function=element_type_prompt_dict[KnowledgeType.Reward], 
               reward_format=element_type_format_dict[KnowledgeType.Reward], reward_code_string=code_str_format(reward_code_string), 
               observation_space_desc=OBSERVATION_SPACE_DESC, action_space_desc=ACTION_SPACE_DESC, obs=obs, action=action, next_obs=next_obs)
    return raw_prompt
