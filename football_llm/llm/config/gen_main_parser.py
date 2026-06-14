import argparse
import os
from pydantic import BaseModel, Field


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


imaginary_obv_prompt_v0 = """
        I want you to act a football game simulator. I wii provide you with the current text observation of the football match, and the action set for the active player.
        
        You should generate the JSON format of feedback for next time step of the football texted observation.
        
        About the current texted observation: 
        
        Here, I will show you the text observation of the current football match. 
        First, it provides information such as the and score of the match. 
        Then, which side has control of the ball, and the active player in the Left team you need to propose corresponding policies to him. 
        Next is the position of the players and ball: In this text description, the football grass field is divided into 240 zones. 
        We use zone (x, y) to express the position of the player. "x" is the distance from the left team's penalty area to the right team's penalty area, ranging from 1 to 20, and y is the distance from the lower corner to the upper corner flag, ranging from 1 to 12.
        This means that the center circle position of the field is zone (10, 6), where the game start.
        The lower left corner position of the Left team is (1, 1), and the upper right corner position of the Right team is (20, 12). 
        Venues never interchange or change. Following the position information is direction, meaning the direction the player is currently facing and the direction of future actions.
        
        ---------------------\n
        
        {current_texted_obs}.
        
        ---------------------\n
        
        About the current action for all players:
        
       
        Possible Action information is below,
        action set = (
            "0": "action_idle, a no-op action, sticky actions are not affected, player maintains his directional movement.",
            "1": "action_left, sticky action, player will continue to move left until another action is taken.",
            "2": "action_top_left, sticky action, player will continue to move top left until another action is taken.",
            "3": "action_top, sticky action, player will continue to move top until another action is taken.",
            "4": "action_top_right, sticky action, player will continue to move top right until another action is taken.",
            "5": "action_right, sticky action, player will continue to move right until another action is taken.",
            "6": "action_bottom_right, sticky action, player will continue to move bottom right until another action is taken.",
            "7": "action_bottom, sticky action, player will continue to move bottom until another action is taken.",
            "8": "action_bottom_left, sticky action, player will continue to move bottom left until another action is taken.",
            "9": "action_long_pass, player will try to pass the ball to a teammate.",
            "10": "action_high_pass, player will try to pass the ball to a teammate.",
            "11": "action_short_pass, player will try to pass the ball to a teammate.",
            "12": "action_shot, player will try to shoot the ball.",
            "13": "action_sprint, player will sprint.",
            "14": "action_release_direction, player will stop moving in the current direction.",
            "15": "action_release_sprint, player will stop sprinting.",
            "17": "action_dribble, player will try to dribble.",
            "18": "action_release_dribble, player will stop dribbling."
        )
        ---------------------\n
        
        {action_set}
        
        ---------------------\n
        
        each player will take the action as shown above.
        
        Please note that you as a football simulaor, you should only gnerate one step futhure texted observation based on the current texted observation and the current action set. For example, you may know from the current texted observation, the Left team player 3 in zone (12,4) in current step, and the Left team player 3 will take action "Top" in the current step. Then, you should generate the next step texted observation, which is the Left team player 3 in zone (12,5) in next step.
        
        Response example: the next time step texted observation (you should resposne in the following order):
        
        {{
            
            "score": The current socre of the match, should be a list of two integers, such as [0,0], the first integer is the score of the left team, the second integer is the score of the right team.
            "step": The current step of the match, should be an integer that should add 1 to the current text observation step.
            "active_left_player": The active players of the left team, you mainly control the players. If the left team has the ball, then the active player should be the ball controller. If the left team does not control the ball, then the active player should be the left team player closest to the ball. It should be an integer, ranging from 0 to 10. Such as 6.
            "active_right_player": The active players of the right team, you mainly control the players. If the left team has the ball, then the active player should be the ball controller. If the left team does not control the ball, then the active player should be the left team player closest to the ball. It should be an integer, ranging from 0 to 10. Such as 6.
            "ball_ownership": the team that currently has the ball, should be an integer, 0 means the left team has the ball, 1 means the right team has the ball, -1 means no team has the ball, or the ball is during the passing.
            "ball_ownership_player": the player that currently has the ball, should be an integer, if no team control te ball, it should be -1. If the left team has the ball, it should be an integer and as same as the active player, ranging from 0 to 10. Such as 6. If right team has the ball, it should be an integer, ranging from 11 to 21. Such as 18.
            "ball_zone": the zone of the ball, should be a list of two integers, the first integer is the x coordinate of the ball which ranging from 1 to 20., the second integer is the y coordinate of the ball which ranging from 1 to 12. For example, [17,2].
            "left_active_player_zone:": The zone of the left team active player at the next time step; you should first check the result generated in the key "active_left_player", then check the zone of that player in the current time step observation, and then based on the left team active player provided above action to output. The output should be a list of two integers, the first being the x-coordinate of player 0, ranging from 1 to 20. The second integer is the y-coordinate of player 0, ranging from 1 to 12. For example, [7,6].
            "right_acctive_player_zone:": The zone of the right team acctive player at the next time step, you should first check the result generated in the key "active_right_player", then check the zone of this player in current time step observation, and then based on the right team active player provided above action to output. The output should be a list of two integers, the first being the x-coordinate of player 0, ranging from 1 to 20. The second integer is the y-coordinate of player 0, ranging from 1 to 12. For example, [5,6].
        }}
        
"""

imaginary_obv_prompt_v1 = """
       In this simulation, you will act as a football game simulator. Given the current text observation of a football match and the action set for the active player from the left team, your task is to generate a JSON-formatted feedback for the next time step of the football match's text observation.

        
        ### Current Texted Observation
        
        - **Description**: The text observation provides details of the ongoing football match, including the score and which team is in possession of the ball. 
        - **Player and Ball Position in zones**: The football field is divided into 240 zones, represented as (x, y), where "x" ranges from 1 to 20 (left to right from the left team's perspective) and "y" ranges from 1 to 12 (bottom to top). For instance, the center circle is at (10, 6), and the lower left corner from the left team's view is at (1, 1), and the upper right corner position of the Right team is (20, 12). 
        - **Player Direction**: The direction of the player, which means the player is moving in the corresponding direction (north, south, east, west, northeast, northwest, southeast, southwest).
        - **Information**: Venues never interchange or change. Following the position information is direction, meaning the direction the player is currently facing and the direction of future actions.
        
        ---------------------\n
        
        {current_texted_obs}.
        
        ---------------------\n
        
        ### Current Action Set for Left Team's Active Player
        
        The active player can perform one of the following actions:
        
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
            9, # action_long_pass, the player will long pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A long pass covers a large distance on the field. \n
            10, # action_high_pass, the player will high pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A high pass sends the ball into the air, often over obstacles, to reach a teammate.  \n
            11, # action_short_pass, the player will short pass to their teammate in his current direcetion. Before you pass, you should tune make sure the diresction is fine, using action 0-8 to chenge the direction. A short pass is a quick and close-range exchange between teammates, commonly used to maintain possession and build an attack. \n
            12, # action_shot, players will try to shoot. When near to the oppenent penalty area , such as zone(x, y), x>18, 4<y<8, try to shoot.\n
            13, # action_sprint, when the player will chose this action, the agent will sprint with sticky action's direction, it will make agent run faster.\n
            14, # action_release_direction, player will stop moving in the current direction. Choose it when you want this agent to change the direction, after this action, you should choose another action from 0-8 to change it.\n
            15, # action_release_sprint, player will stop sprinting, only when the agent's during the sprint. \n
            16, # action_sliding, the player will try to slide the tackle. If your position is on the opponent's path with the ball, you can intercept it. However, if the sliding tackle fails, you will be separated by a large distance, allowing the opponent to ignore the defense.. \n
            17, # action_dribble, players will try to dribble. When they have the ball, dribbling will greatly improve the success rate of dribbling, especially in multi-person double-teams and difficult-to-handle situations..\n
            18, # action_release_dribble, player will stop dribbling.\n
            )

        The left team active player will take the action as shown below,
        
        ---------------------\n
        
        {action}
        
        ---------------------\n
        
        ### Task

        Based on the current texted observation and the action set, generate the next step text observation. For instance, if the current observation shows the Left team player 3 in zone (12,4) and the chosen action is "Top," the next observation should show the player in zone (12,5).

        ### Response Format
        
        Your response should be a JSON object containing the following keys:
        
        1. `score`: The current socre of the match, should be a list of two integers, such as [0,0], the first integer is the score of the left team, the second integer is the score of the right team.
        2. `step`: The current step of the match, should be an integer that should add 1 to the current text observation step.
        3. `active_left_player`: The active players of the left team, If an active player chooses a pass or shoot (actions 9, 10, 11, 12) or you think the ball is most likely been intercepted by the right team, the active player must be replaced. If the left team has the ball, then the active player should be the ball controller. If the left team does not control the ball, then the active player should be the left team player closest to the ball. It should be an integer, ranging from 0 to 10. Such as 6.
        4. `ball_ownership`: The team that currently has the ball, No matter which team passes the ball, shoots or is intercepted, the ball rights will be overwritten. As long as any player passes the ball, the ball's ownership will be 0 and no one will control the ball. A team is judged to be in possession of the ball only when the ball is at a player's feet. The output should be an integer, 1 means the left team has the ball, 2 means the right team has the ball, 0 means no team has the ball, or the ball is during the passing.
        5. `ball_ownership_player`: The player currently holding the ball should be an integer and match the result generated in the key "ball_ownership" above. If no team controls the ball, it should be -1. If the left team has the ball, this should be an integer, the same as the active player, ranging from 0 to 10. Such as 6. If the right team has the ball, it should be an integer, ranging from 11 to 21. For example, 18. When either team passes the ball and shoots or is likely to be intercepted, pay attention to switching the ball-carrying personnel.
        6. `ball_zone`:  This description elaborates on the mechanism for determining the position of the ball in a football game simulator. In this simulator, the ball's position is represented by a list containing two integers. The first integer indicates the ball's x-coordinate, with a value range from 1 to 20, while the second integer represents the y-coordinate, ranging from 1 to 12. For instance, the coordinates [17,2] signify a specific location of the ball on the field. When an active player from the left team controls the ball, the position of the ball changes according to the player's movements. During each timestep, if the player chooses one of the actions between 0 to 8, or if the sticky action list includes an action that the player continuously performs, the ball will move at least one grid space, depending on the direction of the player's movement. For example, if the current player opts to move left and control the ball (action 1), and the current position of the ball is [10,5], then the position of the ball should approximately update to [9,5] in the next moment. In the case of passing, the ball moves at a faster speed. The player first determines the intended teammate for the pass and then simulates the flight of the ball. In this scenario, the ball can move between 1 to 4 grid spaces per timestep. For example, if a player chooses to make a long pass (action 9), with the current position of the ball at [4,5] and the target teammate at [11,7], then the position of the ball in the next moment is likely to be around [7,6].
        7. `left_active_player_zone`: The zone of the left team active player at the next time step; you should first check the result generated in the key "active_left_player", then check the zone of that player in the current time step observation, and then based on the left team active player provided above action to output. The output should be a list of two integers, the first being the x-coordinate of player 0, ranging from 1 to 20. The second integer is the y-coordinate of player 0, ranging from 1 to 12. For example, [7,6].
        8. `thought`: Explanation for the generated observation.


        ### Response example: 
        
        The next time step texted observation (you should resposne in the following order):
        
        {{
            "score": [0,0],
            "step": 2,
            "active_left_player": 3,
            "ball_ownership": 1,
            "ball_ownership_player": 3,
            "ball_zone": [9,5],
            "left_active_player_zone": [9,5],
            "thought": "based on the current texted observation and the action set, the ball and player 3 are in the zone [9,4], and the active player is controlling the ball, and choose the action go top, thus consider the right have very low chance to interape the ball, the ball will be in the zone [9,5] in the next time step."
        }}
        
"""

imaginary_obv_prompt_v2 = """
        In this simulation, you will act as a football game simulator. Given the current text observation of a football match and the action set for the active player from the left team, your task is to generate a JSON-formatted feedback for the next time step of the football match's text observation.

        ### About the prior Knowledge: 
        
        I will first provide you with a few pre-code snippets of the most relevant Transition and Reward functions code from the tutorial, written in Python style. 
        You can refer to these pre-coded snippets to make your decision. 
    
        ### Transition code 
        - **Description**: The transition function describes the mechanism for updating the position of the ball and players in the football game simulator.
                
        ---------------------\n
        {transition_code}.
        ---------------------\n
        
        ### Reward code 
        - **Description**: The reward function describes the mechanism for calculating the reward of the football game simulator.
        
        ---------------------\n
        {reward_code}.
        ---------------------\n
        
        ### Current Texted Observation
        
        - **Description**: The text observation provides details of the ongoing football match, including the score and which team is in possession of the ball. 
        - **Player and Ball Position in zones**: The football field is divided into 240 zones, represented as (x, y), where "x" ranges from 1 to 20 (left to right from the left team's perspective) and "y" ranges from 1 to 12 (bottom to top). For instance, the center circle is at (10, 6), and the lower left corner from the left team's view is at (1, 1), and the upper right corner position of the Right team is (20, 12). 
        - **Player Direction**: The direction of the player, which means the player is moving in the corresponding direction (north, south, east, west, northeast, northwest, southeast, southwest).
        - **Information**: Venues never interchange or change. Following the position information is direction, meaning the direction the player is currently facing and the direction of future actions.
        
        ---------------------\n
        
        {current_texted_obs}.
        
        ---------------------\n
        
        ### Current Action Set for Left Team's Active Player
        
        The active player can perform one of the following actions:
        
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
            9, # action_long_pass, the player will long pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A long pass covers a large distance on the field. \n
            10, # action_high_pass, the player will high pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A high pass sends the ball into the air, often over obstacles, to reach a teammate.  \n
            11, # action_short_pass, the player will short pass to their teammate in his current direcetion. Before you pass, you should tune make sure the diresction is fine, using action 0-8 to chenge the direction. A short pass is a quick and close-range exchange between teammates, commonly used to maintain possession and build an attack. \n
            12, # action_shot, players will try to shoot. When near to the oppenent penalty area , such as zone(x, y), x>18, 4<y<8, try to shoot.\n
            13, # action_sprint, when the player will chose this action, the agent will sprint with sticky action's direction, it will make agent run faster.\n
            14, # action_release_direction, player will stop moving in the current direction. Choose it when you want this agent to change the direction, after this action, you should choose another action from 0-8 to change it.\n
            15, # action_release_sprint, player will stop sprinting, only when the agent's during the sprint. \n
            16, # action_sliding, the player will try to slide the tackle. If your position is on the opponent's path with the ball, you can intercept it. However, if the sliding tackle fails, you will be separated by a large distance, allowing the opponent to ignore the defense.. \n
            17, # action_dribble, players will try to dribble. When they have the ball, dribbling will greatly improve the success rate of dribbling, especially in multi-person double-teams and difficult-to-handle situations..\n
            18, # action_release_dribble, player will stop dribbling.\n
            )

        The left team active player will take the action as shown below,
        
        ---------------------\n
        
        {action}
        
        ---------------------\n
        
        ### Task

        Based on the current texted observation and the action set, generate the next step text observation. For instance, if the current observation shows the Left team player 3 in zone (12,4) and the chosen action is "Top," the next observation should show the player in zone (12,5).

        ### Response Format
        
        Your response should be a JSON object containing the following keys:
        
        1. `score`: The current socre of the match, should be a list of two integers, such as [0,0], the first integer is the score of the left team, the second integer is the score of the right team.
        2. `step`: The current step of the match, should be an integer that should add 1 to the current text observation step.
        3. `active_left_player`: The active players of the left team, If an active player chooses a pass or shoot (actions 9, 10, 11, 12) or you think the ball is most likely been intercepted by the right team, the active player must be replaced. If the left team has the ball, then the active player should be the ball controller. If the left team does not control the ball, then the active player should be the left team player closest to the ball. It should be an integer, ranging from 0 to 10. Such as 6.
        4. `ball_ownership`: The team that currently has the ball, No matter which team passes the ball, shoots or is intercepted, the ball rights will be overwritten. As long as any player passes the ball, the ball's ownership will be 0 and no one will control the ball. A team is judged to be in possession of the ball only when the ball is at a player's feet. The output should be an integer, 1 means the left team has the ball, 2 means the right team has the ball, 0 means no team has the ball, or the ball is during the passing.
        5. `ball_ownership_player`: The player currently holding the ball should be an integer and match the result generated in the key "ball_ownership" above. If no team controls the ball, it should be -1. If the left team has the ball, this should be an integer, the same as the active player, ranging from 0 to 10. Such as 6. If the right team has the ball, it should be an integer, ranging from 11 to 21. For example, 18. When either team passes the ball and shoots or is likely to be intercepted, pay attention to switching the ball-carrying personnel.
        6. `ball_zone`:  This description elaborates on the mechanism for determining the position of the ball in a football game simulator. In this simulator, the ball's position is represented by a list containing two integers. The first integer indicates the ball's x-coordinate, with a value range from 1 to 20, while the second integer represents the y-coordinate, ranging from 1 to 12. For instance, the coordinates [17,2] signify a specific location of the ball on the field. When an active player from the left team controls the ball, the position of the ball changes according to the player's movements. During each timestep, if the player chooses one of the actions between 0 to 8, or if the sticky action list includes an action that the player continuously performs, the ball will move at least one grid space, depending on the direction of the player's movement. For example, if the current player opts to move left and control the ball (action 1), and the current position of the ball is [10,5], then the position of the ball should approximately update to [9,5] in the next moment. In the case of passing, the ball moves at a faster speed. The player first determines the intended teammate for the pass and then simulates the flight of the ball. In this scenario, the ball can move between 1 to 4 grid spaces per timestep. For example, if a player chooses to make a long pass (action 9), with the current position of the ball at [4,5] and the target teammate at [11,7], then the position of the ball in the next moment is likely to be around [7,6].
        7. `left_active_player_zone`: The zone of the left team active player at the next time step; you should first check the result generated in the key "active_left_player", then check the zone of that player in the current time step observation, and then based on the left team active player provided above action to output. The output should be a list of two integers, the first being the x-coordinate of player 0, ranging from 1 to 20. The second integer is the y-coordinate of player 0, ranging from 1 to 12. For example, [7,6].
        8. `dense_reward`: The dense reward of the current step's action and state, depends on the reward code above.
        9. `thought`: Explanation for the generated observation.


        ### Response example: 
        
        The next time step texted observation (you should resposne in the following order):
        
        {{
            "score": [0,0],
            "step": 2,
            "active_left_player": 3,
            "ball_ownership": 1,
            "ball_ownership_player": 3,
            "ball_zone": [9,5],
            "left_active_player_zone": [9,5],
            "dense_reward:": 0.5,
            "thought": "based on the current texted observation and the action set, the ball and player 3 are in the zone [9,4], and the active player is controlling the ball, and choose the action go top, thus consider the right have very low chance to interape the ball, the ball will be in the zone [9,5] in the next time step."
        }}
        
"""

imaginary_obv_prompt_v2_shorter =  """
        In this simulation, you will act as a football game simulator.

        ### Prior Knowledge: 
        
        Few pre-code snippets of the most relevant Transition and Reward functions code from the tutorial, written in Python style. 
    
        ### Transition code: The transition function describes the mechanism for updating the position of the ball and players in the football game simulator.
        ---------------------
        {transition_code}.
        ---------------------
        ### Reward code: The reward function describes the mechanism for calculating the reward of the football game simulator.
        ---------------------
        {reward_code}.
        ---------------------
        ### Current Texted Observation
        - **Description**: The text observation provides details of the ongoing football match, including the score and which team is in possession of the ball. 
        - **Player and Ball Position in zones**: The football field is split into 240 zones using coordinates (x, y), with "x" from 1 to 20 (left to right) and "y" from 1 to 12 (bottom to top). The center circle is at (10, 6), the lower left corner from the left team's side is at (1, 1), and the upper right corner from the right team's side is at (20, 12).
        - **Player Direction**: The direction of the player (north, south, east, west, northeast, northwest, southeast, southwest).
        ---------------------
        {current_texted_obs}.
        ---------------------
        ### Current Action
        
        "0": idle, "1": left, "2": left, "3": top, "4": top_right, "5": right, "6": bottom_right, "7": bottom, "8": bottom_left, "9": long_pass, "10": high_pass, "11": short_pass, "12": shot, "13": sprint, "14": release_direction, "15": release_sprint, "17": dribble, "18": release_dribble,
        
        The left team active player will take the action as shown below,
        ---------------------
        {action}
        ---------------------
        ### Task

        Based on the current texted observation and the action, generate the next step text observation. 

        ### Response Format
        
        Your response should be a JSON object containing the following keys:
        
        1. `score`: Current match score as a list of two integers: [left team score, right team score].
        2. `step`: Increment the current match step by 1.
        3. `active_left_player`: The left team's active player, as an integer (0-10). Change player if there's a pass, shoot, or likely interception, or based on ball possession.
        4. `ball_ownership`: Indicates ball possession: 1 for left team, 2 for right team, 0 for no possession or during a pass.
        5. `ball_ownership_player`: Player holding the ball, as an integer. -1 for no possession, 0-10 for left team, 11-21 for right team. Change player on pass, shoot, or likely interception.
        6. `ball_zone`: Ball's position as [x, y] coordinates.
        7. `left_active_player_zone`: Next step's zone for the left team's active player, as a list [x, y], with x ranging 1-20 and y 1-12.
        8. `dense_reward`: Calculated based on the current step's action and state.
        
        ### Response example: 
        
        The next time step texted observation (you should resposne in the following order):
        
        {{
            "score": [0,0],
            "step": 2,
            "active_left_player": 3,
            "ball_ownership": 1,
            "ball_ownership_player": 3,
            "ball_zone": [9,5],
            "left_active_player_zone": [9,5],
            "dense_reward:": 0.5,
        }}
        
"""


raw_policy_prompt_all= """
        I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator. 
        
        You should make the optimal decision for the current state of the football match.\n
        
        About the prior Knowledge: \n
        I will first provide you with a few pre-code snippets of the most relevant football manager policy from the tutorial, written in Python style. 
        You can refer to these pre-coded snippets to make your decision. \n
        
        ---------------------\n
        
        {context_str}.
        
        ---------------------\n     
         
        About the current state: \n
         
        Here, I will show you the text observation of the current football match. 
        First, it provides information such as the time and score of the match. 
        Second, which side has control of the ball, and the active player in the Left team you need to propose corresponding policies to him. 
        Next is the position and role information of each player: In this text description, the football grass field is divided into 240 zones. 
        We use zone (x, y) to express the position of the player. "x" is the distance from the left team's penalty area to the right team's penalty area, ranging from 1 to 20, and y is the distance from the lower corner to the upper corner flag, ranging from 1 to 12.
        This means that the center circle position of the field is zone (10, 6), where the game start.
        The lower left corner position of the Left team is (1, 1), and the upper right corner position of the Right team is (20, 12). 
        Venues never interchange or change. Following the position information is direction, meaning the direction the player is currently facing and the direction of future actions.
        
        ---------------------\n
        
        {text_obs}.
        
        ---------------------\n
        
        About the optimal decision:\n
        
        - You should take one of the action in the action set for the active player as the optimal decision for the current state of the football match: \n
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
            9, # action_long_pass, the player will long pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A long pass covers a large distance on the field. \n
            10, # action_high_pass, the player will high pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A high pass sends the ball into the air, often over obstacles, to reach a teammate.  \n
            11, # action_short_pass, the player will short pass to their teammate in his current direcetion. Before you pass, you should tune make sure the diresction is fine, using action 0-8 to chenge the direction. A short pass is a quick and close-range exchange between teammates, commonly used to maintain possession and build an attack. \n
            12, # action_shot, players will try to shoot. When near to the oppenent penalty area , such as zone(x, y), x>18, 4<y<8, try to shoot.\n
            13, # action_sprint, when the player will chose this action, the agent will sprint with sticky action's direction, it will make agent run faster.\n
            14, # action_release_direction, player will stop moving in the current direction. Choose it when you want this agent to change the direction, after this action, you should choose another action from 0-8 to change it.\n
            15, # action_release_sprint, player will stop sprinting, only when the agent's during the sprint. \n
            16, # action_sliding, the player will try to slide the tackle. If your position is on the opponent's path with the ball, you can intercept it. However, if the sliding tackle fails, you will be separated by a large distance, allowing the opponent to ignore the defense.. \n
            17, # action_dribble, players will try to dribble. When they have the ball, dribbling will greatly improve the success rate of dribbling, especially in multi-person double-teams and difficult-to-handle situations..\n
            18, # action_release_dribble, player will stop dribbling.\n
            )
        - You can only control the active player in the Left team, but the active player is dynamically selected by the simulator as the player closest to the ball. You can make shot to the player you would like to active to better achieve your objective.\n

        Requirements:

        About the action choosing:
        - Please choose the action that best fits the code logic.

        About the format:
        - you should answer in pure JSON format with the key: 'action': a int number from 0 to 18, 'thought': why you choose this action. without any other information or code. For example, you should not add the ```json``` tag in the answer.
                
        Response example (you should resposne in the following order):        
        {{
            "action": 0,
            "thought": "based on your thought, tell me the optimal action you would like to select in the action set.",
        }}
            
  """  

raw_policy_prompt_step1_all_code = """
        I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator. 
        
        You should make the optimal decision for the current state of the football match.\n
        
        About the prior Knowledge: \n
        I will first provide you with a few pre-code snippets of the most relevant football manager policy from the tutorial, written in Python style. 
        You can refer to these pre-coded snippets to make your decision. \n
        
        
        ### Related Transition code
        
        Those transition code is extract from the tutorial, it may not align with the current football game simulator, what you should do is to refer to these pre-coded snippets to generate a new transition code for the current football game simulator and based on the current time step texted observation.
        
        ---------------------\n
        
        {transition_string}.
        
        ---------------------\n     
        
        ### Related Reward code
        
        Those reward code is extract from the tutorial, it may may not align with the current football game simulator, what you should do is to refer to these pre-coded snippets to generate a new reward code for the current football game simulator and based on the current time step texted observation.
        
        ---------------------\n
        
        {reward_string}.
        
        ---------------------\n
        
        ### Related Policy code
        
        Those reward code is extract from the tutorial, it may may not align with the current football game simulator, what you should do is to refer to these pre-coded snippets to generate a new policy code for the current football game simulator and based on the current time step texted observation.
        
        ---------------------\n
        
        {policy_string}.
        
        ---------------------\n
         
        About the current state: \n
         
        Here, I will show you the text observation of the current football match. 
        First, it provides information such as the time and score of the match. 
        Second, which side has control of the ball, and the active player in the Left team you need to propose corresponding policies to him. 
        Next is the position and role information of each player: In this text description, the football grass field is divided into 240 zones. 
        We use zone (x, y) to express the position of the player. "x" is the distance from the left team's penalty area to the right team's penalty area, ranging from 1 to 20, and y is the distance from the lower corner to the upper corner flag, ranging from 1 to 12.
        This means that the center circle position of the field is zone (10, 6), where the game start.
        The lower left corner position of the Left team is (1, 1), and the upper right corner position of the Right team is (20, 12). 
        Venues never interchange or change. Following the position information is direction, meaning the direction the player is currently facing and the direction of future actions.
        
        ---------------------\n
        
        {text_obs}.
        
        ---------------------\n
        
        About the optimal decision:\n
        
        - You should take one of the action in the action set for the active player as the optimal decision for the current state of the football match: \n
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
            9, # action_long_pass, the player will long pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A long pass covers a large distance on the field. \n
            10, # action_high_pass, the player will high pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A high pass sends the ball into the air, often over obstacles, to reach a teammate.  \n
            11, # action_short_pass, the player will short pass to their teammate in his current direcetion. Before you pass, you should tune make sure the diresction is fine, using action 0-8 to chenge the direction. A short pass is a quick and close-range exchange between teammates, commonly used to maintain possession and build an attack. \n
            12, # action_shot, players will try to shoot. When near to the oppenent penalty area , such as zone(x, y), x>18, 4<y<8, try to shoot.\n
            13, # action_sprint, when the player will chose this action, the agent will sprint with sticky action's direction, it will make agent run faster.\n
            14, # action_release_direction, player will stop moving in the current direction. Choose it when you want this agent to change the direction, after this action, you should choose another action from 0-8 to change it.\n
            15, # action_release_sprint, player will stop sprinting, only when the agent's during the sprint. \n
            16, # action_sliding, the player will try to slide the tackle. If your position is on the opponent's path with the ball, you can intercept it. However, if the sliding tackle fails, you will be separated by a large distance, allowing the opponent to ignore the defense.. \n
            17, # action_dribble, players will try to dribble. When they have the ball, dribbling will greatly improve the success rate of dribbling, especially in multi-person double-teams and difficult-to-handle situations..\n
            18, # action_release_dribble, player will stop dribbling.\n
            )
        - You can only control the active player in the Left team, but the active player is dynamically selected by the simulator as the player closest to the ball. You can make shot to the player you would like to active to better achieve your objective.\n
        
        About your code-writing task:\n
        - Please provide the PYTHON-tyle presudo code as detailed as you can.
        - Your task is to rewrite a code that descicribe a policy which is suitable to current state.  \n
        - You should make the optimal decision based on the analyze current state of the football match. output the anaysis to "analyze to current state" \n
        - After analyzing the current state of the football match, you should rewrite the pseudocode to make it most suitable to derive the optimal action for some downstream tasks.  \n        
        - The code will be repeatedly used in the next 1 minute of the game in the simulator, so you should make sure that the code can be generalized in the next 1 minute of the game. Thus your code should consider all of the possible situations that might happend in the future, including the chaning of active player, ball positions, and opponent positions. \n
        - The code should be as detailed as possible, so that the downstream tasks can run the code directly.
        - Since I will delete the original code after getting your aggregated code,  you cannot call the presudocodes that I provided in the prompt. If it is necessary to call the presudocode, please still return the presudocode as an individual function in the answer.
        - Since I will try to run your code in the simulator directly, you should not implement any function that return with placeholder value.

        About the format:
            - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.


        Response example (you should resposne in the following order):
        {{
            "policy_code": "the policy code you rewrite based on the analyze to current state",
            "reward_code": "the rewad code you rewrite based on the analyze to current state",
            "transition_code": "the transition code you rewrite based on the analyze to current state",
            "thought": "why you rewrite the code in this way",
        }}   
  """  

raw_policy_prompt_step1= """
        I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator. 
        
        You should make the optimal decision for the current state of the football match.\n
        
        About the prior Knowledge: \n
        I will first provide you with a few pre-code snippets of the most relevant football manager policy from the tutorial, written in Python style. 
        You can refer to these pre-coded snippets to make your decision. \n
        
        ---------------------\n
        
        {context_str}.
        
        ---------------------\n     
         
        About the current state: \n
         
        Here, I will show you the text observation of the current football match. 
        First, it provides information such as the time and score of the match. 
        Second, which side has control of the ball, and the active player in the Left team you need to propose corresponding policies to him. 
        Next is the position and role information of each player: In this text description, the football grass field is divided into 240 zones. 
        We use zone (x, y) to express the position of the player. "x" is the distance from the left team's penalty area to the right team's penalty area, ranging from 1 to 20, and y is the distance from the lower corner to the upper corner flag, ranging from 1 to 12.
        This means that the center circle position of the field is zone (10, 6), where the game start.
        The lower left corner position of the Left team is (1, 1), and the upper right corner position of the Right team is (20, 12). 
        Venues never interchange or change. Following the position information is direction, meaning the direction the player is currently facing and the direction of future actions.
        
        ---------------------\n
        
        {text_obs}.
        
        ---------------------\n
        
        About the optimal decision:\n
        
        - You should take one of the action in the action set for the active player as the optimal decision for the current state of the football match: \n
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
            9, # action_long_pass, the player will long pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A long pass covers a large distance on the field. \n
            10, # action_high_pass, the player will high pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A high pass sends the ball into the air, often over obstacles, to reach a teammate.  \n
            11, # action_short_pass, the player will short pass to their teammate in his current direcetion. Before you pass, you should tune make sure the diresction is fine, using action 0-8 to chenge the direction. A short pass is a quick and close-range exchange between teammates, commonly used to maintain possession and build an attack. \n
            12, # action_shot, players will try to shoot. When near to the oppenent penalty area , such as zone(x, y), x>18, 4<y<8, try to shoot.\n
            13, # action_sprint, when the player will chose this action, the agent will sprint with sticky action's direction, it will make agent run faster.\n
            14, # action_release_direction, player will stop moving in the current direction. Choose it when you want this agent to change the direction, after this action, you should choose another action from 0-8 to change it.\n
            15, # action_release_sprint, player will stop sprinting, only when the agent's during the sprint. \n
            16, # action_sliding, the player will try to slide the tackle. If your position is on the opponent's path with the ball, you can intercept it. However, if the sliding tackle fails, you will be separated by a large distance, allowing the opponent to ignore the defense.. \n
            17, # action_dribble, players will try to dribble. When they have the ball, dribbling will greatly improve the success rate of dribbling, especially in multi-person double-teams and difficult-to-handle situations..\n
            18, # action_release_dribble, player will stop dribbling.\n
            )
        - You can only control the active player in the Left team, but the active player is dynamically selected by the simulator as the player closest to the ball. You can make shot to the player you would like to active to better achieve your objective.\n
        
        About your code-writing task:\n
        - Please provide the PYTHON-tyle presudo code as detailed as you can.
        - Your task is to rewrite a code that descicribe a policy which is suitable to current state.  \n
        - You should make the optimal decision based on the analyze current state of the football match. output the anaysis to "analyze to current state" \n
        - After analyzing the current state of the football match, you should rewrite the pseudocode to make it most suitable to derive the optimal action for some downstream tasks.  \n        
        - The code will be repeatedly used in the next 1 minute of the game in the simulator, so you should make sure that the code can be generalized in the next 1 minute of the game. Thus your code should consider all of the possible situations that might happend in the future, including the chaning of active player, ball positions, and opponent positions. \n
        - The code should be as detailed as possible, so that the downstream tasks can run the code directly.
        - Since I will delete the original code after getting your aggregated code,  you cannot call the presudocodes that I provided in the prompt. If it is necessary to call the presudocode, please still return the presudocode as an individual function in the answer.
        - Since I will try to run your code in the simulator directly, you should not implement any function that return with placeholder value.

        About the format:
            - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.


        Response example (you should resposne in the following order):
        {{
            "analyze": "the analyze to current state",
            "code": "the code you rewrite based on the analyze to current state",
        }}   
       
        
  """  

raw_policy_prompt_step2= """
      
      The pseudocode for the policy you want this active player to implement is as follows:\n
      {policy_code}
      ---------------------\n
      The texted observation for the current state:\n
      {text_obs}
      ---------------------\n
      Possible Action information is below,
      0 = action_idle, a no-op action, sticky actions are not affected (player maintains his directional movement etc.).\n
      1 = action_left, sticky action and will change the player's direction, player will continue to move left until another action is taken. Such as, from zone(11,4) to zone(10,4)\n
      2 = action_top_left, sticky action and will change the player's direction, player will continue to move top left until another action is taken. Such as, from zone(11,4) to zone(10,5)\n
      3 = action_top, sticky action and will change the player's direction, player will continue to move top until another action is taken. Such as, from zone(11,4) to zone(11,5)\n
      4 = action_top_right, sticky action and will change the player's direction, player will continue to move top right until another action is taken. Such as, from zone(11,4) to zone(12,5)\n
      5 = action_right, sticky action and will change the player's direction, player will continue to move right until another action is taken. Such as, from zone(11,4) to zone(12,4)\n
      6 = action_bottom_right, sticky action and will change the player's direction, player will continue to move bottom right until another action is taken. Such as, from zone(11,4) to zone(12,3)\n
      7 = action_bottom, sticky action and will change the player's direction, player will continue to move bottom until another action is taken.Such as, from zone(11,4) to zone(11,3)\n
      8 = action_bottom_left, sticky action and will change the player's direction, player will continue to move bottom left until another action is taken. Such as, from zone(11,4) to zone(10,3)\n
      9 = action_long_pass, the player will long pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A long pass covers a large distance on the field. \n
      10 = action_high_pass, the player will high pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A high pass sends the ball into the air, often over obstacles, to reach a teammate.  \n
      11 = action_short_pass, the player will short pass to their teammate in his current direcetion. Before you pass, you should tune make sure the diresction is fine, using action 0-8 to chenge the direction. A short pass is a quick and close-range exchange between teammates, commonly used to maintain possession and build an attack. \n
      12 = action_shot, players will try to shoot. When near to the oppenent penalty area , such as zone(x, y), x>18, 4<y<8, try to shoot.\n
      13 = action_sprint, when the player will chose this action, the agent will sprint with sticky action's direction, it will make agent run faster.\n
      14 = action_release_direction, player will stop moving in the current direction. Choose it when you want this agent to change the direction, after this action, you should choose another action from 0-8 to change it.\n
      15 = action_release_sprint, player will stop sprinting, only when the agent's during the sprint. \n
      16 = action_sliding, the player will try to slide the tackle. If your position is on the opponent's path with the ball, you can intercept it. However, if the sliding tackle fails, you will be separated by a large distance, allowing the opponent to ignore the defense.. \n
      17 = action_dribble, players will try to dribble. When they have the ball, dribbling will greatly improve the success rate of dribbling, especially in multi-person double-teams and difficult-to-handle situations..\n
      18 = action_release_dribble, player will stop dribbling.\n
      ---------------------\n
      
      For example,if the active player is Player 2 and you want him to close to the ball and control it, in the texted observation, 
      
      - Forward Player 2 is at Zone(9,9).
      - The ball is at Zone(11,8).
      
      Given these coordinates, the ball is diagonally one zone to the right (east) and one zone down (south) from the player's current position. The most direct route to the ball would indeed be diagonally towards the bottom right. \n

      Therefore, the most appropriate action for Forward Player 2 in this situation would be: 

      6 = action_bottom_right: This sticky action will allow the player to move diagonally in the bottom-right direction (southeast), which is the direct path to where the ball is currently located in Zone(11,8).
      By choosing action_bottom_right, Forward Player 2 can close the distance to the ball more effectively, aligning their movement directly with the ball's current location. Once the player reaches the ball, the subsequent action can be decided based on the situation at that moment (e.g., dribbling, passing, or shooting).
    
      ---------------------\n
      
      Question: What is the next action you want this active player to take?  \n
      Answer: 
  """
  
  
llm_agent_prompt= """
      The texted observation for the current state:\n
      {text_obs}
      ---------------------\n
      Possible Action information is below,
      0 = action_idle, a no-op action, sticky actions are not affected (player maintains his directional movement etc.).\n
      1 = action_left, sticky action and will change the player's direction, player will continue to move left until another action is taken. Such as, from zone(11,4) to zone(10,4)\n
      2 = action_top_left, sticky action and will change the player's direction, player will continue to move top left until another action is taken. Such as, from zone(11,4) to zone(10,5)\n
      3 = action_top, sticky action and will change the player's direction, player will continue to move top until another action is taken. Such as, from zone(11,4) to zone(11,5)\n
      4 = action_top_right, sticky action and will change the player's direction, player will continue to move top right until another action is taken. Such as, from zone(11,4) to zone(12,5)\n
      5 = action_right, sticky action and will change the player's direction, player will continue to move right until another action is taken. Such as, from zone(11,4) to zone(12,4)\n
      6 = action_bottom_right, sticky action and will change the player's direction, player will continue to move bottom right until another action is taken. Such as, from zone(11,4) to zone(12,3)\n
      7 = action_bottom, sticky action and will change the player's direction, player will continue to move bottom until another action is taken.Such as, from zone(11,4) to zone(11,3)\n
      8 = action_bottom_left, sticky action and will change the player's direction, player will continue to move bottom left until another action is taken. Such as, from zone(11,4) to zone(10,3)\n
      9 = action_long_pass, the player will long pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A long pass covers a large distance on the field. \n
      10 = action_high_pass, the player will high pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A high pass sends the ball into the air, often over obstacles, to reach a teammate.  \n
      11 = action_short_pass, the player will short pass to their teammate in his current direcetion. Before you pass, you should tune make sure the diresction is fine, using action 0-8 to chenge the direction. A short pass is a quick and close-range exchange between teammates, commonly used to maintain possession and build an attack. \n
      12 = action_shot, players will try to shoot. When near to the oppenent penalty area , such as zone(x, y), x>18, 4<y<8, try to shoot.\n
      13 = action_sprint, when the player will chose this action, the agent will sprint with sticky action's direction, it will make agent run faster.\n
      14 = action_release_direction, player will stop moving in the current direction. Choose it when you want this agent to change the direction, after this action, you should choose another action from 0-8 to change it.\n
      15 = action_release_sprint, player will stop sprinting, only when the agent's during the sprint. \n
      16 = action_sliding, the player will try to slide the tackle. If your position is on the opponent's path with the ball, you can intercept it. However, if the sliding tackle fails, you will be separated by a large distance, allowing the opponent to ignore the defense.. \n
      17 = action_dribble, players will try to dribble. When they have the ball, dribbling will greatly improve the success rate of dribbling, especially in multi-person double-teams and difficult-to-handle situations..\n
      18 = action_release_dribble, player will stop dribbling.\n
      ---------------------\n
      
      For example,if the active player is Player 2 and you want him to close to the ball and control it, in the texted observation, 
      
      - Forward Player 2 is at Zone(9,9).
      - The ball is at Zone(11,8).
      
      Given these coordinates, the ball is diagonally one zone to the right (east) and one zone down (south) from the player's current position. The most direct route to the ball would indeed be diagonally towards the bottom right. \n

      Therefore, the most appropriate action for Forward Player 2 in this situation would be: 

      6 = action_bottom_right: This sticky action will allow the player to move diagonally in the bottom-right direction (southeast), which is the direct path to where the ball is currently located in Zone(11,8).
      By choosing action_bottom_right, Forward Player 2 can close the distance to the ball more effectively, aligning their movement directly with the ball's current location. Once the player reaches the ball, the subsequent action can be decided based on the situation at that moment (e.g., dribbling, passing, or shooting).
    
      ---------------------\n
      
      Question: What is the next action you want this active player to take?  \n
      Answer: 
  """ 

llm_rag_prompt= """
      
      The pseudocode for the policy you want this active player to implement is as follows:\n
      {policy_code}
      ---------------------\n
      The texted observation for the current state:\n
      {text_obs}
      ---------------------\n
      Possible Action information is below,
      0 = action_idle, a no-op action, sticky actions are not affected (player maintains his directional movement etc.).\n
      1 = action_left, sticky action and will change the player's direction, player will continue to move left until another action is taken. Such as, from zone(11,4) to zone(10,4)\n
      2 = action_top_left, sticky action and will change the player's direction, player will continue to move top left until another action is taken. Such as, from zone(11,4) to zone(10,5)\n
      3 = action_top, sticky action and will change the player's direction, player will continue to move top until another action is taken. Such as, from zone(11,4) to zone(11,5)\n
      4 = action_top_right, sticky action and will change the player's direction, player will continue to move top right until another action is taken. Such as, from zone(11,4) to zone(12,5)\n
      5 = action_right, sticky action and will change the player's direction, player will continue to move right until another action is taken. Such as, from zone(11,4) to zone(12,4)\n
      6 = action_bottom_right, sticky action and will change the player's direction, player will continue to move bottom right until another action is taken. Such as, from zone(11,4) to zone(12,3)\n
      7 = action_bottom, sticky action and will change the player's direction, player will continue to move bottom until another action is taken.Such as, from zone(11,4) to zone(11,3)\n
      8 = action_bottom_left, sticky action and will change the player's direction, player will continue to move bottom left until another action is taken. Such as, from zone(11,4) to zone(10,3)\n
      9 = action_long_pass, the player will long pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A long pass covers a large distance on the field. \n
      10 = action_high_pass, the player will high pass to their teammate in his current direcetion. Before you pass, you should relase the sprint if the agent is doing sprint. A high pass sends the ball into the air, often over obstacles, to reach a teammate.  \n
      11 = action_short_pass, the player will short pass to their teammate in his current direcetion. Before you pass, you should tune make sure the diresction is fine, using action 0-8 to chenge the direction. A short pass is a quick and close-range exchange between teammates, commonly used to maintain possession and build an attack. \n
      12 = action_shot, players will try to shoot. When near to the oppenent penalty area , such as zone(x, y), x>18, 4<y<8, try to shoot.\n
      13 = action_sprint, when the player will chose this action, the agent will sprint with sticky action's direction, it will make agent run faster.\n
      14 = action_release_direction, player will stop moving in the current direction. Choose it when you want this agent to change the direction, after this action, you should choose another action from 0-8 to change it.\n
      15 = action_release_sprint, player will stop sprinting, only when the agent's during the sprint. \n
      16 = action_sliding, the player will try to slide the tackle. If your position is on the opponent's path with the ball, you can intercept it. However, if the sliding tackle fails, you will be separated by a large distance, allowing the opponent to ignore the defense.. \n
      17 = action_dribble, players will try to dribble. When they have the ball, dribbling will greatly improve the success rate of dribbling, especially in multi-person double-teams and difficult-to-handle situations..\n
      18 = action_release_dribble, player will stop dribbling.\n
      ---------------------\n
      
      For example,if the active player is Player 2 and you want him to close to the ball and control it, in the texted observation, 
      
      - Forward Player 2 is at Zone(9,9).
      - The ball is at Zone(11,8).
      
      Given these coordinates, the ball is diagonally one zone to the right (east) and one zone down (south) from the player's current position. The most direct route to the ball would indeed be diagonally towards the bottom right. \n

      Therefore, the most appropriate action for Forward Player 2 in this situation would be: 

      6 = action_bottom_right: This sticky action will allow the player to move diagonally in the bottom-right direction (southeast), which is the direct path to where the ball is currently located in Zone(11,8).
      By choosing action_bottom_right, Forward Player 2 can close the distance to the ball more effectively, aligning their movement directly with the ball's current location. Once the player reaches the ball, the subsequent action can be decided based on the situation at that moment (e.g., dribbling, passing, or shooting).
    
      ---------------------\n
      
      Question: What is the next action you want this active player to take?  \n
      Answer: 
  """
  

  

raw_policy_prompt_step2_shorter= """
      
    The pseudocode for the policy you want this active player to implement is as follows:\n
    {policy_code}
    ---------------------\n
    The texted observation for the current state:\n
    {text_obs}
    ---------------------\n
    Possible Action information is below,
    action set = (
        "0": "action_idle, a no-op action, sticky actions are not affected, player maintains his directional movement.",
        "1": "action_left, sticky action, player will continue to move left until another action is taken.",
        "2": "action_top_left, sticky action, player will continue to move top left until another action is taken.",
        "3": "action_top, sticky action, player will continue to move top until another action is taken.",
        "4": "action_top_right, sticky action, player will continue to move top right until another action is taken.",
        "5": "action_right, sticky action, player will continue to move right until another action is taken.",
        "6": "action_bottom_right, sticky action, player will continue to move bottom right until another action is taken.",
        "7": "action_bottom, sticky action, player will continue to move bottom until another action is taken.",
        "8": "action_bottom_left, sticky action, player will continue to move bottom left until another action is taken.",
        "9": "action_long_pass, player will try to pass the ball to a teammate.",
        "10": "action_high_pass, player will try to pass the ball to a teammate.",
        "11": "action_short_pass, player will try to pass the ball to a teammate.",
        "12": "action_shot, player will try to shoot the ball.",
        "13": "action_sprint, player will sprint.",
        "14": "action_release_direction, player will stop moving in the current direction.",
        "15": "action_release_sprint, player will stop sprinting.",
        "17": "action_dribble, player will try to dribble.",
        "18": "action_release_dribble, player will stop dribbling."
    )
    ---------------------\n
      
    Requirements:

    About the action choosing:
    - Please choose the action that best fits the code logic.

    About the format:
        - you should answer in pure JSON format with the key: 'action': a int number from 0 to 18.
            
    Response example (you should resposne in the following order):        
    {{
        "action": 0
    }}
  """
  




gloabl_prompt_step2= """
    I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator. 
    
    I will give you the presudocode snippets to define the specific policy for the active player in the football game. 
    
    These codes represent absolute correctness, do not add other common logic. Your task is to select the action that best fits and executes the code based on the logic of the given code.
    
    Requirements:

    About the action choosing:
    - Please choose the action that best fits the code logic.

    About the format:
        - you should answer in pure JSON format with the key: 'action': a int number from 0 to 18, 'thought': why you choose this action. without any other information or code. For example, you should not add the ```json``` tag in the answer.
            
    Response example (you should resposne in the following order):        
    {{
        "action": 0,
        "thought": "based on your thought, tell me the optimal action you would like to select in the action set.",
    }}

"""

def parse_args(return_parser=False):
    parser = argparse.ArgumentParser(description='Football Environment Configuration')
    
    parser.add_argument('--num_players', type=int, default=1, help='Number of players')
    
    parser.add_argument('--num_timesteps', type=int, default=1500, help='Number of timesteps per episode')
    parser.add_argument('--write_goal_dumps', default=False, action="store_true", help='Write goal dumps')
    parser.add_argument('--write_full_episode_dumps', default=False, action="store_true", help='Write full episode dumps')
    
    project_root = os.environ.get("PLFB_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    data_root = os.environ.get("PLFB_DATASET_PATH", os.path.join(project_root, "data"))
    artifact_root = os.environ.get("PLFB_ARTIFACT_ROOT")
    default_filter_path = os.environ.get("PLFB_FILTER_PATH")
    if default_filter_path is None:
        if artifact_root:
            default_filter_path = os.path.join(artifact_root, "book_derived", "retrieval")
        else:
            default_filter_path = os.path.join(data_root, "v4-gpt-3.5-turbo-1106-level-strict")

    parser.add_argument('--json_log_path', type=str, default=os.environ.get("PLFB_JSON_LOG_PATH", os.path.join(project_root, "llm", "result")), help='The path of the json log file')
    
    # Llama index related
    parser.add_argument('--use_llama_index', default=False, action="store_true", help='Use llama index or not')
    parser.add_argument('--encoder_type', type=str, default='defult', help='Type of the encoder (defult)')
    parser.add_argument('--llm_type', type=str, default='gpt-3', help='Type of the index (llama or gpt-3, gpt-4)')
    
    parser.add_argument('--filter_path', type=str, default=default_filter_path, help='Json of the Policy, Reward and Transition file path')
    
    parser.add_argument('--imaginary_obv_prompt', type=str, default=imaginary_obv_prompt_v1, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--imaginary_obv_prompt_v2', type=str, default=imaginary_obv_prompt_v2, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--imaginary_obv_prompt_v2_shorter', type=str, default=imaginary_obv_prompt_v2_shorter, help='Ask Policy, through, QA prompt template string')
    
    parser.add_argument('--raw_policy_prompt', type=str, default=raw_policy_prompt_all, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--raw_policy_prompt_step1', type=str, default=raw_policy_prompt_step1, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--raw_policy_prompt_step1_all_code', type=str, default=raw_policy_prompt_step1_all_code, help='Ask Policy, through, QA prompt template string')
    
    parser.add_argument('--raw_policy_prompt_step2', type=str, default=raw_policy_prompt_step2, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--raw_policy_prompt_step2_shorter', type=str, default=raw_policy_prompt_step2_shorter, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--gloabl_prompt_step2', type=str, default=gloabl_prompt_step2, help='Ask Policy, through, QA prompt template string')
    
    parser.add_argument('--llm_agent_prompt', type=str, default=llm_agent_prompt, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--llm_rag_prompt', type=str, default=llm_rag_prompt, help='Ask Policy, through, QA prompt template string')
    
    parser.add_argument('--use_openai_compat_client', default=_env_flag("PLFB_USE_OPENAI_COMPAT_CLIENT", False), action="store_true", help='Use the OpenAI-compatible REST client; honors OPENAI_BASE_URL for compatible providers')

    parser.add_argument('--offline_dataset_collection', default=False, action="store_true", help='Use llama index or not')
    parser.add_argument('--use_pydantic_basemodel', default=False, action="store_true", help='Use llama index or not')
    
    parser.add_argument('--number', type=int, default=0, help='Number of the current task')

    if return_parser:
        return parser
    else:
        return parser.parse_args()
  
  

