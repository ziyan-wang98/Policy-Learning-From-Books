import argparse
import os
from pathlib import Path
from pydantic import BaseModel, Field


def _artifact_path(*parts):
    return str(Path(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts")).joinpath(*parts))


def _work_path(*parts):
    return str(Path(os.environ.get("PLFB_WORK_ROOT", "runs")).joinpath(*parts))

# Gpt as the decision policy
raw_policy_prompt= """
      You are a football coach. You are coaching a football game. \n
      You are coaching the Left team. \n
      The Right team is the opponent team. \n
      Context information is below.\n
      ---------------------\n
      {context_str}\n
      ---------------------\n
      Given the context information and not prior knowledge, 
      answer the query in the style of a football coach.\n
      Language Observation for this football game at current time step is below.\n
      Language Observation for this football game at current time step is below.\n
      The football court is divided into 240 zones, each zone is 3.5m x 5m in size. \n
      The zones are numbered from (1,1) to (20,12) from left to right, bottom to top. \n
      If the some players or ball are in the zone (10,7), which means the ball is in the middle of the football court. \n
      ---------------------\n
      {query_str}\n
      ---------------------\n
      Question: What is the next action you want this active player to take? you can only choose from the following actions (0 to 18), only answer a int number: \n
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
      Objective: when you do not have ball, you cant shot or pass, you need to run to the ball as soon as possible and try to intercept it, and when you have the ball, you should try to score.  please be sure to give them according to their duty/role. For example, the forwards should be responsible for offense and the defenders should be responsible for defense.. \n\n
      For defence, if you want to defence, you should try to use action 0-8 move the defender to intercept it by blocking the opponent's path. \n
      Please directly type the number of the action you want the active player to take, without any other words. \n
      response example (you should use the following format to construct your response): 0\n   
      Answer: 
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


fc_policy_prompt= """
      You are a football coach. You are coaching a football game. \n
      You are coaching the Left team. \n
      The Right team is the opponent team. \n
      Context information is below.\n
      ---------------------\n
      {context_str}\n
      ---------------------\n
      Given the context information and not prior knowledge, 
      answer the query in the style of a football coach.\n
      Language Observation for this football game at current time step is below.\n
      The football court is divided into 240 zones, each zone is 3.5m x 5m in size. \n
      The zones are numbered from (1,1) to (20,12) from left to right, bottom to top. \n
      If the some players or ball are in the zone (10,7), which means the ball is in the middle of the football court. \n
      ---------------------\n
      {query_str}\n
      ---------------------\n
      Question: What is the next action function you want this active player to take? 
      Answer: 
  """  

tool = [
    {
        "type": "function",
        "function": {
            "name": "action_0",
            "description":  "action_0 is the action_idle, a no-op action, sticky actions are not affected, player maintains his directional movement."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_1",
            "description":  "action_1 is the action_left, sticky action, player will continue to move left until another action is taken."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_2",
            "description":  "action_2 is the action_top_left, sticky action, player will continue to move top left until another action is taken."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_3",
            "description":  "action_3 is the action_top, sticky action, player will continue to move top until another action is taken."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_4",
            "description":  "action_4 is the action_top_right, sticky action, player will continue to move top right until another action is taken."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_5",
            "description":  "action_5 is the action_right, sticky action, player will continue to move right until another action is taken."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_6",
            "description":  "action_6 is the action_bottom_right, sticky action, player will continue to move bottom right until another action is taken."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_7",
            "description":  "action_7 is the action_bottom, sticky action, player will continue to move bottom until another action is taken."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_8",
            "description":  "action_8 is the action_bottom_left, sticky action, player will continue to move bottom left until another action is taken."
        }
    },
    {
        "type": "function",
         "function": {
            "name": "action_9",
            "description":  "action_9 is the action_long_pass, player will try to pass the ball to a teammate."
        }                           
    },
    {
        "type": "function",
        "function": {
            "name": "action_10",
            "description":  "action_10 is the action_high_pass, player will try to pass the ball to a teammate."
        }   
    },
    {
        "type": "function",
        "function": {
            "name": "action_11",
            "description":  "action_11 is the action_short_pass, player will try to pass the ball to a teammate."
        }   
    },
    {
        "type": "function",
        "function": {
            "name": "action_12",
            "description":  "action_12 is the action_shot, player will try to shoot the ball."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_13",
            "description":  "action_13 is the action_sprint, player will sprint."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_14",
            "description":  "action_14 is the action_release_direction, player will stop moving in the current direction."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_15",
            "description":  "action_15 is the action_release_sprint, player will stop sprinting."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_16",
            "description":  "action_16 is the action_sliding, player will try to slide."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_17",
            "description":  "action_17 is the action_dribble, player will try to dribble."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "action_18",
            "description":  "action_18 is the action_release_dribble, player will stop dribbling."
        }
    }
]


no_ball_tool = [
    {
        "type": "function",
        "function": {
            "name": "assign_actions",
            "description": "Assigns actions to players based on given strategies or rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_actions": {
                        "type": "list",
                        "description": "An action list contains 11 players' action from action 0 to 18, from Left team Player1 to Left team Player11, the action of each player."
                    }, 
                },
                "metadata": {
                    "actions": {
                        "type": "dict",
                        "action_0": "action_idle, a no-op action, sticky actions are not affected, player maintains his directional movement.",
                        "action_1": "action_left, sticky action, player will continue to move left until another action is taken.",
                        "action_2": "action_top_left, sticky action, player will continue to move top left until another action is taken.",
                        "action_3": "action_top, sticky action, player will continue to move top until another action is taken.",
                        "action_4": "action_top_right, sticky action, player will continue to move top right until another action is taken.",
                        "action_5": "action_right, sticky action, player will continue to move right until another action is taken.",
                        "action_6": "action_bottom_right, sticky action, player will continue to move bottom right until another action is taken.",
                        "action_7": "action_bottom, sticky action, player will continue to move bottom until another action is taken.",
                        "action_8": "action_bottom_left, sticky action, player will continue to move bottom left until another action is taken.",
                        "action_13": "action_sprint, player will sprint.",
                        "action_14": "action_release_direction, player will stop moving in the current direction.",
                        "action_15": "action_release_sprint, player will stop sprinting.",
                        "action_16": "action_sliding, player will try to slide."
                    }
                },
                "required": ["player_actions"]
            },
        }
    }
]


ball_ower_tool = [
    {
        "type": "function",
        "function": {
            "name": "assign_actions",
            "description": "Assigns actions to players based on given strategies or rules.",
            "parameters": [
                {
                    "name": "player_actions",
                    "type": "list",
                    "description": "A list of actions for each player."
                }
            ],
            "return": {
                "type": "dict",
                "description": "A dictionary mapping each player to their respective action."
            },
            "metadata": {
                "actions": {
                    "action_0": "action_idle, a no-op action, sticky actions are not affected, player maintains his directional movement.",
                    "action_1": "action_left, sticky action, player will continue to move left until another action is taken.",
                    "action_2": "action_top_left, sticky action, player will continue to move top left until another action is taken.",
                    "action_3": "action_top, sticky action, player will continue to move top until another action is taken.",
                    "action_4": "action_top_right, sticky action, player will continue to move top right until another action is taken.",
                    "action_5": "action_right, sticky action, player will continue to move right until another action is taken.",
                    "action_6": "action_bottom_right, sticky action, player will continue to move bottom right until another action is taken.",
                    "action_7": "action_bottom, sticky action, player will continue to move bottom until another action is taken.",
                    "action_8": "action_bottom_left, sticky action, player will continue to move bottom left until another action is taken.",
                    "action_9": "action_long_pass, player will try to pass the ball to a teammate.",
                    "action_10": "action_high_pass, player will try to pass the ball to a teammate.",
                    "action_11": "action_short_pass, player will try to pass the ball to a teammate.",
                    "action_12": "action_shot, player will try to shoot the ball.",
                    "action_13": "action_sprint, player will sprint.",
                    "action_14": "action_release_direction, player will stop moving in the current direction.",
                    "action_15": "action_release_sprint, player will stop sprinting.",
                    "action_17": "action_dribble, player will try to dribble.",
                    "action_18": "action_release_dribble, player will stop dribbling."
                }
            }
        }
    }
]







ma_fc_policy_prompt= """
      You are a football coach. You are coaching a football game. \n
      You are coaching the Left team. \n
      The Right team is the opponent team. \n
      Context information is below.\n
      ---------------------\n
      {context_str}\n
      ---------------------\n
      Given the context information and not prior knowledge, 
      answer the query in the style of a football coach.\n
      Language Observation for this football game at current time step is below.\n
      The football court is divided into 240 zones, each zone is 3.5m x 5m in size. \n
      The zones are numbered from (1,1) to (20,12) from left to right, bottom to top. \n
      If the some players or ball are in the zone (10,7), which means the ball is in the middle of the football court. \n
      ---------------------\n
      {query_str}\n
      ---------------------\n
      Question: What actions do you want all players to take from player 1 to player 11? 
      Answer: 
  """  

ma_raw_policy_prompt= """
      You are a football coach. You are coaching a football game. \n
      You are coaching the Left team. \n
      The Right team is the opponent team. \n
      Context information is below.\n
      ---------------------\n
      {context_str}\n
      ---------------------\n
      Given the context information and not prior knowledge, 
      answer the query in the style of a football coach.\n
      Language Observation for this football game at current time step is below.\n
      Language Observation for this football game at current time step is below.\n
      The football court is divided into 240 zones, each zone is 3.5m x 5m in size. \n
      The zones are numbered from (1,1) to (20,12) from left to right, bottom to top. \n
      If the some players or ball are in the zone (10,7), which means the ball is in the middle of the football court. \n
      ---------------------\n
      {query_str}\n
      ---------------------\n
      Question: What actions do you want all players to take? You can only choose from the following actions (0 to 18), and answer in a JSON array format: \n
      0 = action_idle, a no-op action, sticky actions are not affected (player maintains his directional movement etc.).\n
      1 = action_left, sticky action, player will continue to move left until another action is taken.\n
      2 = action_top_left, sticky action, player will continue to move top left until another action is taken.\n
      3 = action_top, sticky action, player will continue to move top until another action is taken.\n
      4 = action_top_right, sticky action, player will continue to move top right until another action is taken.\n
      5 = action_right, sticky action, player will continue to move right until another action is taken.\n
      6 = action_bottom_right, sticky action, player will continue to move bottom right until another action is taken.\n
      7 = action_bottom, sticky action, player will continue to move bottom until another action is taken.\n
      8 = action_bottom_left, sticky action, player will continue to move bottom left until another action is taken.\n
      9 = action_long_pass, player will try to pass the ball to a teammate.\n
      10 = action_high_pass, player will try to pass the ball to a teammate.\n
      11 = action_short_pass, player will try to pass the ball to a teammate.\n
      12 = action_shot, player will try to shoot the ball.\n
      13 = action_sprint, player will sprint.\n
      14 = action_release_direction, player will stop moving in the current direction.\n
      15 = action_release_sprint, player will stop sprinting.\n
      16 = action_sliding, player will try to slide.\n
      17 = action_dribble, player will try to dribble.\n
      18 = action_release_dribble, player will stop dribbling.\n
      Please directly type the actions you want all 11 players to take in the following format, without any other words.
      Response example (you should use the following format to construct your response): [[11],[9],[8],[3],[7],[3],[2],[10],[1],[0],[4]] \n
      As a football coach, you should encourage your players to take more actions to achieve goal, rather than choose the idle actipm. \n
      Answer: 
  """


def parse_args(return_parser=False):
    parser = argparse.ArgumentParser(description='Football Environment Configuration')
    
    parser.add_argument('--environment', type=str, default='11_vs_11_hard_stochastic', help='Type of the environment')
    parser.add_argument('--algo', type=str, default='baseline_v1_single_agent', help='Type of the algorithm')
    parser.add_argument('--representation', type=str, default='simple115v3', help='Type of representation')
    parser.add_argument('--logdir', type=str, default=os.environ.get("PLFB_GFOOTBALL_LOGDIR", _work_path("gfootball_res")), help='Directory for logs')
    parser.add_argument('--render', default=False, action="store_true", help='Render environment or not')
    parser.add_argument('--num_players', type=int, default=1, help='Number of players')
    parser.add_argument('--logall', default=True, action="store_true", help='Log all information or not')
    
    parser.add_argument('--reward_prompt', type=str, default=None, help='Ask Reward, through QA prompt template string')
    parser.add_argument('--transition_prompt', type=str, default=None, help='Ask Transition, though QA prompt template string')
    parser.add_argument('--num_timesteps', type=int, default=1500, help='Number of timesteps per episode')
    parser.add_argument('--write_goal_dumps', default=False, action="store_true", help='Write goal dumps')
    parser.add_argument('--write_full_episode_dumps', default=False, action="store_true", help='Write full episode dumps')
    

    # Llama index related
    parser.add_argument('--use_llama_index', default=False, action="store_true", help='Use llama index or not')
    
    parser.add_argument('--encoder_type', type=str, default='defult', help='Type of the encoder (defult)')
    parser.add_argument('--llm_type', type=str, default='gpt-3', help='Type of the index (llama or gpt-3, gpt-4)')
    parser.add_argument('--filter_path', type=str, default=os.environ.get("PLFB_FILTER_PATH", _artifact_path("book_derived", "v4-gpt-3.5-turbo-1106-level-strict")), help='Json of the Policy, Reward and Transition file path')
    parser.add_argument('--raw_policy_prompt', type=str, default=raw_policy_prompt, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--fc_policy_prompt', type=str, default=fc_policy_prompt, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--raw_policy_prompt_step1', type=str, default=raw_policy_prompt_step1, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--raw_policy_prompt_step2', type=str, default=raw_policy_prompt_step2, help='Ask Policy, through, QA prompt template string')
    
    parser.add_argument('--gloabl_prompt_step2', type=str, default=gloabl_prompt_step2, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--offline_dataset_collection', default=False, action="store_true", help='Use llama index or not')
    
    parser.add_argument('--use_function_calling', default=False, action="store_true", help='Use function calling to get the policy')
    parser.add_argument('--function_calling_tool', type=str, default=tool, help='The tools that can be called')
    parser.add_argument('--use_pydantic_basemodel', default=False, action="store_true", help='Use llama index or not')
    
    # obs2text related
    parser.add_argument('--raw_text_flag',default=False, action="store_true", help='Type of the obs2text (default)')
    parser.add_argument('--block_mode', type=str, default='240', help='Type of the obs2text (default)')
    
    
    # Multi-agent related
    parser.add_argument('--individual_policy_flag', default=False, action="store_true", help='Wheather use individual policy for each player, cost a lot, since it will call the gpt/llama query for each player each time step')
    parser.add_argument('--no_ball_tool', type=str, default=no_ball_tool, help='The tools that can be called')
    parser.add_argument('--ball_ower_tool', type=str, default=ball_ower_tool, help='The tools that can be called')
    parser.add_argument('--ma_fc_policy_prompt', type=str, default=ma_fc_policy_prompt, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--ma_raw_policy_prompt', type=str, default=ma_raw_policy_prompt, help='Ask Policy, through, QA prompt template string')
    
    parser.add_argument('--e2e', default=False, action="store_true", help='Use function calling to get the policy')
    
    parser.add_argument('--json_log_path', type=str, default='football_llm/llm/result', help='The path of the json log file')
    if return_parser:
        return parser
    else:
        return parser.parse_args()
  
  


