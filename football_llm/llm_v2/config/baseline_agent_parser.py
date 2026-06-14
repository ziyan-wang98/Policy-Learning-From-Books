import argparse
import os
from pathlib import Path
from pydantic import BaseModel, Field


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
        
        I will provide you with a few pre-code snippets of the most relevant football manager policy from the tutorial, written in Python style.
        
        {context_str}
        
        These codes may share some logic. Your task is to combine the general logic of the given code with a specific football scenario given below to return a policy that is most suitable for this moment and what is the optimal policy for the Left team's active player for the following period of time.
        
        Here, I will give you the texted observation of the current football game, which is a list of 240 numbers, each number represents a zone in the football court.
        
        {query_str}
        
        Requirements:

        About the aggregated code:
        - Please provide the PYTHON-tyle presudo code as detailed as you can to cover the most information of the original content.
        - Using the least number of presudocode items.
        - Covering all of the code. However, please feel free to add more presudocode items if needed.
        - Since I will delete the original code after getting your aggregated code,  you cannot call the presudocodes that I provided in the prompt. If it is necessary to call the presudocode, please still return the presudocode as an individual item in the answer.
        
        About the format:
            - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.
            
        The response example:
        
        {
            "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
            "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
            "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
        }
        
        
      You are a football coach. You are coaching a football game. \n
      You are coaching the Left team. \n
      The Right team is the opponent team. \n
      Context information is below.\n
      ---------------------\n
      \n
      ---------------------\n
      Given the context information and not prior knowledge, 
      answer the query in the style of a football coach.\n
      Language Observation for this football game at current time step is below.\n
      Language Observation for this football game at current time step is below.\n
      The football court is divided into 240 zones, each zone is 3.5m x 5m in size. \n
      The zones are numbered from (1,1) to (20,12) from left to right, bottom to top. \n
      If the some players or ball are in the zone (10,7), which means the ball is in the middle of the football court. \n
      ---------------------\n
      \n
      ---------------------\n
  """  


raw_policy_prompt_step2= """
      You are a football coach. You are coaching a football game. \n
      You are coaching the Left team. \n
      The Right team is the opponent team. \n
      The pseudocode for the policy you want this active player to implement is as follows:\n
      ---------------------\n
      {policy_str}\n
      ---------------------\n
      Possible Action information is below.\n
      ---------------------\n
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
      Question: What is the next action you want this active player to take? you can only choose from the following actions (0 to 18), only answer a int number: \n
      Answer: 
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


def parse_args():
    parser = argparse.ArgumentParser(description='Football Environment Configuration')
    
    parser.add_argument('--environment', type=str, default='11_vs_11_easy_stochastic', help='Type of the environment')
    parser.add_argument('--algo', type=str, default='baseline_v1_single_agent', help='Type of the algorithm')
    parser.add_argument('--representation', type=str, default='simple115v3', help='Type of representation')
    parser.add_argument('--logdir', type=str, default=os.environ.get("PLFB_GFOOTBALL_LOGDIR", _work_path("gfootball_res")), help='Directory for logs')
    parser.add_argument('--render', default=False, action="store_true", help='Render environment or not')
    parser.add_argument('--num_players', type=int, default=1, help='Number of players')
    
    parser.add_argument('--reward_prompt', type=str, default=None, help='Ask Reward, through QA prompt template string')
    parser.add_argument('--transition_prompt', type=str, default=None, help='Ask Transition, though QA prompt template string')
    parser.add_argument('--num_timesteps', type=int, default=1500, help='Number of timesteps per episode')
    parser.add_argument('--write_goal_dumps', default=False, action="store_true", help='Write goal dumps')
    parser.add_argument('--write_full_episode_dumps', default=False, action="store_true", help='Write full episode dumps')
    

    # Llama index related
    parser.add_argument('--encoder_type', type=str, default='defult', help='Type of the encoder (defult)')
    parser.add_argument('--llm_type', type=str, default='gpt-3', help='Type of the index (llama or gpt-3, gpt-4)')
    project_root = os.environ.get("PLFB_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    data_root = os.environ.get("PLFB_DATASET_PATH", os.path.join(project_root, "data"))
    parser.add_argument('--filter_path', type=str, default=os.environ.get("PLFB_FILTER_PATH", os.path.join(data_root, "v2-football_intelligence_res-gpt-3.5-turbo-1106-level-strict")), help='Json of the Policy, Reward and Transition file path')
    parser.add_argument('--raw_policy_prompt', type=str, default=raw_policy_prompt, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--fc_policy_prompt', type=str, default=fc_policy_prompt, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--raw_policy_prompt_step1', type=str, default=raw_policy_prompt_step1, help='Ask Policy, through, QA prompt template string')
    parser.add_argument('--raw_policy_prompt_step2', type=str, default=raw_policy_prompt_step2, help='Ask Policy, through, QA prompt template string')
    
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
    
    return parser.parse_args()
  
  

