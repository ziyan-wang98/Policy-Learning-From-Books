from llm.utils.openai_compat import openai_chat_query
import numpy as np
import json
from termcolor import colored
import os
import re

MAX_ATTAMPTS = 500

def llm_agent(current_dict_obs, model=None):
    model = model or os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini')
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
    global_prompt= """
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

    action_prompt = llm_agent_prompt.format(text_obs=current_dict_obs)
    for attempt in range(MAX_ATTAMPTS):
        try:
            action_response = openai_chat_query(global_prompt, action_prompt, model_name=model, req_json=True)
            action_r = json.loads(action_response)
            action_rj = json.loads(action_r["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
            action = action_rj["action"]
            # action_thought = action_rj["thought"]
            break
        except Exception as e:
            print(colored(f"Action Error: {e}, retrying...", "red"))
            print("action_response: ", action_response)
            if attempt == MAX_ATTAMPTS - 1:
                raise e
            continue
    return action

def llm_rag(current_dict_obs, policy_str, model=None):
    model = model or os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini')
    llm_rag_prompt= """

      The relative policy from the book you want this active player to implement is as follows:\n
      {policy_str}
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
    global_prompt= """
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

    action_prompt = llm_rag_prompt.format(text_obs=current_dict_obs, policy_str=policy_str)
    for attempt in range(MAX_ATTAMPTS):
        try:
            action_response = openai_chat_query(global_prompt, action_prompt, model_name=model, req_json=True)
            action_r = json.loads(action_response)
            action_rj = json.loads(action_r["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
            action = action_rj["action"]
            # action_thought = action_rj["thought"]
            break
        except Exception as e:
            print(colored(f"Action Error: {e}, retrying...", "red"))
            print("action_response: ", action_response)
            if attempt == MAX_ATTAMPTS - 1:
                raise e
            continue
    return action

def RT2(current_dict_obs):

    from gradientai import Gradient
    adapter_id = os.environ.get("GRADIENT_RT2_MODEL_ADAPTER_ID")
    if not os.environ.get("GRADIENT_ACCESS_TOKEN") or not os.environ.get("GRADIENT_WORKSPACE_ID") or not adapter_id:
        raise RuntimeError(
            "Set GRADIENT_ACCESS_TOKEN, GRADIENT_WORKSPACE_ID, and "
            "GRADIENT_RT2_MODEL_ADAPTER_ID before using the RT2 baseline."
        )
    gradient = Gradient()

    new_model_adapter = gradient.get_model_adapter(model_adapter_id=adapter_id)


    for attempt in range(MAX_ATTAMPTS):
        try:
            instruction = llm_rt_policy_prompt(current_dict_obs)

            RT2_prompt = f"<s>[INST] <<SYS>>\nYou are an instruction-following model. \n<</SYS>>\n\n{instruction} [/INST] "

            action_response = new_model_adapter.complete(query=RT2_prompt, max_generated_token_count=100).generated_output

            first_number = re.search(r'\d+', action_response).group()
            action = int(first_number)

            break
        except Exception as e:
            print(colored(f"Get Action: Error: {e}, retrying...", "red"))
            print("action_response: ", action_response)
            if attempt == MAX_ATTAMPTS - 1:
                raise e
            continue

    return action


from simulator_info import *
from CONFIG import *
def llm_rt_policy_prompt(obs):
    knowledge_type = KnowledgeType.Policy
    raw_policy_prompt = """
        I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator.

        The action space of the football simulator:
        {action_space_desc}

        The current observation you are facing in the football simulator:

        {obs}

        You Should response the best action number directly. Response Example:

        0
    """.format(knowledge_type=knowledge_type, knowledge_function=element_type_prompt_dict[knowledge_type], knowledge_format=element_type_format_dict[knowledge_type],
                       knwoledge_function_example=element_type_example_dict[knowledge_type],
                       observation_space_desc=OBSERVATION_SPACE_DESC, action_space_desc=ACTION_SPACE_DESC, obs=obs)
    return raw_policy_prompt
