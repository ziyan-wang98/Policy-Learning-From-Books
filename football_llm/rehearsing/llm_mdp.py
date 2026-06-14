import json
import copy
from termcolor import colored
import re
import numpy as np
from llm.utils.openai_compat import openai_chat_query
from llm.utils.obs2text import get_zons_240_list, vector_to_direction,vector_to_direction

from simulator_info import DIRECTIONS, PLAYER_ROLES, sticky_list, sticky_list_map_dir
from CONFIG import KnowledgeType
from rehearsing.prompt import llm_policy_prompt, llm_transition_reward_prompt, llm_transition_prompt, llm_reward_prompt
from llm.openai_server import OpenAIServer
from llm.utils.obs2text import img_obs_to_text, imaginary_data_observation_stacked_info
from funcs import json_str_clean
from json_repair import repair_json


def code_str_format(code_str):
    if code_str == 'EMPTY' or code_str is None:
        return 'EMPTY'
    else:
        # return '=== \n ' + '\n'.join(code_str['code']) + '\n ==== \n'
        return '=== \n Reason to generate the code:' + '\n'.join(code_str['analyze']) + '\n \n' + '\n'.join(code_str['code']) + '\n ==== \n'

MAX_ATTAMPTS = 4

class RolloutMDP:
    def __init__(self, model, seed, skip_knowledge, simulator_info):
        self.model = model
        self.openai_server = OpenAIServer(model=model, temp=0.1, max_token=1024, seed=seed)
        self.skip_knowledge = skip_knowledge
        self.sim_info = simulator_info

    def recent_obs_to_text(self, recent_obs):
        raise NotImplementedError

    def get_current_obs(self, recent_obs):
        raise NotImplementedError

    def img_obs_to_text_fn(self, img_obs, stacked):
        raise NotImplementedError

    def post_process_img_next_obs(self, img_next_obs, recent_obs, action, other_information):
        raise NotImplementedError

    def _get_response(self, input_prompt, knowledge_type):

        img_data = None
        img_thought = 'INVALID'
        response = None
        for _ in range(MAX_ATTAMPTS):
            try:
                if 'gpt' in self.model:
                    response = openai_chat_query("", input_prompt, model_name=self.model, req_json=True)
                    response = json.loads(response)
                    response = response["choices"][0]["message"]["content"]
                    rj = json.loads(repair_json(json_str_clean(response, single_line=True)))
                else:
                    response = self.openai_server.chat(input_prompt)
                    rj = json.loads(repair_json(json_str_clean(response, single_line=True)))
                img_thought = rj["thought"]
                if knowledge_type == KnowledgeType.Policy:
                    img_data = rj["action"]
                    img_data = int(img_data) if img_data is not None else 0
                elif knowledge_type == KnowledgeType.Dynamics:
                    img_data = rj
                    del img_data['thought']
                elif knowledge_type == KnowledgeType.Reward:
                    img_data = rj["dense_reward"]
                break
            except json.JSONDecodeError as e:
                print(colored(f"Error: {e}, retrying...", "red"))
                print(f"{knowledge_type} response: ", response)
            except Exception as e:
                print(colored(f"Error: {e}, retrying...", "red"))
                print(f"{knowledge_type} response: ", response)
        if response is None:
            raise Exception("Failed to get response")
        return img_data, img_thought


    def rollout_one_step(self, recent_obs, policy_code, reward_code, transition_code, other_information):
        text_obs = self.recent_obs_to_text(recent_obs)
        # fake_code = {'analyze' = [''], 'code' = ['']}
        if self.skip_knowledge:
            policy_code = 'EMPTY'
            reward_code = 'EMPTY'
            transition_code = 'EMPTY'
        action_prompt = self.llm_policy_prompt(code_string=policy_code, obs=text_obs)
        action, action_thought = self._get_response(action_prompt, KnowledgeType.Policy)
        next_obs_prompt = self.llm_transition_prompt(code_string=transition_code, obs=text_obs, action=action)
        next_text_obs = None
        is_end = False
        for _ in range(MAX_ATTAMPTS):
            next_obs, transition_thought = self._get_response(next_obs_prompt, KnowledgeType.Dynamics)
            next_obv, next_text_obs, fixed_img_obs, is_end = self.post_process_img_next_obs(copy.deepcopy(next_obs), recent_obs, action, other_information)
            if next_text_obs is not None:
                break
        if next_text_obs is None: raise Exception("Failed to change format")
        dense_reward_prompt = self.llm_reward_prompt(reward_code, text_obs, action, next_text_obs, is_end)
        dense_rewards, reward_thought = self._get_response(dense_reward_prompt, KnowledgeType.Reward)
        print("\n===")
        print("action: ", action, "action_thought: ", action_thought)
        print("---")
        print("next_obs: ", next_obs,  "\n transition_thought: ", transition_thought, "\n post_process_obs:", fixed_img_obs)
        print("---")
        print("dense_rewards: ", dense_rewards, "reward_thought: ", reward_thought)
        print("===\n")
        return next_obv, action, dense_rewards, {'transition_thought': transition_thought, 'action_thought': action_thought, 'reward_thought': reward_thought}

    def llm_policy_prompt(self, code_string, obs):
        knowledge_type = KnowledgeType.Policy
        raw_policy_prompt = """
            I want you to act {task_info}.

            I will give you an observation which you are facing in a simulator, your task is to response a correct results serving as a {knowledge_type} function, which is used for {knowledge_function}.
            For example, {knwoledge_function_example}.

            Formally, the format of {knowledge_type} function is {knowledge_format}.

            To help you complete the task, I will provide you
            (1) pieces of relevant knowledge from the tutorial, written in Python-style pseudocode. You should output your result based on the logic of the pseudocode.
            (2) the observation, action of the simulator;
            (3) the current observation which you are facing in the simulator.


            Python-style relevant knowledge which is retrieved from the tutorial books:

            {code_string}

            The observation space of the simulator:
            {observation_space_desc}

            The action space of the simulator:
            {action_space_desc}

            The current observation you are facing in the simulator: {obs}

            Hints for your inference:

            {hint}

            Requirements:

            About the action choosing:
            - Please choose the action that best fits the code logic.

            About the format:
                - you should answer in pure JSON format with the key: 'action': a int number from 0 to 18, 'thought': why you choose this action. without any other information or code. For example, you should not add the ```json``` tag in the answer.

            Response format (you should resposne in the following order):

            {format_example}

        """.format(task_info=self.sim_info.TASK_DESC,
                   knowledge_type=knowledge_type, knowledge_function=self.sim_info.element_type_prompt_dict[knowledge_type], knowledge_format=self.sim_info.element_type_format_dict[knowledge_type],
                        knwoledge_function_example=self.sim_info.element_type_example_dict[knowledge_type], code_string=code_str_format(code_string),
                        hint=self.sim_info.element_type_infer_hint[knowledge_type], format_example=self.sim_info.element_type_infer_format[knowledge_type],
                        observation_space_desc=self.sim_info.OBSERVATION_SPACE_DESC, action_space_desc=self.sim_info.ACTION_SPACE_DESC, obs=obs)
        return raw_policy_prompt


    def llm_transition_prompt(self, code_string, obs, action):
        knowledge_type = KnowledgeType.Dynamics
        raw_prompt = """
            I want you to act {task_info}.

            I will give you an observation which you are facing in a the target simulator, your task is to response a correct results serving as a {dynamics} function code from the tutorial, written in Python style.

            Formally, the format of {dynamics} function is {dynamics_format}. The {dynamics} function describes the mechanism for updating the position of the ball and players in the target simulator.

            To help you complete the task, I will provide you
            (1) pieces of relevant knowledge from the tutorial, written in Python-style pseudocode. You should output your result based on the logic of the pseudocode.
            (2) the observation, action of the simulator;
            (3) the current observation which you are facing in the simulator and the action which made by the active player.

            Python-style relevant knowledge which is retrieved from the tutorial books:

            {dynamics} function:

            {dynamics_code_string}

            The observation space of the simulator:
            {observation_space_desc}

            The action space of the simulator:
            {action_space_desc}

            The current observation and action you are facing in the simulator:

            observation: {obs}

            action: {action}


            Hints for your inference:

            {hint}


            ### Response example: (you should resposne with the JSON format in the following order, without any other information, explanation or comments):

            {format_example}

        """.format(task_info=self.sim_info.TASK_DESC, dynamics=KnowledgeType.Dynamics, dynamics_function=self.sim_info.element_type_prompt_dict[KnowledgeType.Dynamics],
                dynamics_format=self.sim_info.element_type_format_dict[KnowledgeType.Dynamics], dynamics_code_string=code_str_format(code_string),
                        hint=self.sim_info.element_type_infer_hint[knowledge_type], format_example=self.sim_info.element_type_infer_format[knowledge_type],
                observation_space_desc=self.sim_info.OBSERVATION_SPACE_DESC, action_space_desc=self.sim_info.ACTION_SPACE_DESC, obs=obs, action=action)
        return raw_prompt

    def llm_reward_prompt(self, code_string, obs, action, next_obs, is_end=False):
        if is_end:
            next_obs = str(next_obs) + "(reached the end of the game)"
        knowledge_type = KnowledgeType.Reward
        raw_prompt = """
            I want you to act {task_info}.

            I will give you an observation which you are facing in a simulator, your task is to response a correct results serving as a  {reward} function code from the tutorial, written in Python style.

            Formally, the format of {reward} function is {reward_format}. The {reward} function describes the mechanism for calculating the reward of the  game simulator.

            To help you complete the task, I will provide you
            (1) pieces of relevant knowledge from the tutorial, written in Python-style pseudocode. You should output your result based on the logic of the pseudocode.
            (2) the observation, action of the  simulator;
            (3) the current observation which you are facing in the simulator and the action which made by the active player.

            Python-style relevant knowledge which is retrieved from the tutorial books:


            {reward} function:

                    {reward_code_string}

            The observation space of the  simulator:
            {observation_space_desc}

            The action space of the simulator:
            {action_space_desc}

            The current observation you are facing, the action taken by you, and the next observation (after opponent player acts) returned by the simulator :

            observation: {obs}

            action: {action}

            next_observation: {next_obs}


            Hints for your inference:

            {hint}


            ### Response example:  (you should resposne with the JSON format in the following order, without any other information, explanation or comments):

            {format_example}

        """.format(task_info=self.sim_info.TASK_DESC, reward=KnowledgeType.Reward, reward_function=self.sim_info.element_type_prompt_dict[KnowledgeType.Reward],
                reward_format=self.sim_info.element_type_format_dict[KnowledgeType.Reward], reward_code_string=code_str_format(code_string),
                hint=self.sim_info.element_type_infer_hint[knowledge_type], format_example=self.sim_info.element_type_infer_format[knowledge_type],
                observation_space_desc=self.sim_info.OBSERVATION_SPACE_DESC, action_space_desc=self.sim_info.ACTION_SPACE_DESC, obs=obs, action=action, next_obs=next_obs)
        return raw_prompt


class FootBallRolloutMDP(RolloutMDP):
    def __init__(self, model, seed, skip_knowledge, simulator_info):
        super().__init__(model, seed, skip_knowledge, simulator_info)

    def img_obs_to_text_fn(self, img_obs, stacked):
        return img_obs_to_text(img_obs, stacked)

    def recent_obs_to_text(self, recent_obs):
        return img_obs_to_text(imaginary_data_observation_stacked_info(recent_obs), stacked=True)

    def get_current_obs(self, recent_obs):
        return recent_obs[-1]

    def post_process_img_next_obs(self, img_next_obs, recent_obs, action, other_information):
        next_obs = img_next_obs
        next_text_obs = None
        current_obs = self.get_current_obs(recent_obs)
        try:
            if next_obs is None:
                next_obv = current_obs
                fixed_img_obs = None
            else:
                next_obv, fixed_img_obs = change_format(current_obs, next_obs, action, other_information)
            next_text_obs = self.img_obs_to_text_fn(next_obv, stacked=False)
        except Exception as e:
            print("Error: ", e)
            print("next_obs: ", next_obs)
            print("next_text_obs: ", next_text_obs)
        return next_obv, next_text_obs, fixed_img_obs, False


class TicTacToeRolloutMDP(RolloutMDP):
    def __init__(self, env, oppo_strategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.oppo_strategy = oppo_strategy
        from envs.tictactoe import TicTacToeEnv
        from tictactoe.ttt_gym_player import MinimaxPlayer
        self.oppo_player_pi = MinimaxPlayer('O')
        assert isinstance(env, TicTacToeEnv)

    def img_obs_to_text_fn(self, img_obs, stacked):
        return self.env.vector_state_to_text(img_obs)

    def recent_obs_to_text(self, recent_obs):
        return self.img_obs_to_text_fn(recent_obs[-1], False)

    def get_current_obs(self, recent_obs):
        return recent_obs[-1]

    def post_process_img_next_obs(self, img_next_obs, recent_obs, action, other_information):
        next_obs = img_next_obs
        next_text_obs = None
        next_obv = None
        fixed_img_obs = None
        current_obs = self.get_current_obs(recent_obs)
        is_end = False
        try:
            if next_obs is None:
                next_obv = current_obs[:-1]
                fixed_img_obs = None
            else:
                next_obs = img_next_obs['board']
                if not self.env.check_win(next_obs) and current_obs[action] == 0 and len(self.env.available_moves(next_obs)) > 0:
                    if self.oppo_strategy == 'random':
                        oppo_action = np.random.choice(self.env.available_moves(next_obs))
                    elif self.oppo_strategy == 'minimax':
                        if np.random.rand() < 0.1:
                            oppo_action = np.random.choice(self.env.available_moves(next_obs))
                        else:
                            oppo_action = self.oppo_player_pi.get_action(self.env, next_obs)
                    # TODO: here we simplify the task. we assume the opponent is random and compute the next state directly.
                    next_obs[oppo_action] = other_information['oppo_player_symbol']
                else:
                    is_end = True
                next_obv = fixed_img_obs = next_obs
                # next_obv, fixed_img_obs = change_format(current_obs, next_obs, action, other_information)
            next_text_obs = self.img_obs_to_text_fn(next_obv, stacked=False)
        except Exception as e:
            print("Error: ", e)
            print("next_obs: ", next_obs)
            print("next_text_obs: ", next_text_obs)

        return next_obv, next_text_obs, fixed_img_obs, is_end

# ===== funcs for football =====
def get_direction_from_text(current_zone, current_direction, next_zone):
    if current_zone == next_zone:
        return current_direction
    else:
        delta_x = next_zone[0] - current_zone[0]
        delta_y = next_zone[1] - current_zone[1]
        return vector_to_direction(delta_x, delta_y)


def change_format(current_obs, next_obs, action, other_player_action=None):
    """
    TODO: verify this conversion logic before changing football observation formats
    """
    res = {}

    game_modes = ["Noramal", "KickOff", "GoalKick", "FreeKick", "Corner", "hrowIn", "Penalty"]  # Replace with actual game modes
    role_list = ["Goalkeeper", "Forward", "Forward", "Defender", "Defender", "Defender", "Defender", "Midfielder", "Midfielder", "Midfielders", "Forward"]


    #sticky_actions

    sticky_action_list = [0] * len(sticky_list)
    stick_actions_for_dir_map = []
    for a in current_obs["sticky_actions"]:
        sticky_action_list[sticky_list.index(a)] = 1


    # Sticky action
    if action in range(1,9):
        # sticky_action_list[0:8] should be 0
        sticky_action_list[0:8] = [0] * 8
        sticky_action_list[action-1] = 1
    elif action == 13:
        sticky_action_list[8] = 1
        sticky_action_list[9] = 0
    elif action == 17:
        sticky_action_list[9] = 1
        sticky_action_list[8] = 0
    # if current_obs["active_player"] == next_obs["active_left_player"]:

    # else:
    #     sticky_action_list = [0 for _ in range(10)]

    sticky_action_list = np.array(sticky_action_list)
    if len(sticky_action_list) > 0 and np.any(sticky_action_list != 0):
        indices = np.where(sticky_action_list == 1)[0]
        selected_actions = [sticky_list[idx] for idx in indices]
        res["sticky_actions"] = selected_actions
        stick_actions_for_dir_map = [sticky_list_map_dir[idx] for idx in indices]
    else:
        res["sticky_actions"] = []
        stick_actions_for_dir_map = []

    # Game mode
    res["game_mode"] = "Normal"


    # Score
    res["score"] = next_obs["score"]

    # Step
    res["step"] = next_obs["step"]

    # Time
    time =  next_obs["step"] * 1.8 # 3000 timpstep = 90 mins
    min = int(time // 60)
    sec = int(time % 60)
    time_text = f"The current time is {min} minutes {sec} seconds. "
    res["time"] = time_text



    # Ball direction
    current_ball_zone = current_obs["ball_zone"]
    current_ball_direction = current_obs["ball_direction"]
    next_obs["ball_direction"] = get_direction_from_text(current_ball_zone, current_ball_direction, next_obs["ball_zone"])


    bc_active_player_list = []
    ball_change_flag_list = []
    # Player
    for i in range(22):
        player = {}
        if i < 11:
            player["team"] = "Left"
        else:
            player["team"] = "Right"
        player["role"] = role_list[i%11]
        if i == current_obs["active_player"]:
            player["zone"] = next_obs["left_active_player_zone"]
            current_player_zone = current_obs[f"player_{i}"]["zone"]
            current_player_directioin = current_obs[f"player_{i}"]["direction"]
            player["direction"] = get_direction_from_text(current_player_zone, current_player_directioin, player["zone"])
            for idx, direction in enumerate(DIRECTIONS):
                if direction in stick_actions_for_dir_map:
                    player["direction"] = direction
            ball_change_flag_list.append(False)
        else:
            zone_diff_x  = np.round(other_player_action[i][0][0])
            zone_diff_y = np.round(other_player_action[i][0][1])
            own_the_ball_state_diff = np.clip(np.round(other_player_action[i][0][2]), -1, 1)
            ball_direction_diff = np.round(other_player_action[i][0][3])
            ball_change_flag = ((ball_direction_diff != 0) & (own_the_ball_state_diff != 0))

            # for players
            player["zone"] = [current_obs[f"player_{i}"]["zone"][0] + zone_diff_x, current_obs[f"player_{i}"]["zone"][1] + zone_diff_y]
            current_player_zone = current_obs[f"player_{i}"]["zone"]
            current_player_directioin = current_obs[f"player_{i}"]["direction"]
            player["direction"] = get_direction_from_text(current_player_zone, current_player_directioin, player["zone"])

            # active player
            own_the_ball_state = current_obs["ball_ownership_player"]  == i
            next_own_the_ball_state = own_the_ball_state + own_the_ball_state_diff

            if next_own_the_ball_state == 1:
                bc_active_player_list.append(i)

            # ball direction
            ball_change_flag_list.append(ball_change_flag)

        res[f"player_{i}"] = player

    if len(bc_active_player_list) == 0:
        # no one control the ball or the human player is playing
        # if current_obs["ball_ownership_player"] != current_obs['active_player']:
        #     next_obs["ball_ownership_player"] = -1
        #     next_obs["ball_ownership"] = 0
        # else:
        if action in (9,10,11,12): # Force no ball owner after a pass action.
            next_obs["ball_ownership_player"] = -1
            next_obs["ball_ownership"] = 0
        if next_obs["ball_ownership"] == 0:
            next_obs["ball_ownership_player"] = -1
        elif next_obs["ball_ownership"] == 1:
            next_obs["ball_ownership_player"] = -1 # Override using the nearest-distance logic for the final active player.
        else:
            # On the opponent side, choose the opponent player nearest to the ball.
            active_player = 11
            active_player_zone_diff = np.inf
            for i in range(11, 22):
                current_zone = res[f"player_{i}"]["zone"]
                current_zone_diff = np.sum(np.abs(np.array(current_zone) - np.array(next_obs["ball_zone"])))
                if current_zone_diff < active_player_zone_diff:
                    active_player = i
                    active_player_zone_diff = current_zone_diff
            if current_zone_diff > 1: # If opponent possession is detected, require the nearest player to be close to the ball; otherwise reset to no possession.
                print("[WARN] ball is not near the closest player (right team), reset ball ownership to no one", current_zone_diff, "zone", current_zone, "ball_zone", next_obs["ball_zone"])
                next_obs["ball_ownership_player"] = -1
                next_obs["ball_ownership"] = 0
            else:
                next_obs["ball_ownership_player"] = active_player
    else:
        if len(bc_active_player_list) > 1:
            next_obs["ball_ownership_player"] = np.random.choice(bc_active_player_list)
        else:
            next_obs["ball_ownership_player"] = bc_active_player_list[0]
        next_obs['ball_zone'] = res[f"player_{next_obs['ball_ownership_player']}"]['zone']

        if next_obs["ball_ownership_player"] < 11:
            next_obs["ball_ownership"] = 1
            next_obs["active_left_player"] = next_obs["ball_ownership_player"]
        else:
            next_obs["ball_ownership"] = 2
        if ball_change_flag_list[next_obs["ball_ownership_player"]]:
            ball_direction_diff = np.round(other_player_action[next_obs["ball_ownership_player"]][0][3])
            ball_current_direction = current_obs["ball_direction"]
            next_obs["ball_direction"] = DIRECTIONS[int(DIRECTIONS.index(ball_current_direction) + ball_direction_diff) % len(DIRECTIONS)]
        else:
            next_obs["ball_direction"] = res[f"player_{next_obs['ball_ownership_player']}"]["direction"]

    # for i in range(22):
    #     if i == next_obs["ball_ownership_player"]:
    #         if ball_change_flag_list[i]:
    #             ball_direction_diff = np.round(other_player_action[i][0][3])
    #             ball_current_direction = current_obs["ball_direction"]
    #             next_obs["ball_direction"] = DIRECTIONS[int(DIRECTIONS.index(ball_current_direction) + ball_direction_diff) % len(DIRECTIONS)]
    #         else:
    #             next_obs["ball_direction"] = res[f"player_{i}"]["direction"]

    # Active player
    clost_player = 0
    clost_player_zone_diff = np.sum(np.abs(np.array(res["player_0"]["zone"]) - np.array(next_obs["ball_zone"])))
    for i in range(11):
        if i == next_obs["ball_ownership_player"]:
            clost_player = i
            break
        else:
            # If the BC policy does not identify a new ball owner, ownership is decided by the language model.
            current_zone = res[f"player_{i}"]["zone"]
            current_zone_diff = np.sum(np.abs(np.array(current_zone) - np.array(next_obs["ball_zone"])))
            if current_zone_diff < clost_player_zone_diff:
                clost_player = i
                clost_player_zone_diff = current_zone_diff
    res["active_player_role"] = role_list[clost_player]
    res["active_player"] = clost_player
    if next_obs["ball_ownership"] == 1:
        if clost_player_zone_diff > 1:
            print("[WARN] ball is not near the closest player (left team), reset active player to no one", clost_player_zone_diff, "zone", current_zone, "ball_zone", next_obs["ball_zone"])
            next_obs["ball_ownership"] = 0
            next_obs["ball_ownership_player"] = -1
        else:
            res[f"player_{clost_player}"]["zone"] = next_obs["ball_zone"]
            next_obs["ball_ownership_player"] = clost_player
    if res["active_player"] != current_obs["active_player"]:
        print("[WARN] change the active player", current_obs["active_player"], "->", res["active_player"])
        res["sticky_actions"] = []

    # Active player role
    # if next_obs["active_left_player"] > 10:
    #     print("[WARNING] active player is not in the left team, replace with the clost player")

    # else:
    #     res["active_player_role"] = role_list[next_obs["active_left_player"]]

    # Ball ownership
    res["ball_ownership"] = next_obs["ball_ownership"]

    # Ball ownership player
    res["ball_ownership_player"] = next_obs["ball_ownership_player"]

    # Ball zone
    res["ball_zone"] = next_obs["ball_zone"]

    # Ball direction
    res["ball_direction"] = next_obs["ball_direction"]

    return res, next_obs
