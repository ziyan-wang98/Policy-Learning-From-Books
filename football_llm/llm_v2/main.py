import copy
import os
from pathlib import Path
import sys

sys.path.append('football_llm')
import tqdm
import llm.config.baseline_agent_parser as baseline_agent_parser
from llm.utils.obs2text import observation_to_text_human, observation_to_text_raw, long_obs_list_to_text
from llm_v2.algo.baseline_v1_single_agent import BaselineV1SingleAgent # , Baseline_v1_multi_agent
from llm_v2.algo.baseline_v1_multi_agent import BaselineV1MultiAgent
import gfootball.env as football_env
import json
from llm.utils.obs2text import *
from book_scripts.utils import *
from gfootball.env.wrappers import Simple115StateWrapper_ball_owned_player

from algo.rule_base_2.gfootball import agent_dict as rule_base_2_agent_dict

if os.environ.get('PLFB_CUDA_VISIBLE_DEVICES'):
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['PLFB_CUDA_VISIBLE_DEVICES']
# Set your API keys here
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# os.environ["REPLICATE_API_TOKEN"] = "YOUR_REPLICATE_API_TOKEN"
assert os.environ["OPENAI_API_KEY"] != None, "Please set your OPENAI_API_KEY in the environment variable"
assert os.environ["REPLICATE_API_TOKEN"] != None, "Please set your REPLICATE_API_TOKEN in the environment variable"

sys.stdout = open("output.txt", "w")

def setup_directories(args):
    """
    Creates necessary directories for saving outputs and returns their paths.

    :param args: ArgumentParser object containing necessary attributes.
    :return: A tuple containing the paths to the video and JSON directories.
    """
    # Create the base result directory
    result_dir = './result'
    os.makedirs(result_dir, exist_ok=True)

    # Create a unique subdirectory
    index = 0
    unique_dir_name = f"{args.algo}_{args.environment}_{index}"
    unique_dir_path = os.path.join(result_dir, unique_dir_name)
    while os.path.exists(unique_dir_path):
        index += 1
        unique_dir_name = f"{args.algo}_{args.environment}_{index}"
        unique_dir_path = os.path.join(result_dir, unique_dir_name)

    os.makedirs(unique_dir_path, exist_ok=True)

    # Create subdirectories for videos and JSON files
    video_dir = os.path.join(unique_dir_path, 'video')
    json_dir = os.path.join(unique_dir_path, 'json')
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    return video_dir, json_dir

raw_policy_prompt= """
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
        We use zone (x, y) to express the position of the player. "x" is the discretized coordinate parallelized with the line from the left team's penalty area to the right team's penalty area, ranging from 1 to 20, and y is the discretized coordinate parallelized with the line from the lower corner to the upper corner flag, ranging from 1 to 12.
        The left team's penalty area is zone (1, 4)-(1, 8)-(3, 8)-(3,4), and the right team's penalty area is zone (20, 4)-(20, 8)-(18, 8)-(18, 4).
        This means that the center circle position of the field is zone (10, 6), where the game start.
        The lower left corner position of the Left team is (1, 1), and the upper right corner position of the Right team is (20, 12).
        Venues never interchange or change. Following the position information is direction, meaning the direction the player is currently facing and the direction of future actions.

        ---------------------\n

        {query_str}.

        ---------------------\n

        """
program_code_rerank_prompt = """
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
            12, # action_shot, players will try to shoot. When near to the oppenent penalty area, such as zone(x, y), x>18, 4<y<8, try to shoot.\n
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
        - You should not implement any function that with placeholder.

        About the format:
            - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.


        Response example (you should resposne in the following order):
        {
            "analyze to current state": "the analyze to current state",
            "code": "the code you rewrite based on the analyze to current state",
        }
"""
case_prompt = """


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
        - You should make the optimal decision based on the analyze current state of the football match. output the anaysis to "analyze to current state" \n

        About the code simulation: \n
        - Note that parts of the presudo code have not been implemented and just use comments to describe the function of the code. You should implement the code to make it work. \n
        - Then simulation code to get the result based on the current state of the football match and your common sense. the result should be put in the key "simulation result of prior-knowledge code" \n
        - You can make the optimal decision based on the results of the prior knowledge code. put your results to the value in key 'action' \n



        About the format:
            - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.

        Response example (you should resposne in the following order):
        {
            "analyze to current state": "the analyze to current state",
            "simulation result of prior-knowledge code": "the simulation result of the prior-knowledge code. You should simulate the deterministic results bsed on current state.",
            "thought": "regarding the code as the principle to conduct your action, beside with your common sense, think about the optimal decision-making you should make.",
            "thought-to-action": "based on your thought, tell me the optimal action you would like to select in the action set. You should give me the deterministic action, not the probability.",
            "action": 0,

        }


  """

if __name__ == "__main__":
    args = baseline_agent_parser.parse_args()
    file_name = 'agg-best.jsonl'
    policy_root = Path(os.environ.get("PLFB_POLICY_SOURCE_DIR", str(Path(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts")) / "book_derived" / "v4-gpt-3.5-turbo-1106-level-strict")))
    data_path = policy_root / "Policy" / "best" / file_name
    doc_code = load_jsonl_list(str(data_path))
    doc_list = []
    out_data_path = Path(os.environ.get("PLFB_POLICY_CONTEXT_PATH", str(policy_root / "best" / "policy" / "agg_postprocess.jsonl")))
    out_data_path.parent.mkdir(parents=True, exist_ok=True)
    for item in doc_code:
        for k, v in item.items():
            if 'code' in k:
                doc = f"purpose of the functions: {k} \n\n" + "\n".join(v)
                doc_list.append(doc)
                pprint.pprint(doc)
    selected_idx = [0, 1, 4, 5, 6, 7, 9, 10, 11, 12]
    doc_emb_list = None
    # doc_emb_list = [get_embedding(doc) for doc in tqdm.tqdm(doc_list)]
    threshold = 0.75
    filter_num = 5
    def find_match_doc(current_state, threshold, filter_num, do_print=False):
        cur_state_emb = get_embedding(current_state)
        sim_coef = get_cosine_similarity(doc_emb_list, cur_state_emb)

        match_idx = np.argsort(sim_coef)[-filter_num:]
        match_idx_threshold = match_idx[np.where(sim_coef[match_idx] > threshold)]
        matched_doc = []
        for idx in match_idx_threshold:
            doc = doc_list[idx]
            if do_print:
                pprint.pprint(str(doc))
            matched_doc.append(doc)
        print(f"print matched doc ({sim_coef[match_idx_threshold]}): \n\n")
        return matched_doc

    env = football_env.create_environment(env_name=args.environment, representation='raw', \
                                stacked=False, logdir=args.logdir, write_goal_dumps=args.write_goal_dumps, \
                                write_full_episode_dumps=args.write_full_episode_dumps, render=args.render,\
                                number_of_left_players_agent_controls=args.num_players)

    wrapper = Simple115StateWrapper_ball_owned_player(env)

    model_name_stage_1 = os.environ.get('PLFB_OPENAI_CODE_MODEL', os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini'))
    model_name_stage_2 = os.environ.get('PLFB_OPENAI_ACTION_MODEL', os.environ.get('PLFB_OPENAI_CHAT_MODEL', 'gpt-4o-mini'))

    # video_dir, json_dir = setup_directories(args)

    obs = env.reset()
    json_outputs_list = []
    steps = 0
    repeat = 2
    last_time_acs = 0
    repeat_code_gen = 10
    code = ""
    last_time_gen = 0
    while True:
        rule_action_dict = rule_base_2_agent_dict(obs[0].copy(), ret_dict=True)
        rule_action = rule_action_dict['action'].value
        if (obs[0]['game_mode'] == 0 and rule_action_dict['group_name'] == 'own_goal') or rule_action_dict['group_name'] == 'defence_memory_patterns':
            action = rule_action
            print("group_name: ", rule_action_dict['group_name'], "pattern_name: ", rule_action_dict['pattern'], "action: ", action, obs[0]['game_mode'])
        else:
            if (steps - last_time_gen) > repeat_code_gen:
                text_obs = observation_to_text_human(wrapper.observation(obs)[0], obs[0], steps, block_mode="240")
                matched_docs = list(np.array(doc_list)[selected_idx]) # find_match_doc(text_obs, threshold, filter_num)
                global_prompt = raw_policy_prompt.format(context_str="\n\n".join(matched_docs), query_str=text_obs)
                code_res = query(global_prompt, program_code_rerank_prompt, model_name_stage_1, print_user_prompt=False, req_json=True)
                if code_res is not None and 'code' in code_res.keys():
                    code = code_res['code']
                    last_time_gen = steps

            if (steps - last_time_acs) > repeat:
                # observation, raw_obs, step, block_mode=None
                text_obs = observation_to_text_human(wrapper.observation(obs)[0], obs[0], steps, block_mode="240")
                # stage2: make the decision
                global_prompt = raw_policy_prompt.format(context_str=code, query_str=text_obs)
                acs_res = query(global_prompt, case_prompt, model_name_stage_2, print_user_prompt=False, req_json=True)
                if acs_res is not None and 'action' in acs_res.keys() and type(acs_res['action']) == int:
                    action = acs_res['action']
                    if action in [9, 10, 11, 12] and 'gpt-4' not in model_name_stage_2:
                        print(" ----- check with gpt4")
                        acs_res2 = query(global_prompt, case_prompt, os.environ.get('PLFB_OPENAI_CODE_MODEL', model_name_stage_2), print_user_prompt=False, req_json=False, print_global_prompt=False)
                        if acs_res2 is not None and 'action' in acs_res.keys() and type(acs_res2['action']) == int:
                            action = acs_res2['action']
            if action == 12:
                print("reach shot action")
            if False and action != rule_action:
                old_acs = action
                if rule_action == 12 and rule_action_dict["group_name"] == "offence_memory_patterns": # shot
                    action = rule_action
                elif rule_action == 16: # sliding
                    action = rule_action
                elif rule_action == 13: # sprint
                    action = rule_action
                elif rule_action == 17: # dribble
                    action = rule_action
                elif action == 0: # idle
                    action = rule_action

                if action == rule_action:
                    print("group_name: ", rule_action_dict['group_name'], "pattern_name: ", rule_action_dict['pattern'])
                    print("replace action: ", old_acs, "->", rule_action)
        obs, rew, done, info = env.step(action)
        steps += 1
        sys.stdout.flush()
        print("Step {} Reward: {}. done {}".format(steps, rew, done))
        if steps==args.num_timesteps or done:
            break
    # print("Steps: %d Reward: %.2f" % (steps, rew))
    env.close()


    # 1. analyze state based on principle in the presudo code -> get the extracted state
    # 2. output the response based on the extracted state.



