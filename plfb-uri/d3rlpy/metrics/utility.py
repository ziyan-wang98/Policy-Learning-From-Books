import numpy as np
import tqdm
from ..interface import QLearningAlgoProtocol, StatefulTransformerAlgoProtocol
from ..types import GymEnv
import json
from llm.utils.llama_index_compat import BaseReader, Document, OpenAIEmbedding, SimpleDirectoryReader, index_from_documents, make_service_context
from llm.algo.rule_base_2.gfootball import agent_dict as rule_base_2_agent_dict
from llm.utils.obs2text import observation_to_text_human, observation_to_text_raw, get_zons_240, format_code, imaginary_data_observation
import time
from llm.llm_baseline import llm_agent, llm_rag, RT2
from llm.utils.obs2text import img_obs_to_text

from gfootball.env.wrappers import Simple115StateWrapper_ball_owned_player
from llm.utils.obs2text import imaginary_data_observation, imaginary_data_to_vector, imaginary_data_observation_v2
__all__ = [
    "evaluate_qlearning_with_environment",
    "evaluate_transformer_with_environment",
    "evaluate_qlearning_with_img_obs_environment",
]

HORIZON = 1500
import copy

def rule_based_postprocess(wrapper_obs, obs, action, version='all_replaced'):

    if version == 'no':
        return action, False
    replace=False
    rule_policy_dic = rule_base_2_agent_dict(obs.copy(), ret_dict=True)
    rule_action = rule_policy_dic['action'].value
    team_owner = np.where(wrapper_obs[94:97] == 1)[0][0] # 0: no one, 1: left team, 2: right team
    team_owner = np.where(wrapper_obs[94:97] == 1)[0][0] # 0: no one, 1: left team, 2: right team
    ball_zone = get_zons_240(wrapper_obs[88], wrapper_obs[89])
    active_player_index = np.where(wrapper_obs[97:108] == 1)[0][0]
    active_player_zone = get_zons_240(wrapper_obs[2*active_player_index], wrapper_obs[2*active_player_index+1])
    if team_owner != 1 and active_player_zone == ball_zone:
        action = rule_action
        replace = True
    # enmergency actions:
    # 9 = action_long_pass, player will try to pass the ball to a teammate.\n
    # 10 = action_high_pass, player will try to pass the ball to a teammate.\n
    # 11 = action_short_pass, player will try to pass the ball to a teammate.\n
    # 12 = action_shot, players will try to shoot, the only way to score. When the opportunity is right, more attempts to shoot will win.\n
    # 13 = action_sprint, the player will sprint, gaining high speed, use it whenever possible, providing a great advantage when attacking and returning to defense .\n
    # 14 = action_release_direction, player will stop moving in the current direction.\n
    # 15 = action_release_sprint, player will stop sprinting.\n
    # 16 = action_sliding, the player will try to slide the tackle. If your position is on the opponent's path with the ball, you can intercept it. However, if the sliding tackle fails, you will be separated by a large distance, allowing the opponent to ignore the defense.. \n
    # 17 = action_dribble, players will try to dribble. When they have the ball, dribbling will greatly improve the success rate of dribbling, especially in multi-person double-teams and difficult-to-handle situations..\n
    enmergency_action_set = [9, 10, 11, 12, 13, 14, 15, 16, 17]
    # TODO: replace if game mode is not normal
    if team_owner == 1 and action != rule_action:
        for rtz in obs['right_team']:
            zone = get_zons_240(rtz[0], rtz[1])
            if version == 'all_replaced':
                if active_player_zone == zone: # and rule_action in [9, 10, 11, 12, 13, 14, 15, 16, 17]:
                    action = rule_action
                    replace = True
            elif version == 'enmergency_replaced':
                if active_player_zone == zone and rule_action in enmergency_action_set:
                    action = rule_action
                    replace = True
            elif version == 'parts_keeped':
                if active_player_zone == zone:
                    # Trust extra opportunities or risks discovered by the policy.
                    if action not in [9, 10, 11, 12, 13]:
                        action = rule_action
                        replace = True
                    # Still replace if the policy also identifies an emergency.
                    elif rule_action in enmergency_action_set:
                        action = rule_action
                        replace = True
            else:
                raise ValueError(f"Unsupported version: {version}")
    return action, replace




def evaluate_ttt_environment(
    algo: QLearningAlgoProtocol,
    env: GymEnv,
    oppo_players: list,
    n_trials: int = 10,
    epsilon: float = 0.0,
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.

    Returns:
        average score.
    """
    import time 
    res_dict = {'rew': [], 'win': [], 'draw': [], 'lose': []}
    for oppo_player in oppo_players:
        if oppo_player == 'minimax':
            from tictactoe.ttt_gym_player import MinimaxPlayer
            oppo_player_pi = MinimaxPlayer('O')
        episode_rewards = []
        for n in tqdm.tqdm(range(n_trials), desc=f"evaluating {env.env_name}"):
            step = 0
            observation = env.reset()
            episode_reward = 0.0
            while True:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if isinstance(observation, np.ndarray):
                        observation = np.expand_dims(observation, axis=0)
                    elif isinstance(observation, (tuple, list)):
                        observation = [
                            np.expand_dims(o, axis=0) for o in observation
                        ]
                    else:
                        raise ValueError(
                            f"Unsupported observation type: {type(observation)}"
                        )
                    action = algo.predict(observation)[0]
                    # algo.predict_value(observation, np.array(8))
                observation, reward, done, info = env.step(action)
                episode_reward += float(reward)
                step += 1
                if done or step > HORIZON:
                    break
                if oppo_player == 'random':
                    free_space = np.where(observation==0)[0]
                    # random pick a position
                    action = np.random.choice(free_space)
                elif oppo_player == 'minimax':
                    action = oppo_player_pi.get_action(env, copy.deepcopy(observation))
                else:
                    raise ValueError(f"Unsupported oppo_player: {oppo_player}")
                observation, reward, done, info = env.step(action)
                if done:
                    episode_reward += float(reward)
                    break
            episode_rewards.append(episode_reward)
            if episode_reward > 1:
                pass
        cur_res_dict = {
                f"rew_{oppo_player}": float(np.mean(episode_rewards)), 
                f"win_{oppo_player}": float(np.mean(np.array(episode_rewards)>0)), 
                f"draw_{oppo_player}": float(np.mean(np.array(episode_rewards)==0)), 
                f"lose_{oppo_player}": float(np.mean(np.array(episode_rewards)<0))}
       
        res_dict.update(cur_res_dict)
        res_dict["rew"].append(float(np.mean(episode_rewards)))
        res_dict["win"].append(float(np.mean(np.array(episode_rewards)>0)))
        res_dict["draw"].append(float(np.mean(np.array(episode_rewards)==0)))
        res_dict["lose"].append(float(np.mean(np.array(episode_rewards)<0)))
    res_dict["rew"] = float(np.mean(res_dict["rew"]))
    res_dict["win"] = float(np.mean(res_dict["win"]))
    res_dict["draw"] = float(np.mean(res_dict["draw"]))
    res_dict["lose"] = float(np.mean(res_dict["lose"]))
    return res_dict



def evaluate_qlearning_with_img_obs_environment(
    algo: QLearningAlgoProtocol,
    env_list: GymEnv,
    n_trials: int = 10,
    epsilon: float = 0.0,
    update_stack_obs=None,
    stack_obs_len=None,
    acs_replace_strategy='parts_keeped',
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.

    Returns:
        average score.
    """
    import time 
    res_dict = {'rew': [], 'win': [], 'draw': [], 'lose': [], 'replaced_rate': []}
    for env in env_list:
        episode_rewards = []
        replaced_rates = []
        wrapper_func = Simple115StateWrapper_ball_owned_player
        wrapper = wrapper_func(env)
        stack_obs = np.zeros(stack_obs_len)
        for n in tqdm.tqdm(range(n_trials), desc=f"evaluating {env.env_name}"):
            step = 0
            raw_obs = env.reset()
            episode_reward = 0.0
            replace_times = 0
            
            while True:
                # t1 = time.time()
                wrap_obs = wrapper.observation(copy.deepcopy(raw_obs))
                observation = imaginary_data_observation(copy.deepcopy(wrap_obs[0]), copy.deepcopy(raw_obs[0]), 
                                                         step, ret_type='vector', TODO_missing=True)
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if isinstance(observation, np.ndarray):
                        observation = np.expand_dims(observation, axis=0)
                    elif isinstance(observation, (tuple, list)):
                        observation = [
                            np.expand_dims(o, axis=0) for o in observation
                        ]
                    else:
                        raise ValueError(
                            f"Unsupported observation type: {type(observation)}"
                        )
                    stack_obs =update_stack_obs(observation[0], stack_obs)
                    action = algo.predict(np.expand_dims(stack_obs, axis=0))[0]
                action, replace = rule_based_postprocess(copy.deepcopy(wrap_obs[0]), copy.deepcopy(raw_obs[0]), action, version=acs_replace_strategy)
                raw_obs, reward, done, info = env.step(action)
                # t2 = time.time()
                # print(t2 - t1)
                replace_times += int(replace)
                episode_reward += float(reward)
                step += 1
                if done or step > HORIZON:
                    break
            replaced_rates.append(replace_times/step)
            episode_rewards.append(episode_reward)
            print(episode_reward)
            if episode_reward > 1:
                pass
        cur_res_dict = {
                f"rew_{env.env_name}": float(np.mean(episode_rewards)), 
                f"win_{env.env_name}": float(np.mean(np.array(episode_rewards)>0)), 
                f"draw_{env.env_name}": float(np.mean(np.array(episode_rewards)==0)), 
                f"lose_{env.env_name}": float(np.mean(np.array(episode_rewards)<0)), 
                f"replaced_rate_{env.env_name}": float(np.mean(replaced_rates))}
       
        res_dict.update(cur_res_dict)
        res_dict["rew"].append(float(np.mean(episode_rewards)))
        res_dict["win"].append(float(np.mean(np.array(episode_rewards)>0)))
        res_dict["draw"].append(float(np.mean(np.array(episode_rewards)==0)))
        res_dict["lose"].append(float(np.mean(np.array(episode_rewards)<0)))
        res_dict["replaced_rate"].append(float(np.mean(replaced_rates)))
    res_dict["rew"] = float(np.mean(res_dict["rew"]))
    res_dict["win"] = float(np.mean(res_dict["win"]))
    res_dict["draw"] = float(np.mean(res_dict["draw"]))
    res_dict["lose"] = float(np.mean(res_dict["lose"]))
    res_dict["replaced_rate"] = float(np.mean(res_dict["replaced_rate"]))
    return res_dict


def evaluate_qlearning_with_environment(
    algo: QLearningAlgoProtocol,
    env_list: GymEnv,
    n_trials: int = 10,
    epsilon: float = 0.0,
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.

    Returns:
        average score.
    """
    res_dict = {}
    for env in env_list:
        episode_rewards = []
        wrapper_func = Simple115StateWrapper_ball_owned_player
        wrapper = wrapper_func(env)
        
        for n in range(n_trials):
            step = 0
            observation = env.reset()
            episode_reward = 0.0
            print("evaluation trial", n, "in env", env.env_name)
            while True:
                observation = wrapper.observation(observation)
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if isinstance(observation, np.ndarray):
                        observation = np.expand_dims(observation, axis=0)
                    elif isinstance(observation, (tuple, list)):
                        observation = [
                            np.expand_dims(o, axis=0) for o in observation
                        ]
                    else:
                        raise ValueError(
                            f"Unsupported observation type: {type(observation)}"
                        )
                    action = algo.predict(observation)[0]

                observation, reward, done, info = env.step(action)
                episode_reward += float(reward)
                step += 1
                if done or step > HORIZON:
                    break
            episode_rewards.append(episode_reward)
        res_dict.update({
                f"rew_{env.env_name}": float(np.mean(episode_rewards)), 
                f"win_{env.env_name}": float(np.mean(np.array(episode_rewards)>0)), 
                f"draw_{env.env_name}": float(np.mean(np.array(episode_rewards)==0)), 
                f"lose_{env.env_name}": float(np.mean(np.array(episode_rewards)<0))})
    return res_dict


def evaluate_transformer_with_environment(
    algo: StatefulTransformerAlgoProtocol,
    env: GymEnv,
    n_trials: int = 10,
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.

    Returns:
        average score.
    """
    episode_rewards = []
    for _ in range(n_trials):
        algo.reset()
        observation, reward = env.reset()[0], 0.0
        episode_reward = 0.0

        while True:
            # take action
            action = algo.predict(observation, reward)

            observation, _reward, done, truncated, _ = env.step(action)
            reward = float(_reward)
            episode_reward += reward

            if done or truncated:
                break
        episode_rewards.append(episode_reward)
    return float(np.mean(episode_rewards))

def evaluate_rule_based_with_environment(
    env_list: GymEnv,
    n_trials: int = 10,
) -> float:

    # episode_rewards = []
    # for _ in range(n_trials):
    #     observation, reward = env.reset()[0], 0.0
    #     episode_reward = 0.0

    #     while True:
    #         # take action
    #         action = rule_base_2_agent_dict(observation.copy())['action'].value

    #         observation, _reward, done, truncated, _ = env.step(action)
    #         reward = float(_reward)
    #         episode_reward += reward

    #         if done or truncated:
    #             break
    #     episode_rewards.append(episode_reward)
    # return float(np.mean(episode_rewards))
    
    res_dict = {'rew': [], 'win': [], 'draw': [], 'lose': []}
    for env in env_list:
        episode_rewards = []
        for _ in tqdm.tqdm(range(n_trials), desc=f"evaluating {env.env_name}"):
            step = 0
            raw_obs = env.reset()
            episode_reward = 0.0
            while True:
                action = rule_base_2_agent_dict(raw_obs.copy()[0]).value
                raw_obs, reward, done, info = env.step(action)
                episode_reward += float(reward)
                step += 1
                if done or step > HORIZON:
                    break
            episode_rewards.append(episode_reward)
        cur_res_dict = {
                f"rew_{env.env_name}": float(np.mean(episode_rewards)), 
                f"win_{env.env_name}": float(np.mean(np.array(episode_rewards)>0)), 
                f"draw_{env.env_name}": float(np.mean(np.array(episode_rewards)==0)), 
                f"lose_{env.env_name}": float(np.mean(np.array(episode_rewards)<0))}
        res_dict.update(cur_res_dict)
        print(cur_res_dict)
        res_dict["rew"].append(float(np.mean(episode_rewards)))
        res_dict["win"].append(float(np.mean(np.array(episode_rewards)>0)))
        res_dict["draw"].append(float(np.mean(np.array(episode_rewards)==0)))
        res_dict["lose"].append(float(np.mean(np.array(episode_rewards)<0)))
    res_dict["rew"] = float(np.mean(res_dict["rew"]))
    res_dict["win"] = float(np.mean(res_dict["win"]))
    res_dict["draw"] = float(np.mean(res_dict["draw"]))
    res_dict["lose"] = float(np.mean(res_dict["lose"]))
    return res_dict
    
    
def load_jsonl_list(file_path):
    res_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            res_list.append(json.loads(line))
    return res_list

def get_rag_retriever():
    
    class JsonFileReader(BaseReader):
        def load_data(self, file, extra_info=None):
            json_list = load_jsonl_list(file)
            doc_list = []
            for res in json_list:
                doc_list.append(Document(text=str(res), extra_info=extra_info or {}))
            # load_data returns a list of Document objects
            return doc_list
    
    service_context_policy = make_service_context(embed_model=OpenAIEmbedding())
    policy_path = os.environ.get("PLFB_RAG_POLICY_PATH", os.path.join(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts"), "book_derived", "rag"))
    policy_doc = SimpleDirectoryReader(policy_path, file_extractor={'.jsonl': JsonFileReader()}).load_data()
    
    policy_index = index_from_documents(policy_doc, service_context=service_context_policy)
    policy_retriever = policy_index.as_retriever()

    return policy_retriever


    

def evaluate_llm_with_environment(
    llm_version: str,
    env_list: GymEnv,
    n_trials: int = 10,
    epsilon: float = 0.0,
    acs_replace_strategy='parts_keeped',
) -> float:
    res_dict = {'rew': [], 'win': [], 'draw': [], 'lose': [], 'replaced_rate': []}
    
    if llm_version == 'llm_rag':
        policy_retriever = get_rag_retriever()
    for env in env_list:
        episode_rewards = []
        replaced_rates = []
        wrapper_func = Simple115StateWrapper_ball_owned_player
        wrapper = wrapper_func(env)
        
        for n in tqdm.tqdm(range(n_trials), desc=f"evaluating {env.env_name}"):
            step = 0
            raw_obs = env.reset()
            episode_reward = 0.0
            replace_times = 0
            
            with tqdm.tqdm(total=HORIZON, desc=f"Step / HORIZON") as pbar:
                while True:
                    wrap_obs = wrapper.observation(copy.deepcopy(raw_obs))
                    # observation = imaginary_data_observation(copy.deepcopy(wrap_obs[0]), copy.deepcopy(raw_obs[0]), 
                    #                                           step, ret_type='dict', TODO_missing=True)
                    
                    observation = imaginary_data_observation_v2(copy.deepcopy(raw_obs[0]), step)
                
                    # generate action
                    
                    replace = False
                    wos = copy.deepcopy(wrap_obs[0])
                    oos = copy.deepcopy(raw_obs[0])
                    
                    team_owner = np.where(wos[94:97] == 1)[0][0] # 0: no one, 1: left team, 2: right team
                    team_owner = np.where(wos[94:97] == 1)[0][0] # 0: no one, 1: left team, 2: right team
                    ball_zone = get_zons_240(wos[88], wos[89])
                    active_player_index = np.where(wos[97:108] == 1)[0][0]
                    active_player_zone = get_zons_240(wos[2*active_player_index], wos[2*active_player_index+1])
                    rule_policy_dic = rule_base_2_agent_dict(oos.copy(), ret_dict=True)
                    rule_action = rule_policy_dic['action'].value
                    
                    
                    # start time counting
                    
                    init_time = time.time()
                    
                    if team_owner != 1 and active_player_zone == ball_zone:
                        action = rule_action
                        replace = True
                    else:
                        if llm_version == 'llm_agent':
                            action = llm_agent(observation)
                        elif llm_version == 'llm_rag':
                            rag_obs = json.dumps(observation)
                            policy_nodes = policy_retriever.retrieve(rag_obs)
                            policy_context_str = "\n\n".join([n.node.get_content() for n in policy_nodes])
                            action = llm_rag(observation, policy_context_str)
                        elif llm_version == 'RT2':
                            text_obs = img_obs_to_text(observation)
                            action = RT2(text_obs)
                        else:
                            raise ValueError(f"Unsupported version: {llm_version}")
                        
                    if team_owner == 1 and action != rule_action:
                        for rtz in oos['right_team']:
                            zone = get_zons_240(rtz[0], rtz[1])
                            if active_player_zone == zone: 
                                action = rule_action
                                replace = True    
                   
                   # end time counting
                    end_time = time.time()
                    
                    print("time cost", end_time - init_time)

                    
                    raw_obs, reward, done, info = env.step(action)
                    replace_times += int(replace)
                    episode_reward += float(reward)
                    step += 1
                    pbar.update(1)
                    
                    if done or step > HORIZON:
                        break
            replaced_rates.append(replace_times/step)
            episode_rewards.append(episode_reward)
        cur_res_dict = {
                f"rew_{env.env_name}": float(np.mean(episode_rewards)), 
                f"win_{env.env_name}": float(np.mean(np.array(episode_rewards)>0)), 
                f"draw_{env.env_name}": float(np.mean(np.array(episode_rewards)==0)), 
                f"lose_{env.env_name}": float(np.mean(np.array(episode_rewards)<0)), 
                f"replaced_rate_{env.env_name}": float(np.mean(replaced_rates))}
        res_dict.update(cur_res_dict)
        print(cur_res_dict)
        res_dict["rew"].append(float(np.mean(episode_rewards)))
        res_dict["win"].append(float(np.mean(np.array(episode_rewards)>0)))
        res_dict["draw"].append(float(np.mean(np.array(episode_rewards)==0)))
        res_dict["lose"].append(float(np.mean(np.array(episode_rewards)<0)))
        res_dict["replaced_rate"].append(float(np.mean(replaced_rates)))
    res_dict["rew"] = float(np.mean(res_dict["rew"]))
    res_dict["win"] = float(np.mean(res_dict["win"]))
    res_dict["draw"] = float(np.mean(res_dict["draw"]))
    res_dict["lose"] = float(np.mean(res_dict["lose"]))
    res_dict["replaced_rate"] = float(np.mean(res_dict["replaced_rate"]))
    return res_dict


