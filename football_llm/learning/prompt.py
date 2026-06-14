
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
