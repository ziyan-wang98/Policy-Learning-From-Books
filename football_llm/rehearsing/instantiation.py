

import os
import json
import numpy as np
from CONFIG import KnowledgeType
from retrieval.state_filed_retrieval import KnowledgeRetrieval
from llm.openai_server import OpenAIServer

from termcolor import colored
from funcs import load_jsonl_list, npz_extractor, json_str_clean


MAX_ATTAMPTS = 3

class CodeInstantiation(object):
    def __init__(self, retrieval_model:KnowledgeRetrieval, cache_path:str, vector_to_obs_fn:callable, 
                 knowledge_type:str, task_info:str, element_type_format_dict:dict, element_type_example_dict:dict, element_type_prompt_dict:dict,
                 obs_desc:str, acs_desc:str, one_step_using=False
                 ):
        self.retrieval_model = retrieval_model
        self.one_step_using = one_step_using
        self.cache_path = cache_path
        self.vector_to_obs_fn = vector_to_obs_fn
        self.knowledge_type = knowledge_type
        self.task_info = task_info
        self.code_instantiated_dict = {}
        self.element_type_format_dict = element_type_format_dict
        self.element_type_example_dict = element_type_example_dict
        self.element_type_prompt_dict = element_type_prompt_dict
        self.obs_desc = obs_desc
        self.acs_desc = acs_desc
        self.openai_server_plus = OpenAIServer(model="gpt-4o", top_p=0.9, temp=0.1, max_token=1500)
        os.makedirs(self.cache_path, exist_ok=True)
    
    def check_and_load_cache(self):
        for file in os.listdir(self.cache_path):
            if file.startswith("state-scope-cache"):
                with open(os.path.join(self.cache_path, file), 'r') as f:
                    self.code_instantiated_dict.update(json.load(f))

    def get_code(self, vector_obs:str):
        obs_key = str(vector_obs)
        self.check_and_load_cache() # for multiprocess safe.
        if obs_key in self.code_instantiated_dict:
            print(colored(f"Get {self.knowledge_type} Code from Cache: {obs_key}", "green"))
            return self.code_instantiated_dict[obs_key]
        else:
            text_obs = self.vector_to_obs_fn(vector_obs)
            code_knowledge = self.retrieval_model.retrieve_knowledge(text_obs)
            code_instant_prompt = self.code_instantiation_prompt(text_obs, code_knowledge)
            instant_code = self.code_instantiation(code_instant_prompt)
            if instant_code is not None:
                self.code_instantiated_dict[obs_key] = instant_code
                self.save_cache(obs_key, instant_code)
            return instant_code

    def code_instantiation_prompt(self, text_obs, code):
        return self.meta_code_instantiation_prompt('\n\n\n === another code knowledge: \n\n'.join(code), text_obs)
    
    def save_cache(self, obs_key:str, instant_code:dict):
        with open(os.path.join(self.cache_path, f"state-scope-cache-{obs_key.replace(' ', '-').replace(',', '_')}.json"), 'w') as file:
            json.dump({obs_key: instant_code}, file, indent=4)

    def meta_code_instantiation_prompt(self, code_string, obs):
        raw_policy_prompt_step1_all_code = """
                I want you to act {task_info}. 
                
                I will give you an observation which you are facing in the simulator, your task is to instantiated a code that serve as a {knowledge_type} function, which is used for {knowledge_function}.
                For example, {knwoledge_function_example}

                Formally, the format of {knowledge_type} function is {knowledge_format}.

                To help you complete the task, I will provide you 
                (1) pieces of relevant knowledge from the tutorial, written in Python-style pseudocode. You can refer to these pre-coded snippets to instantiated a code.
                (2) the observation, action of the target simulator;
                (3) the current observation which you are facing in the target simulator.

                
                Python-style relevant knowledge from the tutorial: \n{code_string}
                
                The observation space of the target simulator: \n{observation_space_desc}

                The action space of the target simulator: \n{action_space_desc}

                The current observation you are facing in the target simulator: \n{obs}

                About your code-instantiation task:\n
                - Please provide the PYTHON-tyle presudo code as detailed as you can.
                - Your task is to rewrite a code that descicribe a {knowledge_type} which is suitable to current observation.  \n
                - You should make the optimal decision based on the analyze current observation. To achive this, you should output the anaysis to "analyze to current observation" \n
                - After analyzing the current observation, you should rewrite the pseudocode to make it most suitable to derive the {knowledge_type} function for some downstream tasks.  \n        
                - Please keep a main {knowledge_type} function, named "{knowledge_type}" in the code, and you can also add some inner functions if necessary. \n

                """.format(task_info=self.task_info, knowledge_type=self.knowledge_type, 
                        knowledge_function=self.element_type_prompt_dict[self.knowledge_type], knowledge_format=self.element_type_format_dict[self.knowledge_type],
                        knwoledge_function_example=self.element_type_example_dict[self.knowledge_type], code_string=code_string, 
                        observation_space_desc=self.obs_desc, action_space_desc=self.acs_desc, 
                        obs=obs) + \
                """
                NOTE: 
                - Since I will delete the original code after getting your instantiated code, you cannot call the presudocodes that I provided in the prompt.
                - The process of variable assignment example usage are allowed to be ignored/simplified, however, you should implment the logic of the code as detailed as possible so that others who without the relevant knowledge can also implment it
                - You should not implement any function that return with placeholder.
                """
        if self.one_step_using:
            raw_policy_prompt_step1_all_code += """
                - Please Keep the code as short, direct, and easy-to-understand as possible , and focus on the main logic of the code.\n
                """
        else:
            raw_policy_prompt_step1_all_code += """
                - The code will be repeatedly used in the next 1 minute of the game in the simulator, so you should make sure that the code can be generalized in the next 1 minute of the game. Thus your code should consider all of the possible situations that might happend in the future, including the chaning of active player, ball positions, and opponent positions. \n
                """
        raw_policy_prompt_step1_all_code += """
                About the format:
                    - you should answer in pure JSON format directly, without any other information. For example, you should not add the ```json``` tag in the answer.
                

                Response example (you SHOULD resposne in the following key order and format):
                {
                    "analyze": "the analyze to current observation. Focus on how to instantiated a code based on the current observation.",   
                    "code": "the code you rewrite based on the analyze to current observation...",
                }

                Output:
                """
        return raw_policy_prompt_step1_all_code


    def code_instantiation(self, knowledge_prompt):
        code_res = None
        code_json_load = None
        code_response = None
        for attempt in range(MAX_ATTAMPTS):
            try:
                code_response = self.openai_server_plus.chat(knowledge_prompt)
                try:
                    code_response_clean = json_str_clean(code_response)
                    code_res = json.loads(code_response_clean)
                except json.decoder.JSONDecodeError as e:
                    code = code_response_clean.split('"code": "')[1].replace("\\n", "\n")
                    analyze = code_response_clean.split('"code": "')[0].replace("\\n", "\n")
                    code_res = {"code": code, "analyze": analyze}
                for k, v in code_res.items():
                    code_res[k] = v.split('\n')
                break
            except Exception as e:
                print(colored(f"Get Code Error: {e}, retrying...", "red"))
                print("code_json_load: ", code_response)
                continue
        return code_res