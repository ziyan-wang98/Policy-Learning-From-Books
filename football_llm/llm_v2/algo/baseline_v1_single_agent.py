import os
from llm.utils.llama_index_compat import (
    BaseRetriever,
    ChatMessage,
    ChatPromptTemplate,
    OpenAI,
    OpenAIQuestionGenerator,
    OpenAIPydanticProgram,
    PromptTemplate,
    QueryBundle,
    ToolMetadata,
)
from pydantic import BaseModel, Field
from typing import List

import json
import openai
from openai import OpenAI as OpenAIClient
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

# from ...rule_base_1 import agent as rule_base_1_agent
# from rule_base_1.agent import agent as rule_base_1_agent
from algo.rule_base_2.gfootball import agent as rule_base_2_agent
from algo.rule_base_2.gfootball import get_memory_patterns_new as rule_base_2_get_memory_patterns
from enum import Enum

from utils.obs2text import observation_to_text_human, observation_to_text_raw
from utils.index import get_engine, get_index, index_llm

import numpy as np
class Action(Enum):
    Idle = 0
    Left = 1
    TopLeft = 2
    Top = 3
    TopRight = 4
    Right = 5
    BottomRight = 6
    Bottom = 7
    BottomLeft = 8
    LongPass= 9
    HighPass = 10
    ShortPass = 11
    Shot = 12
    Sprint = 13
    ReleaseDirection = 14
    ReleaseSprint = 15
    Slide = 16
    Dribble = 17
    ReleaseDribble = 18


sticky_index_to_action = [
    Action.Left,
    Action.TopLeft,
    Action.Top,
    Action.TopRight,
    Action.Right,
    Action.BottomRight,
    Action.Bottom,
    Action.BottomLeft,
    Action.Sprint,
    Action.Dribble
]


ACTION_TEXT = [
    "idle",
    "go left",
    "go top left",
    "go top",
    "go top right",
    "go right",
    "go bottom right",
    "go bottom",
    "go bottom left",
    "do long_pass",
    "do high_pass",
    "do short_pass",
    "shot",
    "do sprint",
    "release_direction",
    "release_sprint",
    "sliding",
    "dribble",
    "release_dribble",
]

GPT_MODEL = os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini")
# Set PLFB_OPENAI_CHAT_MODEL to override the default chat model.

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
    }
    json_data = {"model": GPT_MODEL, "messages": messages}
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

class SingleAgentFunctionCalling(BaseModel):
    """What is the next action you want this active player to take? you can only choose from the following actions (0 to 18)"""
    action_picked: int = Field(description=""" The action picked by the active player, choose from the following actions (0 to 18), only answer a int number: \n
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
                        12 = action_shot, players will try to shoot, the only way to score. When the opportunity is right, more attempts to shoot will win.\n
                        13 = action_sprint, the player will sprint, gaining high speed, use it whenever possible, providing a great advantage when attacking and returning to defense .\n
                        14 = action_release_direction, player will stop moving in the current direction.\n
                        15 = action_release_sprint, player will stop sprinting.\n
                        16 = action_sliding, the player will try to slide the tackle. If your position is on the opponent's path with the ball, you can intercept it. However, if the sliding tackle fails, you will be separated by a large distance, allowing the opponent to ignore the defense.. \n
                        17 = action_dribble, players will try to dribble. When they have the ball, dribbling will greatly improve the success rate of dribbling, especially in multi-person double-teams and difficult-to-handle situations..\n
                        18 = action_release_dribble, player will stop dribbling.\n
                    """)
    thought: str = Field(description="The thought of the active player, in the style of a football coach.")


class BaselineV1SingleAgent:
    
    def __init__(self, args):
        self.args = args
        # Llama index related
        self.encorder_type = args.encoder_type
        self.llm_type = args.llm_type
        self.filter_path = args.filter_path
        self.policy_index, self.reward_index, self.transition_index = get_index(self.encorder_type, self.llm_type, self.filter_path)
        
        self.policy_query_engine, self.reward_query_engine, self.transition_query_engine = get_engine(self.encorder_type, self.llm_type, self.filter_path)

        self.index_llm = index_llm(self.llm_type)
        
    def get_action(self, obs, wrapper_obs, step):
        """
        simple115_v2

        22 - (x,y) coordinates of left team players
        22 - (x,y) direction of left team players
        22 - (x,y) coordinates of right team players
        22 - (x, y) direction of right team players
        3 - (x, y and z) - ball position
        3 - ball direction
        3 - one hot encoding of ball ownership (noone, left, right)
        11 - one hot encoding of which player is active
        7 - one hot encoding of game_mode
        Entries for players that are not active (either due to red cards or if number of player is less than 11) are set to -1.    
        
        
        Bottom left/right corner of the field is located at [-1, 0.42] and [1, 0.42], respectively.
        Top left/right corner of the field is located at [-1, -0.42] and [1, -0.42], respectively.
        Left/right goal is located at -1 and 1 X coordinate, respectively. They span between -0.044 and 0.044 in Y coordinates.
        Speed vectors represent a change in the position of the object within a single step.
        
        """
        
        ## Step1: change the observation to text
        import pdb; pdb.set_trace()
        if self.args.raw_text_flag:
            text_obs = observation_to_text_raw(wrapper_obs)
        else:
            text_obs = observation_to_text_human(wrapper_obs, step, block_mode=self.args.block_mode) # block_mode = 240 or None
        print("--"*20)
        print(text_obs)
        print("--"*20)
        
        
        # Step1.5: rule based agent policy
        rule_policy = rule_base_2_agent(obs)
        
        ## Step2: query the policy engine
    
        if self.args.use_function_calling:
            if self.args.use_pydantic_basemodel: # use pydantic basemodel to do the function calling
                self.retriever = self.policy_index.as_retriever()
                nodes = self.retriever.retrieve(text_obs)
                context_str = "\n\n".join([n.node.get_content() for n in nodes])
                current_obs_prompt = self.args.fc_policy_prompt.format(context_str=context_str, query_str=text_obs)
                messages = [
                    {
                        "role": "system",
                        "content": """ You are a football coach, and you are coaching a football game. Left team is your team, and right team is the opponent team. \n
                                        Base on the most related policy content from the football coach book below, please choose the next action for the active player base on the current game text observation. \n
                                   """
                    },
                    {
                        "role": "user",
                        "content": current_obs_prompt
                    }
                ]
                prompt = ChatPromptTemplate(
                    message_templates=messages
                )
                program = OpenAIPydanticProgram.from_defaults(
                    output_cls=SingleAgentFunctionCalling,
                    llm=self.index_llm,
                    prompt=prompt,
                    verbose=True,
                )
                action_str = program(transcript=context_str)
                action = action_str.action_picked
                assert isinstance(action, int)
                action_thought = action_str.thought
                print("Coach's thought: ", action_thought)
                
            else:  
                # use the tools to do the function calling
                tools = self.args.function_calling_tool
                self.retriever = self.policy_index.as_retriever() #TODO: only policy, cam add reward and transition here later
                nodes = self.retriever.retrieve(text_obs)
                context_str = "\n\n".join([n.node.get_content() for n in nodes])
                current_obs_prompt = self.args.fc_policy_prompt.format(context_str=context_str, query_str=text_obs)
                messages = [
                    {
                        "role": "system",
                        "content": """ You are a football coach, and you are coaching a football game. Left team is your team, and right team is the opponent team. \n
                                        Base on the most related policy content from the football coach book below, please choose the next action for the active player base on the current game text observation. \n
                                   """
                    },
                    {
                        "role": "user",
                        "content": current_obs_prompt
                    }
                ]
                action_str = chat_completion_request(
                    messages=messages, tools=tools
                )
  
                action_str=action_str.json()["choices"][0]["message"]
                action_thought = action_str['content']
                print(action_str)
                if  action_str.json()["choices"][0]["message"]["tool_calls"]:
                    action=int(action_str["tool_calls"][0]["function"]['name'].split("_")[-1])
                else:
                    action = re.search(r'_(\d+)', action_str.json()["choices"][0]["message"]["content"]).group(1)
                assert isinstance(action, int)
                print("Coach's thought: ", action_thought)
        elif self.args.e2e:
            assert self.args.raw_policy_prompt != None, "Please set your raw_policy_prompt in the environment variable"
            # if self.args.raw_policy_prompt != None: 
            self.raw_policy_prompt = PromptTemplate(self.args.raw_policy_prompt)
            self.policy_query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": self.raw_policy_prompt}
            )
            action_str = self.policy_query_engine.query(text_obs).response
            try:
                action = int(action_str)
            except:
                #find the first int
                import re
                action = int(re.findall(r"\d+", action_str)[0])
        else:
            assert self.args.raw_policy_prompt_step1 and self.args.raw_policy_prompt_step2 != None, "Please set your raw_policy_prompt_step1 and step2 in the environment variable"
            # if self.args.raw_policy_prompt != None: 
            self.raw_policy_prompt_step1 = PromptTemplate(self.args.raw_policy_prompt_step1)
            self.policy_query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": self.raw_policy_prompt_step1}
            )
            policy_code = self.policy_query_engine.query(text_obs).response
            
            current_action_prompt = self.args.raw_policy_prompt_step2.format(policy_code=policy_code)
            
            client = OpenAIClient(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

            global_prompt = """
                I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator. 
                
                I will give you the presudocode snippets to define the specific policy for the active player in the football game. 
                
                These codes represent absolute correctness, do not add other common logic. Your task is to select the action that best fits and executes the code based on the logic of the given code.
                
                Requirements:

                About the action choosing:
                - Please choose the action that best fits the code logic.
               
                About the format:
                    - you should answer in pure JSON format with the key: action and a int number from 0 to 18, without any other information or code. For example, you should not add the ```json``` tag in the answer.
                    
                The response example: 
                {
                    "action": 4
                }
            """
            
            action = client.chat.completions.create(
                messages=[
                {"role": "system", "content": global_prompt},
                {"role": "user", "content": current_action_prompt}
                ],
                model=GPT_MODEL,
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            # try:
            #     action = int(action_str)
            # except:
            #     #find the first int
            #     import re
            #     action = int(re.findall(r"\d+", action_str)[0])
        
        print(f"The football coach choose action:  {Action(action)}, for the player, {np.where(wrapper_obs[97:108] == 1)[0][0] + 1}")
        print("--"*20)
        
        rule_action = rule_policy[0]
        
        if action != rule_action:
            if action == 12:
                action = rule_action
            elif action == 16:
                action = rule_action
    
        return action
    



class NewBaselineV1SingleAgent:
    
    def __init__(self, args):
        self.args = args
        # Llama index related
        self.encorder_type = args.encoder_type
        self.llm_type = args.llm_type
        self.filter_path = args.filter_path
        self.policy_index, self.reward_index, self.transition_index = get_index(self.encorder_type, self.llm_type, self.filter_path)
        
        self.policy_query_engine, self.reward_query_engine, self.transition_query_engine = get_engine(self.encorder_type, self.llm_type, self.filter_path)

        self.index_llm = index_llm(self.llm_type)
        
    def get_action(self, obs, step):
        """
        simple115_v2

        22 - (x,y) coordinates of left team players
        22 - (x,y) direction of left team players
        22 - (x,y) coordinates of right team players
        22 - (x, y) direction of right team players
        3 - (x, y and z) - ball position
        3 - ball direction
        3 - one hot encoding of ball ownership (noone, left, right)
        11 - one hot encoding of which player is active
        7 - one hot encoding of game_mode
        Entries for players that are not active (either due to red cards or if number of player is less than 11) are set to -1.    
        
        
        Bottom left/right corner of the field is located at [-1, 0.42] and [1, 0.42], respectively.
        Top left/right corner of the field is located at [-1, -0.42] and [1, -0.42], respectively.
        Left/right goal is located at -1 and 1 X coordinate, respectively. They span between -0.044 and 0.044 in Y coordinates.
        Speed vectors represent a change in the position of the object within a single step.
        
        """
        
        ## Step1: change the observation to text
        
        if self.args.raw_text_flag:
            text_obs = observation_to_text_raw(obs)
        else:
            text_obs = observation_to_text_human(obs, step, block_mode=self.args.block_mode) # block_mode = 240 or None
        print("--"*20)
        print(text_obs)
        print("--"*20)
        
        
        ## Step2: query the policy engine
    
        if self.args.use_function_calling:
            if self.args.use_pydantic_basemodel: # use pydantic basemodel to do the function calling
                self.retriever = self.policy_index.as_retriever()
                nodes = self.retriever.retrieve(text_obs)
                context_str = "\n\n".join([n.node.get_content() for n in nodes])
                current_obs_prompt = self.args.fc_policy_prompt.format(context_str=context_str, query_str=text_obs)
                messages = [
                    {
                        "role": "system",
                        "content": """ You are a football coach, and you are coaching a football game. Left team is your team, and right team is the opponent team. \n
                                        Base on the most related policy content from the football coach book below, please choose the next action for the active player base on the current game text observation. \n
                                   """
                    },
                    {
                        "role": "user",
                        "content": current_obs_prompt
                    }
                ]
                prompt = ChatPromptTemplate(
                    message_templates=messages
                )
                program = OpenAIPydanticProgram.from_defaults(
                    output_cls=SingleAgentFunctionCalling,
                    llm=self.index_llm,
                    prompt=prompt,
                    verbose=True,
                )
                action_str = program(transcript=context_str)
                action = action_str.action_picked
                assert isinstance(action, int)
                action_thought = action_str.thought
                print("Coach's thought: ", action_thought)
                
            else:  
                # use the tools to do the function calling
                tools = self.args.function_calling_tool
                self.retriever = self.policy_index.as_retriever() #TODO: only policy, cam add reward and transition here later
                nodes = self.retriever.retrieve(text_obs)
                context_str = "\n\n".join([n.node.get_content() for n in nodes])
                current_obs_prompt = self.args.fc_policy_prompt.format(context_str=context_str, query_str=text_obs)
                messages = [
                    {
                        "role": "system",
                        "content": """ You are a football coach, and you are coaching a football game. Left team is your team, and right team is the opponent team. \n
                                        Base on the most related policy content from the football coach book below, please choose the next action for the active player base on the current game text observation. \n
                                   """
                    },
                    {
                        "role": "user",
                        "content": current_obs_prompt
                    }
                ]
                action_str = chat_completion_request(
                    messages=messages, tools=tools
                )
  
                action_str=action_str.json()["choices"][0]["message"]
                action_thought = action_str['content']
                print(action_str)
                if  action_str.json()["choices"][0]["message"]["tool_calls"]:
                    action=int(action_str["tool_calls"][0]["function"]['name'].split("_")[-1])
                else:
                    action = re.search(r'_(\d+)', action_str.json()["choices"][0]["message"]["content"]).group(1)
                assert isinstance(action, int)
                print("Coach's thought: ", action_thought)
        elif self.args.e2e:
            assert self.args.raw_policy_prompt != None, "Please set your raw_policy_prompt in the environment variable"
            # if self.args.raw_policy_prompt != None: 
            self.raw_policy_prompt = PromptTemplate(self.args.raw_policy_prompt)
            self.policy_query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": self.raw_policy_prompt}
            )
            action_str = self.policy_query_engine.query(text_obs).response
            try:
                action = int(action_str)
            except:
                #find the first int
                import re
                action = int(re.findall(r"\d+", action_str)[0])
        else:
            assert self.args.raw_policy_prompt_step1 and self.args.raw_policy_prompt_step2 != None, "Please set your raw_policy_prompt_step1 and step2 in the environment variable"
            # if self.args.raw_policy_prompt != None: 
            self.raw_policy_prompt = PromptTemplate(self.args.raw_policy_prompt)
            self.policy_query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": self.raw_policy_prompt}
            )
            action_str = self.policy_query_engine.query(text_obs).response
            try:
                action = int(action_str)
            except:
                #find the first int
                import re
                action = int(re.findall(r"\d+", action_str)[0])
        
        print(f"The football coach choose action:  {Action(action)}, for the player, {np.where(obs[97:108] == 1)[0][0] + 1}")
        print("--"*20)
        
        
            
        return action