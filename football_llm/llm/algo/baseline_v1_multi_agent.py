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


from utils.obs2text import observation_to_text_human, observation_to_text_raw
from utils.index import get_engine, get_index, index_llm

import numpy as np

GPT_MODEL = os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini")
ACTION_TEXT = [
    "action_idle",
    "action_left",
    "action_top_left",
    "action_top",
    "action_top_right",
    "action_right",
    "action_bottom_right",
    "action_bottom",
    "action_bottom_left",
    "action_long_pass",
    "action_high_pass",
    "action_short_pass",
    "action_shot",
    "action_sprint",
    "action_release_direction",
    "action_release_sprint",
    "action_sliding",
    "action_dribble",
    "action_release_dribble",
]
# Set PLFB_OPENAI_CHAT_MODEL to override the default chat model.

def flatten_list(l):
    flattened_list = [item for sublist in l for item in sublist]
    return flattened_list

def gpt_json_query(user_prompt):
    client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Please interpret and reformat the following string into a valid JSON format, \
                    Response example (you should use the following format to construct your response): [[0],[1],[2],...], make sure the response have 11 elements and each element is from 0 to 18."},
            {"role": "user", "content": user_prompt}
        ],
        model=GPT_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content

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
                        12 = action_shot, player will try to shoot the ball.\n
                        13 = action_sprint, player will sprint.\n
                        14 = action_release_direction, player will stop moving in the current direction.\n
                        15 = action_release_sprint, player will stop sprinting.\n
                        16 = action_sliding, player will try to slide.\n
                        17 = action_dribble, player will try to dribble.\n
                        18 = action_release_dribble, player will stop dribbling.\n
                    """)
    thought: str = Field(description="The thought of the active player, in the style of a football coach.")


class BaselineV1MultiAgent:
    
    def __init__(self, args):
        self.args = args
        # Llama index related
        self.encorder_type = args.encoder_type
        self.llm_type = args.llm_type
        self.filter_path = args.filter_path
        self.policy_index, self.reward_index, self.transition_index = get_index(self.encorder_type, self.llm_type, self.filter_path)
        
        self.policy_query_engine, self.reward_query_engine, self.transition_query_engine = get_engine(self.encorder_type, self.llm_type, self.filter_path)

        self.index_llm = index_llm (self.llm_type)
        
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
        
        
        action_list = []
        if self.args.individual_policy_flag:
            for i in range(self.num_players):
                current_obs = obs[i]
                ## Step1: change the observation to text
                if self.args.raw_text_flag:
                    text_obs = observation_to_text_raw(current_obs)
                else:
                    text_obs = observation_to_text_human(current_obs, step, block_mode=self.args.block_mode) # block_mode = 240 or None
                print("--"*20)
                print(text_obs)
                print("--"*20)
                ## Step2: get the action
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
                        action=int(action_str["tool_calls"][0]["function"]['name'].split("_")[-1])
                        assert isinstance(action, int)
                        print("Coach's thought: ", action_thought)
                else:
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
                print("The football coach choose action: ", str(action), " for the player", {np.where(obs[97:108] == 1)[0][0] + 1})
                print("--"*20)
                action_list.append(action)
                
        else:
            # use the same action for all the players
            current_obs = obs[0] # except the active player, other dimantion is the same
            ball_ownership = np.where(current_obs[94:97] == 1)[0][0] # 0: no one, 1: left team, 2: right team
            
            ## Step1: change the observation to text
            if self.args.raw_text_flag:
                text_obs = observation_to_text_raw(current_obs)
            else:
                text_obs = observation_to_text_human(current_obs, step, block_mode=self.args.block_mode) # block_mode = 240 or None #TODO: add ma, remove the active player text
            print("--"*20)
            print(text_obs)
            print("--"*20)
            
            
            
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
                    if ball_ownership == 0 or 2: # no one
                        # All agnets controled by no_ball_policy 
                        # action_space = 0 - 8 + 13, 14, 15, 16 (no_ball_tool)
                        tools = self.args.no_ball_tool
                    elif ball_ownership == 1: # left team
                        # ball_ower policy
                        # action_space = 0 - 15, 17, 18 (ball_ower_tool)
                        ball_owner= np.where(current_obs[-22:] == 1)[0][0]
                        # others policy
                        # action_space = 0 - 8, 13, 14, 15, 16 (no_ball_tool)
                        tools = self.args.no_ball_tool
                        
                    self.retriever = self.policy_index.as_retriever() #TODO: only policy, cam add reward and transition here later
                    nodes = self.retriever.retrieve(text_obs) #TODO: for different player, we can chenge the text_obs to "player i 's role description" + text_obs. For now, we just use the same text_obs for all the players.
                    context_str = "\n\n".join([n.node.get_content() for n in nodes])
                    current_obs_prompt = self.args.ma_fc_policy_prompt.format(context_str=context_str, query_str=text_obs)
                    messages = [
                        {
                            "role": "system",
                            "content": """ You are a football coach, and you are coaching a football game. Left team is your team, and right team is the opponent team. \n
                                            Base on the most related policy content from the football coach book below, please choose the next action for all players you coached base on the current game text observation. \n
                                    """
                        },
                        {
                            "role": "user",
                            "content": current_obs_prompt
                        }
                    ]
                    action_str = chat_completion_request(messages=messages, tools= tools)
                    import pdb; pdb.set_trace()
                    action_str=action_str.json()["choices"][0]["message"]
                    
                    
                    
                    
                    if ball_owner:
                        # Change policy for the ball owner
                        ball_owner_obs = obs[ball_owner]
                        if self.args.raw_text_flag:
                            text_obs = observation_to_text_raw(ball_owner_obs)
                        else:
                            text_obs = observation_to_text_human(ball_owner_obs, step, block_mode=self.args.block_mode) # block_mode = 240 or None #TODO: add ma, remove the active player text
                        ball_owner_tools = self.args.ball_ower_tool
                        self.retriever = self.policy_index.as_retriever() #TODO: only policy, cam add reward and transition here later
                        nodes = self.retriever.retrieve(text_obs) #TODO: for different player, we can chenge the text_obs to "player i 's role description" + text_obs. For now, we just use the same text_obs for all the players.
                        context_str = "\n\n".join([n.node.get_content() for n in nodes])
                        current_obs_prompt = self.args.fc_policy_prompt.format(context_str=context_str, query_str=text_obs)
                        messages = [
                            {
                                "role": "system",
                                "content": """ You are a football coach, and you are coaching a football game. Left team is your team, and right team is the opponent team. \n
                                            Base on the most related policy content from the football coach book below, please choose the next action for the active player base on the current game text observation.
                                        """
                            },
                            {
                                "role": "user",
                                "content": current_obs_prompt
                            }
                        ]
                        ball_owner_action_str = chat_completion_request(
                            messages=messages, tools= ball_owner_tools
                        )
                        ball_owner_action_str=action_str.json()["choices"][0]["message"]
                        ball_owner_action_thought = ball_owner_action_str['content']
                        ball_owner_action= int(ball_owner_action_str["tool_calls"][0]["function"]['name'].split("_")[-1])
                        
                        action_list[ball_owner] = ball_owner_action
                    
                    
                    import pdb; pdb.set_trace()
                    
                    # action_thought = action_str['content']
                    # action=int(action_str["tool_calls"][0]["function"]['name'].split("_")[-1])
                    # assert isinstance(action, int)
                    # print("Coach's thought: ", action_thought)
                        


                    
                for i, action in enumerate(action_list):
                    print(f"The football coach chooses action {ACTION_TEXT[action]} for player {i + 1} in Left team.")
            else:
                # MA Raw policy
                assert self.args.ma_raw_policy_prompt != None, "Please set your ma_raw_policy_prompt in the environment variable"

                self.ma_raw_policy_prompt = PromptTemplate(self.args.ma_raw_policy_prompt)
                self.policy_query_engine.update_prompts(
                    {"response_synthesizer:text_qa_template": self.ma_raw_policy_prompt}
                )
                action_str = self.policy_query_engine.query(text_obs).response
                try:
                    action_list = json.loads(action_str)
                except:
                    reformatted_action_str = gpt_json_query(action_str)
                    try:
                        # Attempt to parse the new JSON string
                        print("WARNING!!!: Attempting to parse the reformatted action string as JSON.")
                        action_list = json.loads(reformatted_action_str)
                    except json.JSONDecodeError:
                        # Handle case where even the reformatted string is not valid JSON
                        print("Failed to parse the reformatted action string as JSON.")
                for i, action in enumerate(action_list):
                    action_index = action[0]  # assuming each action list contains one action index
                    print(f"The football coach chooses action {ACTION_TEXT[action_index]} for player {i + 1} in Left team.")
                action_list = flatten_list(action_list)

                

            print("--"*20)

            
            
            

                
                
                
                
        return action_list