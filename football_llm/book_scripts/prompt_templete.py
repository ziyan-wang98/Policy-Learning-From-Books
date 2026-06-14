
from llm.utils.obs2text import OBS_TEXT
filter_stringency_level_prompt_dict = {
    'loose': 'You can put all related content that is helpful to derive {0} element/function here',
    'medium': 'You should only put the content that can derive {0} element/function directly here',
    'strict': 'You should only put the content that can derive {0} element/function directly here, and the content should be very close to the element/function',
}

element_type_prompt_dict = {
    "Policy": '"Policy function": The football manager policy is to give the tactics and strategies for all players in the team, such as how players should be used in the frontcourt during a match, or when forwards should take shots. For example: "When watching defenders you have to assess how they respond to their opponents as well as the ball." ',
    "Dynamics": '"Environment Dynamics function": Dynamics is to give the dynamics function or related rules of the football game under the football manager policy\'s action, such as after shotting, the ball will be in the goal or not. For example: "When the direction of shotting is vertical to the goal, the ball will be easy to the goal." ',
    "Rewards": '"Rewards function": Reward is to give the reward or punishment of the football manager policy. For example: "When the forwards are restricted, the midfielder can support and take away the defenders, which is a very encouraging behavior." ',
}

element_type_example_dict = {
    "Policy": "4-4-2 is a good formation for a team with a strong midfield, because it allows the team to control the ball and keep possession. To play this formation, the team should have two central midfielders, two wingers, and two strikers. The central midfielders should be able to pass the ball well and control the game. The wingers should be able to run up and down the wing, and cross the ball into the box. The strikers should be able to score goals.",
    "Rewards": "The behavior that is encouraged is when the forwards are restricted, the midfielder can support and take away the defenders. This is a very encouraging behavior because it allows the team to keep possession of the ball and control the game. You should only identify 5 types of rewards: 2: Optimal behavior; 1: Encouraging behavior; 0: Borderline behavior; -1: Punishing behavior; -2: Worest behavior.",
    "Dynamics": "When the direction of shotting is vertical to the goal, the ball will be easy to the goal."
}

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

parser_dynamics_info = '\n'.join([ "- " + k + ": " + str(v) for k, v in OBS_TEXT.items()])
element_type_code_example_dict = {
    "Rewards": "The presudocode should be one or a group of functions that take any observations and actions of the players in the football as the input, and output one of the following results: 2: Optimal behavior; 1: Encouraging behavior; 0: Borderline behavior; -1: Punishing behavior; -2: Worest behavior. ",
    "Policy": f"The presudocode should be one or a group of functions that take any observations of the players in the football as the input, and output the macro tactics and strategies for all players in the team, such as how players should be used in the frontcourt during a match, or when forwards should take shots. Or actions like ({','.join(ACTION_TEXT)})",
    "Dynamics": f"The presudocode should be one or a group of functions that take any observations and actions of the players in the football as the input, and output the dynamics function or related rules of the football game under the football manager policy\'s action. The output observation can consider the following information: [[ \n\n {parser_dynamics_info} ]]"
}

def code_agg_prompt(filter_stringency_level, derive_element_type):
    filter_stringency_prompt = filter_stringency_level_prompt_dict[filter_stringency_level]
    global_prompt = """
        I want you to act like a football manager and also an expert in python coding and Reinforcement Learning that want to learn a football manager policy in a football simulator. 
        
        I will give you several presudocode snippets to define the specific theorem, principle, rule, and law of the related elements or concepts: 
            {0}. {1}.
        
        These codes might share some common logic. Your tasks is to aggregate the common logic of the given codes into a single code snippet, while still covering all of the given codes.
        
        """.format(element_type_prompt_dict[derive_element_type], filter_stringency_prompt.format(derive_element_type)) + \
        """
        Requirements:

        About the aggregated code:
        - Please provide the PYTHON-tyle presudo code as detailed as you can to cover the most information of the original content.
        - Using the least number of presudocode items.
        - Covering all of the code. However, please feel free to add more presudocode items if needed.
        - Since I will delete the original code after getting your aggregated code,  you cannot call the presudocodes that I provided in the prompt. If it is necessary to call the presudocode, please still return the presudocode as an individual item in the answer.
        
            
        About the presudocode snippet: 
        - {0}
        About the format:
            - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.
        """.format(element_type_code_example_dict[derive_element_type]) + \
        """
        
        The response example:
        
        {
            "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
            "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
            "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
        }
        """
    return global_prompt
    

def agg_prompt(filter_stringency_level, derive_element_type):
    filter_stringency_prompt = filter_stringency_level_prompt_dict[filter_stringency_level]
    global_prompt = """
            I want you to act like a football manager and also an expert in Reinforcement Learning that want to learn a football manager policy in a football simulator. 
            
            I will give you several paragraphs and the corresponding summary written by code from several football-related books.
            
            You need to analyze the given paragraph step-by-step from a football-related context to aggregate the specific theorem, principle, rule, and law of the related elements or concepts: 
            {0}. {1}.
            """.format(element_type_prompt_dict[derive_element_type], filter_stringency_prompt.format(derive_element_type)) + \
            """
            Requirements:

            About the anylysis:
            - You should write the specific theorem, principle, rule, and law of the related elements via presudo code.
            - Please provide the PYTHON-tyle presudo code as detailed as you can to cover the most information of the original content.
            - You should aggregate the given information as much as you can that 
                1. using the least number of presudocode items.
                2. covering all of the code and most of the original texts. However, please feel free to add more presudocode items if needed.
                3. Since I will delete the original code after getting your aggregated code,  you cannot call the presudocodes that I provided in the prompt. If it is necessary to call the presudocode, please still return the presudocode as an individual item in the answer.

            
            About the presudocode snippet: 
            - {0}
            About the format:
                - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.
            """.format(element_type_code_example_dict[derive_element_type]) + \
            """
            
            The response example:
            {
                "aggregated_pseudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
                "aggregated_pseudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
            }
            """
    return global_prompt


def stage12_prompt(filter_stringency_level, derive_element_type):
    filter_stringency_prompt = filter_stringency_level_prompt_dict[filter_stringency_level]
    global_prompt = """
            I want you to act like a football manager and also an expert in Reinforcement Learning that want to learn a football manager policy in a football simulator. 
            
            You need to analyze the given paragraph step-by-step from a football-related context to derive the specific theorem, principle, rule, and law of the related elements or concepts: 
            {0}. {1}.
            """.format(element_type_prompt_dict[derive_element_type], filter_stringency_prompt.format(derive_element_type)) + \
            """
            Requirements:

            About the answer:
            - If you think the paragraph contains the above elements, please answer 1. The answer is 1 only when you can write the specific theorem, principle, rule, and law of the related elements into presudocode snippet.
            - If you think the paragraph does not contain the above elements, please answer 0.
            - If you think the given paragraph is not clear enough to answer, please answer 2. Then I will give you the following paragraph to help you answer.
            - If you think the paragraph contains the above elements but the content is not clear enough to derive the specific theorem, principle, rule, and law of the related elements, please answer 2. Then I will give you the following paragraph to help you answer.
            NOTE: {0}.

            About the anylysis:
            - If the answer is 1, you should give the specific theorem, principle, rule, and law of the related elements.
            - You should write the specific theorem, principle, rule, and law of the related elements via presudo code.
            - Please provide the PYTHON-tyle presudo code as detailed as you can to cover the most information of the original content.
            
            About the presudocode snippet: 
            - {1}
            About the format:
                - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.
            """.format(filter_stringency_prompt.format(derive_element_type), element_type_code_example_dict[derive_element_type]) + \
            """
            
            The response example:
            {
                "answer": 1,
                "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
                "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
                "presudocode of/about XX": ["PYTHON presudocode snippet1", "PYTHON presudocode snippet2", ...],
            }
            """
    return global_prompt

# meta_prompt 
def stage2_prompt(filter_stringency_level, derive_element_type):
    filter_stringency_prompt = filter_stringency_level_prompt_dict[filter_stringency_level]
    global_prompt = """
            I want you to act like a football manager and also an expert in Reinforcement Learning that want to learn a football manager policy in a football simulator. 
            
            You need to analyze the given paragraph step-by-step from a football-related context to derive the specific theorem, principle, rule, and law of the related elements or concepts: 
            {0}. {1}.
            """.format(element_type_prompt_dict[derive_element_type], filter_stringency_prompt.format(derive_element_type)) + \
            """
            Requirements:
            
            About the answer:
            - Do not just give me the summary of the paragraph, instead, you should give me the specific theorem, principle, rule, or law that derived from the given paragraph.
            - Since I will use the answer instead of the raw paragraph to build the football simulator and policy, you should give the detailed and specific theorem, principle, rule, and law you derived or learned from the paragraph.
            About the format:
                - you should answer in pure JSON format, without any other information or code.
            """ + \
            """Response example:
            {
            """ + \
            """
               "{0}": ["describe in details about the first theorem/principle/rule/law you learned to design the {0} function", 
                       "describe in details about the second theorem/principle/rule/law you learned to design the {0} function", 
                       "describe in details about the third theorem/principle/rule/law you learned  o design the {0} function", ],
            """.format(derive_element_type, element_type_example_dict[derive_element_type]) + \
            "}"

    return global_prompt

def stage1_prompt(filter_stringency_level, derive_element_type):
    filter_stringency_prompt = filter_stringency_level_prompt_dict[filter_stringency_level]
    global_prompt = """
            I want you to act like a football manager and also an expert in Reinforcement Learning that want to learn a football manager policy in a football simulator. 
            
            You need to analyze the given paragraph from a football-related context to identify if it contains the related elements or concepts in Reinforcement Learning or Markov decision process (MDP): 
            {0}. {1}.
            """.format(element_type_prompt_dict[derive_element_type], filter_stringency_prompt.format(derive_element_type)) + \
            """
            Requirements:
            About the answer:
            - If you think the paragraph contains the above elements, please answer 1. 
            - If you think the paragraph does not contain the above elements, please answer 0.
            - If you think the given paragraph is not clear enough to answer, please answer 2. Then I will give you the following paragraph to help you answer.
            - If you think the paragraph contains the above elements but the content is not clear enough to derive the specific theorem, principle, rule, and law of the related elements, please answer 2. Then I will give you the following paragraph to help you answer.
            NOTE: {0}.
            """.format(filter_stringency_prompt.format(derive_element_type)) + \
            """
            
            About the format:
                - you should answer in pure JSON format, without any other information or code. For example, you should not add the ```json``` tag in the answer.

            
            The response example:
            
            {
                "answer": 1,
                "explanation": "The paragraph contains the related information because xxxx."
            }
            """
    return global_prompt