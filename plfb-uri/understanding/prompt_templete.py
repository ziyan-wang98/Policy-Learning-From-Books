
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
