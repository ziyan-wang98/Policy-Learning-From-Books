import argparse
import os
from pathlib import Path


def _work_path(*parts):
    return str(Path(os.environ.get("PLFB_WORK_ROOT", "runs")).joinpath(*parts))


# Gpt as the decision policy
gpt_raw_policy_prompt= """
      You are a football coach. You are coaching a football game. \n
      You are coaching the Left team. \n
      The Right team is the opponent team. \n
      Context information is below.\n
      ---------------------\n
      {context_str}\n
      ---------------------\n
      Given the context information and not prior knowledge, 
      answer the query in the style of a football coach.\n
      Language Observation for this football game at current time step is below.\n
      ---------------------\n
      {query_str}\n
      ---------------------\n
      Question: What is the next action you want this active player to take? you can only choose from the following actions (0 to 18), only answer a int number: \n
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
      Please directly type the number of the action you want the active player to take, without any other words. \n
      response example (you should use the following format to construct your response): 0\n   
      Answer: 
  """

def parse_args():
    parser = argparse.ArgumentParser(description='Football Environment Configuration')
    parser.add_argument('--environment', type=str, default='11_vs_11_easy_stochastic', help='Type of the environment')
    parser.add_argument('--representation', type=str, default='simple115v3', help='Type of representation')
    parser.add_argument('--logdir', type=str, default=os.environ.get("PLFB_GFOOTBALL_LOGDIR", _work_path("gfootball_res")), help='Directory for logs')
    parser.add_argument('--render', type=bool, default=False, help='Render environment or not')
    parser.add_argument('--num_players', type=int, default=1, help='Number of players')
    parser.add_argument('--qa_prompt_tmpl_str', type=str, default=gpt_raw_policy_prompt, help='QA prompt template string')

    return parser.parse_args()
