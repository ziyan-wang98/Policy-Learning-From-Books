import numpy as np
import copy

from simulator_info import DIRECTIONS


PLAYER_ROLES = ['GoalKeeper', 'Forward', 'Forward', 'Defender', 'Defender', 'Defender', 'Defender', 'Midfielder', 'Midfielder', 'Midfielder', 'Forward']
directions = DIRECTIONS

import zipfile

def get_detailed_zone_raw(x, y):
    # Define horizontal zones
    if x < -0.8:
        horizontal = 'far left'
    elif -0.8 <= x < -0.4:
        horizontal = 'mid left'
    elif -0.4 <= x < 0:
        horizontal = 'left center'
    elif 0 <= x < 0.4:
        horizontal = 'right center'
    elif 0.4 <= x < 0.8:
        horizontal = 'mid right'
    else:
        horizontal = 'far right'

    # Define vertical zones
    if y < -0.28:
        vertical = 'top'
    elif -0.28 <= y < -0.14:
        vertical = 'upper middle'
    elif -0.14 <= y < 0:
        vertical = 'lower middle'
    elif 0 <= y < 0.14:
        vertical = 'just above center'
    elif 0.14 <= y < 0.28:
        vertical = 'just below center'
    else:
        vertical = 'bottom'

    # Special handling for goals
    if -0.044 <= y <= 0.044:
        if x <= -1:
            return 'left goal area'
        elif x >= 1:
            return 'right goal area'

    return f'{vertical} {horizontal}'

def get_specific_zone_term(x, y):
    # Vertical Zones (Front, Middle, Back)
    if x < -0.6:
        vertical = 'Back'
    elif -0.6 <= x < -0.2:
        vertical = 'Defensive Midfield'
    elif -0.2 <= x <= 0.2:
        vertical = 'Central'
    elif 0.2 < x <= 0.6:
        vertical = 'Attacking Midfield'
    else:
        vertical = 'Front'

    # Horizontal Zones (Left, Center, Right)
    if y < -0.28:
        horizontal = 'Left'
    elif -0.28 <= y < -0.14:
        horizontal = 'Left Center'
    elif -0.14 <= y < 0.14:
        horizontal = 'Center'
    elif 0.14 <= y < 0.28:
        horizontal = 'Right Center'
    else:
        horizontal = 'Right'

    # Special handling for penalty boxes
    if -0.044 <= y <= 0.044:
        if x <= -1:
            return 'Own Penalty Box'
        elif x >= 1:
            return 'Opponent\'s Penalty Box'

    return f'{vertical} {horizontal}'


def get_zons_240(x, y):
    # Rectangular zones
    block_width = 2.0/ 20 # 0.1
    block_hight = 0.84 / 12 # 0.07
    
    zone_x = int((x + 1) // block_width) + 1
    zone_y = int((-y + 0.42) // block_hight) + 1
    
    # Zone(1,1) is the left bottom corner
    # Zone(20,12) is the right top corner
    
    return f'Zone({zone_x},{zone_y})'

def get_zons_240_list(x, y):
    # Rectangular zones
    block_width = 2.0/ 20 # 0.1
    block_hight = 0.84 / 12 # 0.07
    
    zone_x = int((x + 1) // block_width) + 1
    zone_y = int((-y + 0.42) // block_hight) + 1
    
    # Zone(1,1) is the left bottom corner
    # Zone(20,12) is the right top corner
    
    return [zone_x, zone_y]

def vector_to_direction(x, y, ret_text=True):
    angle = np.arctan2(y, x)
    # Adjusting the angle to map the specific vector to 'north'
    adjusted_angle = (angle + np.pi / 2) % (2 * np.pi)

    # Standard directions, but the calculation is adjusted
    
    idx = round(adjusted_angle / (2 * np.pi / len(directions))) % len(directions)
    if ret_text:
        return directions[idx]
    else:
        return idx

# def vector_to_direction(x, y):
#     angle = np.arctan2(y, x)
#     directions = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
#     idx = round(angle / (2 * np.pi / len(directions))) % len(directions)
#     return directions[idx]

def observation_to_text_raw(observation):
    text = ""
    # Handle player positions and directions
    for i in range(11):
        text += f"Left team player {i} is at ({observation[2*i]}, {observation[2*i+1]}) with direction ({observation[22 + 2*i]}, {observation[22 + 2*i + 1]}). "
        text += f"Right team player {i} is at ({observation[44 + 2*i]}, {observation[44 + 2*i + 1]}) with direction ({observation[66 + 2*i]}, {observation[66 + 2*i + 1]}). "

    # Handle ball position and direction
    text += f"The ball is at ({observation[88]}, {observation[89]}, {observation[90]}) with direction ({observation[91]}, {observation[92]}, {observation[93]}). "

    # Handle ball ownership
    ownership = ["no one", "left team", "right team"]
    owner_index = np.where(observation[94:97] == 1)[0][0]
    text += f"The ball is currently controlled by {ownership[owner_index]}. "

    # Handle active player
    active_player = np.where(observation[97:108] == 1)[0][0] 
    text += f"The current active player is number {active_player}. "

    # Handle game mode
    game_modes = ["e_GameMode_Normal", "e_GameMode_KickOff", "e_GameMode_GoalKick", "e_GameMode_FreeKick", "e_GameMode_Corner", "e_GameMode_ThrowIn", "e_GameMode_Penalty"]  # Replace with actual game modes
    mode_index = np.where(observation[108:115] == 1)[0][0]
    text += f"The current game mode is {game_modes[mode_index]}. "

    return text

sticky_list = ["Left","TopLeft","Top","TopRight","Right","BottomRight","Bottom","BottomLeft","Sprint","Dribble"]
game_modes = ["Normal", "KickOff", "GoalKick", "FreeKick", "Corner", "hrowIn", "Penalty"]  # Replace with actual game modes
role_list = ["Goalkeeper", "Forward", "Forward", "Defender", "Defender", "Defender", "Defender", "Midfielder", "Midfielder", "Midfielders", "Forward"]


OBS_TEXT = {
    "sticky_actions": sticky_list, 
    "game_modes": game_modes,
    "role_list": role_list,
    "Other Information": """
        First, it provides information such as the time and score of the match. 
        Second, which side has control of the ball, and the active player in the Left team you need to propose corresponding policies to him. 
        Next is the position and role information of each player: 
        - In this text description, the football grass field is divided into 240 zones. 
        - We use zone (x, y) to express the position of the player. "x" is the discretized coordinate parallelized with the line from the left team's penalty area to the right team's penalty area, ranging from 1 to 20, and y is the discretized coordinate parallelized with the line from the lower corner to the upper corner flag, ranging from 1 to 12.
        - The left team's penalty area is zone (1, 4)-(1, 8)-(3, 8)-(3,4), and the right team's penalty area is zone (20, 4)-(20, 8)-(18, 8)-(18, 4).
        - This means that the center circle position of the field is zone (10, 6), where the game start.
        - The lower left corner position of the Left team is (1, 1), and the upper right corner position of the Right team is (20, 12). 
        - Venues never interchange or change. Following the position information is direction, meaning the direction the player is currently facing and the direction of future actions.
        """
}

def get_raw_obs_to_llm_response(raw_obs, step):
    res = {}
    res['score'] = raw_obs['score']
    res['step'] = step
    res['active_left_player'] = raw_obs['active']
    res['ball_ownership'] = raw_obs['ball_owned_team'] + 1
    res['ball_ownership_player'] = raw_obs['ball_owned_player']
    res['ball_zone'] = get_zons_240_list(raw_obs['ball'][0], raw_obs['ball'][1])
    res['left_active_player_zone'] = get_zons_240_list(raw_obs['left_team'][raw_obs['active']][0], raw_obs['left_team'][raw_obs['active']][1])
    res['dense_reward'] = 0
    res['thought']= ""
    return res

def img_obs_to_text(raw_obs, stacked=False):
    """
    raw_obs keys: 
    dict_keys(['sticky_actions', 'game_mode', 'score', 'step', 'time', 'active_player', 'active_player_role', 'ball_ownership', 'ball_ownership_player', 'ball_zone', 'ball_direction', 'player_0', 'player_1', 'player_2', 'player_3', 'player_4', 'player_5', 'player_6', 'player_7', 'player_8', 'player_9', 'player_10', 'player_11', 'player_12', 'player_13', 'player_14', 'player_15', 'player_16', 'player_17', 'player_18', 'player_19', 'player_20', 'player_21'])
    """
    finial_text = ""
    sticky_action_list = np.array(raw_obs['sticky_actions'])
    if len(sticky_action_list) > 0:        
        sticky_text = f"This player is currently using the sticky actions : {sticky_action_list} "
    else:
        sticky_text = f"This player currently doesn't use the sticky actions. "
    # Noramal is a typo and should be Normal. keep it for now for compatibility of historical data.
    if raw_obs['game_mode'] == 'Normal' or raw_obs['game_mode'] == 'Noramal':
        game_mode_text = f" The score is {raw_obs['score'][0]}:{raw_obs['score'][1]}. "
    else:
        game_mode_text = f" The game is in {raw_obs['game_mode']}. "
    time_text = f"The current time is {raw_obs['time']}. "
    active_player_text = f"You as the football coach are instructing Left team. \n The Active Player is: {raw_obs['active_player']} Player {raw_obs['active_player_role']} to make the next action. "
    team_ownership = ["no one", "Left team", "Right team"]
    team_owner_index = raw_obs['ball_ownership']
    player_owner_index = raw_obs['ball_ownership_player']
    player_owner_index = player_owner_index % 11

    if team_owner_index == 0:
        ownership_text = f"Currently, no one is controlling the ball or the ball is in the air or the ball is during passing. "
    elif team_owner_index == 1:
        ownership_text = f"Currently, your Left team is controlling the ball, by Player {player_owner_index}. "
    else:
        ownership_text = f"Currently, the ball is controlled by {team_ownership[team_owner_index]} {role_list[player_owner_index]} Player {11 + player_owner_index}. You lost the control of the ball. "
    ball_zone = raw_obs['ball_zone']
    ball_direction = raw_obs['ball_direction']
    
    if stacked:
        ball_text = "The ball's recent positions and directions are: \n"
        for i in range(len(ball_zone)):
            ball_text += f"step {i}: {{'zone': {ball_zone[i]}, 'moving towards': '{ball_direction[i]}'}}"
            if i != len(ball_zone) - 1:
                ball_text += " ->"
            else:
                ball_text += "\n"
        ball_text += "each step is 1.5 seconds.\n"
    else:
        ball_text = f"The ball is in the {ball_zone} zone moving towards {ball_direction}. "
    L_player_text = ""
    R_player_text = ""
    for i in range(11):
        if i == 0 and 11:
            p = "Goalkeeper"
        else:
            p = "Player"
        left_player_zone = raw_obs[f"player_{i}"]['zone']
        left_player_direction = raw_obs[f"player_{i}"]['direction']
        right_player_zone = raw_obs[f"player_{11 + i}"]['zone']
        right_player_direction = raw_obs[f"player_{11 + i}"]['direction']
        if not stacked:
            L_player_text += f"Left team: " + p + f" {i} is in the {left_player_zone} facing {left_player_direction}. " + "\n"
            R_player_text += f"Right team: " + p + f" {11 + i} is in the {right_player_zone} facing {right_player_direction}. " + "\n"
        else:
            L_player_text += f"Left team: " + p + f"{i} 's recent positions and directions are: \n"
            for j in range(len(left_player_zone)):
                L_player_text += f"step {j}: {{'zone': {left_player_zone[j]}, 'facing': {left_player_direction[j]}}}"
                if j != len(left_player_zone) - 1:
                    L_player_text += " ->"
                else:
                    L_player_text += "\n"
            
            R_player_text += f"Right team: " + p + f"{11 + i} 's recent positions and directions are: \n"
            for j in range(len(right_player_zone)):
                R_player_text += f"step {j}: {{'zone': {right_player_zone[j]}, 'facing': {right_player_direction[j]}}}"
                if j != len(right_player_zone) - 1:
                    R_player_text += " ->"
                else:
                    R_player_text += "\n"
    finial_text = time_text + "\n" + game_mode_text + "\n" + active_player_text + "\n" \
                            +  sticky_text + "\n" \
                            +  ownership_text + "\n" + ball_text + "\n" + "Left Team (our team): "\
                            +  "\n" + L_player_text + "\n" + "Right Team (opponent team): " \
                            +  "\n"+ R_player_text
    finial_text += "each step is 1.5 seconds.\n"
    return finial_text

def observation_to_text_human(observation, raw_obs, step, block_mode=None):
    finial_text = ""
    
    # Stinky action
    sticky_action_list = raw_obs['sticky_actions']
    
    if sticky_action_list.size > 0 and np.any(sticky_action_list != 0):
        
        indices = np.where(sticky_action_list == 1)[0]
        selected_actions = [sticky_list[idx] for idx in indices]
        sticky_text = f"This player is currently using the sticky actions : {selected_actions} "
    else:
        sticky_text = f"This player currently doesn't use the sticky actions. "
    # Handle game mode
    
    mode_index = np.where(observation[108:115] == 1)[0][0]
    if mode_index == 0:
        game_mode_text = f" The score is {raw_obs['score'][0]}:{raw_obs['score'][1]}. "
    else:
        game_mode_text = f" The game is in {game_modes[mode_index]}. "
    time = step * 1.8 # 3000 timpstep = 90 mins
    min = int(time // 60)
    sec = int(time % 60)
    time_text = f"The current time is {min} minutes {sec} seconds. "
    
    # Handle active player
    
    # active_player_index = np.where(observation[97:108] == 1)[0][0]
    active_player_index = raw_obs['active']
    active_player_text = f"You as the football coach are instructing Left team. \n The Active Player is: {role_list[active_player_index]} Player {active_player_index} to make the next action. "
    
    # Handle ball ownership
    team_ownership = ["no one", "Left team", "Right team"]
    team_owner_index = np.where(observation[94:97] == 1)[0][0]
    
    player_owner_index = np.where(observation[-22:] == 1)[0][0] if np.any(observation[-22:] == 1) else -1
    player_owner_index = player_owner_index % 11 
    
    if team_owner_index == 0:
        ownership_text = f"Currently, no one is controlling the ball or the ball is in the air or the ball is during passing. "
    elif team_owner_index == 1:
        ownership_text = f"Currently, your Left team is controlling the ball, by Player {player_owner_index}. "
    else:
        ownership_text = f"Currently, the ball is controlled by {team_ownership[team_owner_index]} {role_list[player_owner_index]} Player {11 + player_owner_index}. You lost the control of the ball. "
    
    
    if block_mode == "240":
        # Handle ball position and direction
        ball_zone = get_zons_240(observation[88], observation[89])
        ball_direction = vector_to_direction(observation[91], observation[92])
        ball_text = f"The ball is in the {ball_zone} zone moving towards {ball_direction}. "

        # Handle player positions and directions
        L_player_text = ""
        R_player_text = ""
        for i in range(11):
            if i == 0 and 11:
                p = "Goalkeeper"
            else:
                p = "Player"
            left_player_zone = get_zons_240(observation[2*i], observation[2*i+1])
            left_player_direction = vector_to_direction(observation[22 + 2*i], observation[22 + 2*i + 1])
            right_player_zone = get_zons_240(observation[44 + 2*i], observation[44 + 2*i + 1])
            right_player_direction = vector_to_direction(observation[66 + 2*i], observation[66 + 2*i + 1])
            L_player_text += f"Left team: " + p + f" {i} is in the {left_player_zone} facing {left_player_direction}. " + "\n"
            R_player_text += f"Right team: " + p + f" {11 + i} is in the {right_player_zone} facing {right_player_direction}. " + "\n"
    
    else:
        # Handle ball position and direction
        ball_zone = get_specific_zone_term(observation[88], observation[89])
        ball_direction = vector_to_direction(observation[91], observation[92])
        ball_text = f"The ball is in the {ball_zone} zone moving towards {ball_direction}. "

        # Handle player positions and directions
        L_player_text = ""
        R_player_text = ""
        for i in range(11):
           
            role = role_list[i]
            left_player_zone = get_specific_zone_term(observation[2*i], observation[2*i+1])
            left_player_direction = vector_to_direction(observation[22 + 2*i], observation[22 + 2*i + 1])
            right_player_zone = get_specific_zone_term(observation[44 + 2*i], observation[44 + 2*i + 1])
            right_player_direction = vector_to_direction(observation[66 + 2*i], observation[66 + 2*i + 1])
            L_player_text += f"Left team {role} Player {i} is in the {left_player_zone} facing {left_player_direction}. " + "\n"
            R_player_text += f"Right team {role} Player {i}  is in the {right_player_zone} facing {right_player_direction}. " + "\n"
    if False:
        finial_text = time_text + "\n" + game_mode_text  + "\n" \
                                +  ownership_text + "\n" + ball_text + "\n" + "Left Team (our team): "\
                                +  "\n" + L_player_text + "\n" + "Right Team (opponent team): " \
                                +  "\n"+ R_player_text
    else:
        finial_text = time_text + "\n" + game_mode_text + "\n" + active_player_text + "\n" \
                                +  sticky_text + "\n" \
                                +  ownership_text + "\n" + ball_text + "\n" + "Left Team (our team): "\
                                +  "\n" + L_player_text + "\n" + "Right Team (opponent team): " \
                                +  "\n"+ R_player_text

    return finial_text


def imaginary_data_to_vector(imagination_obs, TODO_missing=True):
       
    sticky_list = ["Left","TopLeft","Top","TopRight","Right","BottomRight","Bottom","BottomLeft","Sprint","Dribble"]
    game_modes = ["Normal", "KickOff", "GoalKick", "FreeKick", "Corner", "hrowIn", "Penalty"]  # Replace with actual game modes
    role_list = PLAYER_ROLES

    vector_obs = []
    
    # Sticky action
    sticky_action = imagination_obs['sticky_actions']
    sticky_action_list = [0] * len(sticky_list)
    for action in sticky_action:
        sticky_action_list[sticky_list.index(action)] = 1
    vector_obs.extend(sticky_action_list)
    
    # score
    vector_obs.extend(imagination_obs['score'])
    
    # active player
    vector_obs.append(imagination_obs['active_player'])
    
    # team ownership
    vector_obs.append(imagination_obs['ball_ownership'])
    assert TODO_missing, "This path only supports observations with ball_ownership_player."
    if TODO_missing:
        vector_obs.append(imagination_obs['ball_ownership_player'])
    
    # ball zone
    vector_obs.extend(imagination_obs['ball_zone'])
    ball_direction_index = directions.index(imagination_obs['ball_direction'])
    vector_obs.append(ball_direction_index)
    # players
    
    for i in range(22):
        key = f"player_{i}"
        vector_obs.extend(imagination_obs[key]['zone'])
        vector_obs.append(directions.index(imagination_obs[key]['direction']))

    return np.array(vector_obs)

def imaginary_data_observation_stacked_info(obs_dict_list):
    current_obs = copy.deepcopy(obs_dict_list[-1])
    # for i in range(stack_obs_num):
    for i in range(22):
        key = f"player_{i}"
        current_obs[key]['zone'] = [hist_obs[key]['zone'] for hist_obs in obs_dict_list]
        current_obs[key]['direction'] = [hist_obs[key]['direction'] for hist_obs in obs_dict_list]
    current_obs['ball_zone'] = [hist_obs['ball_zone'] for hist_obs in obs_dict_list]
    current_obs['ball_direction'] = [hist_obs['ball_direction'] for hist_obs in obs_dict_list]
    return current_obs


def imaginary_data_observation_v2(raw_obs, step, block_mode=None, ret_type='dict'):
    finial_dic = {}
    vector_obs = []
    # Stinky action
    sticky_action_list = raw_obs['sticky_actions']
    
    if sticky_action_list.size > 0 and np.any(sticky_action_list != 0):
        indices = np.where(sticky_action_list == 1)[0]
        selected_actions = [sticky_list[idx] for idx in indices]
        finial_dic["sticky_actions"] = selected_actions
    else:
        finial_dic["sticky_actions"] = []
    vector_obs.extend(sticky_action_list)
        
    # Handle game mode
    mode_index = raw_obs['game_mode']
    finial_dic["game_mode"] = game_modes[mode_index]
    finial_dic["score"] = raw_obs['score']
    vector_obs.extend(raw_obs['score'])

    time = step * 1.8 # 3000 timpstep = 90 mins
    min = int(time // 60)
    sec = int(time % 60)
    finial_dic["step"] = step
    finial_dic["time"] = f"{min} minutes {sec} seconds"
    
    # active_player_index = np.where(observation[97:108] == 1)[0][0]
    active_player_index = raw_obs['active']
    finial_dic["active_player"] = active_player_index
    finial_dic["active_player_role"]= role_list[active_player_index]
    vector_obs.append(active_player_index)
    
    
    # Handle ball ownership
    team_ownership = ["no one", "Left team", "Right team"]
    team_owner_index = raw_obs['ball_owned_team'] + 1
    player_owner_index = raw_obs['ball_owned_player']
    player_owner_index = player_owner_index % 11 
    
    
    finial_dic["ball_ownership"] = team_owner_index
    vector_obs.append(team_owner_index)
    if team_owner_index == 1:
        finial_dic["ball_ownership_player"] = player_owner_index
    elif team_owner_index == 2:
        finial_dic["ball_ownership_player"] = 11 + player_owner_index
    else:
        finial_dic["ball_ownership_player"] = -1
    vector_obs.append(finial_dic["ball_ownership_player"])

    # Handle ball position and direction
    ball_zone = get_zons_240_list(raw_obs['ball'][0], raw_obs['ball'][1])
    ball_direction = vector_to_direction(raw_obs['ball_direction'][0], raw_obs['ball_direction'][1])
    
    finial_dic["ball_zone"] = ball_zone
    finial_dic["ball_direction"] = ball_direction
    vector_obs.extend(ball_zone)
    vector_obs.append(vector_to_direction(raw_obs['ball_direction'][0], raw_obs['ball_direction'][1], ret_text=False))
    for i in range(22):
        key = f"player_{i}"
        if i < 11:
            finial_dic[key] = {
                "team": "Left",
                "role": role_list[i%11],
                "zone": get_zons_240_list(raw_obs['left_team'][i][0], raw_obs['left_team'][i][1]),
                "direction": vector_to_direction(raw_obs['left_team_direction'][i][0], raw_obs['left_team_direction'][i][1]),
            }
            vector_obs.extend(get_zons_240_list(raw_obs['left_team'][i][0], raw_obs['left_team'][i][1]))
            vector_obs.append(vector_to_direction(raw_obs['left_team_direction'][i][0], raw_obs['left_team_direction'][i][1], ret_text=False))
        else:
            finial_dic[key] = {
                "team": "Right",
                "role": role_list[i%11],
                "zone": get_zons_240_list(raw_obs['right_team'][i%11][0], raw_obs['right_team'][i%11][1]),
                "direction": vector_to_direction(raw_obs['right_team_direction'][i%11][0], raw_obs['right_team_direction'][i%11][1]),
            }
            vector_obs.extend(get_zons_240_list(raw_obs['right_team'][i%11][0], raw_obs['right_team'][i%11][1]))
            vector_obs.append(vector_to_direction(raw_obs['right_team_direction'][i%11][0], raw_obs['right_team_direction'][i%11][1], ret_text=False))

    if ret_type == 'dict':
        return finial_dic
    elif ret_type == 'vector':
        return np.array(vector_obs)
    elif ret_type == 'both':
        return finial_dic, np.array(vector_obs)



def imaginary_data_observation(observation, raw_obs, step, block_mode=None, ret_type='dict', TODO_missing=False):
    finial_dic = {}
    vector_obs = []
    # Stinky action
    sticky_action_list = raw_obs['sticky_actions']
    
    if sticky_action_list.size > 0 and np.any(sticky_action_list != 0):
        indices = np.where(sticky_action_list == 1)[0]
        selected_actions = [sticky_list[idx] for idx in indices]
        finial_dic["sticky_actions"] = selected_actions
    else:
        finial_dic["sticky_actions"] = []
    vector_obs.extend(sticky_action_list)
        
    # Handle game mode
    mode_index = np.where(observation[108:115] == 1)[0][0]
    finial_dic["game_mode"] = game_modes[mode_index]
    finial_dic["score"] = raw_obs['score']
    vector_obs.extend(raw_obs['score'])

    time = step * 1.8 # 3000 timpstep = 90 mins
    min = int(time // 60)
    sec = int(time % 60)
    finial_dic["step"] = step
    finial_dic["time"] = f"{min} minutes {sec} seconds"
    
    # active_player_index = np.where(observation[97:108] == 1)[0][0]
    active_player_index = raw_obs['active']
    finial_dic["active_player"] = active_player_index
    finial_dic["active_player_role"]= role_list[active_player_index]
    vector_obs.append(active_player_index)
    
    
    # Handle ball ownership
    team_ownership = ["no one", "Left team", "Right team"]
    team_owner_index = np.where(observation[94:97] == 1)[0][0]
    player_owner_index = np.where(observation[-22:] == 1)[0][0] if np.any(observation[-22:] == 1) else -1
    player_owner_index = player_owner_index % 11 
    
    
    finial_dic["ball_ownership"] = team_owner_index
    vector_obs.append(team_owner_index)
    if team_owner_index == 1:
        finial_dic["ball_ownership_player"] = player_owner_index
    elif team_owner_index == 2:
        finial_dic["ball_ownership_player"] = 11 + player_owner_index
    else:
        finial_dic["ball_ownership_player"] = -1
    if TODO_missing:
        vector_obs.append(finial_dic["ball_ownership_player"])

    # Handle ball position and direction
    ball_zone = get_zons_240_list(observation[88], observation[89])
    ball_direction = vector_to_direction(observation[91], observation[92])
    
    finial_dic["ball_zone"] = ball_zone
    finial_dic["ball_direction"] = ball_direction
    vector_obs.extend(ball_zone)
    vector_obs.append(vector_to_direction(observation[91], observation[92], ret_text=False))
    for i in range(22):
        
        # wheather 11v1 or 11v11
        
        
        
        key = f"player_{i}"
        if i < 11:
            finial_dic[key] = {
                "team": "Left",
                "role": role_list[i%11],
                "zone": get_zons_240_list(observation[2*i], observation[2*i+1]),
                "direction": vector_to_direction(observation[22 + 2*i], observation[22 + 2*i + 1]),
            }
            vector_obs.extend(get_zons_240_list(observation[2*i], observation[2*i+1]))
            vector_obs.append(vector_to_direction(observation[22 + 2*i], observation[22 + 2*i + 1], ret_text=False))
        else:
            finial_dic[key] = {
                "team": "Right",
                "role": role_list[i%11],
                "zone": get_zons_240_list(observation[22 + 2*i], observation[22 + 2*i + 1]),
                "direction": vector_to_direction(observation[44 + 2*i], observation[44 + 2*i + 1]),
            }
            vector_obs.extend(get_zons_240_list(observation[22 + 2*i], observation[22 + 2*i + 1]))
            vector_obs.append(vector_to_direction(observation[44 + 2*i], observation[44 + 2*i + 1], ret_text=False))

    if ret_type == 'dict':
        return finial_dic
    elif ret_type == 'vector':
        return np.array(vector_obs)
    elif ret_type == 'both':
        return finial_dic, np.array(vector_obs)


def long_obs_list_to_text(obs_list):
    """
    obs_list: a dictionary of obs, key is the timestep, value is the obs
    """
    text = ""
    for timestep, obs in obs_list.items():
        text += observation_to_text_human(obs)
        text += "\n"
    return text

def format_code(text):
    # Split text.
    lines = text.split('", "')
    
    # Initialize formatted code.
    formatted_code = ""
    
    # Current indentation level.
    indent_level = 0

    # Process each line.
    for line in lines:
        # Check whether indentation should increase or decrease.
        indent_increase = line.count('{')
        indent_decrease = line.count('}')

        # Legacy implementation note.
        indent_level -= indent_decrease

        # Legacy implementation note.
        formatted_code += "    " * indent_level + line + "\n"

        # Legacy implementation note.
        indent_level += indent_increase

    return formatted_code


def npz_extractor(npz_file_path):
    file = npz_file_path
    try:
        res = np.load(file, allow_pickle=True)
        if 'obs_before_modified_by_acs' not in res.keys() or 'action' not in res.keys():
            print("skip this file", file)
            return None     
    except zipfile.BadZipFile as e:
        print("BadZipFile, skip this file", file)
        print("error", e)
        return None 
    return res
    