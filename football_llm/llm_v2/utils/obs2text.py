import numpy as np

PLAYER_ROLES = ['GoalKeeper', 'Forward', 'Forward', 'Defender', 'Defender', 'Defender', 'Defender', 'Midfielder', 'Midfielder', 'Midfielders', 'Forward']

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
    zone_y = int((y + 0.42) // block_hight) + 1
    
    # Zone(1,1) is the left bottom corner
    # Zone(20,12) is the right top corner
    
    return f'Zone({zone_x},{zone_y})'


def vector_to_direction(x, y):
    angle = np.arctan2(y, x)
    directions = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    idx = round(angle / (2 * np.pi / len(directions))) % len(directions)
    return directions[idx]

def observation_to_text_raw(observation):
    text = ""
    # Handle player positions and directions
    for i in range(11):
        text += f"Left team player {i+1} is at ({observation[2*i]}, {observation[2*i+1]}) with direction ({observation[22 + 2*i]}, {observation[22 + 2*i + 1]}). "
        text += f"Right team player {i+1} is at ({observation[44 + 2*i]}, {observation[44 + 2*i + 1]}) with direction ({observation[66 + 2*i]}, {observation[66 + 2*i + 1]}). "

    # Handle ball position and direction
    text += f"The ball is at ({observation[88]}, {observation[89]}, {observation[90]}) with direction ({observation[91]}, {observation[92]}, {observation[93]}). "

    # Handle ball ownership
    ownership = ["no one", "left team", "right team"]
    owner_index = np.where(observation[94:97] == 1)[0][0]
    text += f"The ball is currently controlled by {ownership[owner_index]}. "

    # Handle active player
    active_player = np.where(observation[97:108] == 1)[0][0] + 1
    text += f"The current active player is number {active_player}. "

    # Handle game mode
    game_modes = ["e_GameMode_Normal", "e_GameMode_KickOff", "e_GameMode_GoalKick", "e_GameMode_FreeKick", "e_GameMode_Corner", "e_GameMode_ThrowIn", "e_GameMode_Penalty"]  # Replace with actual game modes
    mode_index = np.where(observation[108:115] == 1)[0][0]
    text += f"The current game mode is {game_modes[mode_index]}. "

    return text


def observation_to_text_human(observation,step, block_mode=None):
    finial_text = ""
    
    # Handle game mode
    game_modes = ["Noramal", "KickOff", "GoalKick", "FreeKick", "Corner", "hrowIn", "Penalty"]  # Replace with actual game modes
    mode_index = np.where(observation[108:115] == 1)[0][0]
    if mode_index == 0:
        game_mode_text = f" the game is going on. "
    else:
        game_mode_text = f" the game is in {game_modes[mode_index]}. "
    time = step * 1.8 # 3000 timpstep = 90 mins
    min = int(time // 60)
    sec = int(time % 60)
    time_text = f"The current time is {min} minutes {sec} seconds. "
    
    # Handle active player
    active_player = np.where(observation[97:108] == 1)[0][0] + 1
    active_player_text = f"You as the football coach are instructing Left team: Player {active_player} to make the next action, his duty is {PLAYER_ROLES[active_player-1]}. "
    
    # Handle ball ownership
    team_ownership = ["no one", "Left team", "Right team"]
    tema_owner_index = np.where(observation[94:97] == 1)[0][0]
    
    player_owner_index = np.where(observation[-22:] == 1)[0][0] if np.any(observation[-22:] == 1) else -1
    player_owner_index = player_owner_index % 11 + 1
    
    if tema_owner_index == 0:
        ownership_text = f"Currently, the ball is not controlled by any team. Try to get the ball."
    elif tema_owner_index == 1:
        ownership_text = f"Currently, your Left team is controlling the ball, by Player {player_owner_index}. Try to make a good action to attack and offense. "
    else:
        ownership_text = f"Currently, the ball is controlled by {team_ownership[tema_owner_index]}: Player {player_owner_index}. Try to make a good action to defense. "
    
    
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
            L_player_text += f"Left team: " + p + f"{i+1} is in the {left_player_zone} zone facing {left_player_direction}. " + "\n"
            R_player_text += f"Right team: " + p + f"{i+1} is in the {right_player_zone} zone facing {right_player_direction}. " + "\n"
    
    else:
        # Handle ball position and direction
        ball_zone = get_specific_zone_term(observation[88], observation[89])
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
            left_player_zone = get_specific_zone_term(observation[2*i], observation[2*i+1])
            left_player_direction = vector_to_direction(observation[22 + 2*i], observation[22 + 2*i + 1])
            right_player_zone = get_specific_zone_term(observation[44 + 2*i], observation[44 + 2*i + 1])
            right_player_direction = vector_to_direction(observation[66 + 2*i], observation[66 + 2*i + 1])
            L_player_text += f"Left team: " + p + f"{i+1} is in the {left_player_zone} zone facing {left_player_direction}. " + "\n"
            R_player_text += f"Right team: " + p + f"{i+1} is in the {right_player_zone} zone facing {right_player_direction}. " + "\n"
    if False:
        finial_text = time_text + "\n" + game_mode_text  + "\n" \
                                +  ownership_text + "\n" + ball_text + "\n" + "Left Team (our team): "\
                                +  "\n" + L_player_text + "\n" + "Right Team (opponent team): " \
                                +  "\n"+ R_player_text
    else:
        finial_text = time_text + "\n" + game_mode_text + "\n" + active_player_text + "\n" \
                                +  ownership_text + "\n" + ball_text + "\n" + "Left Team (our team): "\
                                +  "\n" + L_player_text + "\n" + "Right Team (opponent team): " \
                                +  "\n"+ R_player_text

    return finial_text


def long_obs_list_to_text(obs_list):
    """
    obs_list: a dictionary of obs, key is the timestep, value is the obs
    """
    text = ""
    for timestep, obs in obs_list.items():
        text += observation_to_text_human(obs)
        text += "\n"
    return text
