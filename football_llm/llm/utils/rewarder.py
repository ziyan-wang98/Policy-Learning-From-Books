import numpy as np
import yaml
import os

class Rewarder:
    def __init__(self, yaml_name="reward.yaml") -> None:
        
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, yaml_name)
        reward_config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        
        self.player_last_hold_ball = -1
        self.last_ball_owned_team = -1
        self.reward_config = reward_config["reward_config"]
        
        self.fix =  reward_config["reward_config"]['fix']
        
        self.cumulative_shot_reward = None
        self.offense_r_encoder = attack_r()
        self.defense_r_encoder = defense_r()
        self.default_r_encoder = default_r()
        
    def calc_reward_v2(self, rew, im_next_obs, prev_img_obs, im_action):
            
        if im_next_obs["ball_ownership"] == 1:
            self.player_last_hold_ball = im_next_obs["ball_ownership_player"]
        
        if im_next_obs["ball_ownership"] != 0:
            self.last_ball_owned_team = im_next_obs["ball_ownership"]

        if self.reward_config is None:  # default reward shaping
            raise NotImplementedError
            reward = (
                5 * win_reward(obs)
                + 5.0 * preprocess_score(obs, rew, self.player_last_hold_ball)
                + 0.03 * ball_position_reward(obs, self.player_last_hold_ball)
                + yellow_reward(prev_obs, obs)
                - 0.003 * min_dist_reward(obs)
            )

        else:
            active_player = im_next_obs["active_player"]
            if self.cumulative_shot_reward is None:
                self.cumulative_shot_reward = [0] * 11
            shot_reward = [0] * 11
            
            if prev_img_obs["score"][1] < im_next_obs["score"][1]:
                self.cumulative_shot_reward = [0] * 11
            elif prev_img_obs["score"][0] < im_next_obs["score"][0]:
                shot_reward[active_player] = self.cumulative_shot_reward[active_player]
                self.cumulative_shot_reward[active_player] = 0
            else:
                if im_action == 12:
                    self.cumulative_shot_reward[active_player] += 1

            single_shot_reward = [0] * 11
            if im_action == 12:
                single_shot_reward[im_next_obs["active_player"]] += 1


        
            reward = (
                self.reward_config["win_reward"] * win_reward_v2(im_next_obs)
                + self.reward_config["preprocess_score"]
                * preprocess_score_v2(im_next_obs, rew, self.player_last_hold_ball)
                + self.reward_config["ball_position_reward"]
                * ball_position_reward_v2(im_next_obs, self.player_last_hold_ball, self.fix)
                - self.reward_config["min_dist_reward"]
                * min_dist_individual_reward_v2(im_next_obs)
                + self.reward_config["goal_reward"] * goal_reward(prev_img_obs, im_next_obs)
                + self.reward_config["lost_ball_reward"]
                * lost_ball_reward_v2(prev_img_obs, im_next_obs, self.player_last_hold_ball)
                + self.reward_config["player_move_reward"]
                * player_move_reward_v2(prev_img_obs, im_next_obs)
                + self.reward_config["dist_goal_to_line"] * dist_goal_to_line_v2(im_next_obs)
                + self.reward_config["shot_reward"] * single_shot_reward[im_next_obs["active_player"]]
                + self.reward_config["role_based_r"] * role_based_r_v2(prev_img_obs, im_next_obs)
                + self.reward_config["pure_goal"] * pure_goal(prev_img_obs, im_next_obs)
                + self.reward_config["pure_lose_goal"] * pure_lose_goal(prev_img_obs, im_next_obs)
                + self.reward_config["control_ball_reward"] * hold_ball_reward_v2(im_next_obs)
                # + 10 * shot_reward[active_player]
            )

            adaptive_attack_defence = self.reward_config.get(
                "adaptive_attack_defence", 0
            )
            if adaptive_attack_defence:
                if im_next_obs["score"][0] >= im_next_obs["score"][1] + 2:
                    if reward > 0:
                        reward *= 0.5
                elif im_next_obs["score"][0] >= im_next_obs["score"][1] + 4:
                    if reward > 0:
                        reward *= 0.2
                elif im_next_obs["score"][0] >= im_next_obs["score"][1] + 5:
                    if reward > 0:
                        reward = min(reward, 0)
                elif im_next_obs["score"][0] <= im_next_obs["score"][1] - 2:
                    if reward < 0:
                        reward *= 0.5
                elif im_next_obs["score"][0] <= im_next_obs["score"][1] - 4:
                    if reward < 0:
                        reward *= 0.2
                elif im_next_obs["score"][0] <= im_next_obs["score"][1] - 5:
                    if reward < 0:
                        reward = max(reward, 0)
            # return calc_skilled_attack_reward(rew, prev_obs, obs) + shot_reward[obs['active']]

            # goal_r = goal_reward(prev_obs, obs)
            # if goal_r > 0:
            #     shot_reward_c = 5
            # else:
            #     shot_reward_c = 0

        return reward
        
        

    def calc_reward(self, rew, obs, prev_obs, action):
        """
        'score', 'left_team_active', 'right_team_roles', 'right_team_active',
        'right_team_yellow_card', 'left_team_direction', 'right_team_direction',
        'ball_owned_player', 'ball_owned_team', 'right_team_tired_factor', 'steps_left',
        'right_team', 'left_team_yellow_card', 'left_team_tired_factor', 'game_mode',
        'left_team_roles', 'ball', 'ball_rotation', 'left_team', 'ball_direction',
        'designated', 'active', 'sticky_actions'
        """


        # return self.offense_r_encoder.r(obs, prev_obs, action, id) \
        #        + self.defense_r_encoder.r(obs, prev_obs, action, id) \
        #         + self.default_r_encoder.r(obs,  prev_obs)

        # return calc_skilled_attack_reward(rew, prev_obs, obs)
        if obs["ball_owned_team"] == 0:
            self.player_last_hold_ball = obs["ball_owned_player"]

        if obs["ball_owned_team"] != -1:
            self.last_ball_owned_team = obs["ball_owned_team"]

        if self.reward_config is None:  # default reward shaping
            raise NotImplementedError
            reward = (
                5 * win_reward(obs)
                + 5.0 * preprocess_score(obs, rew, self.player_last_hold_ball)
                + 0.03 * ball_position_reward(obs, self.player_last_hold_ball)
                + yellow_reward(prev_obs, obs)
                - 0.003 * min_dist_reward(obs)
            )

        else:

            active_player = obs["active"]

            if self.cumulative_shot_reward is None:
                self.cumulative_shot_reward = [0] * len(obs["left_team_roles"])

            shot_reward = [0] * len(obs["left_team_roles"])

            if prev_obs["score"][1] < obs["score"][1]:
                self.cumulative_shot_reward = [0] * len(obs["left_team_roles"])
            elif prev_obs["score"][0] < obs["score"][0]:
                shot_reward[active_player] = self.cumulative_shot_reward[active_player]
                self.cumulative_shot_reward[active_player] = 0
            else:
                if action == 12:
                    self.cumulative_shot_reward[active_player] += 1

            single_shot_reward = [0] * len(obs["left_team_roles"])
            if action == 12:
                single_shot_reward[obs["active"]] += 1

            reward = (
                self.reward_config["win_reward"] * win_reward(obs)
                + self.reward_config["preprocess_score"]
                * preprocess_score(obs, rew, self.player_last_hold_ball)
                + self.reward_config["ball_position_reward"]
                * ball_position_reward(obs, self.player_last_hold_ball)
                + self.reward_config["yellow_reward"] * yellow_reward(prev_obs, obs)
                - self.reward_config["min_dist_reward"]
                * min_dist_individual_reward(obs)
                + self.reward_config["goal_reward"] * goal_reward(prev_obs, obs)
                + self.reward_config["lost_ball_reward"]
                * lost_ball_reward(prev_obs, obs, self.player_last_hold_ball)
                + self.reward_config["player_move_reward"]
                * player_move_reward(prev_obs, obs)
                + self.reward_config["dist_goal_to_line"] * dist_goal_to_line(obs)
                + self.reward_config["shot_reward"] * single_shot_reward[obs["active"]]
                + self.reward_config["role_based_r"] * role_based_r(prev_obs, obs)
                + self.reward_config["pure_goal"] * pure_goal(prev_obs, obs)
                + self.reward_config["pure_lose_goal"] * pure_lose_goal(prev_obs, obs)
                # + 10 * shot_reward[active_player]
            )

            adaptive_attack_defence = self.reward_config.get(
                "adaptive_attack_defence", 0
            )
            if adaptive_attack_defence:
                if obs["score"][0] >= obs["score"][1] + 2:
                    if reward > 0:
                        reward *= 0.5
                elif obs["score"][0] >= obs["score"][1] + 4:
                    if reward > 0:
                        reward *= 0.2
                elif obs["score"][0] >= obs["score"][1] + 5:
                    if reward > 0:
                        reward = min(reward, 0)
                elif obs["score"][0] <= obs["score"][1] - 2:
                    if reward < 0:
                        reward *= 0.5
                elif obs["score"][0] <= obs["score"][1] - 4:
                    if reward < 0:
                        reward *= 0.2
                elif obs["score"][0] <= obs["score"][1] - 5:
                    if reward < 0:
                        reward = max(reward, 0)
            # return calc_skilled_attack_reward(rew, prev_obs, obs) + shot_reward[obs['active']]

            # goal_r = goal_reward(prev_obs, obs)
            # if goal_r > 0:
            #     shot_reward_c = 5
            # else:
            #     shot_reward_c = 0

        return reward

def role_based_r_v2(pre_obs, obs):
    team_goal_weight = {0: 0.2, 1: 0.2, 2: 0.5, 3: 0.5, 5: 0.7, 6: 1, 7: 1, 9: 1}
    team_lose_weight = {0: 1, 1: 1, 2: 0.7, 3: 0.7, 5: 0.5, 6: 0.2, 7: 0.2, 9: 0.2}

    player_role_list = [0, 7, 9, 2, 1, 1, 3, 5, 5, 5, 6]
    current_role = player_role_list[obs["active_player"]]

    r = 0

    opponent_score_pre = pre_obs["score"][1]
    opponent_score_after = obs["score"][1]
    if opponent_score_after > opponent_score_pre:
        r -= team_lose_weight[current_role]

    current_score_pre = pre_obs["score"][0]
    current_score_after = obs["score"][0]
    if current_score_after > current_score_pre:
        r += team_goal_weight[current_role]

    return r



def role_based_r(pre_obs, obs):
    team_goal_weight = {0: 0.2, 1: 0.2, 2: 0.5, 3: 0.5, 5: 0.7, 6: 1, 7: 1, 9: 1}
    team_lose_weight = {0: 1, 1: 1, 2: 0.7, 3: 0.7, 5: 0.5, 6: 0.2, 7: 0.2, 9: 0.2}

    current_role = obs["left_team_roles"][obs["active"]]

    r = 0

    opponent_score_pre = pre_obs["score"][1]
    opponent_score_after = obs["score"][1]
    if opponent_score_after > opponent_score_pre:
        r -= team_lose_weight[current_role]

    current_score_pre = pre_obs["score"][0]
    current_score_after = obs["score"][0]
    if current_score_after > current_score_pre:
        r += team_goal_weight[current_role]

    return r


def pure_goal(pre_obs, obs):
    r = 0
    current_score_pre = pre_obs["score"][0]
    current_score_after = obs["score"][0]
    if current_score_after > current_score_pre:
        r += 1.0
    return r


def pure_lose_goal(pre_obs, obs):
    penalty = 0.0
    opponent_score_pre = pre_obs["score"][1]
    opponent_score_after = obs["score"][1]
    if opponent_score_after > opponent_score_pre:
        penalty -= 1.0

    return penalty


class attack_r:
    def __init__(self):
        self.lost_ball_penalty = -1
        self.lost_ball_recording = False

        self.steal_ball_reward = 1
        self.steal_ball_recording = False

        self.passing_flag = [False, False, False, False, False]
        self.bad_pass_penalty = -1
        self.good_pass_reward = 1

        self.single_shot_reward = 0
        self.cumulative_shot_reward = None
        self.cumulative_shot_reward_factor = 1

        self.pass_reward_list = None

        self.check_offside = False

    def r(self, obs, prev_obs, action, id):

        if "team_1" in id:
            return 0

        lost_ball_r = self.lost_possession(obs, prev_obs, current_player=obs["active"])
        pass_r = self.pass_reward(obs, action)
        shot_r = self.shot_reward(
            obs, prev_obs, current_player=obs["active"], player_action=action
        )

        return lost_ball_r + pass_r + shot_r

    def lost_possession(self, obs, prev_obs, current_player):
        """
        this will include all scenario losing the ball, including being intercepted, out-of-bound,
        offside, shot gets blocked by opponent goalkeeper
        """
        r = 0
        if prev_obs["score"][0] < obs["score"][0]:
            self.lost_ball_recording = (
                False  # change of ball ownership due to ours goal
            )
            return r

        # if obs['game_mode'] == 3:                     #change mainly dur to we offside, here we penalise offside move
        #     self.lost_ball_recording = False
        #     return r

        if self.lost_ball_recording:
            if obs["ball_owned_team"] == -1:
                pass
            elif obs["ball_owned_team"] == 0:  # back to our team
                self.lost_ball_recording = False
                # can add reward here
            else:  # opponent own it
                if self.last_hold_player == 0:  # our goalkeeper lose the ball
                    self.lost_ball_recording = False

                if obs["active"] == self.last_hold_player:
                    self.lost_ball_recording = False
                    r = self.lost_ball_penalty

        if prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 1:
            if (
                current_player == prev_obs["ball_owned_player"]
            ):  # current player is the last holding player
                r = self.lost_ball_penalty

        elif prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == -1:
            self.lost_ball_recording = True
            self.last_hold_player = prev_obs["ball_owned_player"]

        return r

    def pass_reward(self, obs, player_action):
        r = 0
        for i, p in enumerate(self.passing_flag):
            if p:  # if passing
                # if ball_owned_team == 0:
                #     if obs['ball_owned_player'] == current_player:
                #         pass
                #     elif (obs['ball_owned_player'] != current_player
                #           and current_player == i):
                #         r += self.good_pass_reward
                #         self.passing_flag[i] = False
                #
                # else:
                #     if ball_owned_team == -1:
                #         pass
                #     elif ball_owned_team == 1 and current_player == i:
                #         r += self.bad_pass_penalty
                #         self.passing_flag[i] = False

                if obs["ball_owned_team"] == 0 and obs["active"] == i:
                    pass
                else:
                    if obs["ball_owned_team"] == 0 and obs["ball_owned_player"] != i:
                        self.passing_flag[i] = False
                        # good pass reward?
                    elif obs["ball_owned_team"] == -1:
                        pass
                    elif obs["ball_owned_team"] == 1 and obs["active"] == i:
                        r += self.bad_pass_penalty
                        self.passing_flag[i] = False

        if player_action == 9 or player_action == 10 or player_action == 11:
            if (
                obs["ball_owned_team"] == 0
                and not self.passing_flag[obs["active"]]
                and (obs["active"] == obs["ball_owned_player"])
            ):

                self.passing_flag[obs["active"]] = True

        return r

    def goal_pass_reward(self, obs, prev_obs, action):
        """
        reward passing only after goals
        """

        if self.pass_reward_list is None:
            self.pass_reward_list = [0] * len(obs["left_team_roles"])
        pass_reward = [0] * len(obs["left_team_roles"])

        if (
            prev_obs["score"][1] < obs["score"][1]
        ):  # opponent goal, clear the pass reward
            self.pass_reward_list = [0] * len(obs["left_team_roles"])
        elif prev_obs["score"][0] < obs["score"][0]:
            pass_reward[obs["active"]] = self.pass_reward_list[obs["active"]]
            self.pass_reward_list[obs["active"]] = 0
        else:
            if action == 9 or action == 10 or action == 11:
                self.pass_reward_list[obs["active"]] += 1

        return pass_reward[obs["active"]]

    def shot_reward(self, obs, prev_obs, current_player, player_action):
        """
        reward shotting after goals
        """

        r = 0
        if self.cumulative_shot_reward is None:
            self.cumulative_shot_reward = [0] * len(obs["left_team_roles"])

        shot_reward = [0] * len(obs["left_team_roles"])

        if prev_obs["score"][1] < obs["score"][1]:
            self.cumulative_shot_reward = [0] * len(obs["left_team_roles"])
        elif prev_obs["score"][0] < obs["score"][0]:
            shot_reward[current_player] = self.cumulative_shot_reward[current_player]
            self.cumulative_shot_reward[current_player] = 0
        else:
            if player_action == 12:
                self.cumulative_shot_reward[current_player] += 1

                r += self.single_shot_reward

        r += shot_reward[current_player]

        return r

    def offside_pass_penalty(self, obs, prev_obs, current_player, player_action):
        # when agent pass and at least one of the teammate is at offside position, start checking, if gamemode has changed, highly likely offside
        offside_r = 0

        def is_offside(obs):
            our_team_offside = [0] * obs["left_team_roles"]
            second_last_opponent_x = sorted(obs["right_team"][:, 0])[-2]
            ball_x = obs["ball"][0]
            for player_id, left_player_pos in enumerate(obs["left_team"]):
                if (
                    obs["game_mode"] == 0
                    and left_player_pos[0] > ball_x
                    and left_player_pos[0] > second_last_opponent_x
                ):
                    our_team_offside[player_id] = 1
            return sum(our_team_offside)

        if self.check_offside:
            if obs["game_mode"] == 3:
                offside_r -= 1
                self.check_offside = False
            else:
                if obs["ball_owned_team"] == 0:
                    self.check_offside = False
                elif obs["ball_owned_team"] == 1:
                    self.check_offside = False
                else:
                    pass

        if player_action == 9 or player_action == 10 or player_action == 11:
            if is_offside(obs):
                self.check_offside = True


class defense_r:
    def __init__(self):
        self.steal_ball_reward = 1
        self.steal_ball_recording = False

    def r(self, obs, prev_obs, action, id):

        if "team_1" in id:
            return 0

        steal_ball_reward = self.get_possession(obs, prev_obs)
        min_dist_reward = self.min_dist_reward(obs)

        return steal_ball_reward + min_dist_reward

    def get_possession(self, obs, prev_obs):  # get possessing
        """
        this include some scenarios getting ball possession including intercepting, opponent out-of-bound,
        we ignore when our goalkeeper steal the ball as we dont want them to have too much pressure, and we ignore offside here
        """

        r = 0

        if prev_obs["score"][1] < obs["score"][1]:
            self.steal_ball_recording = (
                False  # change of ball ownership due to opponent's goal
            )
            return r

        if (
            obs["game_mode"] == 3
        ):  # change of ball ownership from free kick, this is likely due to opponent offside
            self.steal_ball_recording = (
                False  # change of ball ownership due to opponent's goal
            )
            return r

        if self.steal_ball_recording:
            if obs["ball_owned_team"] == -1:
                pass
            elif obs["ball_owned_team"] == 1:
                self.steal_ball_recording = False
            elif (
                obs["ball_owned_team"] == 0 and obs["ball_owned_player"] == 0
            ):  # our goalkeeper intercept the ball
                self.steal_ball_recording = False
            elif (
                obs["ball_owned_team"] == 0
                and obs["ball_owned_player"] != 0
                and obs["active"] == obs["ball_owned_player"]
            ):
                self.steal_ball_recording = False
                r += (
                    self.steal_ball_reward
                )  # only reward the agent stealing the ball (can we make it team reward?)

        if (
            prev_obs["ball_owned_team"] == 1 and prev_obs["ball_owned_player"] != 0
        ) and obs["ball_owned_team"] == 0:
            if obs["active"] == obs["ball_owned_player"]:
                r += self.steal_ball_reward

        elif (
            prev_obs["ball_owned_team"] == 1 and prev_obs["ball_owned_player"] != 0
        ) and obs["ball_owned_team"] == -1:
            self.steal_ball_recording = True
        else:
            pass

        return r

    def min_dist_reward(self, obs):

        if obs["ball_owned_team"] != 0:
            ball_position = np.array(obs["ball"][:2])
            left_team_position = obs["left_team"][1:]
            left_team_dist2ball = np.linalg.norm(
                left_team_position - ball_position, axis=1
            )
            min_dist2ball = np.min(left_team_dist2ball)
        else:
            min_dist2ball = 0.0

        return min_dist2ball


class default_r:
    def __init__(self):
        self.player_last_hold_ball = -1

    def r(self, obs, prev_obs):
        if obs["ball_owned_team"] == 0:
            self.player_last_hold_ball = obs["ball_owned_player"]

        win_reward = self.win_reward(obs)
        goal_reward = self.goal_reward(prev_obs, obs)
        yellow_reward = self.yellow_reward(prev_obs, obs)
        ball_pos_reward = self.ball_position_reward(obs, self.player_last_hold_ball)
        hold_ball_reward = self.hold_ball_reward(obs)
        dist_to_goal = self.dist_goal_to_line(obs)

        return (
            win_reward
            + goal_reward
            + yellow_reward
            + ball_pos_reward
            + hold_ball_reward
            + dist_to_goal
        )

    def win_reward(self, obs):
        win_reward = 0.0
        if obs["steps_left"] == 0:
            [my_score, opponent_score] = obs["score"]
            if my_score > opponent_score:
                win_reward = my_score - opponent_score
        return win_reward

    def goal_reward(self, prev_obs, obs):
        penalty = 0.0
        opponent_score_pre = prev_obs["score"][1]
        opponent_score_after = obs["score"][1]
        if opponent_score_after > opponent_score_pre:
            penalty -= 1.0

        current_score_pre = prev_obs["score"][0]
        current_score_after = obs["score"][0]
        if current_score_after > current_score_pre:
            penalty += 1.0

        return penalty

    def yellow_reward(self, prev_obs, obs):
        left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
            prev_obs["left_team_yellow_card"]
        )
        right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
            prev_obs["right_team_yellow_card"]
        )
        yellow_r = right_yellow - left_yellow
        return yellow_r

    def ball_position_reward(self, obs, player_last_hold_ball):
        ball_x, ball_y, ball_z = obs["ball"]
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42
        ball_position_r = 0.0
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
            -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):  # in our penalty area
            ball_position_r = -2.0
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):  #
            ball_position_r = -1.0
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):
            ball_position_r = 0.0
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (
            -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            ball_position_r = 2.0
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):
            ball_position_r = 1.0
        else:
            ball_position_r = 0.0

        if obs["ball_owned_team"] == 0:
            if not obs["active"] == player_last_hold_ball:
                ball_position_r *= 0.5

        return ball_position_r

    def hold_ball_reward(self, obs):
        r = 0.0
        if obs["ball_owned_team"] == 0:
            r += 0.001
        elif obs["ball_owned_team"] == 1:
            r -= 0.001
        else:
            pass
        return r

    def dist_goal_to_line(self, obs):
        ball_position = np.array(obs["ball"][:2])
        dist_goal_to_line = np.linalg.norm(np.array([-1, 0]) - ball_position, axis=0)
        return dist_goal_to_line


def hold_ball_reward_v2(obs):
    r = 0.0
    if obs["ball_ownership"] == 0:
        r += 0.05
    elif obs["ball_ownership"] == 1:
        r -= 0.001
    else:
        pass
    return r



def hold_ball_reward(obs):
    r = 0.0
    if obs["ball_owned_team"] == 0:
        r += 0.001
    elif obs["ball_owned_team"] == 1:
        r -= 0.001
    else:
        pass
    return r


def dist_goal_to_line_v2(obs):
    ball_position = np.array(obs["ball_zone"])
    dist_goal_to_line = np.linalg.norm(np.array([1, 6]) - ball_position, axis=0)
    return dist_goal_to_line * 0.1

def dist_goal_to_line(obs):
    ball_position = np.array(obs["ball"][:2])
    dist_goal_to_line = np.linalg.norm(np.array([-1, 0]) - ball_position, axis=0)
    return dist_goal_to_line

def player_move_reward_v2(prev_obs, obs):
    prev_zone_list = [] 
    cur_zone_list = []
    for i in range(11):
        prev_zone = prev_obs[f'player_{i}']['zone']
        cur_zone = obs[f'player_{i}']['zone']
        prev_zone_list.append(prev_zone)
        cur_zone_list.append(cur_zone)
    prev_zone_array = np.array(prev_zone_list)
    cur_zone_array = np.array(cur_zone_list)
    left_position_move = np.sum((prev_zone_array - cur_zone_array) ** 2)
    return left_position_move

def player_move_reward(prev_obs, obs):
    left_position_move = np.sum((prev_obs["left_team"] - obs["left_team"]) ** 2)
    return left_position_move

def ball_possession_reward(prev_obs, obs, player_last_hold_ball):
    if prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 1:
        if obs["active"] == player_last_hold_ball:
            return -0.2
    elif prev_obs["ball_owned_team"] == 1 and obs["ball_owned_team"] == 2:
        if obs["active"] == player_last_hold_ball:
            return 0.2
    else:
        return 0


def goal_reward(pre_obs, obs):
    penalty = 0.0
    opponent_score_pre = pre_obs["score"][1]
    opponent_score_after = obs["score"][1]
    if opponent_score_after > opponent_score_pre:
        penalty -= 1.0

    current_score_pre = pre_obs["score"][0]
    current_score_after = obs["score"][0]
    if current_score_after > current_score_pre:
        penalty += 1.0

    return penalty


def preprocess_score_v2(obs, rew_signal, player_last_hold_ball):
    if rew_signal > 0:
        factor = 1.0 if obs["active_player"] == player_last_hold_ball else 0.3
    else:
        return rew_signal
    return rew_signal * factor

def preprocess_score(obs, rew_signal, player_last_hold_ball):
    if rew_signal > 0:
        factor = 1.0 if obs["active"] == player_last_hold_ball else 0.3
    else:
        return rew_signal
    return rew_signal * factor



def lost_ball_reward_v2(prev_obs, obs, player_last_hold_ball):
    if prev_obs["ball_ownership"] == 1 and obs["ball_ownership"] == 2:
        if obs["active_player"] == player_last_hold_ball:
            return -0.5
    return -0.1


def lost_ball_reward(prev_obs, obs, player_last_hold_ball):
    if prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 1:
        if obs["active"] == player_last_hold_ball:
            return -0.5
    return -0.1


def win_reward_v2(obs):
    win_reward = 0.0
    # print(f"steps left: {obs['steps_left']}")
   
        # print("STEPS LEFT == 0!")
    [my_score, opponent_score] = obs["score"]
    # if my_score > opponent_score:
    win_reward = my_score - opponent_score
    return win_reward

def win_reward(obs):
    win_reward = 0.0
    # print(f"steps left: {obs['steps_left']}")
    if obs["steps_left"] == 0:
        # print("STEPS LEFT == 0!")
        [my_score, opponent_score] = obs["score"]
        # if my_score > opponent_score:
        win_reward = my_score - opponent_score
    return win_reward


def min_dist_reward(obs):
    if obs["ball_owned_team"] != 0:
        ball_position = np.array(obs["ball"][:2])
        left_team_position = obs["left_team"][1:]
        left_team_dist2ball = np.linalg.norm(left_team_position - ball_position, axis=1)
        min_dist2ball = np.min(left_team_dist2ball)
    else:
        min_dist2ball = 0.0
    return min_dist2ball


def min_dist_individual_reward_v2(obs):
    block_width = 2.0 / 20  # 0.1
    block_height = 0.84 / 12  # 0.07

    if obs["ball_ownership"] != 1:
        ball_zone = np.array(obs["ball_zone"])

        ball_position = (ball_zone - 1) * np.array([block_width, block_height])
        ball_position[0] += block_width / 2  
        ball_position[1] += block_height / 2

        dist2ball_list = []
        for i in range(11):
            player_zone = np.array(obs[f'player_{i}']['zone'])
     
            player_position = (player_zone - 1) * np.array([block_width, block_height])
            player_position[0] += block_width / 2 
            player_position[1] += block_height / 2

    
            dist2ball = np.linalg.norm(player_position - ball_position)
            dist2ball_list.append(dist2ball)

        min_dist2ball = np.min(dist2ball_list)
        min_player_id = np.argmin(dist2ball_list)
        
        if obs["active_player"] == min_player_id:
            return min_dist2ball
        else:
            return 0.0
    else:
        return  0.0




def min_dist_individual_reward(obs):
    if obs["ball_owned_team"] != 0:
        ball_position = np.array(obs["ball"][:2])
        left_team_position = obs["left_team"][1:]
        left_team_dist2ball = np.linalg.norm(left_team_position - ball_position, axis=1)
        # print(f"left team dist to ball: {left_team_dist2ball}")
        min_dist2ball = np.min(left_team_dist2ball)
        min_player_id = (
            np.argmin(left_team_dist2ball) + 1
        )  # int(np.where(left_team_dist2ball == min_dist2ball)[0])
        if obs["active"] == min_player_id:
            return min_dist2ball
        else:
            return 0.0
    else:
        min_dist2ball = 0.0
    return min_dist2ball


def yellow_reward(prev_obs, obs):
    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    # right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
    #     prev_obs["right_team_yellow_card"]
    # )
    yellow_r = -left_yellow
    return yellow_r


def ball_position_reward_v2(obs, player_last_hold_ball, fix = False):

    ball_x_zone, ball_y_zone = obs['ball_zone']

   
    MIDDLE_X_ZONE = [8,12]
    OUR_PENALTY_X_ZONE = [1,4]
    OPP_PENALTY_X_ZONE = [17,20]
    PENALTY_Y_ZONE = [4,8]
    END_X_ZONE = [1,20]
    END_Y_ZONE = [1,12]
    
    if fix:
        ball_position_r_list= [-1.0, -0.5, 0.0, 2.0, 5.0]
    else:
        ball_position_r_list = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    ball_position_r = 0.0
    
    if (OUR_PENALTY_X_ZONE[0] <= ball_x_zone and ball_x_zone < OUR_PENALTY_X_ZONE[1]) and (
        PENALTY_Y_ZONE[0] < ball_y_zone and ball_y_zone < PENALTY_Y_ZONE[1]
    ):  # in our penalty area
        ball_position_r = ball_position_r_list[0]
    elif (END_X_ZONE[0] <= ball_x_zone and ball_x_zone < MIDDLE_X_ZONE[0]) and (
        END_Y_ZONE[0] < ball_y_zone and ball_y_zone < END_Y_ZONE[1]
    ):  # in our half
        ball_position_r = ball_position_r_list[1]
    elif (MIDDLE_X_ZONE[0] <= ball_x_zone and ball_x_zone <= MIDDLE_X_ZONE[1]) and (
        END_Y_ZONE[0] < ball_y_zone and ball_y_zone < END_Y_ZONE[1]
    ): # in mid zone
        ball_position_r = ball_position_r_list[2]
    elif (OPP_PENALTY_X_ZONE[0] < ball_x_zone and ball_x_zone <= OPP_PENALTY_X_ZONE[1]) and (
        PENALTY_Y_ZONE[0] < ball_y_zone and ball_y_zone < PENALTY_Y_ZONE[1]
    ): # in opponent penalty area
        ball_position_r = ball_position_r_list[4]
    elif (MIDDLE_X_ZONE[1] < ball_x_zone and ball_x_zone <= END_X_ZONE[1]) and (
        END_Y_ZONE[0] < ball_y_zone and ball_y_zone < END_Y_ZONE[1]
    ): # in opponent half
        ball_position_r = ball_position_r_list[3]
    else:
        ball_position_r = 0.0

    # if obs["ball_owned_team"] == 0:
    #     if not obs["active"] == player_last_hold_ball:
    #         ball_position_r *= 0.5

    return ball_position_r

def ball_position_reward(obs, player_last_hold_ball):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42
    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):  # in our penalty area
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):  #
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    # if obs["ball_owned_team"] == 0:
    #     if not obs["active"] == player_last_hold_ball:
    #         ball_position_r *= 0.5

    return ball_position_r


def calc_skilled_attack_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    highpass_r = 0
    if prev_obs["ball_owned_team"] == 1 or prev_obs["ball_owned_team"] == 0:
        if (
            obs["ball_owned_team"] == 1
            and prev_obs["ball_owned_player"] != obs["ball_owned_player"]
        ):
            highpass_r = 2.0

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### Encourage player movement.
    left_position_move = np.sum((prev_obs["left_team"] - obs["left_team"]) ** 2)

    reward = (
        2.0 * win_reward
        + 20.0 * rew
        + 0.06 * ball_position_r
        + yellow_r
        + highpass_r
        + left_position_move
    )
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward


def calc_active_attack_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### Encourage player movement.
    left_position_move = np.sum((prev_obs["left_team"] - obs["left_team"]) ** 2)

    reward = (
        2.0 * win_reward
        + 20.0 * rew
        + 0.06 * ball_position_r
        + yellow_r
        + left_position_move
    )
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward


def calc_active_deffend_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    left_team_position = obs["left_team"]
    right_team_position = obs["right_team"]

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### Encourage player movement.
    # left_position_move = np.sum((prev_obs['left_team']-obs['left_team'])**2)

    reward = 2.0 * win_reward + 20.0 * rew + 0.06 * ball_position_r + yellow_r
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward


def calc_skilled_deffend_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    if prev_obs["ball_owned_team"] == -1 and obs["ball_owned_team"] == 1:
        ballowned_r = 1.0
    elif prev_obs["ball_owned_team"] == -1 and obs["ball_owned_team"] == 0:
        ballowned_r = 0.0
    else:
        ballowned_r = -1.0

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### Encourage player movement.
    # left_position_move = np.sum((prev_obs['left_team']-obs['left_team'])**2)

    reward = (
        2.0 * win_reward + 20.0 * rew + 0.06 * ball_position_r + yellow_r + ballowned_r
    )
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward


def calc_offside_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### Encourage player movement.
    # left_position_move = np.sum((prev_obs['left_team']-obs['left_team'])**2)

    reward = 2.0 * win_reward + 5.0 * rew + 0.06 * ball_position_r + 20 * yellow_r
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward