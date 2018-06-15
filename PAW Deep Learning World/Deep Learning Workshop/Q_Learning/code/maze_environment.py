import numpy as np
import random

SCALE = 40
MAZE_SIZE = 7

class Maze_Env(object):
    def __init__(self):
        # Use a fixed random sequence for generating the maze, so that different workers construct identical mazes.
        self.fixed_random = random.Random(25912)

        self.use_display = False
        self.action_space_size = 4

        self.num_cols_or_rows = MAZE_SIZE
        self.wid = 2 * self.num_cols_or_rows + 1  # Must be odd
        self.num_open_corners = (self.num_cols_or_rows + 1) * (self.num_cols_or_rows + 1)

        # Each observation is just the integer ID of the current cell, which is a Markov state for this environment.
        self.observation_space_size = MAZE_SIZE * MAZE_SIZE  # The number of cells in the maze.
        self.observation = 0

        # Define the maze array (of bool)
        self.ma = [[False for x in range(self.wid)] for y in range(self.wid)]

        # Add all the perimeter walls.
        for x in range(self.wid):
            for y in range(self.wid):
                x_is_odd = (x % 2 == 1)
                y_is_odd = (y % 2 == 1)
                if x_is_odd and y_is_odd:
                    # Center of a square
                    self.ma[x][y] = False
                elif x_is_odd:
                    # Horizontal
                    if (y == 0) or (y == self.wid - 1):
                        self.add_wall(x, y)
                elif y_is_odd:
                    # Vertical
                    if (x == 0) or (x == self.wid - 1):
                        self.add_wall(x, y)

        # Randomly add inner walls until all corners are attached to some wall.
        while self.num_open_corners > 0:
            x, y = self.choose_random_unattached_corner()
            direction = self.fixed_random.randint(0, 3)
            if direction == 0:
                self.add_wall(x + 1, y)
            elif direction == 1:
                self.add_wall(x, y + 1)
            elif direction == 2:
                self.add_wall(x - 1, y)
            else:
                self.add_wall(x, y - 1)

        # Place the goal (portal) in a random location.
        self.gx, self.gy = self.choose_random_square_privately()

        # Place the agent in a random location.
        self.place_agent_randomly()
        self.reset_online_test_sums()
        self.reward = 0.

    def place_agent_randomly(self):
        self.ax, self.ay = self.choose_random_square()
        while (self.ax == self.gx) and (self.ay == self.gy):
            self.ax, self.ay = self.choose_random_square()

    def choose_random_square(self):
        col = random.randint(0, self.num_cols_or_rows - 1)
        row = random.randint(0, self.num_cols_or_rows - 1)
        return 1 + 2 * col, 1 + 2 * row

    def choose_random_square_privately(self):
        col = self.fixed_random.randint(0, self.num_cols_or_rows - 1)
        row = self.fixed_random.randint(0, self.num_cols_or_rows - 1)
        return 1 + 2 * col, 1 + 2 * row

    def choose_random_unattached_corner(self):
        # Choose a random unattached corner.
        corner_index = self.fixed_random.randint(0, self.num_open_corners - 1)
        # Find and return that corner.
        index = 0
        for x in range(self.wid):
            if x % 2 == 0:
                for y in range(self.wid):
                    if y % 2 == 0:
                        if not self.ma[x][y]:
                            if index == corner_index:
                                return x, y
                            else:
                                index += 1

    def add_wall(self, x, y):
        # Mark the wall as added.
        self.ma[x][y] = True
        # Mark the corners as attached, if still unattached.
        if x % 2 == 1:
            # Horizontal
            if not self.ma[x - 1][y]:
                self.ma[x - 1][y] = True
                self.num_open_corners -= 1
            if not self.ma[x + 1][y]:
                self.ma[x + 1][y] = True
                self.num_open_corners -= 1
        else:
            # Vertical
            if not self.ma[x][y - 1]:
                self.ma[x][y - 1] = True
                self.num_open_corners -= 1
            if not self.ma[x][y + 1]:
                self.ma[x][y + 1] = True
                self.num_open_corners -= 1

    def move_agent_to(self, x, y):
        if self.use_display:
            self.erase_agent()
        self.ax = x
        self.ay = y
        if self.use_display:
            self.draw_agent()
        return 0

    def move_up(self):
        if not self.ma[self.ax][self.ay + 1]:
            return self.move_agent_to(self.ax, self.ay + 2)
        return 0

    def move_down(self):
        if not self.ma[self.ax][self.ay - 1]:
            return self.move_agent_to(self.ax, self.ay - 2)
        return 0

    def move_left(self):
        if not self.ma[self.ax - 1][self.ay]:
            return self.move_agent_to(self.ax - 2, self.ay)
        return 0

    def move_right(self):
        if not self.ma[self.ax + 1][self.ay]:
            return self.move_agent_to(self.ax + 2, self.ay)
        return 0

    def assemble_current_observation(self, action, reward):
        self.observation = int(MAZE_SIZE * (self.ay - 1) / 2 + (self.ax - 1) / 2)
        return self.observation

    def reset(self):
        self.score = 0.
        return self.assemble_current_observation(0, 0)

    def translate_key_to_action(self, key):
        action = -1
        if key == 'Up':
            action = 1
        elif key == 'Left':
            action = 2
        elif key == 'Down':
            action = 3
        elif key == 'Right':
            action = 0
        return action

    def step(self, action):
        self.reward = 0
        if (self.ax == self.gx) and (self.ay == self.gy):
            self.reward = 1
            if self.use_display:
                self.score += self.reward
                self.erase_agent()
            self.place_agent_randomly()
            if self.use_display:
                self.draw_agent()
                self.draw_goal_square()
        else:
            if action == 0:
                self.move_right()
            elif action == 1:
                self.move_up()
            elif action == 2:
                self.move_left()
            elif action == 3:
                self.move_down()
        self.update_online_test_sums(self.reward)
        obs = self.assemble_current_observation(action, self.reward), self.reward, False
        self.draw_text()
        return obs

    # The rest of the functions are for drawing.

    def x_pix_from_square(self, xs):
        if self.use_display:
            return self.x_orig + xs * self.scale

    def y_pix_from_square(self, ys):
        if self.use_display:
            return self.y_orig + ys * self.scale

    def draw_or_erase_agent(self, color):
        if self.use_display:
            ax_pix = self.x_pix_from_square(self.ax)
            ay_pix = self.y_pix_from_square(self.ay)
            self.t.color(color)
            self.t.setpos(ax_pix, ay_pix)
            self.t.pensize(self.scale / 2)
            self.t.dot(self.scale, color)

    def draw_agent(self):
        if self.use_display:
            self.draw_or_erase_agent('red')
            self.t.write('')  # Sometimes the agent doesn't display without this.

    def erase_agent(self):
        if self.use_display:
            self.draw_or_erase_agent('white')

    def draw_line(self, x1, y1, x2, y2, color):
        if self.use_display:
            self.t.pensize(1)
            self.t.color(color)
            self.t.setpos(x1, y1)
            self.t.pendown()
            self.t.goto(x2, y2)
            self.t.penup()

    def draw_rect(self, x1, y1, x2, y2, color):
        if self.use_display:
            self.t.pensize(1)
            self.t.begin_fill()
            self.t.color(color)
            self.t.setpos(x1, y1)
            self.t.pendown()
            self.t.goto(x1, y2)
            self.t.goto(x2, y2)
            self.t.goto(x2, y1)
            self.t.goto(x1, y1)
            self.t.penup()
            self.t.end_fill()

    def draw_wall(self, x, y):
        if self.use_display:
            self.t.color('black')

            if x % 2 == 1:
                # Center of a horizontal wall
                self.draw_line(self.x_orig + (x - 1) * self.scale, self.y_orig + y * self.scale,
                               self.x_orig + (x + 1) * self.scale, self.y_orig + y * self.scale,
                               'black')
            else:
                # Center of a vertical wall
                self.draw_line(self.x_orig + x * self.scale, self.y_orig + (y - 1) * self.scale,
                               self.x_orig + x * self.scale, self.y_orig + (y + 1) * self.scale,
                               'black')

    def draw_goal_square(self):
        self.draw_rect(self.x_pix_from_square(self.gx - 1) + 1,
                       self.y_pix_from_square(self.gy - 1) + 1,
                       self.x_pix_from_square(self.gx + 1) - 1,
                       self.y_pix_from_square(self.gy + 1) - 1,
                       'yellow')

    def draw(self):
        if self.use_display:
            self.scale = SCALE
            self.x_orig = -(self.wid - 1) * self.scale / 2
            self.y_orig = -(self.wid - 1) * self.scale / 2

            # Display the maze.
            for x in range(self.wid):
                for y in range(self.wid):
                    x_is_odd = (x % 2 == 1)
                    y_is_odd = (y % 2 == 1)
                    if x_is_odd and y_is_odd:
                        # Center of a square
                        if self.ma[x][y]:
                            self.t.setpos(self.x_orig + x * self.scale, self.y_orig + y * self.scale)
                            self.t.color('lightgreen')
                            self.t.dot()
                    elif x_is_odd or y_is_odd:
                        if self.ma[x][y]:
                            self.draw_wall(x, y)

            self.draw_goal_square()

            # Draw the agent.
            self.draw_agent()
            self.draw_text()
            self.t.setpos(0, 0)
            self.t._update()

    def draw_text(self):
        if self.use_display:
            rad = self.num_cols_or_rows * self.scale

            self.draw_rect(-rad, rad + 27, -rad + 800, rad + 66, 'white')
            self.t.setpos(-rad, rad + 26)
            self.t.color('black')
            self.t.write('Last reward:  {:3.1f}         Total reward:  {}'.format(self.reward, self.score), font=("Arial", 16, "normal"))

            self.draw_rect(-rad, rad + 71, -rad + 800, rad + 110, 'white')
            self.t.setpos(-rad, rad + 70)
            self.t.color('black')
            self.t.write('Observation:  {}'.format(self.observation), font=("Arial", 16, "normal"))

    def log_settings(self, summary_file):
        return

    # Online test support.
    # In online testing, each training step is also used for testing the agent.
    # This is permissible only in the infinite data case (like games and simulations), where separate train and test sets are not required.
    # The environment must define its own (real-valued) online test metric, which may be as simple as accumulated reward.
    # To smooth out the reported test results, online testing is divided into contiguous reporting periods of many time steps.

    def reset_online_test_sums(self):
        # Called only by the environment itself.
        self.step_sum = 0
        self.reward_sum = 0.
        self.need_to_reset_sums = False

    def update_online_test_sums(self, reward):
        # Called only by the environment itself.
        if self.need_to_reset_sums:
            # Another thread recently called reduce_online_test_sums(), so the previous counts are stale.
            self.reset_online_test_sums()
        # If another thread happens to call reduce_online_test_sums near this point in time,
        # one sample from this agent might get dropped. But that's a small price to avoid locking.
        self.step_sum += 1
        self.reward_sum += reward

    def report_online_test_metric(self):
        # Called by the reporting manager only.
        self.need_to_reset_sums = True

        # Calculate the final metric for this test period.
        ret = ("{:6.2f}".format(self.reward_sum / self.step_sum))

        # Reset the sums.
        self.reset_online_test_sums()

        return ret
