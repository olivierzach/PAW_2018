import os
import turtle
import time
from copy import deepcopy

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # suppress CPU notification

NUM_WORKERS = 1

if NUM_WORKERS == 1:
    REPORTING_INTERVAL = 1000
    TOTAL_STEPS = 10000
else:
    REPORTING_INTERVAL = 625
    TOTAL_STEPS = 10000000

class Worker(object):
    def __init__(self, train, display):
        self.train = train
        self.display = display
        self.start_time = time.time()
        self.environment = self.create_environment()
        self.agent = self.create_agent(self.environment)
        self.step_num = 0
        self.done = False
        self.total_reward = 0.

    def create_agent(self, environment):
       # from random_agent import RandomAgent
       # agent = RandomAgent(observation_space_size=self.observation_space_size, action_space_size=self.action_space_size)
        from tab_q_agent import TabQAgent
        agent = TabQAgent(observation_space_size=self.observation_space_size, action_space_size=self.action_space_size)
        return agent

    def create_environment(self):
        from maze_environment import Maze_Env
        environment = Maze_Env()
        self.observation_space_size = environment.observation_space_size
        self.action_space_size = environment.action_space_size
        return environment

    def run(self):
        self.init_episode()
        if self.train:
            self.take_n_steps(1000000000, None)
            print("Mean reward per step = {}".format(self.total_reward / self.step_num))
        elif self.display:
            self.init_turtle()
            self.enable_display()
            self.wnd.mainloop()  # After this call, the program runs until the user closes the window.

    def init_turtle(self):
        self.t = turtle.Turtle()
        self.environment.t = self.t
        self.t.hideturtle()
        self.t.speed('fastest')
        self.t.screen.tracer(0, 0)
        self.t.penup()
        self.wnd = turtle.Screen()
        self.wnd.setup(1902, 990, 0, 0)
        self.wnd.onkey(self.move_up, "Up")
        self.wnd.onkey(self.move_down, "Down")
        self.wnd.onkey(self.move_left, "Left")
        self.wnd.onkey(self.move_right, "Right")
        self.wnd.onkey(self.on_space, " ")
        self.wnd.listen()

    def on_space(self):
        self.take_n_steps(1, None)

    def take_n_steps(self, max_steps, action_override):
        if self.step_num == TOTAL_STEPS:
            return
        for i in range(max_steps):
            self.step(action_override)
            if self.done or (self.step_num == TOTAL_STEPS):
                if self.step_num == TOTAL_STEPS:
                    return
                self.init_episode()

    def step(self, action_override):
        self.action = self.agent.step(self.reward, self.observation)
        if action_override != None:
            self.action = action_override
        if self.action < 0:
            self.action = self.environment.translate_action(self.action)
        next_observation, self.reward, self.done = self.environment.step(self.action)
        self.monitor(self.reward)
        self.observation = deepcopy(next_observation)

    def draw_line(self, x1, y1, x2, y2, color):
        self.t.pensize(5)
        self.t.color(color)
        self.t.setpos(x1, y1)
        self.t.pendown()
        self.t.goto(x2, y2)
        self.t.penup()

    def move_up(self):
        self.take_n_steps(1, self.environment.translate_key_to_action('Up'))

    def move_down(self):
        self.take_n_steps(1, self.environment.translate_key_to_action('Down'))

    def move_left(self):
        self.take_n_steps(1, self.environment.translate_key_to_action('Left'))

    def move_right(self):
        self.take_n_steps(1, self.environment.translate_key_to_action('Right'))

    def enable_display(self):
        self.use_display = True
        self.environment.use_display = True
        self.t.clear()
        self.environment.draw()

    def init_episode(self):
        self.observation = self.environment.reset()
        self.reward = 0.
        self.done = False

    def output(self, sz):
        print(sz)

    def monitor(self, reward):
        self.step_num += 1
        self.total_reward += reward

        # Report results periodically.
        if (self.step_num % REPORTING_INTERVAL) == 0:
            sz = "{:10.2f} sec  {:12,d} steps".format(time.time() - self.start_time, self.step_num)
            metric_string = self.environment.report_online_test_metric()
            sz += "  {}".format(metric_string)
            self.output(sz)

