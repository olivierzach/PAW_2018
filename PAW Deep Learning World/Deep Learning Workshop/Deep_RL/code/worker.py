import os
import turtle
import time
from copy import deepcopy

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # suppress CPU notification

NUM_WORKERS = 1

if NUM_WORKERS == 1:
    REPORTING_INTERVAL = 10000
    TOTAL_STEPS = 100000
else:
    REPORTING_INTERVAL = 625
    TOTAL_STEPS = 10000000

global_step_num = 0

class Worker(object):
    def __init__(self, num, num_workers, sess, train, display, model):
        self.num = num
        self.num_workers = num_workers
        self.sess = sess
        self.train = train
        self.display = display
        self.model = model
        self.working_dir = os.getcwd()
        self.initialize()
        self.all_environments = None  # The reporting worker (0) will get a list of all environments.
        self.best_global_metric_value = -1000000000.

    def initialize(self):
        self.start_time = time.time()
        self.environment = self.create_environment()
        self.agent = self.create_agent(str(self.num), self.environment, self.sess)
        self.step_num = 0
        self.global_step_num = 0
        self.done = False
        self.best_metric_value = -1000000000.
        self.total_num_episodes = 0
        self.total_reward = 0.

    def create_agent(self, agent_name, environment, sess):
        # agent = None
        from random_agent import RandomAgent
        agent = RandomAgent(action_space_size=self.action_space_size)
        # from deep_rl_agent import DeepRLAgent
        # agent = DeepRLAgent(agent_name=agent_name,
        #                     observation_space_size=self.observation_space_size,
        #                     action_space_size=self.action_space_size,
        #                     sess=sess)
        return agent

    def create_environment(self):
        from maze_environment import Maze_Env
        environment = Maze_Env()
        self.observation_space_size = environment.observation_space_size
        self.action_space_size = environment.action_space_size
        return environment

    def load_model(self):
        # print("Loading model from {}".format(self.model))
        self.agent.load_model('model\model')

    def run(self):
        self.init_episode()
        self.agent.copy_global_weights()

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
                self.total_num_episodes += 1
                self.current_episode_reward = 0.
                if self.step_num == TOTAL_STEPS:
                    return
                self.init_episode()
                self.agent.copy_global_weights()

    def step(self, action_override):
        self.action = self.agent.step(self.action, self.reward, self.observation)
        if action_override != None:
            self.action = action_override
        if self.action < 0:
            self.action = self.environment.translate_action(self.action)
        next_observation, self.reward, self.done = self.environment.step(self.action)
        self.agent.adapt(self.action, self.reward, self.done)
        if (self.num == 0):
            self.monitor(self.reward)
        self.observation = deepcopy(next_observation)
        self.num_steps_taken_in_episode += 1

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
        self.action = self.agent.reset_state()
        self.observation = self.environment.reset()
        self.reward = 0.
        self.done = False
        self.num_steps_taken_in_episode = 0

    def output(self, sz):
        print(sz)

    def monitor(self, reward):
        self.step_num += 1
        self.total_reward += reward

        # Report results periodically.
        if (self.step_num % REPORTING_INTERVAL) == 0:
            sz = "{:10.2f} sec  {:12,d} reporter steps".format(time.time() - self.start_time, self.step_num)

            if hasattr(self.environment, 'report_online_test_metric'):
                # Assemble a global report from all threads.
                num_global_steps, global_metric_value, global_metric_string, global_metric_units = \
                    self.environment.report_online_test_metric(self.all_environments)
                self.global_step_num += num_global_steps

                # Is this the best online test metric yet?
                saved = False
                if global_metric_value > self.best_global_metric_value:
                    saved = True
                    self.best_global_metric_value = global_metric_value
                    filename_base = "{}\\model\\model".format(self.working_dir)
                    self.agent.save_local_model(filename_base)

                sz += "  {:12,d} global steps   {} {}".format(self.global_step_num, global_metric_string, global_metric_units)
                if saved:
                    sz += "  SAVED"
                else:
                    sz += "       "

            self.output(sz)

