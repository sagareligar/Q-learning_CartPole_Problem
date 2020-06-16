#!/usr/bin/env python

import RPi.GPIO as GPIO


one_degree = 0.0174532    # 2pi/360
six_degrees = 0.1047192
twelve_degrees = 0.2094384
fifty_degrees = 0.87266

# approach based on the example from 'Reinforcement Learning: An Introduction' by Richard S. Sutton and Andrew G. Barto (1998)
# Q-Learning with look-up table


in1 = 24
in2 = 23
en = 25


in3 = 15
in4 = 14
en1 = 18


GPIO.setmode(GPIO.BCM)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en,GPIO.OUT)


GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(en1,GPIO.OUT)



GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)

GPIO.output(in3,GPIO.LOW)
GPIO.output(in4,GPIO.LOW)


p=GPIO.PWM(en,1000)
q=GPIO.PWM(en1,1000)

p.start(25)
q.start(25)






class ReinforcementLearner():

    # initialize a new ReinforcementLearner with some default parameters
  def __init__(self,alpha=1000, beta=0.5, gamma=0.95, lambda_w=0.9, lambda_v=0.8, max_failures=50, max_steps=1000000, max_distance=2.4, max_speed=1, max_angle_factor=12):
    self.n_states = 162         # 3x3x6x3 = 162 states
    self.alpha = alpha          # learning rate for action weights
    self.beta = beta            # learning rate for critic weights
    self.gamma = gamma          # discount factor for critic
    self.lambda_w = lambda_w    # decay rate for action weights
    self.lambda_v = lambda_v    # decay rate for critic weights
    self.max_failures = max_failures
    self.max_steps = max_steps

    self.max_distance = max_distance
    self.max_speed = max_speed
    self.max_angle = max_angle_factor * one_degree

    self.action_weights = [0] * self.n_states    # action weights
    self.critic_weights = [0] * self.n_states    # critic weights
    self.action_weights_elig = [0] * self.n_states    # action weight eligibilities
    self.critic_weights_elig = [0] * self.n_states # critic weight eligibilities

    #position, velocity, angle, angle velocity
    self.x, self.dx, self.t, self.dt = 0, 0, 0, 0

   

  # matches the current state to an integer between 1 and n_states
  # 3 states for position x: -max_distance < x < -0.8, -0.8 < x < 0.8, 0.8 < x < 2.4
  # 3 states for velocity dx: dx < -0.5, -0.5 < dx < 0.5, 0.5 < dx
  # 6 states for angle t: t < -6, -6 < t < -1, -1 < t < 0, 0 < t < 1, 1 < t < 6, t < 6
  # 3 states for angle velocity dt: dt < -50, -50 < dt < 50, 50 < dt
  # --> 3x3x6x3 = 162 states for the look-up table 
  def get_state(self):
    state = 0

    # failed
    if self.x < -self.max_distance or self.x > self.max_distance or self.t < -self.max_angle or self.t > self.max_angle:
      return -1

    #position
    if self.x < -0.8:
      state = 0
    elif self.x < 0.8:
      state = 1
    else:
      state = 2

    #velocity
    if self.dx < -self.max_speed:
      state += 0
    elif self.dx < 0.5:
      state += 3
    else:
      state += 6

    #angle
    if self.t < -six_degrees:
      state += 0
    elif self.t < -one_degree:
      state += 9
    elif self.t < 0:
      state += 18
    elif self.t < one_degree:
      state += 27
    elif self.t < six_degrees:
      state += 36
    else:
      state += 45

    #angle velocity
    if self.dt < -fifty_degrees:
      state += 0
    elif self.dt < fifty_degrees:
      state += 54
    else:
      state += 108

    return state

  # read the variables x, dx, t and dt from vrep
  # def read_variables(self):
  #   self.x = self.controller.get_current_position()[0]
  #   self.dx = self.controller.get_current_ground_speed()[0]
  #   self.t = self.controller.get_current_angle()[1]
  #   #self.t=kalmanY
  #   self.dt = self.controller.get_current_angle_speed()[1]

  # executes action and updates x, dx, t, dt
  # action size is two: left and right
  def do_action(self, action):
    if action == True:
      
      GPIO.output(in1,GPIO.HIGH)
      GPIO.output(in2,GPIO.LOW)
      GPIO.output(in3,GPIO.LOW)
      GPIO.output(in4,GPIO.HIGH)
      p.ChangeDutyCycle(100)
      q.ChangeDutyCycle(100)
    else:
      GPIO.output(in2,GPIO.HIGH)
      GPIO.output(in1,GPIO.LOW)
      GPIO.output(in4,GPIO.LOW)
      GPIO.output(in3,GPIO.HIGH)
      p.ChangeDutyCycle(100)
      q.ChangeDutyCycle(100)

  # update all weights or reset them when failed
  def update_all_weights(self, rhat, failed):
    for i in range(self.n_states):
        self.action_weights[i] += self.alpha * rhat * self.action_weights_elig[i]
        self.critic_weights[i] += self.beta * rhat * self.critic_weights_elig[i]

        if self.critic_weights[i] < -1.0:
          self.critic_weights[i] = self.critic_weights[i]

        if failed == True:
          self.action_weights_elig[i] = 0
          self.critic_weights_elig[i] = 0
        else:
          self.action_weights_elig[i] = self.action_weights_elig[i] * self.lambda_w
          self.critic_weights_elig[i] = self.critic_weights_elig[i] * self.lambda_v