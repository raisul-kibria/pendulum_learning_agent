#!/usr/bin/env python

# MIT License
# Copyright (c) 2017 Massimiliano Patacchiola
# https://mpatacchiola.github.io/blog/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from utils.inverted_pendulum import InvertedPendulum, PARAMS
import random

OUTPUT_DIR = './outputs/group4_random'
os.makedirs(OUTPUT_DIR, exist_ok=True)

my_pole = InvertedPendulum(
    pole_mass=PARAMS["pole_mass"],
    cart_mass=PARAMS["cart_mass"],
    pole_lenght=PARAMS["pole_lenght"],
    delta_t=PARAMS["delta_t"])

cumulated_reward = 0
print("Starting random agent...")
for step in range(100):
    action = random.randint(a=0, b=2)
    observation, reward, done = my_pole.step(action)
    cumulated_reward += reward
    print("Step: " + str(step))
    print("Action: " + str(action))
    print("Angle: " + str(observation[0]))
    print("Velocity: " + str(observation[1]))
    print("Reward: " + str(reward))
    print("")
    if done: break
print("Finished after: " + str(step+1) + " steps")
print("Cumulated Reward: " + str(cumulated_reward))
print(f"Saving the gif in: {OUTPUT_DIR}/inverted_pendulum_random_agent.gif")
my_pole.render(file_path=f'{OUTPUT_DIR}/inverted_pendulum_random_agent.gif', mode='gif')
print("Complete!")
