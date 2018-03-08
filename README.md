# PBT + A2C implementation of SC2 RL bot

The goal of this project to implement a Starcraft II reinforcement learning bot through the SC2LE interface (https://deepmind.com/documents/110/sc2le.pdf) using A2C algorithm published by OpenAI adapted for the SC2 environment, and integrate it with the PBT (population based training) optimalisation algorithm. The results will be evaluated and compared to previous solutions.

The base A2C algorithm implementation was taken from [pekaalto's sc2aibot](https://github.com/pekaalto/sc2aibot) and [OpenAI's baselines](https://github.com/openai/baselines/) and is being (among other things) refactored and modified to fit the needs of this project.

You can modify all the important parameters via config.py. The number of models and environments per model will of course heavily affect the performance, so keep that in mind when tweaking the parameters.

Any feedback or suggestions are welcome. Enjoy!


