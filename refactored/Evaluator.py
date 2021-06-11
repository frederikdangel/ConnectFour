from kaggle_environments import make, evaluate
import torch
from Trainer import Trainer
import numpy as np


class Evaluator:
    def __init__(self, rounds, trainer: Trainer):
        self.rounds = rounds
        self.trainer = trainer

    def agent(self, observation, configuration):
        with torch.no_grad():
            state = torch.tensor(observation['board'], dtype=torch.float)
            reshaped = self.trainer.reshape(state)
            action = self.trainer.takeAction(self.trainer.policy(reshaped)[0], reshaped, 0, False)
            return action

    def winPercentage(self):
        config = {'rows': 6, 'columns': 7, 'inarow': 4}
        outcomes = evaluate("connectx", [self.agent, "random"], config, [], self.rounds // 2)
        # Agent 2 goes first (roughly) half the time
        outcomes += [[b, a] for [a, b] in
                     evaluate("connectx", [self.agent, "random"], config, [], self.rounds - self.rounds // 2)]
        print("Agent 1 Win Percentage:", np.round(outcomes.count([1, -1]) / len(outcomes), 2))
        print("Agent 2 Win Percentage:", np.round(outcomes.count([-1, 1]) / len(outcomes), 2))
        print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
        print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
