from unittest import TestCase
from refactored.Trainer import Trainer
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda

class TestChangeReward(TestCase):

    def test_change_reward_with_reward_1_and_done_true(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(hidden_dim=10, buffer_size=2, gamma=0.8, batch_size=32, device=device)
        self.assertEqual(trainer.change_reward(reward=1, done=True, board=None), 10)

    def test_change_reward_with_reward_None_and_done_true(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(hidden_dim=10, buffer_size=2, gamma=0.8, batch_size=32, device=device)
        self.assertEqual(trainer.change_reward(reward=-1, done=True, board=None), -10)

    def test_change_reward_with_reward_minus_1_and_done_true(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(hidden_dim=10, buffer_size=2, gamma=0.8, batch_size=32, device=device)
        self.assertEqual(trainer.change_reward(reward=None, done=True, board=None), -20)


    def test_take_action(self):
        pass

    def test_reshape(self):
        pass

    def test_preprocess_state(self):
        pass

    def test_train_action_from_policy(self):
        pass

    def test_train_action_from_target(self):
        pass

    def test_train(self):
        pass
