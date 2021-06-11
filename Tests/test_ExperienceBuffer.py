from unittest import TestCase
from refactored.ExperienceBuffer import *
import torch


class TestExperienceReplay(TestCase):
    def testStuff(self):
        # your testcode here
        pass

    def testStuff2(self):
        # your testcode here
        pass


class TestLen(TestExperienceReplay):
    def test_len_at_initialization(self):
        experience_replay = ExperienceReplay(capacity=10)
        self.assertEqual(len(experience_replay), 0)

    def test_len_after_appending_experience(self):
        experience_replay = ExperienceReplay(capacity=10)
        new_experience = Experience(
            state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1,
                   0, 1, 2, 1, 2, 2, 1, 2],
            action=torch.tensor(6),
            reward=0.023809523809523808,
            next_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 2,
                        2, 1, 1, 1, 2, 1, 2, 2, 1, 2],
            done=0.0)
        experience_replay.append(new_experience)
        self.assertEqual(len(experience_replay), 1)