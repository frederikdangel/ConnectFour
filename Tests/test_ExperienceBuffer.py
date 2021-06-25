import sys
from unittest import TestCase
from refactored.ExperienceBuffer import Experience, ExperienceReplay
import torch


class TestLen(TestCase):
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

    def test_len_after_capacity_overflow(self):
        experience_replay = ExperienceReplay(capacity=2)
        new_experience = Experience(
            state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1,
                   0, 1, 2, 1, 2, 2, 1, 2],
            action=torch.tensor(6),
            reward=0.023809523809523808,
            next_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 2,
                        2, 1, 1, 1, 2, 1, 2, 2, 1, 2],
            done=0.0)
        experience_replay.append(new_experience)
        experience_replay.append(new_experience)
        experience_replay.append(new_experience)
        experience_replay.append(new_experience)
        self.assertEqual(len(experience_replay), 2)


class TestAppend(TestCase):

    def test_append(self):
        experience_replay = ExperienceReplay(capacity=2)
        new_experience = Experience(
            state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1,
                   0, 1, 2, 1, 2, 2, 1, 2],
            action=torch.tensor(6),
            reward=0.023809523809523808,
            next_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 2,
                        2, 1, 1, 1, 2, 1, 2, 2, 1, 2],
            done=0.0)

        new_second_experience = Experience(
            state=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1,
                   0, 1, 2, 1, 2, 2, 1, 2],
            action=torch.tensor(6),
            reward=0.023809523809523808,
            next_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 2,
                        2, 1, 1, 1, 2, 1, 2, 2, 1, 2],
            done=0.0)

        new_third_experience = Experience(
            state=[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1,
                   0, 1, 2, 1, 2, 2, 1, 2],
            action=torch.tensor(6),
            reward=0.023809523809523808,
            next_state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 2,
                        2, 1, 1, 1, 2, 1, 2, 2, 1, 2],
            done=0.0)

        experience_replay.append(new_experience)
        experience_replay.append(new_second_experience)
        experience_replay.append(new_third_experience)

        self.assertEqual(experience_replay.buffer[0], new_second_experience)