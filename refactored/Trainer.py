import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Model import Net
from ExperienceBuffer import ExperienceReplay
from kaggle_environments import make


class Trainer:
    def __init__(self, hidden_dim, buffer_size, gamma, batch_size, device):
        self.env = make("connectx", debug=True)
        self.device = device
        self.policy = Net(self.env.configuration.columns * self.env.configuration.rows, hidden_dim,
                          self.env.configuration.columns).to(
            device)

        self.target = Net(self.env.configuration.columns * self.env.configuration.rows, hidden_dim,
                          self.env.configuration.columns).to(
            device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.buffer = ExperienceReplay(buffer_size)
        self.trainingPair = self.env.train([None, "random"])
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.policy.parameters(), lr=0.001)
        self.gamma = gamma
        self.batch_size = batch_size
        self.enemy = "random"
        self.first = True

    def switch(self):
        self.trainingPair = self.env.train([None, "negamax"])
        self.enemy = "negamax"

    def switchPosition(self):
        self.env.reset()
        if self.first:
            self.trainingPair = self.env.train(["random", None])
        else:
            self.trainingPair = self.env.train([None, "random"])
        self.first = not self.first

    def load(self):
        self.policy.load_state_dict(torch.load("../model_state"))

    def synchronize(self):
        self.target.load_state_dict(self.policy.state_dict())

    def save(self, name):
        torch.save(self.policy.state_dict(), name)

    def reset(self):
        self.env.reset()
        return self.trainingPair.reset()

    def step(self, action):
        return self.trainingPair.step(action)

    def addExperience(self, experience):
        self.buffer.append(experience)

    def epsilon(self, maxE, minE, episode, lastEpisode):
        return (maxE - minE) * max((lastEpisode - episode) / lastEpisode, 0) + minE

    def change_reward(self, reward, done, board):
        if done and reward == 1:
            return 10
        if done and reward == -1:
            return -10
        if reward is None and done:
            return -20
        if done:
            return 1
        if reward == 0:
            return 1 / 42
        else:
            return reward

    def longestVerticalStreak(self, player, reshapedBoard, action):
        count = 0
        wasZero = False
        for i in range(5, 0, -1):
            if reshapedBoard[0][player][i][action] == 0:
                wasZero = True
            if reshapedBoard[0][player][i][action] == 1 & wasZero:
                count = 0
                wasZero = False
            count += reshapedBoard[0][player][i][action]
        if reshapedBoard[0][0][0][action] == 0:
            return 0
        return count

    def longestHorizontalStreak(self, player, reshapedBoard, action):
        count = 0
        rowOfAction = self.rowOfAction(player, reshapedBoard, action)
        wasZero = False
        for i in range(7):
            if reshapedBoard[0][player][rowOfAction][i] == 0:
                wasZero = True
            if reshapedBoard[0][player][rowOfAction][i] == 1 & wasZero:
                count = 0
                wasZero = False
            count += reshapedBoard[0][player][rowOfAction][i]
        return count

    def longestDiagonalStreak(self, player, reshapedBoard, action):
        rowOfAction = self.rowOfAction(player, reshapedBoard, action)
        for row in range(4):
            for col in range(5):
                if reshapedBoard[0][player][row][col] == reshapedBoard[0][player][row + 1][col + 1] == \
                        reshapedBoard[0][player][row + 2][col + 2] == 1 and self.actionInDiagonal1(action, row, col,
                                                                                                   rowOfAction):
                    return 3
        for row in range(5, 1, -1):
            for col in range(4):
                if reshapedBoard[0][player][row][col] == reshapedBoard[0][player][row - 1][col + 1] == \
                        reshapedBoard[0][player][row - 2][col + 2] == 1 and self.actionInDiagonal2(action, row, col,
                                                                                                   rowOfAction):
                    return 3
        return 0

    def actionInDiagonal1(self, action, row, col, rowOfAction):
        return (rowOfAction == row and action == col or
                rowOfAction == row + 1 and action == col + 1 or
                rowOfAction == row + 2 and action == col + 2)

    def actionInDiagonal2(self, action, row, col, rowOfAction):
        return (rowOfAction == row and action == col or
                rowOfAction == row - 1 and action == col + 1 or
                rowOfAction == row - 2 and action == col + 2)

    def rowOfAction(self, player, reshapedBoard, action):
        rowOfAction = 10
        for i in range(6):
            if reshapedBoard[0][player][i][action] == 1:
                rowOfAction = min(i, rowOfAction)
        return rowOfAction

    def policyAction(self, board, episode, lastEpisode, minEp=0.1, maxEp=0.9):
        reshaped = self.reshape(torch.tensor(board))
        output = self.policy(reshaped).view(-1)
        return self.takeAction(output, reshaped, self.epsilon(maxEp, minEp, episode, lastEpisode))

    def takeAction(self, actionList: torch.tensor, board, epsilon, train=True):
        if (np.random.random() < epsilon) & train:
            # invalide actions rein=geht nicht
            # return torch.tensor(np.random.choice(len(actionList))).item()
            return np.random.choice([i for i in range(len(actionList)) if board[0][0][0][i] == 1])
        else:
            for i in range(7):
                if board[0][0][0][i] == 0:
                    actionList[i] = float('-inf')
            return torch.argmax(actionList).item()

    def reshape(self, board: torch.tensor, unsqz=True):
        tensor = board.view(-1, 7).long()
        # [0] = wo kann er reinwerfen(da wo es geht, steht eine 1), [1] = player1 (da wo es geht steht eine 0), [2] = player2 (da wo es geht steht eine 0)
        a = F.one_hot(tensor, 3).permute([2, 0, 1])
        b = a[:, :, :]
        if unsqz:
            return torch.unsqueeze(b, 0).float().to(self.device)
        return b.float().to(self.device)

    def preprocessState(self, state):
        state = self.reshape(torch.tensor(state), True)
        return state

    def trainActionFromPolicy(self, state, action):
        state = self.preprocessState(state)
        value = self.policy(state).view(-1).to(self.device)
        return value[action].to(self.device)

    def trainActionFromTarget(self, next_state, reward, done):
        next_state = self.preprocessState(next_state)
        target = self.target(next_state)
        target = torch.max(target, 1)[0].item()
        target = reward + ((self.gamma * target) * (1 - done))
        return torch.tensor(target).to(self.device)

    def train(self):
        if len(self.buffer) > self.batch_size:
            self.optimizer.zero_grad()
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size, self.device)
            meanLoss = 0
            for i in range(self.batch_size):
                value = self.trainActionFromPolicy(states[i], actions[i])
                target = self.trainActionFromTarget(next_states[i], rewards[i], dones[i])
                loss = self.loss_function(value, target)
                loss.backward()
                meanLoss += loss
            self.optimizer.step()
            return meanLoss / self.batch_size
