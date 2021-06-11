import torch
from Trainer import Trainer
from ExperienceBuffer import Experience
from Evaluator import Evaluator

episodes = 2000
batch_size = 32
discount = 0.99
hidden_dim = 300
experienceSize = 20000
epsilon_min_after = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainer = Trainer(hidden_dim, experienceSize, discount, batch_size, device)

for e in range(episodes):
    observation = trainer.reset()
    done = False
    batchReward = 0
    steps = 0
    while not done:
        action = trainer.policyAction(observation.board, e, epsilon_min_after)
        old_obs = observation
        observation, reward, done, _ = trainer.step(action)
        reward = trainer.change_reward(reward, done, None)
        next_state = observation.board
        exp = Experience(old_obs.board, action, reward, next_state, float(done))
        trainer.addExperience(exp)
        batchReward += reward
        loss = trainer.train()
        steps += 1
    if e % 20 == 0:
        trainer.synchronize()
        trainer.save()
        print("episode: " + str(e) + " meanReward generateEpisodes: " + str(batchReward) + " meanLoss: " + str(loss))
        print("steps: " + str(steps))
        evaluator = Evaluator(100, trainer)
        evaluator.winPercentage()
        with torch.no_grad():
            print(trainer.policy(trainer.reshape(torch.tensor(trainer.reset().board)))[0])
