import torch
from Trainer import Trainer
from ExperienceBuffer import Experience
from Evaluator import Evaluator
from kaggle_environments import make

episodes = 30000
batch_size = 32
discount = 0.8
hidden_dim = 300
experienceSize = 20000
epsilon_min_after = 1500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainer = Trainer(hidden_dim, experienceSize, discount, batch_size, device)
for e in range(episodes):
    observation = trainer.reset()
    done = False
    batchReward = 0
    steps = 0
    while not done:
        action = trainer.policyAction(observation['board'], e, epsilon_min_after)
        old_obs = observation
        observation, reward, done, _ = trainer.step(int(action))
        reward = trainer.change_reward(reward, done, None)
        next_state = observation['board']
        exp = Experience(old_obs['board'], action, reward, next_state, int(done))
        trainer.addExperience(exp)
        batchReward += reward
        loss = trainer.train()
        steps += 1
    if (e % 1000 == 0) and (e > 0):
        trainer.switchPosition()
    if e % 50 == 0:
        trainer.synchronize()
    if e % 500 == 0:
        trainer.save("model_state")
        print("episode: " + str(e) + " meanReward generateEpisodes: " + str(batchReward) + " meanLoss: " + str(loss))
        print("steps: " + str(steps))
        with torch.no_grad():
            print(trainer.policy(trainer.reshape(torch.tensor(trainer.reset()['board'])))[0])
        evaluator = Evaluator(100, trainer)
        evaluator.winPercentage()

    # if e % 25000 == 0:
    #     trainer.switch()
    #     trainer.save("model_state_"+str(e))
print("done")