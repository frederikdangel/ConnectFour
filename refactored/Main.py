import torch
from Trainer import Trainer
from ExperienceBuffer import Experience
from Evaluator import Evaluator
from kaggle_environments import make
from torch.utils.tensorboard import SummaryWriter
import time


def run(episodes, discount, useStreak):
    start_time = time.time()
    print('run with: ' + str(discount) + str(useStreak))
    batch_size = 32
    hidden_dim = 300
    experienceSize = 20000
    epsilon_min_after = 1500
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter('runs/run_discount_' + str(discount) + '_streak_' + str(useStreak))
    trainer = Trainer(hidden_dim, experienceSize, discount, batch_size, device, writer)

    for e in range(episodes):
        observation = trainer.reset()
        done = False
        batchReward = 0
        steps = 0
        threes = 0
        while not done:
            action = trainer.policyAction(observation['board'], e, epsilon_min_after)
            old_obs = observation
            observation, reward, done, _ = trainer.step(int(action))
            reshaped = trainer.reshape(torch.tensor(observation['board']))
            threes += trainer.streakReward(trainer.player, reshaped, int(action))
            if useStreak:

                reward = trainer.change_reward_streak(reward, done, reshaped, int(action), useStreak)
            else:
                reward = trainer.change_reward(reward, done)
            next_state = observation['board']
            exp = Experience(old_obs['board'], action, reward, next_state, int(done))
            trainer.addExperience(exp)
            batchReward += reward
            loss = trainer.train()
            steps += 1
        threes /= 3
        if loss != None:
            writer.add_scalar('trainLoss', loss, e)
        writer.add_scalar('batchReward', batchReward, e)
        writer.add_scalar('steps', steps, e)
        writer.add_scalar('threes', threes, e)
        if (e % 1000 == 0) and (e > 0):
            trainer.switchPosition()
        if e % 50 == 0:
            trainer.synchronize()
            firstStep = str(trainer.policy(trainer.reshape(torch.tensor(trainer.reset()['board'])))[0])
            writer.add_text('first_qs', firstStep, e)
        if e % 500 == 0:
            trainer.save("model_state_discount_" + str(discount) + '_useStreak_' + str(useStreak))
            print(e)
            # print("episode: " + str(e) + " meanReward generateEpisodes: " + str(batchReward) + " meanLoss: " + str(loss))
            # print("steps: " + str(steps))
            # firstStep = str(trainer.policy(trainer.reshape(torch.tensor(trainer.reset()['board'])))[0])
            # with torch.no_grad():
            #     print(firstStep)
            evaluator = Evaluator(100, trainer)
            evaluator.winPercentage(e)

        # if e % 25000 == 0:
        #     trainer.switch()
        #     trainer.save("model_state_"+str(e))
    print("--- %s seconds ---" % (time.time() - start_time))
run(10, 0.7, False)
run(10, 0.8, False)
run(10, 0.9, False)
print("done")
