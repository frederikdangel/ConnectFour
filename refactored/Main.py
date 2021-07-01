import torch
from Trainer import Trainer
from ExperienceBuffer import Experience
from Evaluator import Evaluator
from kaggle_environments import make
from torch.utils.tensorboard import SummaryWriter
import time


def evalFile(file):
    print(file)
    batch_size = 32
    hidden_dim = 300
    experienceSize = 20000
    useStreak = False
    discount = 0.9
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/run_discount_' + str(discount) + '_streak_' + str(useStreak))
    trainer = Trainer(hidden_dim, experienceSize, discount, batch_size, device, writer)
    trainer.load(file)
    evaluator = Evaluator(100, trainer)
    evaluator.winPercentage(1)

batch_size = 32
hidden_dim = 300
experienceSize = 20000
useStreak = False
discount = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('runs/architecture')
trainer = Trainer(hidden_dim, experienceSize, discount, batch_size, device, writer)
#äwriter.add_graph(trainer.policy, trainer.reshape(torch.tensor(trainer.reset()['board'])))
#evalFile('run_fail/run_fail/model_state_discount_0.4_useStreak_True')
#evalFile('run_fail/run_fail/model_state_discount_0.5_useStreak_True')
#evalFile('run_fail/run_fail/model_state_discount_0.6_useStreak_True')
evalFile('run_fail/run_fail/model_state_discount_0.7_useStreak_True')
#evalFile('run_fail/run_fail/model_state_discount_0.8_useStreak_True')
#evalFile('run_fail/run_fail/model_state_discount_0.9_useStreak_True')

#evalFile('run_fail/run_fail/model_state_discount_0.4_useStreak_False')
#evalFile('run_fail/run_fail/model_state_discount_0.5_useStreak_False')
#evalFile('run_fail/run_fail/model_state_discount_0.6_useStreak_False')
evalFile('run_fail/run_fail/model_state_discount_0.7_useStreak_False')
#evalFile('run_fail/run_fail/model_state_discount_0.8_useStreak_False')
#evalFile('run_fail/run_fail/model_state_discount_0.9_useStreak_False')
#writer.close()
# evalFile('runs_04_27_06/model_state_discount_0.4_useStreak_True')
# evalFile('runs_04_27_06/model_state_discount_0.5_useStreak_True')
# evalFile('runs_04_27_06/model_state_discount_0.6_useStreak_True')
# evalFile('runs_04_27_06/model_state_discount_0.7_useStreak_True')
# evalFile('runs_04_27_06/model_state_discount_0.8_useStreak_True')
# evalFile('runs_04_27_06/model_state_discount_0.9_useStreak_True')
# evalFile('runs_04_27_06/model_state_discount_1.0_useStreak_True')
# evalFile('runs_04_27_06/model_state_discount_1.0_useStreak_True_invalidActions')
#
# evalFile('runs_05_27_06/model_state_discount_0.4_useStreak_False')
# evalFile('runs_05_27_06/model_state_discount_0.5_useStreak_False')
# evalFile('runs_05_27_06/model_state_discount_0.6_useStreak_False')
# evalFile('runs_05_27_06/model_state_discount_0.7_useStreak_False')
# evalFile('runs_05_27_06/model_state_discount_0.8_useStreak_False')
# evalFile('runs_05_27_06/model_state_discount_0.9_useStreak_False')
# evalFile('runs_05_27_06/model_state_discount_1.0_useStreak_False')
#evalFile('runs_04_27_06/model_state_discount_1.0_useStreak_False_invalidActions')
#
# evalFile('run_fail/run_fail/model_state_discount_0.4_useStreak_False')
# evalFile('run_fail/run_fail/model_state_discount_0.4_useStreak_True')
# evalFile('run_fail/run_fail/model_state_discount_0.5_useStreak_False')
# evalFile('run_fail/run_fail/model_state_discount_0.5_useStreak_True')
# evalFile('run_fail/run_fail/model_state_discount_0.6_useStreak_False')
# evalFile('run_fail/run_fail/model_state_discount_0.6_useStreak_True')
# evalFile('run_fail/run_fail/model_state_discount_0.7_useStreak_False')
# evalFile('run_fail/run_fail/model_state_discount_0.7_useStreak_True')
# evalFile('run_fail/run_fail/model_state_discount_0.8_useStreak_False')
# evalFile('run_fail/run_fail/model_state_discount_0.8_useStreak_True')
# evalFile('run_fail/run_fail/model_state_discount_0.9_useStreak_False')
# evalFile('run_fail/run_fail/model_state_discount_0.9_useStreak_True')

def run(episodes, discount, useStreak):
    start_time = time.time()
    print('run with: ' + str(discount) + str(useStreak))
    batch_size = 32
    hidden_dim = 300
    experienceSize = 20000
    epsilon_min_after = 1500
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # writer = SummaryWriter('runs/run_discount_' + str(discount) + '_streak_' + str(useStreak))
    writer = SummaryWriter('test_run')
    trainer = Trainer(hidden_dim, experienceSize, discount, batch_size, device, writer)
    evaluator = Evaluator(100, trainer)
    evaluator.winPercentage(1)
    # for e in range(episodes):
    #     observation = trainer.reset()
    #     done = False
    #     batchReward = 0
    #     steps = 0
    #     threes = 0
    #     while not done:
    #         action = trainer.policyAction(observation['board'], e, epsilon_min_after)
    #         old_obs = observation
    #         observation, reward, done, _ = trainer.step(int(action))
    #         reshaped = trainer.reshape(torch.tensor(observation['board']))
    #         threes += trainer.streakReward(trainer.player, reshaped, int(action))
    #         if useStreak:
    #             reward = trainer.change_reward_streak(reward, done, reshaped, int(action), useStreak)
    #         else:
    #             reward = trainer.change_reward(reward, done)
    #         next_state = observation['board']
    #         exp = Experience(old_obs['board'], action, reward, next_state, int(done))
    #         trainer.addExperience(exp)
    #         batchReward += reward
    #         loss = trainer.train()
    #         steps += 1
    #     threes /= 3
    #     #if loss != None:
    #     #     writer.add_scalar('trainLoss', loss, e)
    #     # writer.add_scalar('batchReward', batchReward, e)
    #     # writer.add_scalar('steps', steps, e)
    #     # writer.add_scalar('threes', threes, e)
    #     if (e % 1000 == 0) and (e > 0):
    #         trainer.switchPosition()
    #     if e % 50 == 0:
    #         trainer.synchronize()
    #         firstStep = str(trainer.policy(trainer.reshape(torch.tensor(trainer.reset()['board'])))[0])
    #         writer.add_text('first_qs', firstStep, e)
    #     if e % 50 == 0:
    #         trainer.save("model_state_discount_" + str(discount) + '_useStreak_' + str(useStreak))
    #         print(e)
    #         print("episode: " + str(e) + " meanReward generateEpisodes: " + str(batchReward) + " meanLoss: " + str(loss))
    #         print("steps: " + str(steps))
    #         firstStep = str(trainer.policy(trainer.reshape(torch.tensor(trainer.reset()['board'])))[0])
    #         with torch.no_grad():
    #             print(firstStep)
    #         evaluator = Evaluator(100, trainer)
    #         evaluator.winPercentage(e)

    # if e % 25000 == 0:
    #     trainer.switch()
    #     trainer.save("model_state_"+str(e))
    print("--- %s seconds ---" % (time.time() - start_time))


# run(1000, 0.9, True)
# run(10, 0.8, False)
# run(10, 0.9, False)
print("done")
