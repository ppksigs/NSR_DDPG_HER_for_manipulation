import gym
from agent import Agent
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from play import Play
import mujoco_py
import random
from mpi4py import MPI
import psutil
import time
from copy import deepcopy as dc
import os
import torch
import argparse
import csv

def parse_arguments():
    parser = argparse.ArgumentParser(description="Configuration for the training script")

    # Environment name
    parser.add_argument("--env_name", type=str, default="FetchPickAndPlace-v1", help="Name of the environment")

    # Random weights used
    parser.add_argument("--rnd_weights_used", type=str, default="rnd_norm", choices=["rnd_norm", "ablation_mean", "ablation_mean_distrib", "not_used", "train_counts"], help="Type of random weights used")

    # Introduction flag
    parser.add_argument("--intro", action="store_true", help="Flag for introduction")
    parser.set_defaults(intro=False)

    # Training flag
    parser.add_argument("--train", action="store_true", help="Flag to enable training")
    parser.set_defaults(train=True)

    # Play flag
    parser.add_argument("--play", action="store_true", help="Flag to enable playing")
    parser.set_defaults(play=False)

    # Maximum epochs
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs")

    # Maximum cycles
    parser.add_argument("--max_cycles", type=int, default=50, help="Maximum number of cycles")

    # Number of updates
    parser.add_argument("--num_updates", type=int, default=40, help="Number of updates")

    # Maximum episodes
    parser.add_argument("--max_episodes", type=int, default=1, help="Maximum number of episodes")

    # Memory size
    parser.add_argument("--memory_size", type=int, default=7e+5 // 50, help="Size of the memory")

    # Batch size
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")

    # Actor learning rate
    parser.add_argument("--actor_lr", type=float, default=1e-3, help="Learning rate for the actor")

    # Critic learning rate
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="Learning rate for the critic")

    # Discount factor
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")

    # Tau
    parser.add_argument("--tau", type=float, default=0.05, help="Tau value")

    # Future steps
    parser.add_argument("--k_future", type=int, default=4, help="Number of future steps")

    # Random seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Added bias on rnd_norm
    parser.add_argument("--rnd_bias", type=float, default=1.0, help="Added bias on rnd_norm")

    # rnd_norm_min
    parser.add_argument("--rnd_norm_min", type=float, default=1.0, help="rnd_norm_min")

    # rnd_norm_max
    parser.add_argument("--rnd_norm_max", type=float, default=3.0, help="rnd_norm_max")

    parser.add_argument("--t_nums", type=int, default=50, help="t_nums")

    parser.add_argument("--threshold", type=float, default=0.02, help="threshold")

    args = parser.parse_args()
    return args

# Get the arguments
args = parse_arguments()

ENV_NAME = args.env_name
RND_WEIGHTS_USED = args.rnd_weights_used
INTRO = args.intro
Train = args.train
Play_FLAG = args.play
MAX_EPOCHS = args.max_epochs
MAX_CYCLES = args.max_cycles
num_updates = args.num_updates
MAX_EPISODES = args.max_episodes
memory_size = args.memory_size
batch_size = args.batch_size
actor_lr = args.actor_lr
critic_lr = args.critic_lr
gamma = args.gamma
tau = args.tau
k_future = args.k_future
randomSeed = args.seed
rnd_bias = args.rnd_bias
rnd_min = args.rnd_norm_min
rnd_max = args.rnd_norm_max
t_nums = args.t_nums
threshold = args.threshold
data_saved_path = f"./data_{ENV_NAME}_{rnd_bias}_{rnd_min}_{rnd_max}_num_updates_{num_updates}_t_{t_nums}_thr_{threshold}"
os.makedirs(data_saved_path, exist_ok=True)

test_env = gym.make(ENV_NAME)
state_shape = test_env.observation_space.spaces["observation"].shape
n_actions = test_env.action_space.shape[0]
n_goals = test_env.observation_space.spaces["desired_goal"].shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['IN_MPI'] = '1'


def eval_agent(env_, agent_):
    total_success_rate = []
    running_r = []
    for ep in range(10):
        per_success_rate = []
        env_dictionary = env_.reset()
        s = env_dictionary["observation"]
        ag = env_dictionary["achieved_goal"]
        g = env_dictionary["desired_goal"]
        while np.linalg.norm(ag - g) <= threshold:
            env_dictionary = env_.reset()
            s = env_dictionary["observation"]
            ag = env_dictionary["achieved_goal"]
            g = env_dictionary["desired_goal"]
        ep_r = 0
        for t in range(t_nums):
            with torch.no_grad():
                a = agent_.choose_action(s, g, train_mode=False)
            observation_new, r, _, info_ = env_.step(a)
            s = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info_['is_success'])
            ep_r += r
        total_success_rate.append(per_success_rate)
        if ep == 0:
            running_r.append(ep_r)
        else:
            running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return global_success_rate / MPI.COMM_WORLD.Get_size(), running_r, ep_r


if INTRO:
    print(f"state_shape:{state_shape[0]}\n"
          f"number of actions:{n_actions}\n"
          f"action boundaries:{action_bounds}\n"
          f"max timesteps:{test_env._max_episode_steps}")
    for _ in range(3):
        done = False
        test_env.reset()
        while not done:
            action = test_env.action_space.sample()
            test_state, test_reward, test_done, test_info = test_env.step(action)
            # substitute_goal = test_state["achieved_goal"].copy()
            # substitute_reward = test_env.compute_reward(
            #     test_state["achieved_goal"], substitute_goal, test_info)
            # print("r is {}, substitute_reward is {}".format(r, substitute_reward))
            test_env.render()
    exit(0)

env = gym.make(ENV_NAME)
env.seed(MPI.COMM_WORLD.Get_rank()+randomSeed)
random.seed(MPI.COMM_WORLD.Get_rank()+randomSeed)
np.random.seed(MPI.COMM_WORLD.Get_rank()+randomSeed)
torch.manual_seed(MPI.COMM_WORLD.Get_rank()+randomSeed)
agent = Agent(n_states=state_shape,
              n_actions=n_actions,
              n_goals=n_goals,
              action_bounds=action_bounds,
              capacity=memory_size,
              action_size=n_actions,
              batch_size=batch_size,
              actor_lr=actor_lr,
              critic_lr=critic_lr,
              gamma=gamma,
              tau=tau,
              k_future=k_future,
              env=dc(env),
              rnd_weights=RND_WEIGHTS_USED)
if Train:
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("ENV_NAME:{}".format(ENV_NAME))
        print("RND_WEIGHTS_USED:{}".format(RND_WEIGHTS_USED))
        print("Seed:seed{}".format(randomSeed))
    t_success_rate = []
    total_ac_loss = []
    total_cr_loss = []
    Running_reward = []
    EP_reward = []
    Actor_Loss = []
    Critic_Loss = []
    Duration = []
    step = 0
    all_epochs_start_time = time.time()
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        for cycle in range(0, MAX_CYCLES):
            if ENV_NAME == "FetchReach-v1":
                start_time = time.time()
            mb = []
            cycle_actor_loss = 0
            cycle_critic_loss = 0
            for episode in range(MAX_EPISODES):
                episode_dict = {
                    "state": [],
                    "action": [],
                    "info": [],
                    "achieved_goal": [],
                    "desired_goal": [],
                    "next_state": [],
                    "next_achieved_goal": []}
                env_dict = env.reset()
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
                while np.linalg.norm(achieved_goal - desired_goal) <= threshold:
                    env_dict = env.reset()
                    state = env_dict["observation"]
                    achieved_goal = env_dict["achieved_goal"]
                    desired_goal = env_dict["desired_goal"]
                for t in range(t_nums):
                    action = agent.choose_action(state, desired_goal)
                    next_env_dict, reward, done, info = env.step(action)

                    next_state = next_env_dict["observation"]
                    next_achieved_goal = next_env_dict["achieved_goal"]
                    next_desired_goal = next_env_dict["desired_goal"]

                    episode_dict["state"].append(state.copy())
                    episode_dict["action"].append(action.copy())
                    episode_dict["achieved_goal"].append(achieved_goal.copy())
                    episode_dict["desired_goal"].append(desired_goal.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()

                episode_dict["state"].append(state.copy())
                episode_dict["achieved_goal"].append(achieved_goal.copy())
                episode_dict["desired_goal"].append(desired_goal.copy())
                episode_dict["next_state"] = episode_dict["state"][1:]
                episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                mb.append(dc(episode_dict))

            agent.store(mb)
            for n_update in range(num_updates):
                actor_loss, critic_loss = agent.train(rnd_bias, rnd_min, rnd_max)
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss

            epoch_actor_loss += cycle_actor_loss / num_updates
            epoch_critic_loss += cycle_critic_loss /num_updates
            agent.update_networks()
            step += 1

            if ENV_NAME=="FetchReach-v1":
                ram = psutil.virtual_memory()
                success_rate, running_reward, episode_reward = eval_agent(env, agent)
                if MPI.COMM_WORLD.Get_rank() == 0:
                    Running_reward.append(running_reward[-1])
                    t_success_rate.append(success_rate)
                    Duration.append(time.time() - start_time)
                    EP_reward.append(episode_reward)
                    Actor_Loss.append(cycle_actor_loss)
                    Critic_Loss.append(cycle_critic_loss)
                    print(f"Step:{step}| "
                        f"Running_reward:{running_reward[-1]:.3f}| "
                        f"EP_reward:{episode_reward:.3f}| "
                        f"Memory_length:{len(agent.memory)}| "
                        f"Duration:{time.time() - start_time:.3f}| "
                        f"Actor_Loss:{actor_loss:.3f}| "
                        f"Critic_Loss:{critic_loss:.3f}| "
                        f"Success rate:{success_rate:.3f}| "
                        f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")

        if ENV_NAME!="FetchReach-v1":
            ram = psutil.virtual_memory()
            success_rate, running_reward, episode_reward = eval_agent(env, agent)
            total_ac_loss.append(epoch_actor_loss)
            total_cr_loss.append(epoch_critic_loss)
            if MPI.COMM_WORLD.Get_rank() == 0:
                Running_reward.append(running_reward[-1])
                t_success_rate.append(success_rate)
                Duration.append(time.time() - start_time)
                EP_reward.append(episode_reward)
                Actor_Loss.append(actor_loss)
                Critic_Loss.append(critic_loss)
                print(f"Epoch:{epoch}| "
                    f"Running_reward:{running_reward[-1]:.3f}| "
                    f"EP_reward:{episode_reward:.3f}| "
                    f"Memory_length:{len(agent.memory)}| "
                    f"Duration:{time.time() - start_time:.3f}| "
                    f"Actor_Loss:{actor_loss:.3f}| "
                    f"Critic_Loss:{critic_loss:.3f}| "
                    f"Success rate:{success_rate:.3f}| "
                    f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")
                agent.save_weights()

        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                agent.save_weights()

    all_epochs_end_time = time.time()
    total_training_time = all_epochs_end_time - all_epochs_start_time 

    if MPI.COMM_WORLD.Get_rank() == 0 and ENV_NAME!="FetchReach-v1":        
        csvName = data_saved_path + ENV_NAME + "_" + RND_WEIGHTS_USED + "_" + "seed" + str(randomSeed) + ".csv"

        with SummaryWriter("logs") as writer:
            for i, success_rate in enumerate(t_success_rate):
                writer.add_scalar("Success_rate", success_rate, i)

        with open(csvName, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Iteration", "Success_Rate","Running_reward", "EP_reward","Duration", "Actor_Loss", "Critic_Loss"])
            for i,success_rate,running_reward,episode_reward,duration, actor_loss,critic_loss in zip(range(MAX_EPOCHS),t_success_rate,Running_reward,EP_reward,Duration,Actor_Loss,Critic_Loss):
                writer.writerow([i, success_rate, running_reward, episode_reward, actor_loss, critic_loss])

            writer.writerow([])
            writer.writerow(f"Total training time: {total_training_time} seconds")

    elif MPI.COMM_WORLD.Get_rank() == 0 and ENV_NAME=="FetchReach-v1":
        csvName = data_saved_path + ENV_NAME + "_" + RND_WEIGHTS_USED + "_" + "seed" + str(randomSeed) + ".csv"

        with SummaryWriter("logs") as writer:
            for i, success_rate in enumerate(t_success_rate):
                writer.add_scalar("Success_rate", success_rate, i)

        with open(csvName, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Iteration", "Success_Rate","Running_reward", "EP_reward","Duration", "Actor_Loss", "Critic_Loss"])
            for i,success_rate,running_reward,episode_reward,duration, actor_loss,critic_loss in zip(range(MAX_EPOCHS*MAX_CYCLES),t_success_rate,Running_reward,EP_reward,Duration,Actor_Loss,Critic_Loss):
                writer.writerow([i, success_rate, running_reward, episode_reward, actor_loss, critic_loss])

            writer.writerow([])
            writer.writerow(f"Total training time: {total_training_time} seconds")

elif Play_FLAG:
    player = Play(env, agent, max_episode=100)
    player.evaluate()