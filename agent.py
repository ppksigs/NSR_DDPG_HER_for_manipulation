import torch
from torch import from_numpy, device
import numpy as np
from models import Actor, Critic
from memory import Memory
from torch.optim import Adam
from mpi4py import MPI
from normalizer import Normalizer
from rnd import RND

class Agent:
    def __init__(self, n_states, n_actions, n_goals, action_bounds, capacity, env, rnd_weights,
                 k_future,
                 batch_size,
                 action_size=1,
                 tau=0.05,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 rnd_lr = 1e-4,
                 gamma=0.98):
        self.device = device("cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.device = torch.device("cuda")
            # print("Using GPU:", torch.cuda.get_device_name(0))
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.k_future = k_future
        self.action_bounds = action_bounds
        self.action_size = action_size
        self.env = env
        self.rnd_weights = rnd_weights
        self.rnd = RND(n_states).to(self.device)
        self.actor = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(self.device)
        self.critic = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(self.device)
        self.sync_networks(self.actor)
        self.sync_networks(self.critic)
        self.actor_target = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(self.device)
        self.critic_target = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(self.device)
        self.init_target_networks()
        self.tau = tau
        self.gamma = gamma

        self.capacity = capacity
        self.memory = Memory(self.capacity, self.k_future, self.env)

        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.rnd_lr = rnd_lr
        self.actor_optim = Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), self.critic_lr)
        self.rnd_optim = Adam(self.rnd.parameters(), self.rnd_lr)
        self.rnd_list = []

        self.state_normalizer = Normalizer(self.n_states[0], default_clip_range=5)
        self.goal_normalizer = Normalizer(self.n_goals, default_clip_range=5)

    def choose_action(self, state, goal, train_mode=True):
        state = self.state_normalizer.normalize(state)
        goal = self.goal_normalizer.normalize(goal)
        state = np.expand_dims(state, axis=0)
        goal = np.expand_dims(goal, axis=0)

        with torch.no_grad():
            x = np.concatenate([state, goal], axis=1)
            x = from_numpy(x).float().to(self.device)
            action = self.actor(x)[0].cpu().data.numpy()

        if train_mode:
            action += 0.2 * np.random.randn(self.n_actions)
            action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

            random_actions = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1],
                                               size=self.n_actions)
            action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)

        return action

    def store(self, mini_batch):
        for batch in mini_batch:
            self.memory.add(batch)
        self._update_normalizer(mini_batch)

    def init_target_networks(self):
        self.hard_update_networks(self.actor, self.actor_target)
        self.hard_update_networks(self.critic, self.critic_target)

    @staticmethod
    def hard_update_networks(local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())

    @staticmethod
    def soft_update_networks(local_model, target_model, tau=0.05):
        for t_params, e_params in zip(target_model.parameters(), local_model.parameters()):
            t_params.data.copy_(tau * e_params.data + (1 - tau) * t_params.data)

    def train(self, bias=1.0, min=1.0, max=3.0):
        states, actions, rewards, next_states, goals = self.memory.sample(self.batch_size)

        states = self.state_normalizer.normalize(states)
        next_states = self.state_normalizer.normalize(next_states)
        goals = self.goal_normalizer.normalize(goals)
        inputs = np.concatenate([states, goals], axis=1)
        next_inputs = np.concatenate([next_states, goals], axis=1)

        inputs = torch.Tensor(inputs).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_inputs = torch.Tensor(next_inputs).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)

        rnd_loss = self.rnd(next_states).to(self.device)
        rnd_loss_nograd = rnd_loss.clone().detach()
        rnd_loss_mean = rnd_loss_nograd.mean().item()
        rnd_loss_std = rnd_loss_nograd.std().item()
        rnd_loss_norm = (rnd_loss-rnd_loss_mean)/(rnd_loss_std+1e-6)
        

        if self.rnd_weights != "train_counts":
            if self.rnd_weights == "rnd_norm":
                used_rnd_weights = torch.clamp((rnd_loss_norm/rnd_loss_std)+bias, min, max).unsqueeze(-1)
            elif self.rnd_weights == "ablation_mean":
                used_rnd_weights = torch.clamp((rnd_loss_norm/rnd_loss_std)+bias, min, max).unsqueeze(-1)
                used_rnd_weights = used_rnd_weights.mean()
            elif self.rnd_weights == "ablation_mean_distrib":
                used_rnd_weights = torch.clamp((rnd_loss_norm/rnd_loss_std)+bias, min, max).unsqueeze(-1)
                used_rnd_weights_mean = used_rnd_weights.mean()
                used_rnd_weights_std = used_rnd_weights.std()
                used_rnd_weights = torch.normal(used_rnd_weights_mean.item(), used_rnd_weights_std.item(), size=rnd_loss_norm.shape).to(self.device)
            else:
                used_rnd_weights = torch.Tensor([1.0]).to(self.device)


            with torch.no_grad():
                target_q = self.critic_target(next_inputs, self.actor_target(next_inputs))
                target_returns = rewards + self.gamma * target_q.detach()
                target_returns = torch.clamp(target_returns, -1 / (1 - self.gamma), 0)
            q_eval = self.critic(inputs, actions)
            critic_loss = ((target_returns - q_eval)*used_rnd_weights.detach()).pow(2).mean()

            a = self.actor(inputs)
            actor_loss = -(self.critic(inputs, a)*used_rnd_weights.detach()).mean()
            actor_loss += a.pow(2).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.sync_grads(self.actor)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.sync_grads(self.critic)
            self.critic_optim.step()


        else:
            train_counts = torch.clamp((rnd_loss_norm/rnd_loss_std).int()+bias, min, max).unsqueeze(-1)

            for _ in range(3): 
                with torch.no_grad():
                    target_q = self.critic_target(next_inputs, self.actor_target(next_inputs))
                    target_returns = rewards + self.gamma * target_q.detach()
                    target_returns = torch.clamp(target_returns, -1 / (1 - self.gamma), 0)
                q_eval = self.critic(inputs, actions)

                critic_loss = ((target_returns - q_eval)*((train_counts>0).float().detach())).pow(2).mean()
                a = self.actor(inputs)
                actor_loss = -(self.critic(inputs, a)*((train_counts>0).float().detach())).mean()
                actor_loss += (a.pow(2)*((train_counts>0).float().detach())).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.sync_grads(self.actor)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.sync_grads(self.critic)
                self.critic_optim.step()

                train_counts = train_counts - 1
                

        self.rnd_optim.zero_grad()
        rnd_loss.mean().backward()
        self.rnd_optim.step()

            
        return actor_loss.mean().item(), critic_loss.mean().item()

    def save_weights(self):
        torch.save({"actor_state_dict": self.actor.state_dict(),
                    "state_normalizer_mean": self.state_normalizer.mean,
                    "state_normalizer_std": self.state_normalizer.std,
                    "goal_normalizer_mean": self.goal_normalizer.mean,
                    "goal_normalizer_std": self.goal_normalizer.std}, "FetchPickAndPlace.pth")

    def load_weights(self):

        checkpoint = torch.load("FetchPickAndPlace.pth")
        actor_state_dict = checkpoint["actor_state_dict"]
        self.actor.load_state_dict(actor_state_dict)
        state_normalizer_mean = checkpoint["state_normalizer_mean"]
        self.state_normalizer.mean = state_normalizer_mean
        state_normalizer_std = checkpoint["state_normalizer_std"]
        self.state_normalizer.std = state_normalizer_std
        goal_normalizer_mean = checkpoint["goal_normalizer_mean"]
        self.goal_normalizer.mean = goal_normalizer_mean
        goal_normalizer_std = checkpoint["goal_normalizer_std"]
        self.goal_normalizer.std = goal_normalizer_std

    def set_to_eval_mode(self):
        self.actor.eval()
        # self.critic.eval()

    def update_networks(self):
        self.soft_update_networks(self.actor, self.actor_target, self.tau)
        self.soft_update_networks(self.critic, self.critic_target, self.tau)

    def _update_normalizer(self, mini_batch):
        states, goals = self.memory.sample_for_normalization(mini_batch)

        self.state_normalizer.update(states)
        self.goal_normalizer.update(goals)
        self.state_normalizer.recompute_stats()
        self.goal_normalizer.recompute_stats()

    @staticmethod
    def sync_networks(network):
        comm = MPI.COMM_WORLD
        flat_params = _get_flat_params_or_grads(network, mode='params')
        comm.Bcast(flat_params, root=0)
        _set_flat_params_or_grads(network, flat_params, mode='params')

    @staticmethod
    def sync_grads(network):
        flat_grads = _get_flat_params_or_grads(network, mode='grads')
        comm = MPI.COMM_WORLD
        global_grads = np.zeros_like(flat_grads)
        comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
        _set_flat_params_or_grads(network, global_grads, mode='grads')


def _get_flat_params_or_grads(network, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(
            torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()
