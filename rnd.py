import torch.nn as nn

class RND(nn.Module):
    def __init__(self, n_states, hidden_dim=128):
        n_states = n_states[0]
        super(RND, self).__init__()
        self.target = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, requires_grad=True):
        target_output = self.target(x)
        predict_output = self.predictor(x)
        intrinsic_reward = ((target_output - predict_output) ** 2).sum(-1)
        if not requires_grad:
            return intrinsic_reward.detach()
        return intrinsic_reward
