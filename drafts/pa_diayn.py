# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer

from rlnav import PointMazeV0  # Added by hed

# from envs.wenv import Wenv  # Removed by hed
# from envs.config_env import config  # Removed by hed
# from src.utils.wandb_utils import send_matrix  # Removed by hed


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "contrastive_test_2"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    use_hp_file: bool = False
    """if toggled, will load the hyperparameters from file"""
    hp_file: str = "hyper_parameters_sac.json"
    """the path to the hyperparameters json file"""
    sweep_mode: bool = False
    """if toggled, will log the sweep id to wandb"""

    # GIF
    make_gif: bool = True
    """if toggled, will make gif """
    plotly: bool = False
    """if toggled, will use plotly instead of matplotlib"""
    fig_frequency: int = 1000
    """the frequency of logging the figures"""
    metric_freq: int = 1000
    """the frequency of ploting metric"""

    # Algorithm specific arguments
    env_id: str = "Maze-Ur-v0"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e7)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""
    learning_frequency: int = 2
    """the frequency of training the Q network"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.1
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""
    num_envs: int = 4
    """ num of parallel envs """

    # DIAYN specific arguments
    nb_skill: int = 4
    """the number of skills"""
    lr_classifier_diayn: float = 1e-3
    """the learning rate of the classifier"""
    nb_epoch_before_training: int = 4
    """ nb epoch between each training """
    diayn_epochs: int = 32
    """ nb epoch for diayn training """

    keep_extrinsic_reward: bool = False
    """if toggled, the extrinsic reward will be kept"""
    coef_intrinsic: float = 5.0
    """the coefficient of the intrinsic reward"""
    coef_extrinsic: float = 1.0
    """the coefficient of the extrinsic reward"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():


        maze_array = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]
        env = PointMazeV0(maze_array=maze_array, goal_conditioned=False, action_noise=0.0)

        # env = Wenv(env_id=env_id, xp_id=run_name, **config[env_id])  # Removed by hed
        # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # if capture_video:
        #     if idx == 0:
        #         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ClipAction(env)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, nb_skill):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape) + nb_skill,
            256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, z, a):
        x = torch.cat([x, z, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, nb_skill):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + nb_skill, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x, z):
        x = torch.cat([x, z], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x, z, eps=None):
        mean, log_std = self(x, z)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() if eps is None else mean + (
                    std * eps)  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Classifier(torch.nn.Module):
    def __init__(self,
                 observation_space,
                 nb_skill,
                 device,
                 env_id,
                 feature_extractor=False):
        super(Classifier, self).__init__()
        self.relu = torch.nn.ReLU()
        self.nb_skill = nb_skill
        self.env_id = env_id
        self.feature_extractor = feature_extractor
        if feature_extractor:
            self.fcz1 = torch.nn.Linear(2, 128, device=device)
            self.fcz2 = torch.nn.Linear(128, 64, device=device)
            self.fcz3 = torch.nn.Linear(64, nb_skill, device=device)
        else:
            self.fcz1 = torch.nn.Linear(observation_space.shape[0], 128, device=device)
            self.fcz2 = torch.nn.Linear(128, 64, device=device)
            self.fcz3 = torch.nn.Linear(64, nb_skill, device=device)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.feature(x) if self.feature_extractor else x
        x = self.relu(self.fcz1(x))
        x = self.relu(self.fcz2(x))
        return self.softmax(self.fcz3(x))

    def mlh_loss(self, obs, z):
        # change dtype to int
        z = z.type(torch.int64)
        p_z = self.forward(obs)
        p_z_i = torch.sum(p_z * z, dim=-1)
        return -torch.mean(torch.log(p_z_i + 1e-3))

    def feature(self, x):
        x = x[:, :, 2] if x.dim() == 3 else x[:, 2]
        return x


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    if args.use_hp_file:
        import json

        with open(args.hp_file, "r") as f:
            type_id = 'maze'
            hp = json.load(f)['hyperparameters'][type_id][args.exp_name]
            for k, v in hp.items():
                setattr(args, k, v)

    # DIAYN Setup
    args.num_envs = args.nb_skill  # overide
    z = np.eye(args.nb_skill, dtype=np.float32)  # skill one hot vector

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        if args.sweep_mode:
            wandb.init()
            # set config from sweep
            wandb.config.update(args)
        else:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_step = config[args.env_id]['kwargs']['max_episode_steps']
    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs, args.nb_skill).to(device)
    qf1 = SoftQNetwork(envs, args.nb_skill).to(device)
    qf2 = SoftQNetwork(envs, args.nb_skill).to(device)
    qf1_target = SoftQNetwork(envs, args.nb_skill).to(device)
    qf2_target = SoftQNetwork(envs, args.nb_skill).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    classifier_diayn = Classifier(observation_space=envs.single_observation_space, nb_skill=args.nb_skill,
                                  device=device, env_id=args.env_id, feature_extractor=True)
    classifier_diayn_optimizer = optim.Adam(list(classifier_diayn.parameters()), lr=args.lr_classifier_diayn)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=args.num_envs
    )
    # add time
    rb.times = np.zeros((args.buffer_size, args.num_envs), dtype=int)
    rb.zs = np.zeros((args.buffer_size, args.num_envs, args.nb_skill), dtype=np.float32)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # coverage assessment
        env_check.update_coverage(obs)
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device), torch.Tensor(z).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                    wandb.log({
                        "charts/episodic_return": info["episode"]["r"],
                        "charts/episodic_length": info["episode"]["l"],
                    }, step=global_step) if args.track else None

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        rb.times[rb.pos - 1 if not rb.full else rb.buffer_size - 1] = infos['l']
        rb.zs[rb.pos - 1 if not rb.full else rb.buffer_size - 1] = z

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # DIAYN TRAINING
        if global_step % args.nb_epoch_before_training * max_step == 0 and global_step > args.learning_starts:
            mean_diayn_loss = 0.0
            for _ in range(args.diayn_epochs):
                # for _ in range(int(args.nb_epoch_before_training*max_step/args.vae_batch_size)):
                b_inds = np.random.randint(0, rb.pos if not rb.full else rb.buffer_size, args.batch_size)
                b_inds_envs = np.random.randint(0, args.num_envs, args.batch_size)
                b_obs = torch.tensor(rb.observations[b_inds, b_inds_envs], device=device)
                b_z = torch.tensor(rb.zs[b_inds, b_inds_envs], device=device)
                diayn_loss = classifier_diayn.mlh_loss(b_obs, b_z)
                classifier_diayn_optimizer.zero_grad()
                diayn_loss.backward()
                mean_diayn_loss += diayn_loss.item()
            wandb.log({
                "losses/diayn_loss": mean_diayn_loss / args.diayn_epochs,
            }, step=global_step) if args.track else None

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.learning_frequency == 0:
            with torch.no_grad():
                b_inds = np.random.randint(0, rb.pos if not rb.full else rb.buffer_size, args.batch_size)
                b_inds_envs = np.random.randint(0, args.num_envs, args.batch_size)
                b_obs = torch.tensor(rb.observations[b_inds, b_inds_envs], device=device)
                b_next_obs = torch.tensor(rb.next_observations[b_inds, b_inds_envs], device=device)
                b_actions = torch.tensor(rb.actions[b_inds, b_inds_envs], device=device)
                b_rewards = torch.tensor(rb.rewards[b_inds, b_inds_envs], device=device)
                b_dones = torch.tensor(rb.dones[b_inds, b_inds_envs], device=device)
                b_z = torch.tensor(rb.zs[b_inds, b_inds_envs], device=device)
                next_state_actions, next_state_log_pi, _ = actor.get_action(b_next_obs, b_z)
                qf1_next_target = qf1_target(b_next_obs, b_z, next_state_actions)
                qf2_next_target = qf2_target(b_next_obs, b_z, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                intrinsic_reward = torch.log(torch.sum(classifier_diayn(b_obs) * b_z, dim=-1) + 1e-3) - torch.log(
                    torch.tensor(1 / args.nb_skill))
                extrinsic_reward = b_rewards.flatten()
                if args.keep_extrinsic_reward:
                    rewards = extrinsic_reward * args.coef_extrinsic + intrinsic_reward * args.coef_intrinsic
                else:
                    rewards = intrinsic_reward * args.coef_intrinsic
                next_q_value = rewards + (1 - b_dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(b_obs, b_z, b_actions).view(-1)
            qf2_a_values = qf2(b_obs, b_z, b_actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                        args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(b_obs, b_z)
                    qf1_pi = qf1(b_obs, b_z, pi)
                    qf2_pi = qf2(b_obs, b_z, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(b_obs, b_z)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                wandb.log({
                    "losses/qf1_values": qf1_a_values.mean().item(),
                    "losses/qf2_values": qf2_a_values.mean().item(),
                    "losses/qf1_loss": qf1_loss.item(),
                    "losses/qf2_loss": qf2_loss.item(),
                    "losses/qf_loss": qf_loss.item() / 2.0,
                    "losses/actor_loss": actor_loss.item(),
                    "losses/alpha": alpha,
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                    "losses/alpha_loss": alpha_loss.item() if args.autotune else 0.0,
                    "specific/intrisic_reward_mean": intrinsic_reward.mean().item(),
                    "specific/intrisic_reward_max": intrinsic_reward.max().item(),
                    "specific/intrisic_reward_min": intrinsic_reward.min().item(),
                }, step=global_step) if args.track else None


    envs.close()