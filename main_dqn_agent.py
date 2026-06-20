"""
DQN Agent Implementation
Deep Q-Network for autonomous car navigation
"""

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from main_constant import (
    MEMORY_SIZE,
    LEARNING_RATE,
    GAMMA,
    TRAIN_MAX_EPSILON,
    TRAIN_MIN_EPSILON,
    EPSILON_DECAY,
    BATCH_SIZE,
    TARGET_UPDATE_FREQ,
    DQN_HIDDEN_SIZES,
    GRAD_CLIP_MAX_NORM,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Supported algorithm identifiers
# ---------------------------------------------------------------------------
ALGO_DQN = "dqn"
ALGO_DDQN = "ddqn"
ALGO_D3QN = "d3qn"
SUPPORTED_ALGOS = {ALGO_DQN, ALGO_DDQN, ALGO_D3QN}


def _validate_algo(algo: str) -> str:
    """Normalise and validate the algorithm identifier."""
    algo = str(algo).strip().lower()
    if algo not in SUPPORTED_ALGOS:
        raise ValueError(
            f"Unknown algorithm '{algo}'. Supported: {sorted(SUPPORTED_ALGOS)}"
        )
    return algo


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture (used by DQN and DDQN)."""

    def __init__(self, state_size, action_size, hidden_sizes=None):
        super(DQNNetwork, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = DQN_HIDDEN_SIZES

        layers = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN architecture (Wang et al., 2016).

    Separates Q into a *state-value* stream V(s) and an *advantage* stream A(s,a).
    Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]

    This is the network used by D3QN (Dueling Double DQN).
    The shared feature layers feed into two independent heads so the agent can
    learn which states are valuable independently of the effect of each action.
    """

    def __init__(self, state_size, action_size, hidden_sizes=None):
        super(DuelingDQNNetwork, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = DQN_HIDDEN_SIZES

        # Shared feature layers (all hidden layers except the last)
        shared_layers = []
        input_size = state_size
        for hidden_size in hidden_sizes[:-1]:
            shared_layers.append(nn.Linear(input_size, hidden_size))
            shared_layers.append(nn.ReLU())
            input_size = hidden_size
        self.feature = nn.Sequential(*shared_layers) if shared_layers else nn.Identity()
        feature_out = input_size

        # If only one hidden size, the last one becomes the stream size
        stream_hidden = hidden_sizes[-1] if hidden_sizes else 64

        # Value stream: feature -> stream_hidden -> 1
        self.value_stream = nn.Sequential(
            nn.Linear(feature_out, stream_hidden),
            nn.ReLU(),
            nn.Linear(stream_hidden, 1),
        )

        # Advantage stream: feature -> stream_hidden -> action_size
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_out, stream_hidden),
            nn.ReLU(),
            nn.Linear(stream_hidden, action_size),
        )

        # Expose a .network attribute so that existing code that iterates
        # over linear layers (formula logging, neuron trace) still works.
        # It lists all Linear layers in order: shared, value, advantage.
        _all_layers = []
        for mod in self.feature:
            _all_layers.append(mod)
        for mod in self.value_stream:
            _all_layers.append(mod)
        for mod in self.advantage_stream:
            _all_layers.append(mod)
        self.network = nn.Sequential(*_all_layers)

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)            # (batch, 1)
        advantage = self.advantage_stream(features)    # (batch, action_size)
        # Mean-centering: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, discount):
        self.buffer.append((state, action, reward, next_state, done, discount))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, discounts = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(discounts, dtype=np.float32),
        )
        
    def __len__(self):
        return len(self.buffer)

    def state_dict(self):
        """Return a torch-checkpoint-friendly replay buffer payload."""
        transitions = []
        for state, action, reward, next_state, done, discount in self.buffer:
            transitions.append(
                {
                    "state": np.asarray(state, dtype=np.float32).tolist(),
                    "action": int(action),
                    "reward": float(reward),
                    "next_state": np.asarray(next_state, dtype=np.float32).tolist(),
                    "done": bool(done),
                    "discount": float(discount),
                }
            )
        return {
            "capacity": int(self.buffer.maxlen or len(self.buffer)),
            "transitions": transitions,
        }

    def load_state_dict(self, state):
        """Restore replay buffer payload saved by state_dict()."""
        if not isinstance(state, dict):
            return

        capacity = int(state.get("capacity") or self.buffer.maxlen or MEMORY_SIZE)
        restored = deque(maxlen=capacity)
        for item in state.get("transitions", []):
            if not isinstance(item, dict):
                continue
            restored.append(
                (
                    np.asarray(item.get("state", []), dtype=np.float32),
                    int(item.get("action", 0)),
                    float(item.get("reward", 0.0)),
                    np.asarray(item.get("next_state", []), dtype=np.float32),
                    bool(item.get("done", False)),
                    float(item.get("discount", 1.0)),
                )
            )
        self.buffer = restored

def _build_network(algo, state_size, action_size):
    """Instantiate the correct network architecture for the chosen algorithm."""
    algo = _validate_algo(algo)
    if algo == ALGO_D3QN:
        return DuelingDQNNetwork(state_size, action_size)
    # DQN and DDQN both use the standard network
    return DQNNetwork(state_size, action_size)


class DQNAgent:
    """DQN / DDQN / D3QN agent with replay buffer and target network.

    algo:
        'dqn'  – vanilla DQN  (Mnih et al., 2015)
        'ddqn' – Double DQN    (van Hasselt et al., 2016)
        'd3qn' – Dueling Double DQN  (Wang et al., 2016 + Double)
    """

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=TRAIN_MAX_EPSILON,
        epsilon_min=TRAIN_MIN_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        memory_size=MEMORY_SIZE,
        algo=ALGO_DQN,
    ):
        self.algo = _validate_algo(algo)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.memory_size = int(memory_size)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0


        self.policy_net = _build_network(self.algo, state_size, action_size).to(device)
        self.target_net = _build_network(self.algo, state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)


        self.memory = ReplayBuffer(capacity=self.memory_size)


        self.loss_history = []

        self.last_traininfo = None
        self.last_checkpoint = None

    def select_action(self, state, training=True, debug=False):
        """Select action using epsilon-greedy policy.

        If `debug=True`, returns `(action, info_dict)` where info_dict includes:
        - explore: bool (whether random action was used)
        - rand: float (random draw used for epsilon decision, None if not training)
        - epsilon: float
        """
        rand_value = None
        explore = False
        if training:
            rand_value = random.random()
            explore = rand_value < self.epsilon

        if explore:
            action = random.randrange(self.action_size)
            if debug:
                return action, {"explore": True, "rand": rand_value, "epsilon": float(self.epsilon)}
            return action

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
            if debug:
                return action, {"explore": False, "rand": rand_value, "epsilon": float(self.epsilon)}
            return action

    def get_q_values(self, state):
        """Return Q-values for a given state as a 1D numpy array.

        This is used only for visualization/debugging and does not affect training.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        return q_values

    @staticmethod
    def _formula_float(value):
        return f"{float(value):+.6f}"

    def _formula_action_labels(self):
        if int(self.action_size) == 6:
            return [
                "slow_left",
                "slow_straight",
                "slow_right",
                "fast_left",
                "fast_straight",
                "fast_right",
            ]
        return [f"a{i}" for i in range(int(self.action_size))]

    def get_q_values_with_calculation(self, state, tag=None):
        """Return Q-values while printing a detailed forward-pass breakdown."""
        step_tag = str(tag) if tag is not None else "?/?"
        state_array = np.asarray(state, dtype=np.float64).reshape(-1)
        linear_layers = [
            module
            for module in self.policy_net.network
            if isinstance(module, nn.Linear)
        ]
        total_linear_layers = len(linear_layers)
        action_labels = self._formula_action_labels()

        input_line = ", ".join(
            f"x[{idx}]={self._formula_float(value)}"
            for idx, value in enumerate(state_array)
        )
        print(f"[FORMULA][{step_tag}] input = [{input_line}]")

        activations = np.array(state_array, dtype=np.float64, copy=True)
        linear_index = 0

        for module in self.policy_net.network:
            if not isinstance(module, nn.Linear):
                continue

            linear_index += 1
            is_output_layer = linear_index == total_linear_layers
            weights = module.weight.detach().cpu().numpy().astype(np.float64, copy=False)
            bias = module.bias.detach().cpu().numpy().astype(np.float64, copy=False)
            input_symbol = "x" if linear_index == 1 else f"a{linear_index - 1}"
            z_symbol = "q" if is_output_layer else f"z{linear_index}"
            layer_name = "OUTPUT" if is_output_layer else f"L{linear_index}"

            print(
                f"[FORMULA][{step_tag}][Q_CALCULATION] {layer_name} Linear("
                f"{weights.shape[1]} -> {weights.shape[0]})"
            )

            pre_activation = np.zeros(weights.shape[0], dtype=np.float64)

            for out_idx in range(weights.shape[0]):
                symbol_label = (
                    action_labels[out_idx]
                    if is_output_layer and out_idx < len(action_labels)
                    else str(out_idx)
                )
                term_formula = " + ".join(
                    f"(W{linear_index}[{out_idx},{in_idx}]*{input_symbol}[{in_idx}])"
                    for in_idx in range(weights.shape[1])
                )
                term_substitution = " + ".join(
                    f"({self._formula_float(weights[out_idx, in_idx])}*"
                    f"{self._formula_float(activations[in_idx])})"
                    for in_idx in range(weights.shape[1])
                )
                term_products_list = [
                    float(weights[out_idx, in_idx] * activations[in_idx])
                    for in_idx in range(weights.shape[1])
                ]
                term_products = " + ".join(
                    self._formula_float(product) for product in term_products_list
                )
                weighted_sum = float(np.sum(term_products_list))
                z_value = float(bias[out_idx] + weighted_sum)
                pre_activation[out_idx] = z_value

                if is_output_layer:
                    print(
                        f"[FORMULA][{step_tag}] "
                        f"output[{symbol_label}] formula: "
                        f"q[{out_idx}] = b{linear_index}[{out_idx}] + {term_formula}"
                    )
                    print(
                        f"[FORMULA][{step_tag}] "
                        f"output[{symbol_label}] values : "
                        f"q[{out_idx}] = {self._formula_float(bias[out_idx])} + "
                        f"{term_substitution}"
                    )
                    print(
                        f"[FORMULA][{step_tag}] "
                        f"output[{symbol_label}] prods  : "
                        f"q[{out_idx}] = {self._formula_float(bias[out_idx])} + "
                        f"{term_products}"
                    )
                    print(
                        f"[FORMULA][{step_tag}] "
                        f"output[{symbol_label}] result : "
                        f"q[{out_idx}] = {self._formula_float(bias[out_idx])} + "
                        f"{self._formula_float(weighted_sum)} = {self._formula_float(z_value)}"
                    )
                else:
                    print(
                        f"[FORMULA][{step_tag}] "
                        f"L{linear_index}[{out_idx}] formula: "
                        f"{z_symbol}[{out_idx}] = b{linear_index}[{out_idx}] + {term_formula}"
                    )
                    print(
                        f"[FORMULA][{step_tag}] "
                        f"L{linear_index}[{out_idx}] values : "
                        f"{z_symbol}[{out_idx}] = {self._formula_float(bias[out_idx])} + "
                        f"{term_substitution}"
                    )
                    print(
                        f"[FORMULA][{step_tag}] "
                        f"L{linear_index}[{out_idx}] prods  : "
                        f"{z_symbol}[{out_idx}] = {self._formula_float(bias[out_idx])} + "
                        f"{term_products}"
                    )
                    print(
                        f"[FORMULA][{step_tag}] "
                        f"L{linear_index}[{out_idx}] result : "
                        f"{z_symbol}[{out_idx}] = {self._formula_float(bias[out_idx])} + "
                        f"{self._formula_float(weighted_sum)} = {self._formula_float(z_value)}"
                    )

            if is_output_layer:
                activations = pre_activation
                continue

            post_activation = np.maximum(pre_activation, 0.0)
            for out_idx, (z_value, a_value) in enumerate(
                zip(pre_activation, post_activation)
            ):
                print(
                    f"[FORMULA][{step_tag}][Q_CALCULATION] "
                    f"L{linear_index}[{out_idx}] relu   : "
                    f"a{linear_index}[{out_idx}] = max(0, {z_symbol}[{out_idx}]="
                    f"{self._formula_float(z_value)}) = {self._formula_float(a_value)}"
                )
            activations = post_activation

        output_line = ", ".join(
            f"{label}={self._formula_float(value)}"
            for label, value in zip(action_labels, activations)
        )
        print(f"[FORMULA][{step_tag}][Q_CALCULATION] output = [{output_line}]")
        return activations.astype(np.float32)

    def remember(self, state, action, reward, next_state, done, discount=None):
        """Store experience in replay buffer.

        discount is the bootstrapping factor applied to maxQ(next_state).
        - 1-step transition: discount = gamma
        - N-step transition: discount = gamma ** N
        """
        if discount is None:
            discount = self.gamma
        self.memory.push(state, action, reward, next_state, done, discount)

    def train_step(self, formula=False, tag=None, traininfo=False):
        """Perform one training step.

        When `formula=True`, prints detailed per-step calculations (sampled from batch[0]).
        """
        if len(self.memory) < self.batch_size:
            if formula:
                step_tag = str(tag) if tag is not None else "?/?"
                print(
                    f"[FORMULA][{step_tag}][TRAIN_SKIP] len(memory)={len(self.memory)} < batch_size={self.batch_size}"
                )
            if traininfo:
                self.last_traininfo = {
                    "skipped": True,
                    "reason": "buffer",
                    "buffer_len": int(len(self.memory)),
                    "batch_size": int(self.batch_size),
                }
            return None


        states, actions, rewards, next_states, dones, discounts = self.memory.sample(
            self.batch_size
        )


        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        discounts = torch.FloatTensor(discounts).to(device)


        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))


        with torch.no_grad():
            if self.algo == ALGO_DQN:
                # Vanilla DQN: action selection AND evaluation by target_net
                # y = r + γ * max_a' Q_target(s', a')
                next_q_values = self.target_net(next_states).max(1)[0]
            else:
                # DDQN / D3QN: action selection by policy_net,
                # evaluation by target_net  (van Hasselt et al., 2016)
                # a* = argmax_a' Q_policy(s', a')
                # y  = r + γ * Q_target(s', a*)
                best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q_values = rewards + (1 - dones) * discounts * next_q_values


        a0 = r0 = d0 = g0 = max_next0 = y0 = q0 = delta0 = se0 = None
        if formula or traininfo:
            try:
                a0 = int(actions[0].item())
                r0 = float(rewards[0].item())
                d0 = float(dones[0].item())
                g0 = float(discounts[0].item())
                max_next0 = float(next_q_values[0].item())
                y0 = float(target_q_values[0].item())
                q0 = float(current_q_values[0].item())
                delta0 = y0 - q0
                se0 = delta0 * delta0
            except Exception:
                pass


        if formula:
            step_tag = str(tag) if tag is not None else "?/?"
            try:

                td_errors = (target_q_values - current_q_values.squeeze(1)).detach()
                manual_loss = float((td_errors * td_errors).mean().item())

                print(
                    f"[FORMULA][{step_tag}][TD_TARGET] y = r + (1-d)*discount*maxQ_next"
                    f" = {r0:.3f} + (1-{d0:.0f})*{g0:.3f}*{max_next0:.3f} = {y0:.3f}"
                )
                print(
                    f"[FORMULA][{step_tag}][CURRENT_Q] Q(s,a) = policy_net(s)[a={a0}] = {q0:.3f}"
                )
                print(f"[FORMULA][{step_tag}][TD_ERROR] delta = y - Q = {y0:.3f} - {q0:.3f} = {delta0:.3f}")
                print(f"[FORMULA][{step_tag}][SQ_ERROR] (y-Q)^2 = ({delta0:.3f})^2 = {se0:.3f}")
                print(f"[FORMULA][{step_tag}][LOSS] loss = mean((y - Q)^2) = {manual_loss:.3f}")
            except Exception as e:
                print(f"[FORMULA][{step_tag}][WARN] train_step formula logging failed: {e}")


        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)


        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=GRAD_CLIP_MAX_NORM,
        )
        self.optimizer.step()

        if formula:
            step_tag = str(tag) if tag is not None else "?/?"
            try:

                with torch.no_grad():
                    post_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
                q0_after = float(post_q_values[0].item())
                print(
                    f"[FORMULA][{step_tag}][BACKPROP] grad_norm(before_clip) = {float(grad_norm):.3f}"
                )
                print(
                    f"[FORMULA][{step_tag}][BACKPROP] Q(s,a) before -> after = {float(current_q_values[0].item()):.3f} -> {q0_after:.3f}"
                )
            except Exception as e:
                print(f"[FORMULA][{step_tag}][WARN] post-update logging failed: {e}")


        self.update_counter += 1
        updated_target = False
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            updated_target = True
            if formula:
                step_tag = str(tag) if tag is not None else "?/?"
                print(
                    f"[FORMULA][{step_tag}][TARGET_UPDATE] hard update: target_net <- policy_net (update_counter={self.update_counter}, freq={self.target_update_freq})"
                )

        loss_value = loss.item()
        self.loss_history.append(loss_value)

        if traininfo:
            try:
                sample_state = states[0].detach().cpu().numpy().tolist()
                sample_next_state = next_states[0].detach().cpu().numpy().tolist()
            except Exception:
                sample_state = None
                sample_next_state = None
            self.last_traininfo = {
                "skipped": False,
                "batch_size": int(self.batch_size),
                "buffer_len": int(len(self.memory)),
                "sample": {
                    "state": sample_state,
                    "action": a0,
                    "reward": r0,
                    "next_state": sample_next_state,
                    "done": d0,
                    "discount": g0,
                },
                "target": {
                    "r": r0,
                    "d": d0,
                    "discount": g0,
                    "max_next": max_next0,
                    "y": y0,
                },
                "q": {
                    "current": q0,
                },
                "td_error": delta0,
                "sq_error": se0,
                "loss": float(loss_value),
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
                "updated_target": bool(updated_target),
                "update_counter": int(self.update_counter),
                "target_update_freq": int(self.target_update_freq),
            }

        return loss_value

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_memory(self):
        """Clear replay buffer (useful when environment distribution shifts)."""
        self.memory = ReplayBuffer(capacity=self.memory_size)

    def hard_update_target(self):
        """Immediate sync: target_net <- policy_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath, training_state=None):
        """Save model, optimizer, replay buffer, and training state."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'algo': self.algo,
            'update_counter': self.update_counter,
            'agent_config': {
                'state_size': int(self.state_size),
                'action_size': int(self.action_size),
                'gamma': float(self.gamma),
                'epsilon_min': float(self.epsilon_min),
                'epsilon_decay': float(self.epsilon_decay),
                'batch_size': int(self.batch_size),
                'target_update_freq': int(self.target_update_freq),
                'memory_size': int(self.memory_size),
                'algo': self.algo,
            },
            'replay_buffer': self.memory.state_dict(),
            'training_state': training_state or {},
            'checkpoint_version': 2,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath, checkpoint=None, load_replay_buffer=True):
        """Load model, optimizer, replay buffer, and training state."""
        if checkpoint is None:
            try:
                checkpoint = torch.load(
                    filepath, map_location=device, weights_only=False
                )
            except TypeError:
                checkpoint = torch.load(filepath, map_location=device)

        config = checkpoint.get('agent_config', {}) if isinstance(checkpoint, dict) else {}
        if isinstance(config, dict):
            self.gamma = float(config.get('gamma', self.gamma))
            self.epsilon_min = float(config.get('epsilon_min', self.epsilon_min))
            self.epsilon_decay = float(config.get('epsilon_decay', self.epsilon_decay))
            self.batch_size = int(config.get('batch_size', self.batch_size))
            self.target_update_freq = int(
                config.get('target_update_freq', self.target_update_freq)
            )
            self.memory_size = int(config.get('memory_size', self.memory_size))

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = float(checkpoint['epsilon'])
        self.loss_history = checkpoint.get('loss_history', [])
        self.update_counter = int(checkpoint.get('update_counter', 0))
        if load_replay_buffer and checkpoint.get('replay_buffer') is not None:
            self.memory = ReplayBuffer(capacity=self.memory_size)
            self.memory.load_state_dict(checkpoint.get('replay_buffer'))
            self.memory_size = int(self.memory.buffer.maxlen or self.memory_size)
        self.last_checkpoint = checkpoint
        print(f"Model loaded from {filepath}")
        return checkpoint
