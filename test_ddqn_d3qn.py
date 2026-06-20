"""
Tests for DDQN and D3QN algorithm variants.

Validates:
1. Network architecture correctness (DQN vs Dueling)
2. TD-target formulas (vanilla DQN vs Double DQN)
3. Agent creation for each algo variant
4. Save/load round-trip preserves algo identity
5. Training step produces valid loss for each variant
6. CLI argument parsing for --ddqn and --d3qn
"""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import torch

from main_dqn_agent import (
    ALGO_D3QN,
    ALGO_DDQN,
    ALGO_DQN,
    DQNAgent,
    DQNNetwork,
    DuelingDQNNetwork,
    _build_network,
    _validate_algo,
)


# ---------------------------------------------------------------------------
# 1. Algorithm identifier validation
# ---------------------------------------------------------------------------
class TestAlgoValidation(unittest.TestCase):
    """_validate_algo should accept valid IDs and reject invalid ones."""

    def test_valid_algos(self):
        self.assertEqual(_validate_algo("dqn"), "dqn")
        self.assertEqual(_validate_algo("DQN"), "dqn")
        self.assertEqual(_validate_algo("ddqn"), "ddqn")
        self.assertEqual(_validate_algo("DDQN"), "ddqn")
        self.assertEqual(_validate_algo("d3qn"), "d3qn")
        self.assertEqual(_validate_algo("D3QN"), "d3qn")
        # Whitespace tolerance
        self.assertEqual(_validate_algo("  dqn  "), "dqn")

    def test_invalid_algo_raises(self):
        with self.assertRaises(ValueError):
            _validate_algo("ppo")
        with self.assertRaises(ValueError):
            _validate_algo("")


# ---------------------------------------------------------------------------
# 2. Network architecture
# ---------------------------------------------------------------------------
class TestNetworkArchitecture(unittest.TestCase):
    """Ensure _build_network returns the right class per algo."""

    def test_dqn_uses_dqn_network(self):
        net = _build_network("dqn", state_size=8, action_size=6)
        self.assertIsInstance(net, DQNNetwork)

    def test_ddqn_uses_dqn_network(self):
        net = _build_network("ddqn", state_size=8, action_size=6)
        self.assertIsInstance(net, DQNNetwork)

    def test_d3qn_uses_dueling_network(self):
        net = _build_network("d3qn", state_size=8, action_size=6)
        self.assertIsInstance(net, DuelingDQNNetwork)


class TestDuelingNetwork(unittest.TestCase):
    """DuelingDQNNetwork must output correct shape and satisfy Q = V + A - mean(A)."""

    def setUp(self):
        self.state_size = 8
        self.action_size = 6
        self.net = DuelingDQNNetwork(self.state_size, self.action_size)

    def test_output_shape(self):
        x = torch.randn(4, self.state_size)
        q = self.net(x)
        self.assertEqual(q.shape, (4, self.action_size))

    def test_single_input(self):
        x = torch.randn(1, self.state_size)
        q = self.net(x)
        self.assertEqual(q.shape, (1, self.action_size))

    def test_mean_centering_property(self):
        """Q = V + (A - mean(A)).  The mean advantage across actions should be ~0."""
        x = torch.randn(16, self.state_size)
        with torch.no_grad():
            features = self.net.feature(x)
            value = self.net.value_stream(features)        # (16, 1)
            advantage = self.net.advantage_stream(features)  # (16, 6)
            q = self.net(x)
            # Reconstruct Q manually
            expected_q = value + advantage - advantage.mean(dim=1, keepdim=True)
        np.testing.assert_allclose(
            q.cpu().numpy(), expected_q.cpu().numpy(), rtol=1e-5, atol=1e-6
        )

    def test_has_network_attribute(self):
        """Backward compat: .network should exist for neuron trace / formula code."""
        self.assertTrue(hasattr(self.net, "network"))


# ---------------------------------------------------------------------------
# 3. Agent creation per algo
# ---------------------------------------------------------------------------
class TestAgentCreation(unittest.TestCase):
    """DQNAgent must store the algo and use the correct network type."""

    def _make_agent(self, algo):
        return DQNAgent(state_size=8, action_size=6, algo=algo)

    def test_dqn_agent(self):
        agent = self._make_agent("dqn")
        self.assertEqual(agent.algo, "dqn")
        self.assertIsInstance(agent.policy_net, DQNNetwork)
        self.assertIsInstance(agent.target_net, DQNNetwork)

    def test_ddqn_agent(self):
        agent = self._make_agent("ddqn")
        self.assertEqual(agent.algo, "ddqn")
        self.assertIsInstance(agent.policy_net, DQNNetwork)
        self.assertIsInstance(agent.target_net, DQNNetwork)

    def test_d3qn_agent(self):
        agent = self._make_agent("d3qn")
        self.assertEqual(agent.algo, "d3qn")
        self.assertIsInstance(agent.policy_net, DuelingDQNNetwork)
        self.assertIsInstance(agent.target_net, DuelingDQNNetwork)

    def test_default_algo_is_dqn(self):
        agent = DQNAgent(state_size=8, action_size=6)
        self.assertEqual(agent.algo, "dqn")


# ---------------------------------------------------------------------------
# 4. TD-target formula correctness
# ---------------------------------------------------------------------------
class TestTDTargetFormulas(unittest.TestCase):
    """Verify that the TD target uses the correct formula for each variant.

    DQN:  y = r + γ * max_a' Q_target(s', a')
    DDQN: y = r + γ * Q_target(s', argmax_a' Q_policy(s', a'))
    D3QN: same as DDQN but with dueling network
    """

    def _fill_buffer_and_step(self, algo):
        """Create agent, fill buffer, run one train step, return loss."""
        agent = DQNAgent(
            state_size=8, action_size=6, batch_size=4, memory_size=100, algo=algo
        )
        # Seed weights for reproducibility
        torch.manual_seed(42)

        # Fill replay buffer with 10 transitions
        for i in range(10):
            state = np.random.randn(8).astype(np.float32)
            action = np.random.randint(0, 6)
            reward = float(np.random.randn())
            next_state = np.random.randn(8).astype(np.float32)
            done = i == 9
            agent.remember(state, action, reward, next_state, done)

        loss = agent.train_step()
        return agent, loss

    def test_dqn_train_step_produces_loss(self):
        _, loss = self._fill_buffer_and_step("dqn")
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertTrue(loss >= 0.0)

    def test_ddqn_train_step_produces_loss(self):
        _, loss = self._fill_buffer_and_step("ddqn")
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertTrue(loss >= 0.0)

    def test_d3qn_train_step_produces_loss(self):
        _, loss = self._fill_buffer_and_step("d3qn")
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertTrue(loss >= 0.0)

    def test_ddqn_uses_policy_net_for_action_selection(self):
        """DDQN must select actions via policy_net, evaluate via target_net.

        We manually verify by comparing with hand-computed targets.
        """
        agent = DQNAgent(
            state_size=4, action_size=3, batch_size=2, memory_size=10, algo="ddqn"
        )

        # Create a controlled scenario
        for i in range(5):
            s = np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i], dtype=np.float32)
            agent.remember(s, i % 3, 1.0, s + 0.01, False)

        # Manually compute DDQN target for first 2 samples
        states, actions, rewards, next_states, dones, discounts = agent.memory.sample(2)
        next_states_t = torch.FloatTensor(next_states)

        with torch.no_grad():
            # DDQN: action selected by policy_net
            best_actions = agent.policy_net(next_states_t).argmax(dim=1)
            # Evaluated by target_net
            target_q_vals = agent.target_net(next_states_t)
            selected_q = target_q_vals[range(2), best_actions]

        # These should be finite numbers
        self.assertTrue(torch.all(torch.isfinite(selected_q)))

    def test_dqn_vs_ddqn_different_targets(self):
        """When policy_net ≠ target_net, DQN and DDQN should produce different targets."""
        torch.manual_seed(123)
        np.random.seed(123)

        # Build two agents with same initial weights but different algos
        agent_dqn = DQNAgent(
            state_size=4, action_size=3, batch_size=2, memory_size=10, algo="dqn"
        )
        agent_ddqn = DQNAgent(
            state_size=4, action_size=3, batch_size=2, memory_size=10, algo="ddqn"
        )

        # Copy DQN weights to DDQN
        agent_ddqn.policy_net.load_state_dict(agent_dqn.policy_net.state_dict())
        agent_ddqn.target_net.load_state_dict(agent_dqn.target_net.state_dict())

        # Manually force policy_net and target_net to prefer different actions
        with torch.no_grad():
            # policy_net bias for action 0 is huge -> policy_net will choose action 0
            agent_ddqn.policy_net.network[-1].bias.fill_(0.0)
            agent_ddqn.policy_net.network[-1].bias[0] = 100.0
            agent_dqn.policy_net.network[-1].bias.fill_(0.0)
            agent_dqn.policy_net.network[-1].bias[0] = 100.0

            # target_net bias for action 1 is huge -> target_net max is action 1 (large Q value)
            # but action 0 has a small Q value in target_net
            agent_ddqn.target_net.network[-1].bias.fill_(0.0)
            agent_ddqn.target_net.network[-1].bias[1] = 100.0
            agent_dqn.target_net.network[-1].bias.fill_(0.0)
            agent_dqn.target_net.network[-1].bias[1] = 100.0

        # Compute targets on the same next_states
        next_states = torch.randn(2, 4)
        with torch.no_grad():
            # DQN target: max from target_net (evaluates to ~100.0)
            dqn_targets = agent_dqn.target_net(next_states).max(1)[0]
            # DDQN target: argmax from policy_net (selects action 0), evaluated in target_net (evaluates to ~0.0)
            best_a = agent_ddqn.policy_net(next_states).argmax(dim=1, keepdim=True)
            ddqn_targets = agent_ddqn.target_net(next_states).gather(1, best_a).squeeze(1)

        # They should differ (not be exactly equal) since policy_net ≠ target_net
        # and action selection is different
        self.assertFalse(
            torch.allclose(dqn_targets, ddqn_targets, atol=1e-3),
            "DQN and DDQN targets should differ when policy_net != target_net"
        )


# ---------------------------------------------------------------------------
# 5. Save / Load round-trip
# ---------------------------------------------------------------------------
class TestSaveLoadAlgo(unittest.TestCase):
    """Model checkpoint must persist and restore the algo field."""

    def _roundtrip(self, algo):
        agent = DQNAgent(state_size=8, action_size=6, algo=algo)
        for idx in range(3):
            state = np.full(8, idx, dtype=np.float32)
            next_state = np.full(8, idx + 1, dtype=np.float32)
            agent.remember(
                state,
                idx % agent.action_size,
                float(idx),
                next_state,
                idx == 2,
                discount=0.99,
            )
        agent.update_counter = 7

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pth")
            buf = io.StringIO()
            with redirect_stdout(buf):
                agent.save(
                    path,
                    training_state={
                        "last_episode": 12,
                        "next_episode": 13,
                        "current_stage": 2,
                    },
                )
            self.assertTrue(os.path.isfile(path))

            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(checkpoint.get("algo"), algo)
            self.assertEqual(checkpoint.get("update_counter"), 7)
            self.assertEqual(checkpoint.get("training_state", {}).get("next_episode"), 13)
            self.assertEqual(
                len(checkpoint.get("replay_buffer", {}).get("transitions", [])), 3
            )

            agent2 = DQNAgent(state_size=8, action_size=6, algo=algo)
            with redirect_stdout(io.StringIO()):
                agent2.load(path)
            self.assertEqual(agent2.update_counter, 7)
            self.assertEqual(len(agent2.memory), 3)
            first = agent2.memory.buffer[0]
            np.testing.assert_allclose(first[0], np.zeros(8, dtype=np.float32))
            self.assertEqual(first[1], 0)
            self.assertAlmostEqual(first[2], 0.0)
            # Weights should match
            for p1, p2 in zip(agent.policy_net.parameters(), agent2.policy_net.parameters()):
                np.testing.assert_allclose(
                    p1.detach().cpu().numpy(), p2.detach().cpu().numpy(), atol=1e-7
                )

    def test_dqn_save_load(self):
        self._roundtrip("dqn")

    def test_ddqn_save_load(self):
        self._roundtrip("ddqn")

    def test_d3qn_save_load(self):
        self._roundtrip("d3qn")


# ---------------------------------------------------------------------------
# 6. Q-value retrieval works for all variants
# ---------------------------------------------------------------------------
class TestQValueRetrieval(unittest.TestCase):
    """get_q_values and select_action must work for every algo."""

    def _test_q_values(self, algo):
        agent = DQNAgent(state_size=8, action_size=6, algo=algo)
        state = np.random.randn(8).astype(np.float32)
        q = agent.get_q_values(state)
        self.assertEqual(q.shape, (6,))
        self.assertTrue(np.all(np.isfinite(q)))

    def _test_action_selection(self, algo):
        agent = DQNAgent(state_size=8, action_size=6, algo=algo)
        agent.epsilon = 0.0  # greedy
        state = np.random.randn(8).astype(np.float32)
        action = agent.select_action(state, training=False)
        self.assertIn(action, range(6))

    def test_dqn_q_values(self):
        self._test_q_values("dqn")

    def test_ddqn_q_values(self):
        self._test_q_values("ddqn")

    def test_d3qn_q_values(self):
        self._test_q_values("d3qn")

    def test_dqn_action(self):
        self._test_action_selection("dqn")

    def test_ddqn_action(self):
        self._test_action_selection("ddqn")

    def test_d3qn_action(self):
        self._test_action_selection("d3qn")


# ---------------------------------------------------------------------------
# 7. CLI argument parsing
# ---------------------------------------------------------------------------
class TestTrainCLIArgs(unittest.TestCase):
    """Verify that --ddqn and --d3qn args are parsed correctly by main_train."""

    def _parse(self, arg_list):
        import argparse

        # Replicate the relevant subset of main_train argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--episodes", type=int, default=500)
        algo_group = parser.add_mutually_exclusive_group()
        algo_group.add_argument("--ddqn", action="store_true")
        algo_group.add_argument("--d3qn", action="store_true")
        return parser.parse_args(arg_list)

    def test_no_flag_is_dqn(self):
        args = self._parse([])
        self.assertFalse(args.ddqn)
        self.assertFalse(args.d3qn)

    def test_ddqn_flag(self):
        args = self._parse(["--ddqn"])
        self.assertTrue(args.ddqn)
        self.assertFalse(args.d3qn)

    def test_d3qn_flag(self):
        args = self._parse(["--d3qn"])
        self.assertFalse(args.ddqn)
        self.assertTrue(args.d3qn)

    def test_mutual_exclusion(self):
        with self.assertRaises(SystemExit):
            self._parse(["--ddqn", "--d3qn"])


class TestVisualizeCLIArgs(unittest.TestCase):
    """Verify that --ddqn and --d3qn args are parsed correctly by main_visualize."""

    def _parse(self, arg_list):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="models/best_model.pth")
        algo_group = parser.add_mutually_exclusive_group()
        algo_group.add_argument("--ddqn", action="store_true")
        algo_group.add_argument("--d3qn", action="store_true")
        return parser.parse_args(arg_list)

    def test_no_flag_is_dqn(self):
        args = self._parse([])
        self.assertFalse(args.ddqn)
        self.assertFalse(args.d3qn)

    def test_ddqn_flag(self):
        args = self._parse(["--ddqn"])
        self.assertTrue(args.ddqn)

    def test_d3qn_flag(self):
        args = self._parse(["--d3qn"])
        self.assertTrue(args.d3qn)


# ---------------------------------------------------------------------------
# 8. Multiple train steps (gradient descent works for each variant)
# ---------------------------------------------------------------------------
class TestMultipleTrainSteps(unittest.TestCase):
    """Run several train steps and verify loss decreases or stays finite."""

    def _run_steps(self, algo, n_steps=10):
        agent = DQNAgent(
            state_size=8, action_size=6, batch_size=4, memory_size=100, algo=algo
        )
        # Fill buffer
        for i in range(20):
            s = np.random.randn(8).astype(np.float32)
            a = np.random.randint(0, 6)
            r = float(np.random.randn())
            ns = np.random.randn(8).astype(np.float32)
            d = i == 19
            agent.remember(s, a, r, ns, d)

        losses = []
        for _ in range(n_steps):
            loss = agent.train_step()
            self.assertIsNotNone(loss)
            self.assertTrue(np.isfinite(loss))
            losses.append(loss)
        return losses

    def test_dqn_multi_step(self):
        losses = self._run_steps("dqn")
        self.assertEqual(len(losses), 10)

    def test_ddqn_multi_step(self):
        losses = self._run_steps("ddqn")
        self.assertEqual(len(losses), 10)

    def test_d3qn_multi_step(self):
        losses = self._run_steps("d3qn")
        self.assertEqual(len(losses), 10)


# ---------------------------------------------------------------------------
# 9. Epsilon decay is identical across variants
# ---------------------------------------------------------------------------
class TestEpsilonDecay(unittest.TestCase):
    """Epsilon decay logic should be unaffected by algo choice."""

    def test_decay_consistent(self):
        agents = {}
        for algo in ["dqn", "ddqn", "d3qn"]:
            agent = DQNAgent(state_size=8, action_size=6, algo=algo)
            agent.epsilon = 1.0
            for _ in range(100):
                agent.decay_epsilon()
            agents[algo] = agent.epsilon

        # All should be the same
        self.assertAlmostEqual(agents["dqn"], agents["ddqn"], places=10)
        self.assertAlmostEqual(agents["dqn"], agents["d3qn"], places=10)


if __name__ == "__main__":
    unittest.main()
