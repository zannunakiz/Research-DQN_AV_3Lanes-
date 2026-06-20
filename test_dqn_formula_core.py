import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import torch

from main_dqn_agent import DQNAgent


class TestDQNFormulaCore(unittest.TestCase):
    def test_q_calculation_detail_matches_standard_forward(self):
        agent = DQNAgent(state_size=8, action_size=6)

        with torch.no_grad():
            linear_seed = 1
            for module in agent.policy_net.network:
                if isinstance(module, torch.nn.Linear):
                    module.weight.fill_(0.001 * linear_seed)
                    module.bias.fill_(0.01 * linear_seed)
                    linear_seed += 1

        state = np.array([0.91, 0.77, 0.65, 0.58, 0.44, 0.33, 0.22, 0.11], dtype=np.float32)
        expected_q = agent.get_q_values(state)

        stdout_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer):
            actual_q = agent.get_q_values_with_calculation(state, tag="TEST")

        np.testing.assert_allclose(actual_q, expected_q, rtol=1e-6, atol=1e-6)

        output = stdout_buffer.getvalue()
        self.assertIn("[FORMULA][TEST] input =", output)
        self.assertIn("output[slow_left] formula:", output)
        self.assertIn("[FORMULA][TEST][Q_CALCULATION] output =", output)


if __name__ == "__main__":
    unittest.main()
