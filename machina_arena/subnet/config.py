"""Machina Arena — Subnet configuration."""
import os

# Subnet configuration
SUBNET_NAME = "Machina Arena"
SUBNET_DESCRIPTION = "Decentralized robotics policy competition. Train in simulation, prove it in the arena."

# Chain configuration
NETWORK = os.environ.get("BT_NETWORK", "finney")
NETUID = int(os.environ.get("MA_NETUID", "0"))  # Set after registration

# Task configuration
DEFAULT_TASK = "pickcube-v1"
EPISODES_PER_EVAL = 100
EVAL_TEMP = 360  # blocks per tempo (~72 min)

# Scoring
WEIGHT_TEMPERATURE = 0.1  # softmax temperature (lower = more competitive)

# Validator requirements
MIN_VALIDATOR_STAKE = 1.0  # TAO minimum to be a validator
MAX_UIDS = 256  # max miners

# Rendering (for top policies)
RENDER_TOP_N = 5  # render top 5 miners' policies each tempo
