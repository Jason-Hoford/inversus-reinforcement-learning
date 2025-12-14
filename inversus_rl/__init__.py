"""Reinforcement Learning components for INVERSUS."""

# Lazy imports to avoid requiring torch at module level
__all__ = [
    'SingleInversusRLEnv',
    'MultiEnvRunner',
    'InversusCNNPolicy',
    'make_policy_from_env',
    'PPOAgent',
]

def __getattr__(name):
    """Lazy import for RL components."""
    if name == 'SingleInversusRLEnv' or name == 'MultiEnvRunner':
        from .env_wrappers import SingleInversusRLEnv, MultiEnvRunner
        return SingleInversusRLEnv if name == 'SingleInversusRLEnv' else MultiEnvRunner
    elif name == 'InversusCNNPolicy' or name == 'make_policy_from_env':
        from .policies import InversusCNNPolicy, make_policy_from_env
        return InversusCNNPolicy if name == 'InversusCNNPolicy' else make_policy_from_env
    elif name == 'PPOAgent':
        from .ppo_agent import PPOAgent
        return PPOAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

