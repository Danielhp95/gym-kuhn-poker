from gym.envs.registration import register

register(
    id='KuhnPoker-v0',
    entry_point='gym_kuhn_poker.envs:KuhnPokerEnv',
)
