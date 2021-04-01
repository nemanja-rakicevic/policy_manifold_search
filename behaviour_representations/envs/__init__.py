
from gym.envs.registration import register


### Biped-Walker

register(
    id='BipedalWalkerAugmentedEnv-v0',
    entry_point='behaviour_representations.envs.box2d.env_bipedal_walker:BipedalWalkerAugmentedEnv',
    max_episode_steps=500,
    reward_threshold=0,
)


### Biped-Kicker

register(
    id='BipedalKickerAugmentedEnv-v0',
    entry_point='behaviour_representations.envs.box2d.env_bipedal_kicker:BipedalKickerAugmentedEnv',
    max_episode_steps=500,
    reward_threshold=0,
)


### Striker

register(
    id='StrikerAugmentedEnv-v0',
    entry_point='behaviour_representations.envs.box2d.env_striker:StrikerAugmentedEnv',
    max_episode_steps=1000,
    reward_threshold=0,
)


### Panda Striker

register(
    id='PandaStrikerEnv-v0',
    entry_point='behaviour_representations.envs.pybullet.env_panda:PandaStrikerEnv',
    max_episode_steps=1000,
    reward_threshold=0,
)
