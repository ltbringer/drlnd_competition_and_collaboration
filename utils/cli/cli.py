import argparse
from constants import constants

parser = argparse.ArgumentParser()


# ================================================================
# IF these are missing, application should crash as they cant
# have defaults
# ================================================================
parser.add_argument(
    '--env-path',
    help='Path where the Unity Reacher environment is located',
)

parser.add_argument(
    '--actor-model-path',
    help='[test-only]: Path where weights for actor are saved'
)

parser.add_argument(
    '--critic-model-path',
    help='[test-only]: Path where weights for critic are saved'
)
# ================================================================


parser.add_argument(
    '--run',
    help='Input one of: [train, test]; DEFAULT=test',
    default='test'
)

parser.add_argument(
    '--qualify-window',
    help='[train-only]: Number of episodes for which an average score of 30 should be maintained',
    default=constants.QUALIFY_WINDOW
)


parser.add_argument(
    '--seed',
    help='Random seed',
    default=constants.SEED
)

parser.add_argument(
    '--episodes',
    help='Number of episodes to test or train'
)

parser.add_argument(
    '--time-steps',
    help='Timesteps per episode',
    default=constants.TIME_STEPS
)

parser.add_argument(
    '--batch-size',
    help='[train-only]: Number of samples that should be there in memory before learning can take place',
    default=constants.BATCH_SIZE
)

parser.add_argument(
    '--buffer-size',
    help='[train-only]: Maximum number of samples that should be there in memory',
    default=constants.BUFFER_SIZE
)

parser.add_argument(
    '--gamma',
    help='[train-only]: Hyperparameter, controls the discount over future rewards',
    default=constants.GAMMA
)

parser.add_argument(
    '--tau',
    help='[train-only]: Hyperparameter, controls the amount of influence local network has over target during a soft update',
    default=constants.TAU
)

parser.add_argument(
    '--actor-lr',
    help='[train-only]: Hyperparameter, learning rate for the actor',
    default=constants.ACTOR_LR
)

parser.add_argument(
    '--critic-lr',
    help='[train-only]: Hyperparameter, learning rate for the critic',
    default=constants.CRITIC_LR
)

parser.add_argument(
    '--critic-layer-1-nodes',
    help='[train-only]: Hyperparameter, Number of neurons in the first layer of critic\'s Neural Network',
    default=constants.FC1_UNITS
)

parser.add_argument(
    '--critic-layer-2-nodes',
    help='[train-only]: Hyperparameter, Number of neurons in the second layer of critic\'s Neural Network',
    default=constants.FC2_UNITS
)

parser.add_argument(
    '--actor-layer-1-nodes',
    help='[train-only]: Hyperparameter, Number of neurons in the first layer of actor\'s Neural Network',
    default=constants.FC1_UNITS
)

parser.add_argument(
    '--actor-layer-2-nodes',
    help='[train-only]: Hyperparameter, Number of neurons in the second layer of actor\'s Neural Network',
    default=constants.FC2_UNITS
)
