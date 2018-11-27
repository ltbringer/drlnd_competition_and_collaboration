import numpy as np
import matplotlib.pyplot as plt

from constants import constants
from utils.cli import cli
from utils.cli.test import test_agent
from utils.cli.train import train_agent

args                    = cli.parser.parse_args()
environment_path        = args.env_path
episodes                = args.episodes
time_steps              = args.time_steps
qualify_window          = args.qualify_window
seed                    = args.seed
gamma                   = args.gamma
tau                     = args.tau
buffer_size             = args.buffer_size
batch_size              = args.batch_size
critic_lr               = args.critic_lr
actor_lr                = args.actor_lr
actor_layer_1_nodes     = args.actor_layer_1_nodes
critic_layer_1_nodes    = args.critic_layer_1_nodes
actor_layer_2_nodes     = args.actor_layer_2_nodes
critic_layer_2_nodes    = args.critic_layer_2_nodes
actor_model_path        = args.actor_model_path
critic_model_path       = args.critic_model_path


def main():
    if not args.env_path:
        raise Exception("""Provide the path for Reacher environment using the --env-path args""")

    if args.run == constants.TRAIN:
        scores = train_agent(
            environment_path,
            episodes=100000,
            max_t=5000,
            qualify_window=qualify_window,
            seed=seed,
            gamma=gamma,
            tau=tau,
            critic_lr=critic_lr,
            actor_lr=actor_lr,
            buffer_size=buffer_size,
            batch_size=batch_size,
            actor_layer_1_nodes=actor_layer_1_nodes,
            critic_layer_1_nodes=critic_layer_1_nodes,
            actor_layer_2_nodes=actor_layer_2_nodes,
            critic_layer_2_nodes=critic_layer_2_nodes,
            actor_model_path=args.actor_model_path,
            critic_model_path = args.critic_model_path
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores) + 1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')

        plt.savefig('scores.png')
        plt.show()
        plt.close()
    else:
        if args.actor_model_path and args.critic_model_path:
            test_agent(
                environment_path,
                episodes=10,
                max_t=5000,
                seed=seed,
                gamma=gamma,
                tau=tau,
                critic_lr=critic_lr,
                actor_lr=actor_lr,
                buffer_size=buffer_size,
                batch_size=batch_size,
                actor_layer_1_nodes=actor_layer_1_nodes,
                critic_layer_1_nodes=critic_layer_1_nodes,
                actor_layer_2_nodes=actor_layer_2_nodes,
                critic_layer_2_nodes=critic_layer_2_nodes,
                actor_model_path=actor_model_path,
                critic_model_path=critic_model_path
            )
        else:
            raise Exception("""Provide the path for actor and critic weights 
                using the --actor-model-path and --critic-model-path args""")


if __name__ == '__main__':
    main()
