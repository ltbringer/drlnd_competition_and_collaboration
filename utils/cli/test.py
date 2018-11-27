import numpy as np
from constants import constants
from utils.initialize import initialize


def test_agent(
        environment_path,
        episodes=10,
        max_t=5000,
        seed=constants.SEED,
        gamma=constants.GAMMA,
        tau=constants.TAU,
        critic_lr=constants.CRITIC_LR,
        actor_lr=constants.ACTOR_LR,
        buffer_size=constants.BUFFER_SIZE,
        batch_size=constants.BATCH_SIZE,
        actor_layer_1_nodes=constants.FC1_UNITS,
        critic_layer_1_nodes=constants.FC2_UNITS,
        actor_layer_2_nodes=constants.FC1_UNITS,
        critic_layer_2_nodes=constants.FC2_UNITS,
        actor_model_path=None,
        critic_model_path=None
):

    tennisEnv, agent = initialize(
        environment_path,
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
        critic_model_path=critic_model_path,
        is_test=True
    )
    for episode in range(episodes):
        agent.reset()
        states = tennisEnv \
            .reset_to_initial_state(train_mode=False) \
            .get_state_snapshot()

        print(states[0])

        scores = []
        for t in range(max_t):
            actions = [agent.act(state, add_noise=False, is_training=False) for state in states]
            next_states, rewards, dones = tennisEnv.step(actions).reaction()
            scores.append(rewards)
            states = next_states

            if any(dones): break

        print('Players gathered rewards = {} in episode {}'.format(np.mean(scores), episode + 1))

