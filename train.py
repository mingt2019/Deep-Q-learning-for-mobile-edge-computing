from fog_env import Offload
from RL_brain import DeepQNetwork
import numpy as np
import random
# import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def reward_fun(delay, max_delay, unfinish_indi):

    # still use reward, but use the negative value
    penalty = - max_delay * 2

    if unfinish_indi:
        reward = penalty
    else:
        reward = - delay

    return reward


def train(iot_RL_list, NUM_EPISODE):

    RL_step = 0

    for episode in range(NUM_EPISODE):

        print(episode)
        print(iot_RL_list[0].epsilon)
        # BITRATE ARRIVAL
        bitarrive = np.random.uniform(env.min_bit_arrive, env.max_bit_arrive, size=[env.n_time, env.n_iot])
        task_prob = env.task_arrive_prob
        bitarrive = bitarrive * (np.random.uniform(0, 1, size=[env.n_time, env.n_iot]) < task_prob)
        bitarrive[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_iot])

        # =================================================================================================
        # ========================================= DRL ===================================================
        # =================================================================================================

        # OBSERVATION MATRIX SETTING
        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for iot_index in range(env.n_iot):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_iot])

        # INITIALIZE OBSERVATION
        observation_all, lstm_state_all = env.reset(bitarrive)

        # TRAIN DRL
        while True:

            # PERFORM ACTION
            action_all = np.zeros([env.n_iot])
            for iot_index in range(env.n_iot):

                observation = np.squeeze(observation_all[iot_index, :])

                if np.sum(observation) == 0:
                    # if there is no task, action = 0 (also need to be stored)
                    action_all[iot_index] = 0
                else:
                    action_all[iot_index] = iot_RL_list[iot_index].choose_action(observation)

                if observation[0] != 0:
                    iot_RL_list[iot_index].do_store_action(episode, env.time_count, action_all[iot_index])

            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)
            observation_all_, lstm_state_all_, done = env.step(action_all)

            # should store this information in EACH time slot
            for iot_index in range(env.n_iot):
                iot_RL_list[iot_index].update_lstm(lstm_state_all_[iot_index,:])

            process_delay = env.process_delay
            unfinish_indi = env.process_delay_unfinish_ind

            # STORE MEMORY; STORE TRANSITION IF THE TASK PROCESS DELAY IS JUST UPDATED
            for iot_index in range(env.n_iot):

                history[env.time_count - 1][iot_index]['observation'] = observation_all[iot_index, :]
                history[env.time_count - 1][iot_index]['lstm'] = np.squeeze(lstm_state_all[iot_index, :])
                history[env.time_count - 1][iot_index]['action'] = action_all[iot_index]
                history[env.time_count - 1][iot_index]['observation_'] = observation_all_[iot_index]
                history[env.time_count - 1][iot_index]['lstm_'] = np.squeeze(lstm_state_all_[iot_index,:])

                update_index = np.where((1 - reward_indicator[:,iot_index]) * process_delay[:,iot_index] > 0)[0]

                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii]
                        iot_RL_list[iot_index].store_transition(history[time_index][iot_index]['observation'],
                                                                history[time_index][iot_index]['lstm'],
                                                                history[time_index][iot_index]['action'],
                                                                reward_fun(process_delay[time_index, iot_index],
                                                                           env.max_delay,
                                                                           unfinish_indi[time_index, iot_index]),
                                                                history[time_index][iot_index]['observation_'],
                                                                history[time_index][iot_index]['lstm_'])
                        iot_RL_list[iot_index].do_store_reward(episode, time_index,
                                                               reward_fun(process_delay[time_index, iot_index],
                                                                          env.max_delay,
                                                                          unfinish_indi[time_index, iot_index]))
                        iot_RL_list[iot_index].do_store_delay(episode, time_index,
                                                              process_delay[time_index, iot_index])
                        reward_indicator[time_index, iot_index] = 1

            # ADD STEP (one step does not mean one store)
            RL_step += 1

            # UPDATE OBSERVATION
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            # CONTROL LEARNING START TIME AND FREQUENCY
            if (RL_step > 200) and (RL_step % 10 == 0):
                for iot in range(env.n_iot):
                    iot_RL_list[iot].learn()

            # GAME ENDS
            if done:
                break
        #  =================================================================================================
        #  ======================================== DRL END=================================================
        #  =================================================================================================


if __name__ == "__main__":

    NUM_IOT = 50
    NUM_FOG = 5
    NUM_EPISODE = 1000
    NUM_TIME_BASE = 100
    MAX_DELAY = 10
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY

    # GENERATE ENVIRONMENT
    env = Offload(NUM_IOT, NUM_FOG, NUM_TIME, MAX_DELAY)

    # GENERATE MULTIPLE CLASSES FOR RL
    iot_RL_list = list()
    for iot in range(NUM_IOT):
        iot_RL_list.append(DeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                        learning_rate=0.01,
                                        reward_decay=0.9,
                                        e_greedy=0.99,
                                        replace_target_iter=200,  # each 200 steps, update target net
                                        memory_size=500,  # maximum of memory
                                        ))

    # TRAIN THE SYSTEM
    train(iot_RL_list, NUM_EPISODE)
    print('Training Finished')
