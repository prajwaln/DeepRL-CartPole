import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =  0.99 # discount factor
INITIAL_EPSILON = 0.6 # starting value of epsilon
FINAL_EPSILON =  0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period

experience = []
BUFFER_SIZE = 50000
BATCH_SIZE = 256
B_1 = 0
B_2 = 0

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM]) #a_t
target_in = tf.placeholder("float", [None]) #loss calculation

# TODO: Define Network Graph

w_initializer = tf.random_normal_initializer(0., 0.01)
b_initializer = tf.constant_initializer(0.01)

HIDDEN_NODES = [16, 48]
H1, H2 = HIDDEN_NODES[0], HIDDEN_NODES[1]
l = tf.layers.dense(state_in, H1, activation = tf.nn.tanh) #relu
l = tf.layers.dense(l, H2, activation = tf.nn.tanh) #relu
l_final = tf.layers.dense(l, ACTION_DIM, activation = None) #linear

# TODO: Network outputs
q_values = l_final
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# TODO: Loss/Optimizer Definition
loss = tf.reduce_mean(tf.square(target_in - q_action)) #mse
optimizer = tf.train.AdamOptimizer().minimize(loss) # learning rates B1 B2

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        e = [state, action, reward, next_state, done]
        experience.append(e)
        # Ensure experience doesn't grow larger than BUFFER_SIZE
        if len(experience) > BUFFER_SIZE:
            experience.pop(0)

        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        target = reward
        if not done:
            target = reward + GAMMA * np.max(nextstate_q_values)
        
        if len(experience) > BATCH_SIZE:
            minibatch = random.sample(experience, BATCH_SIZE)

            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            next_state_batch = [data[3] for data in minibatch]

            target_batch = []
            Q_value_batch = q_values.eval(feed_dict={
                state_in: next_state_batch
            })
            
            for i in range(0, BATCH_SIZE):
                sample_is_done = minibatch[i][4]
                if sample_is_done:
                    target_batch.append(reward_batch[i])
                else:
                    target_val = reward_batch[i] + GAMMA * np.max(Q_value_batch[i])
                    target_batch.append(target_val)

            summary = session.run([optimizer], feed_dict={
                            target_in: target_batch,
                            state_in: state_batch,
                            action_in: action_batch
                        })
##
##        # Do one training step
##        session.run([optimizer], feed_dict={
##            target_in: [target],
##            action_in: [action],
##            state_in: [state]
##        })
        
        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
