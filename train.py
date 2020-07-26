import gym
import numpy as np
import tensorflow as tf
from tensorflow import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


class ActorCriticModel(Model):
    def __init__(self, action_space_size, state_space_size):
        super(ActorCriticModel, self).__init__()
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        # The critic part
        self.val_input = Dense(units=256, input_dim=self.state_space_size,
                               activation='relu', kernel_initializer='he_uniform')
        self.val_output = Dense(units=1, activation='linear')
        # The actor part
        self.policy_input = Dense(units=256, input_dim=self.state_space_size,
                                  activation='relu', kernel_initializer='he_uniform')
        self.policy_output = Dense(
            units=self.action_space_size, activation='softmax')

    def call(self, inputs, **kwargs):
        # The critic part
        val_x = self.val_input(inputs)
        val = self.val_output(val_x)
        # The actor part
        action_x = self.policy_input(inputs)
        action_dist = self.policy_output(action_x)
        return action_dist, val


def sample_action(action_space_size, probs, use_max=False):
    if use_max:
        return np.argmax(probs)
    else:
        return np.random.choice(action_space_size, p=probs/probs.sum())


def eval(model, env, max_eps, action_space_size):
    total_reward = 0.0
    for _ in range(max_eps):
        done = False
        state = env.reset()
        while not done:
            action_dist, _ = model(tf.convert_to_tensor([state]))
            action = sample_action(
                action_space_size, action_dist.numpy()[0], use_max=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
    avg_reward = total_reward / max_eps
    return avg_reward


def compute_discounted_rewards(rewards, gamma):
    discounted_reward = 0
    discounted_rewards = []
    for reward in rewards[::-1]:
        discounted_reward = gamma * discounted_reward + reward
        discounted_rewards.append([discounted_reward])
    return discounted_rewards[::-1]


def train(max_eps=1000, gamma=0.99):
    env = gym.make('CartPole-v0')
    eval_env = gym.make('CartPole-v0')
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape[0]
    print('Initialize with action space size {0} and state space size {1}'.format(
        action_space_size, state_space_size))
    actor_critic_model = ActorCriticModel(action_space_size, state_space_size)
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    for eps in range(max_eps):
        state = env.reset()
        done = False
        rewards, actions, states = [], [], []
        while not done:
            action_dist, _ = actor_critic_model(
                tf.convert_to_tensor([state], dtype=tf.float32))
            action = sample_action(
                action_space_size, action_dist.numpy()[0])
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            actions.append(action)
            states.append(state)
            state = next_state
        # Calculate the gradient after the episode ends
        with tf.GradientTape() as tape:
            probs, vals = actor_critic_model(
                tf.convert_to_tensor(states, dtype=tf.float32))
            q_vals = tf.convert_to_tensor(
                compute_discounted_rewards(rewards, gamma), dtype=tf.float32)
            advantages = q_vals - vals
            value_loss = advantages ** 2
            clipped_probs = tf.clip_by_value(probs, 1e-10, 1-1e-10)
            log_probs = tf.math.log(clipped_probs)
            action_onehot = tf.one_hot(
                actions, action_space_size, dtype=tf.float32)
            policy_loss = -(log_probs * action_onehot) * advantages
            entropy_loss = -tf.reduce_sum(probs * log_probs)
            loss = tf.reduce_mean(0.5 * value_loss) + \
                tf.reduce_mean(policy_loss) + 0.01 * entropy_loss
        gradients = tape.gradient(loss, actor_critic_model.trainable_weights)
        optimizer.apply_gradients(
            zip(gradients, actor_critic_model.trainable_weights))
        eval_score = eval(actor_critic_model, eval_env, 10, action_space_size)
        print(
            'Finished training {0}/{1} with score {2}'.format(eps, max_eps, eval_score))
    env.close()
    print('Done!')


if __name__ == '__main__':
    train()
