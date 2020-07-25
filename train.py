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
        self.val_input = Dense(units=64, input_dim=self.state_space_size,
                               activation='relu', kernel_initializer='he_uniform')
        self.val_dense_0 = Dense(
            units=32, activation='relu', kernel_initializer='he_uniform')
        self.val_output = Dense(units=1, activation='linear')
        # The actor part
        self.policy_input = Dense(units=64, input_dim=self.state_space_size,
                                  activation='relu', kernel_initializer='he_uniform')
        self.policy_dense_0 = Dense(
            units=32, activation='relu', kernel_initializer='he_uniform')
        self.policy_output = Dense(
            units=self.action_space_size, activation='softmax')

    def call(self, inputs, **kwargs):
        # The critic part
        val_x = self.val_input(inputs)
        val_x = self.val_dense_0(val_x)
        val = self.val_output(val_x)
        # The actor part
        action_x = self.policy_input(inputs)
        action_x = self.policy_dense_0(action_x)
        action_dist = self.policy_output(action_x)
        return action_dist, val


def sample_action(action_space_size, probs):
    return np.random.choice(action_space_size, p=probs/probs.sum())


def eval(model, env, max_eps, action_space_size):
    total_reward = 0.0
    for _ in range(max_eps):
        done = False
        state = env.reset()
        while not done:
            action_dist, _ = model(tf.convert_to_tensor([state]))
            action = sample_action(
                action_space_size, action_dist.numpy()[0])
            state, reward, done, _ = env.step(action)
            total_reward += reward
    avg_reward = total_reward / max_eps
    return avg_reward


def compute_discounted_rewards(rewards, gamma):
    discounted_reward = 0
    discounted_rewards = []
    for reward in rewards[::-1]:
        discounted_reward = gamma * discounted_reward + reward
        discounted_rewards.append(discounted_reward)
    return discounted_rewards[::-1]


def train(max_eps=5000, gamma=0.9):
    env = gym.make('CartPole-v0')
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape[0]
    print('Initialize with action space size {0} and state space size {1}'.format(
        action_space_size, state_space_size))
    actor_critic_model = ActorCriticModel(action_space_size, state_space_size)
    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    action_dists, rewards, vals = [], [], []
    for eps in range(max_eps):
        state = env.reset()
        done = False
        with tf.GradientTape() as tape:
            while not done:
                action_dist, val = actor_critic_model(
                    tf.convert_to_tensor([state], dtype=tf.float32))
                action = sample_action(
                    action_space_size, action_dist.numpy()[0])
                state, reward, done, _ = env.step(action)
                action_dists.append(action_dist)
                rewards.append(reward)
                vals.append(val)
            # Calculate the gradient after the episode ends
            q_vals = tf.convert_to_tensor(
                compute_discounted_rewards(rewards, gamma))
            advantages = q_vals - \
                tf.squeeze(tf.convert_to_tensor(vals), axis=[1, 2])
            log_action_dists = tf.math.log(tf.convert_to_tensor(action_dists))
            actor_loss = tf.math.reduce_mean(
                tf.multiply(-log_action_dists, tf.expand_dims(advantages, axis=1)))
            critic_loss = 0.5 * tf.math.reduce_mean(tf.math.pow(advantages, 2))
            loss = tf.add(actor_loss, critic_loss)
        gradients = tape.gradient(loss, actor_critic_model.trainable_weights)
        optimizer.apply_gradients(
            zip(gradients, actor_critic_model.trainable_weights))
        eval_score = eval(actor_critic_model, env, 10, action_space_size)
        print(
            'Finished training {0}/{1} with score {2}'.format(eps, max_eps, eval_score))
    env.close()
    print('Done!')


if __name__ == '__main__':
    train()
