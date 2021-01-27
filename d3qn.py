import tensorflow as tf


from tensorflow import keras
import numpy as np
from replay_buffer import ReplayBuffer, PriorityExperienceReplay



class D3QN(keras.Model):
    def __init__(self, model, n_actions, lr=1e-4):
        super(D3QN, self).__init__()
        self.model = model

        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

        self.opt = tf.keras.optimizers.Adam(lr=lr)

    def call(self, x):
        x = self.model(x)

        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))####dueling DQN的改动，将输出改为V和A reduce——A需要进行初始化，避免action对reward的影响较小
        #print(Q)
        return Q

    def advantage(self, x, decision_mask=None):###计算duelingDQN中的各action的advantages
        x = self.model.predict(x)
        A = self.A(x).numpy() #breaks tf auto gradient tape

        if decision_mask is not None:
            A = A+decision_mask
        return A

    def debug(self, x, A):
        V = self.V(x).numpy()
        Q = (V+(A-np.mean(A, axis=1, keepdims=True)))
        #print("Q-value: ", Q)

    def loss_func(self, y_true, y_pred, weights=None):
        difference = y_true - y_pred
        abs_diff = tf.math.abs(difference)
        abs_diff = tf.math.reduce_sum(abs_diff, axis=-1)

        if weights is not None:

            abs_diff = weights*abs_diff

        return tf.math.reduce_mean(abs_diff, axis=-1)

    def train_func(self, x, y, weights=None):####梯度更新
        with tf.GradientTape() as tape:
            q = self.call(x)

            loss = self.loss_func(y, q, weights=weights)
            #print(loss)
            #loss = loss+self.losses

        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss




class Agent():####两个model，一个为eval 一个为target即next model
    def __init__(self, n_actions, obs_shape,save_weight_name="d3qn"):###n_action 动作数量 obs——shape状态类别？？？
        #self.action_space = list(range(n_actions))
        self.n_actions = n_actions
        self.batch_size=64
        self.epsilon=1.0
        self.gamma = 0.95
        self.eps_dec = 0.0005
        self.eps_min = 0.001
        self.replace = 200###多少次更新一次next_model的值
        self.memory_size=10000
        self.lr=1e-3
        self.opt = tf.keras.optimizers.Adam(lr=self.lr)
        self.eval_model=self.build_model()
        self.next_model=self.build_model()

        self.reward_eval_model=self.reward_model()
        self.reward_target_model=self.reward_model()

        self.save_weight_name = save_weight_name

        self.learn_step_counter = 0


        self.memory = ReplayBuffer(self.memory_size, obs_shape)

        self.q_eval = D3QN(self.eval_model, n_actions, lr=self.lr)####model的build？？,增加Dueling DQN？
        self.q_next = D3QN(self.next_model, n_actions, lr=self.lr)

        #For tf-keras formality
        self.q_eval.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse') #actual loss is not mse
        self.q_next.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        self.reward_eval_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        self.reward_target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')

    def store_transition(self, obs, action, reward,estimate_reward,utility ,new_obs,new_action, done):
        return self.memory.store_transition(obs, action, reward,estimate_reward,utility, new_obs,new_action, done)

    def choose_action(self, obs, decision_mask=None):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            if len(obs.shape) < 3:
                obs = np.array([obs]) #Adding batch dimension if there is no batch dimension
            actions = self.q_eval.advantage(np.reshape(obs,(1,1,3,1,1)), decision_mask=decision_mask)

            action = np.squeeze(np.argmax(actions, axis=1))

        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

        return action
    def estimate_reward(self,obs,action):
        estimate_reward=self.reward_eval_model(np.reshape(np.append(obs,action),(1,4,1)))
        return estimate_reward

    def learn(self):
        for _ in range(5):
            obs, actions, rewards,estimate_rewards,utility, new_obs,new_actions, dones= self.memory.sample_buffer(self.batch_size)

            if self.learn_step_counter % self.replace == 0:
                if self.learn_step_counter == 0:
                    self.q_eval.predict(np.reshape(obs,(self.batch_size,1,3,1,1)))
                    self.q_next.predict(np.reshape(new_obs,(self.batch_size,1,3,1,1)))
                    self.reward_eval_model.predict(np.reshape(np.append(obs,actions),(self.batch_size,4,1)))
                    self.reward_target_model.predict(np.reshape(np.append(new_obs, new_actions), (self.batch_size, 4, 1)))

                self.q_next.set_weights(self.q_eval.get_weights())####对target网络赋值新权重
                self.reward_target_model.set_weights(self.reward_eval_model.get_weights())

            q_target = self.q_eval.predict(np.reshape(obs,(self.batch_size,1,3,1,1)))
            q_next = self.q_next.predict(np.reshape(new_obs,(self.batch_size,1,3,1,1)))


            reward_target=self.reward_target_model(np.reshape(np.append(new_obs, new_actions), (self.batch_size, 4, 1)))
            true_target=rewards+np.reshape(reward_target,(self.batch_size))
            loss2=self.reward_train(np.reshape(np.append(obs,actions),(self.batch_size,4,1)),true_target)

            max_actions = np.argmax(self.q_eval.predict(np.reshape(new_obs,(self.batch_size,1,3,1,1))), axis=1)##DDQN
            for idx, terminal in enumerate(dones):
                q_target[idx, actions[idx]] = utility[idx] + self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))#####DDRN的修改

            loss1=self.q_eval.train_func(np.reshape(obs,(self.batch_size,1,3,1,1)), q_target, weights=None)

            self.learn_step_counter += 1
        return loss1,loss2
    def save_model(self, model_name=None):
        if model_name is None:
            model_name = self.save_weight_name
        self.q_eval.save_weights(model_name)

    def load_model(self, model_name=None):
        if model_name is None:
            model_name = self.save_weight_name
        self.q_eval.load_weights(model_name)

    def build_model(self):
        model=keras.Sequential()
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(32, (2, 1), strides=(2, 1),padding='same', activation='relu'),
                                  input_shape=(1, 3, 1, 1)))
        # input_shape=(time_step, row, col, channels)
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(64, (2, 1), strides=(2, 2),padding='same',  activation='relu')))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(64, (1, 1), strides=(1, 1),padding='same',  activation='relu')))
        model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
        model.add(keras.layers.LSTM(512))
        model.summary()
        return model

    def reward_model(self):
        model=keras.Sequential()
        model.add(keras.layers.LSTM(128,input_shape=(4,1), activation='relu'))
        model.add(keras.layers.Dense(1))
        model.summary()
        return model
    def reward_train(self,x,y_true):
        with tf.GradientTape() as tape1:
             reward_eval = self.reward_eval_model(x)
             y_pred=tf.reshape(reward_eval,[self.batch_size])
             loss=self.loss_func(y_true,y_pred)
             #print(loss)
             grades=tape1.gradient(loss,self.reward_eval_model.trainable_variables)
             self.opt.apply_gradients(zip(grades, self.reward_eval_model.trainable_variables))
        return loss


    def loss_func(self, y_true, y_pred):
        difference = y_true - y_pred
        abs_diff = tf.math.abs(difference)
        abs_diff = tf.math.reduce_mean(abs_diff, axis=-1)

        return abs_diff