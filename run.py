from env_v2 import Env_VR
from d3qn import D3QN,Agent
import numpy as np
from tqdm import tqdm
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
EPISODE_Length = 100
STEP_Lenght = 5000
memory_size=200
env = Env_VR()
Agent1 = Agent(20,obs_shape=[3,])####Agent定义的相关参数
Agent2 = Agent(20,obs_shape=[3,])#
Agent3 = Agent(20,obs_shape=[3,])#

for episode in range(EPISODE_Length):
    env.reset()
    for step in tqdm(range(STEP_Lenght),ncols=100, desc="eposide {}".format(episode)):
        action = []
        s_i = env.OUTPUT_STATE().copy()

        #print(np.reshape(np.hstack((s_i[0],s_i[3],s_i[6])),(1,3,1,1)))
        action1 = Agent1.choose_action(np.hstack((s_i[0],s_i[3],s_i[6])))
        action2 = Agent2.choose_action(np.hstack((s_i[1],s_i[4],s_i[7])))
        action3 = Agent3.choose_action(np.hstack((s_i[2],s_i[5],s_i[8])))
        action.append(int(action1))
        action.append(int(action2))
        action.append(int(action3))
        # 经过VCG Auction用户可以得到用户分配到的通信资源
        reward = env.VR_Watch_Reward()
        payment = env.VCG_Auction(action,reward)
        done=env.KNAP_SACK(action,reward)
        utility = reward - payment
        env.STATE_TRANSITION()
        s_i_ = env.OUTPUT_STATE().copy()###[9*1]

        s_i_1=np.hstack((s_i_[0],s_i_[3],s_i_[6]))
        s_i_2=np.hstack((s_i_[1],s_i_[4],s_i_[7]))
        s_i_3=np.hstack((s_i_[2],s_i_[5],s_i_[8]))

        #print(action)
        '''print(s_i)
        print(reward)
        print(payment)
        print(utility)
        print(s_i_)'''
        Agent1.store_transition(np.hstack((s_i[0],s_i[3],s_i[6])),action[0],reward[0],utility[0],s_i_1,done[0])
        Agent2.store_transition(np.hstack((s_i[1], s_i[4], s_i[7])), action[1], reward[1], utility[1], s_i_2, done[1])
        Agent3.store_transition(np.hstack((s_i[2], s_i[5], s_i[8])), action[2], reward[2], utility[2], s_i_3, done[2])
        if (episode*STEP_Lenght+step+1)>memory_size:
            Agent1.learn()
            Agent2.learn()
            Agent3.learn()


        # State这里需要考虑要具体存储哪些内容
        '''a_i = action
        reward_i = reward[XXX]
        utility_i = utility[XXX]
        env.STATE_TRANSITION()
        
        storetranstion(np.hstack(s_i, a_i, reward_i, utility_i, s_i_))
        if episode * EPISODE_Length + step + 1 > memorycapacity:
            agent Learn
            learning rate decrease'''