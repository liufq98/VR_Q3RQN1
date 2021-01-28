
from env import Env_VR
from d3qn import D3QN,Agent
import numpy as np
from tqdm import tqdm
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import scipy.io
config = ConfigProto()

print(33333)

config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
EPISODE_Length = 200
STEP_Lenght = 100
memory_start=200
env = Env_VR()
Agent1 = Agent(20,obs_shape=[3,])####Agent定义的相关参数
Agent2 = Agent(20,obs_shape=[3,])#
Agent3 = Agent(20,obs_shape=[3,])#
aver_uti=np.zeros((3,EPISODE_Length))
a1_loss=[]
a2_loss=[]
a3_loss=[]
a1_reward=[]
a2_reward=[]
a3_reward=[]
for episode in range(EPISODE_Length):
    env.reset()
    sum_utility1=0
    sum_utility2=0
    sum_utility3=0
    q_sum1=0
    q_sum2=0
    q_sum3=0
    reward_sum1=0
    reward_sum2=0
    reward_sum3=0

    for step in tqdm(range(STEP_Lenght),ncols=100, desc="eposide {}".format(episode)):
        action = []
        s_i = env.OUTPUT_STATE().copy()

        #print(np.reshape(np.hstack((s_i[0],s_i[3],s_i[6])),(1,3,1,1)))
        action1 = Agent1.choose_action(np.hstack((s_i[0],s_i[3],s_i[6])))####会有0.001的概率选择不是Q最大的动作
        action2 = Agent2.choose_action(np.hstack((s_i[1],s_i[4],s_i[7])))
        action3 = Agent3.choose_action(np.hstack((s_i[2],s_i[5],s_i[8])))
        action.append(int(action1))
        action.append(int(action2))
        action.append(int(action3))

        estimated_reward1 = Agent1.estimate_reward(np.hstack((s_i[0],s_i[3],s_i[6])),action1)
        estimated_reward2 = Agent2.estimate_reward(np.hstack((s_i[1],s_i[4],s_i[7])),action2)
        estimated_reward3 = Agent3.estimate_reward(np.hstack((s_i[2],s_i[5],s_i[8])),action3)
        estimated_reward = np.array([estimated_reward1, estimated_reward2, estimated_reward3])
        # 经过VCG Auction用户可以得到用户分配到的通信资源
        payment = env.VCG_Auction(action,np.reshape(estimated_reward,(3)))

        reward = env.VR_Watch_Reward()
        #done=env.KNAP_SACK(action,reward)
        utility = reward - payment
        #print(estimated_reward)
        print('estimate_reward:',estimated_reward)
        print('reward:         ',reward)
        print('payment:        ',payment)
        print('utility:        ',utility)
        env.STATE_TRANSITION()
        s_i_ = env.OUTPUT_STATE().copy()###[9*1]
        s_i_1=np.hstack((s_i_[0],s_i_[3],s_i_[6]))
        s_i_2=np.hstack((s_i_[1],s_i_[4],s_i_[7]))
        s_i_3=np.hstack((s_i_[2],s_i_[5],s_i_[8]))

        action1_ = Agent1.choose_action(s_i_1)
        action2_ = Agent2.choose_action(s_i_2)
        action3_ = Agent3.choose_action(s_i_3)

        sum_utility1+=utility[0]
        sum_utility2+=utility[1]
        sum_utility3+=utility[2]

        Agent1.store_transition(np.hstack((s_i[0],s_i[3],s_i[6])),action[0],reward[0],estimated_reward1,utility[0],s_i_1,action1_,True)
        Agent2.store_transition(np.hstack((s_i[1], s_i[4], s_i[7])), action[1], reward[1],estimated_reward2,utility[1], s_i_2,action2_,True)
        Agent3.store_transition(np.hstack((s_i[2], s_i[5], s_i[8])), action[2], reward[2], estimated_reward3,utility[2],s_i_3,action3_,True)
        if (episode*STEP_Lenght+step+1)>memory_start:
            q_loss1,reward_loss1=Agent1.learn()
            q_loss2,reward_loss2=Agent2.learn()
            q_loss3,reward_loss3=Agent3.learn()
            q_sum1+=q_loss1
            reward_sum1+=reward_loss1
            q_sum2+=q_loss2
            reward_sum2+=reward_loss2
            q_sum3+=q_loss3
            reward_sum3+=reward_loss3

    aver_utility1=sum_utility1/STEP_Lenght
    aver_utility2=sum_utility2/STEP_Lenght
    aver_utility3=sum_utility3/STEP_Lenght
    print('aver_utility1:    ',aver_utility1)
    print('aver_utility2:    ',aver_utility2)
    print('aver_utility3:    ',aver_utility3)

    aver_uti[0][episode]=aver_utility1
    aver_uti[1][episode]=aver_utility2
    aver_uti[2][episode]=aver_utility3
    if episode>(memory_start/STEP_Lenght-1):
       aver_loss1=q_sum1/STEP_Lenght
       aver_loss2=q_sum2/STEP_Lenght
       aver_loss3=q_sum3/STEP_Lenght
       a1_loss.append(np.array(aver_loss1))
       a2_loss.append(np.array(aver_loss2))
       a3_loss.append(np.array(aver_loss3))
       print('a1_loss:       ',a1_loss)

       aver_reward1=reward_sum1/STEP_Lenght
       aver_reward2=reward_sum2/STEP_Lenght
       aver_reward3=reward_sum3/STEP_Lenght
       a1_reward.append(np.array(aver_reward1))
       a2_reward.append(np.array(aver_reward2))
       a3_reward.append(np.array(aver_reward3))
       print('a1_reward:      ',a1_reward)



scipy.io.savemat('aver_utility', mdict={'aver_utility':np.array(aver_uti)})
scipy.io.savemat('a1_loss', mdict={'a1_loss':np.array(a1_loss)})
scipy.io.savemat('a2_loss', mdict={'a2_loss':np.array(a2_loss)})
scipy.io.savemat('a3_loss', mdict={'a3_loss':np.array(a3_loss)})
scipy.io.savemat('a1_reward', mdict={'a1_reward':np.array(a1_reward)})
scipy.io.savemat('a2_reward', mdict={'a2_reward':np.array(a2_reward)})
scipy.io.savemat('a3_reward', mdict={'a3_reward':np.array(a3_reward)})