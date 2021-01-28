import numpy as np

test for github
test 2.0

Hello FangQi, This is Ran Speaking

class Env_VR():
    def __init__(self, USER_NUM=3, Resource_Block_NUM=20):
        self.USER_NUM = USER_NUM
        print('Succ')

        #通信资源
        self.Resource_Block_NUM = Resource_Block_NUM
        self.SNR_Amplify = np.array([0.2, 1, 1.6, 3])
        self.Per_Block_Rate = np.ones(self.USER_NUM) * self.SNR_Amplify[2]
        # self.DataRate_Trans_Prob = np.array([[0.3,0.3,0.4],[0.3,0.4,0.3],[0.4,0.3,0.3]])
        self.DataRate_Trans_Prob = np.array([[0.1, 0.4, 0.8, 1.0], [0.2, 0.5, 0.8, 1.0], [0.1, 0.3, 0.6, 1.0], [0.1, 0.4, 0.7, 1.0]])


        #VR视频属性相关
        self.H_Tile_Bit_Rate = 0.5
        self.L_Tile_Bit_Rate = 0.1
        self.Viewport_Tile_Size = 20

        #VR User相关
        self.User_Data_Rate=np.ones(self.USER_NUM)
        self.Buffer_H = np.zeros(self.USER_NUM)
        self.Buffer_L = np.zeros(self.USER_NUM)
        #这个需要自行设计
        #self.Prediction_Accuracy = np.ones(self.USER_NUM)
        self.Prediction_Accuracy = np.array([0.7,0.8,0.9])
        #self.Prediction_StdDev =np.ones(self.USER_NUM)
        self.Prediction_StdDev =np.array([0.05,0.08,0.1])
        #执行backup操作
        self.Per_Block_Rate_BU = self.Per_Block_Rate.copy()
        self.User_Data_Rate_BU = self.User_Data_Rate.copy()
        self.Buffer_L_BU = self.Buffer_L.copy()
        self.Buffer_H_BU = self.Buffer_H.copy()

    def reset(self):
        self.Per_Block_Rate = self.Per_Block_Rate_BU.copy()
        self.User_Data_Rate = self.User_Data_Rate_BU.copy()
        self.Buffer_H = self.Buffer_H.copy()
        self.Buffer_L = self.Buffer_L.copy()





    def KNAP_SACK(self,Requested_Block_Num,True_Value):
        resArr = np.zeros((self.USER_NUM + 1, self.Resource_Block_NUM + 1))
        temp = np.array([0])
        True_Value = np.hstack((temp,True_Value))
        Requested_Block_Num = np.hstack((temp, Requested_Block_Num))
        for i in range(1, self.USER_NUM + 1):
            for j in range(1, self.Resource_Block_NUM + 1):
                if Requested_Block_Num[i] <= j:
                    resArr[i, j] = max(resArr[i - 1, j - Requested_Block_Num[i]] + True_Value[i], resArr[i - 1, j])
                else:
                    resArr[i, j] = resArr[i - 1, j]
        winner = np.zeros(self.USER_NUM)
        j = self.Resource_Block_NUM
        for i in range(self.USER_NUM,0,-1):
            if resArr[i][j] > resArr[i-1][j]:
                winner[i-1] = True
                j -= Requested_Block_Num[i]
        return winner

    #信道资源拍卖函数，得到最终的信道资源拍卖结果，用户的付款，以及最终的总的能达到的True_Value.
    def VCG_Auction(self,Requested_Block_Num,True_Value):
        #overall_winner决策出当前的需求的resource block与对应true_value下的拍卖结果
        overall_winner = self.KNAP_SACK(Requested_Block_Num,True_Value)
        print('Overall Winner:' +str(overall_winner))
        winner_index = np.where(overall_winner==1)[0]
        overall_true_value = sum(overall_winner * True_Value)
        #payment记录的是每个用户在当前拍卖结果下，需要付出的钱的数量
        payment = np.zeros(self.USER_NUM)
        for i in winner_index:
            temp_True_Value = True_Value.copy()
            temp_True_Value[i] = 0
            temp_winner = self.KNAP_SACK(Requested_Block_Num,temp_True_Value)
            temp_overall_true_value = sum(temp_winner * True_Value)
            payment[i] = temp_overall_true_value - sum(temp_True_Value * overall_winner)
        self.User_Data_Rate = overall_winner * Requested_Block_Num * self.Per_Block_Rate
        print(overall_winner)
        #print(Requested_Block_Num)
        #print(self.Per_Block_Rate)
        print('User Data Rate: '+str(self.User_Data_Rate))
        return payment

    def VR_Watch_Reward(self):
        watch_reward = np.zeros(self.USER_NUM)
        for i in range(self.USER_NUM):
            # Buffer_H与Buffer_L是针对当前要看的ViewPort，已经缓存的高质量&低质量Tile的个数
            Threshold_H = (self.Viewport_Tile_Size - self.Buffer_H[i]) * self.H_Tile_Bit_Rate##20-0*0.5=10
            Threshold_L = (self.Viewport_Tile_Size - self.Buffer_H[i] - self.Buffer_L[i]) * self.L_Tile_Bit_Rate##20-0-0*0.1=2
            miss_tile_num = self.Viewport_Tile_Size - self.Buffer_H[i] - self.Buffer_L[i]##20-0-0
            if self.User_Data_Rate[i] >= Threshold_H:
                watch_reward[i] = 1
            elif self.User_Data_Rate[i]> Threshold_L:
                High_qual_tile_num_nowfetch =  int((self.User_Data_Rate[i] - self.L_Tile_Bit_Rate * miss_tile_num)/(self.L_Tile_Bit_Rate+self.H_Tile_Bit_Rate))
                watch_reward[i] = (self.Buffer_H[i]+High_qual_tile_num_nowfetch)/self.Viewport_Tile_Size
            else:
                watch_reward[i] = -1
        watch_reward+=1
        #返回的是np.array格式的所有用户的观看reward。
        return watch_reward

    def VR_Watch_Buffer_Transition(self):
        #对Buffer根据当前可用剩余带宽（由总数据速率减去当前需要下载的tile所需的速率）、预测准确率进行State Transition。
        for i in range(self.USER_NUM):
            accuracy = np.random.normal(self.Prediction_Accuracy[i],self.Prediction_StdDev[i])###正态分布
            accuracy = np.clip(accuracy,0,1)####标准化到0，1
            predict_err_num = int(20*(1-accuracy))
            Threshold_H = (self.Viewport_Tile_Size - self.Buffer_H[i]) * self.H_Tile_Bit_Rate
            if self.User_Data_Rate[i] > Threshold_H:
                Num_H_Tiles_Downloaded = (self.User_Data_Rate[i] - Threshold_H)/self.H_Tile_Bit_Rate
                if Num_H_Tiles_Downloaded >= self.Viewport_Tile_Size:
                    self.Buffer_H[i] = self.Viewport_Tile_Size - predict_err_num
                    self.Buffer_L[i] = int((self.User_Data_Rate[i] - Threshold_H- self.Viewport_Tile_Size *\
                                        self.H_Tile_Bit_Rate)/self.L_Tile_Bit_Rate)
                    self.Buffer_L[i] = np.clip(self.Buffer_L[i],0,self.Viewport_Tile_Size)
                elif (Num_H_Tiles_Downloaded + predict_err_num > self.Viewport_Tile_Size):
                    self.Buffer_H[i] = self.Viewport_Tile_Size - predict_err_num
                    self.Buffer_L[i] = 0
                elif( Num_H_Tiles_Downloaded + predict_err_num <= self.Viewport_Tile_Size):
                    self.Buffer_H[i] = Num_H_Tiles_Downloaded
                    self.Buffer_L[i] = 0
            else:
                self.Buffer_H[i] = 0
                self.Buffer_L[i] = 0
        return 0

    def Data_Rate_Transtion(self):
        for i in range(self.USER_NUM):
            j=0
            while self.Per_Block_Rate[i] != self.SNR_Amplify[j]:
                j += 1
            tran_prob = self.DataRate_Trans_Prob[j]
            pointer = np.random.uniform(0,1)
            j = 0
            while pointer >= tran_prob[j]:
                j+=1
            self.Per_Block_Rate[i] = self.SNR_Amplify[j]
        return 0

    def OUTPUT_STATE(self):
        #State = np.hstack(self.Resource_Block_NUM)
        #所有用户的所有状态，注意用户的状态可见性
        '''self.Per_Block_Rate
        self.Buffer_L
        self.Buffer_H'''
        return np.hstack((self.Per_Block_Rate,self.Buffer_H,self.Buffer_L))

    def STATE_TRANSITION(self):
        self.Data_Rate_Transtion()
        self.VR_Watch_Buffer_Transition()
        return 0

    def return_bandwidth_reward(self,resource_block_num):
        return_watch_reward = np.zeros(self.USER_NUM)
        Temp_User_Data_Rate = resource_block_num * self.Per_Block_Rate
        for i in range(self.USER_NUM):
            # Buffer_H与Buffer_L是针对当前要看的ViewPort，已经缓存的高质量&低质量Tile的个数
            Threshold_H = (self.Viewport_Tile_Size - self.Buffer_H[i]) * self.H_Tile_Bit_Rate  ##20-0*0.5=10
            Threshold_L = (self.Viewport_Tile_Size - self.Buffer_H[i] - self.Buffer_L[
                i]) * self.L_Tile_Bit_Rate  ##20-0-0*0.1=2
            miss_tile_num = self.Viewport_Tile_Size - self.Buffer_H[i] - self.Buffer_L[i]  ##20-0-0
            if Temp_User_Data_Rate[i] >= Threshold_H:
                return_watch_reward[i] = 1
            elif Temp_User_Data_Rate[i] > Threshold_L:
                High_qual_tile_num_nowfetch = int((Temp_User_Data_Rate[i] - self.L_Tile_Bit_Rate * miss_tile_num) / (
                            self.L_Tile_Bit_Rate + self.H_Tile_Bit_Rate))
                return_watch_reward[i] = (self.Buffer_H[i] + High_qual_tile_num_nowfetch) / self.Viewport_Tile_Size
            else:
                return_watch_reward[i] = -1
        return_watch_reward += 1
        # 返回的是np.array格式的所有用户的观看reward。
        return return_watch_reward

