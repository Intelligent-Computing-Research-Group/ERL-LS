import os
import random
from perf_analysis import xmg_evaluation

import numpy as np
import graphExtractor as GE
import torch
#from convert_list2actions import get_metrics, make_abc_command
from subprocess import check_output
from datetime import datetime
import dgl
from dgl.nn.pytorch import GraphConv
import time
import mtlPy as mtlpy
import abc_py as abcPy

def is_equal(a, b):
    if -1e-5 < a-b < 1e-5:
        return True
    else:
        return False

class EnvGraph_aig(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, aigfile, global_heuristic_result):
        self._abc = abcPy.AbcInterface()
        self._aigfile = aigfile
        self._abc.start()
        print("read start ok")
        self.lenSeq = 0
        self._abc.read(self._aigfile)
        self.length_of_command = 20
        initStats = self._abc.aigStats() # The initial AIG statistics
        self.initNumAnd = float(initStats.numAnd)
        self.initLev = float(initStats.lev)

        # now = datetime.now()
        # dateTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
        # print("TestStartTime for 100 runs of get aig state", dateTime)
        #
        # now = datetime.now()
        # dateTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
        # print("TestEndTime ", dateTime)
        print("Inintial state:")
        print("initNumAnd", self.initNumAnd, "baseline Depth ", self.initLev, "Initial value",
              self.statValue(initStats))

        if is_equal(global_heuristic_result.value, 0):
            print("Init baseline...")
            self.resyn2()
            self.resyn2()
            resyn2Stats = self._abc.aigStats()
            totalReward = self.statValue(initStats) - self.statValue(resyn2Stats)
            self._rewardBaseline = totalReward / self.length_of_command  # 10 is the length of compress2rs sequence
            global_heuristic_result.value = self._rewardBaseline

            print("baseline num of AigNotes ", float(resyn2Stats.numAnd))
            print("rewardBaseline ", self._rewardBaseline)


            # print("test the runtime for each action")
            print('\n')
        else:
            print("Init baseline...inited")
            self._rewardBaseline = global_heuristic_result.value

            # print("baseline num of XmgGates ", global_heuristic_result.value)
            #print("rewardBaseline ", self._rewardBaseline)
            # print("test the runtime for each action")
            #print('\n')

    def resyn2(self):
        self._abc.balance(l=False)
        self._abc.rewrite(l=False)
        self._abc.refactor(l=False)
        self._abc.balance(l=False)
        self._abc.rewrite(l=False)
        self._abc.rewrite(l=False, z=True)
        self._abc.balance(l=False)
        self._abc.refactor(l=False, z=True)
        self._abc.rewrite(l=False, z=True)
        self._abc.balance(l=False)
    def reset(self):
        self.lenSeq = 0
        self._abc.end()
        self._abc.start()
        self._abc.read(self._aigfile)
        self._lastStats = self._abc.aigStats()  # The initial AIG statistics
        self._curStats = self._lastStats  # the current AIG statistics
        self.lastAct = self.numActions() - 1
        self.lastAct2 = self.numActions() - 1
        self.lastAct3 = self.numActions() - 1
        self.lastAct4 = self.numActions() - 1
        self.actsTaken = np.zeros(self.numActions())
        return self.state()
    def close(self):
        self.reset()
    def step(self, actionIdx):
        self.takeAction(actionIdx)
        nextState = self.state()
        reward = self.reward()
        done = False
        if (self.lenSeq >= self.length_of_command):
            done = True
        return nextState, reward, done, 0
    def takeAction(self, actionIdx):
        """
        @return true: episode is end
        """
        # "b -l; rs -K 6 -l; rw -l; rs -K 6 -N 2 -l; rf -l; rs -K 8 -l; b -l; rs -K 8 -N 2 -l; rw -l; rs -K 10 -l; rwz -l; rs -K      10 -N 2 -l; b -l; rs -K 12 -l; rfz -l; rs -K 12 -N 2 -l; rwz -l; b -l
        self.lastAct4 = self.lastAct3
        self.lastAct3 = self.lastAct2
        self.lastAct2 = self.lastAct
        self.lastAct = actionIdx
        #self.actsTaken[actionIdx] += 1
        self.lenSeq += 1
        """
        # Compress2rs actions
        if actionIdx == 0:
            self._abc.balance(l=True) # b -l
        elif actionIdx == 1:
            self._abc.resub(k=6, l=True) # rs -K 6 -l
        #elif actionIdx == 2:
        #    self._abc.resub(k=6, n=2, l=True) # rs -K 6 -N 2 -l
        #elif actionIdx == 3:
        #    self._abc.resub(k=8, l=True) # rs -K 8 -l
        #elif actionIdx == 4:
        #    self._abc.resub(k=10, l=True) # rs -K 10 -l
        #elif actionIdx == 5:
        #    self._abc.resub(k=12, l=True) # rs -K 12 -l
        #elif actionIdx == 6:
        #    self._abc.resub(k=10, n=2, l=True) # rs -K 10 -N 2 -l
        elif actionIdx == 2:
            self._abc.resub(k=12, n=2, l=True) # rs - K 12 -N 2 -l
        elif actionIdx == 3:
            self._abc.rewrite(l=True) # rw -l
        #elif actionIdx == 3:
        #    self._abc.rewrite(l=True, z=True) # rwz -l
        elif actionIdx == 4:
            self._abc.refactor(l=True) # rf -l
        #elif actionIdx == 4:
        #    self._abc.refactor(l=True, z=True) # rfz -l
        elif actionIdx == 5: # terminal
            self._abc.end()
            return True
        else:
            assert(False)
        """
        if actionIdx == 0:
            self._abc.balance(l=False) # b
        elif actionIdx == 1:
            self._abc.rewrite(l=False) # rw
        elif actionIdx == 2:
            self._abc.refactor(l=False) # rf
        elif actionIdx == 3:
            self._abc.rewrite(l=False, z=True) #rw -z
        elif actionIdx == 4:
            self._abc.refactor(l=False, z=True) #rf -z
        elif actionIdx == 5:
            self._abc.end()
            return True
        else:
            assert(False)

        """
        elif actionIdx == 3:
            self._abc.rewrite(z=True) #rwz
        elif actionIdx == 4:
            self._abc.refactor(z=True) #rfz
        """


        # update the statitics
        self._lastStats = self._curStats
        self._curStats = self._abc.aigStats()
        return False
    def state(self):
        """
        @brief current state
        """
        #oneHotAct = np.zeros(self.numActions()) #self.numActions=5
        #print("self.lastAct:", self.lastAct)
        #np.put(oneHotAct, self.lastAct, 1) #oneHotAct[self.lastAct]=1
        lastOneHotActs  = np.zeros(self.numActions())
        lastOneHotActs[self.lastAct2] += 1/3
        lastOneHotActs[self.lastAct3] += 1/3
        lastOneHotActs[self.lastAct] += 1/3
        # stateArray = np.array([self._curStats.numAnd / self.initNumAnd, self._curStats.lev / self.initLev, self._lastStats.numAnd / self.initNumAnd, self._lastStats.lev / self.initLev])
        stateArray = np.array([self._curStats.numAnd / self.initNumAnd, self._lastStats.numAnd / self.initNumAnd])  # dim of state also decreased by 2
        stepArray = np.array([float(self.lenSeq) / self.length_of_command])
        combined = np.concatenate((stateArray, lastOneHotActs, stepArray), axis=-1)
        # print("combined Input state :", combined)
        # combined = np.expand_dims(combined, axis=0)
        # return stateArray.astype(np.float32)
        combined_torch = torch.from_numpy(combined.astype(np.float32)).float()
        # combine_torch share the same memary with combined
        # print("GE\n", datetime.now())
        '''for i in range(100):
            graph = GE.extract_dgl_graph(self._abc)
        print("endGE", datetime.now())'''
        graph = GE.extract_dgl_graph(self._abc)
        #print("input graph:", graph)
        return (combined_torch, graph)
    def reward(self):
        if self.lastAct == 5: #term
            return 0
        #print("lastStats:", self.statValue(self._lastStats), "curStats:", self.statValue(self._curStats),"rewardBaseline:", self._rewardBaseline)
        #print("final rewards:", self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline)
        return self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline
        #return -self._lastStats.numAnd + self._curStats.numAnd - 1
        if (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 2
        elif (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev == self._lastStats.lev):
            return 0
        elif (self._curStats.numAnd == self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 1
        else:
            return -2
    def numActions(self):
        return 5
    def dimState(self):
        return 2 + self.numActions() * 1 + 1
    def returns(self):
        #return [self._curStats.numAnd, self._curStats.lev]
        return self._curStats.numAnd
    def statValue(self, stat):
        #return float(stat.lev)  / float(self.initLev)
        return float(stat.numAnd) / float(self.initNumAnd) #  + float(stat.lev)  / float(self.initLev)
        #return stat.numAnd + stat.lev * 10
    def curStatsValue(self):
        return self.statValue(self._curStats)
    def seed(self, sd):
        pass
    def compress2rs(self):
        self._abc.compress2rs()
class EnvGraph_mtl_xmg(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, xmgfile, global_heuristic_result):
        self._abc = mtlpy.MtlInterface()
        #self._abc = abcPy.AbcInterface()
        self._xmgfile = xmgfile
        self.length_of_command = 10
        # self.PID = processID
        # self.target_end2end_index = end2end_target
        # gate_num=0, latency=1, energy=2, row_usage=3

        self._abc.xmg_start()
        print("read start ok")
        self.lenSeq = 0
        self._abc.xmg_read(self._xmgfile)
        print("read xmgfile ok")
        initStats = self._abc.xmgStats() # The initial XMG statistics
        # self.initNumXmgNodes = float(initStats.numXmgNodes)
        self.initNumXmgGates = float(initStats.numXmgGates)
        # self.initLev = float(initStats.xmg_lev)

        print("Inintial state:")
        # print("initNumXmgNodes", self.initNumXmgNodes)
        print("initNumXmgGates", self.initNumXmgGates)
        print("Initial value", self.statValue(initStats))

        # self.end_to_end_result = self.get_end2end_states()
        # print("Initial end2end result", self.end_to_end_result)
        # self.baselineActions()
        # print("After run of baseline:")
        # resynStats = self._abc.xmgStats()
        # print("baseline num of XmgNodes ",
        #       resynStats.numXmgNodes,
        #       "baseline num of XmgGates ",
        #       float(resynStats.numXmgGates),
        #       " total reward ", self.statValue(resynStats))
        if is_equal(global_heuristic_result.value, 0):
            print("Init baseline...")
            self.baselineActions()
            resyn2Stats = self._abc.xmgStats()
            totalReward = self.statValue(initStats) - self.statValue(resyn2Stats)
            self._rewardBaseline = totalReward / self.length_of_command  # 10 is the length of compress2rs sequence
            global_heuristic_result.value = self._rewardBaseline

            print("baseline num of XmgGates ", float(resyn2Stats.numXmgGates))
            print("rewardBaseline ", self._rewardBaseline)
            # print("test the runtime for each action")
            print('\n')
        else:

            self._rewardBaseline = global_heuristic_result.value

            # print("baseline num of XmgGates ", global_heuristic_result.value)
            print("rewardBaseline ", self._rewardBaseline)
            # print("test the runtime for each action")
            print('\n')
        #self.random_action_test()
        #input()
        #os.system("pause")
        #self.reset()
        #self.test_action_runtime_2()
    def random_action_test(self):

        act_list = [0, 1, 2, 3, 4, 5, 6, 7]
        sum = 0
        epoch = 10
        for j in range(epoch):
            self.reset()
            for i in range(8):
                select = random.choice(act_list)
                self.takeAction(select)
            resyn2Stats = self._abc.xmgStats()
            print(resyn2Stats.numXmgGates)
            sum += resyn2Stats.numXmgGates



        print("average num of random action", sum/epoch)
        return sum/epoch
    def test_action_runtime(self):
        starttime = time.time()
        for i in range(1):
            self._abc.xmg_dco()
        endtime = time.time()
        print("xmg_dco runtime:%s ms", (endtime - starttime)*1000)

        starttime = time.time()
        for i in range(1):
            self._abc.xmg_resub()
        endtime = time.time()
        print("xmg_resub runtime:%s ms", (endtime - starttime) * 1000)

        starttime = time.time()
        for i in range(1):
            self._abc.xmg_cut_rewrite()
        endtime = time.time()
        print("xmg_cut_rewrite runtime:%s ms", (endtime - starttime) * 1000)

        starttime = time.time()
        for i in range(1):
            self._abc.xmg_depth_rewrite(allow_size_increase=True, start='a', overhead=1.2)
        endtime = time.time()
        print("xmg_depth_rewrite(allow_size_increase=True, start='a', overhead=1.2) runtime:%s ms", (endtime - starttime) * 1000)

        starttime = time.time()
        for i in range(1):
            self._abc.xmg_node_resynthesis()
        endtime = time.time()
        print("xmg_node_resyn runtime:%s ms", (endtime - starttime) * 1000)

        starttime = time.time()
        for i in range(1):
            self._abc.xmg_depth_rewrite(start='s', allow_size_increase=False)
        endtime = time.time()
        print("xmg_depth_rewrite(start='s', allow_size_increase=False) runtime:%s ms", (endtime - starttime) * 1000)

        os.system("pause")

    def test_action_runtime_2(self, repeat):
        self.reset()
        list_temp = []
        for action in range(self.numActions()):

            starttime = time.time()
            for i in range(repeat):
                self.takeAction(action)
            endtime = time.time()
            # print(action,":")
            list_temp.append((endtime - starttime)*1000/repeat)
            self.reset()
        print(list_temp)
        return list_temp
        
    def baselineActions(self):

        self._abc.xmg_resub()
        self._abc.xmg_cut_rewrite(cut_size=4)
        self._abc.xmg_cut_rewrite(cut_size=3)
        self._abc.xmg_resub()
        self._abc.xmg_node_resynthesis(cut_size=3)
        self._abc.xmg_resub()
        self._abc.xmg_cut_rewrite(cut_size=4)
        self._abc.xmg_cut_rewrite(cut_size=3)
        self._abc.xmg_resub()


    def reset(self):
        self.lenSeq = 0
        self._abc.xmg_end()
        self._abc.xmg_start()
        self._abc.xmg_read(self._xmgfile)
        self._lastStats = self._abc.xmgStats() # The initial XMG statistics
        # self._lastStats_end2end = self.end_to_end_result
        self._curStats = self._lastStats # the current XMG statistics
        # self._curStats_end2end = self._lastStats_end2end
        self.lastAct = self.numActions() - 1
        self.lastAct2 = self.numActions() - 1
        self.lastAct3 = self.numActions() - 1
        self.lastAct4 = self.numActions() - 1
        self.lastAct5 = self.numActions() - 1
        self.lastAct6 = self.numActions() - 1
        self.lastAct7 = self.numActions() - 1
        self.actsTaken = np.zeros(self.numActions())
        return self.state()

    def close(self):
        self.reset()
    def step(self, actionIdx):
        self.takeAction(actionIdx)
        nextState = self.state()
        reward = self.reward()
        done = False
        if (self.lenSeq >= length_of_command):
            done = True
        return nextState, reward, done, 0
    def takeAction(self, actionIdx):
        """
        @return true: episode is end
        """
        # "b -l; rs -K 6 -l; rw -l; rs -K 6 -N 2 -l; rf -l; rs -K 8 -l; b -l; rs -K 8 -N 2 -l; rw -l; rs -K 10 -l; rwz -l; rs -K      10 -N 2 -l; b -l; rs -K 12 -l; rfz -l; rs -K 12 -N 2 -l; rwz -l; b -l
        self.lastAct7 = self.lastAct6
        self.lastAct6 = self.lastAct5
        self.lastAct5 = self.lastAct4
        self.lastAct4 = self.lastAct3
        self.lastAct3 = self.lastAct2
        self.lastAct2 = self.lastAct
        self.lastAct = actionIdx
        #self.actsTaken[actionIdx] += 1
        self.lenSeq += 1

        if actionIdx == 0:
            self._abc.xmg_depth_rewrite(allow_size_increase=True, start='a', overhead=1.2)
        elif actionIdx == 1:
            self._abc.xmg_depth_rewrite(start='s', allow_size_increase=False)
        elif actionIdx == 2:
            self._abc.xmg_resub()
        elif actionIdx == 3:
            self._abc.xmg_node_resynthesis(cut_size=3)
        elif actionIdx == 4:
            self._abc.xmg_node_resynthesis(cut_size=4)
        elif actionIdx == 5:
            self._abc.xmg_cut_rewrite(cut_size=2)
        elif actionIdx == 6:
            self._abc.xmg_cut_rewrite(cut_size=3)
        elif actionIdx == 7:
            self._abc.xmg_cut_rewrite(cut_size=4)
        elif actionIdx == 8:
            self._abc.end()
            return True
        else:
            assert(False)
        """
        elif actionIdx == 3:
            self._abc.rewrite(z=True) #rwz
        elif actionIdx == 4:
            self._abc.refactor(z=True) #rfz
        """


        # update the statitics
        # self._lastStats = self._curStats
        # self._lastStats_end2end = self._curStats_end2end
        # self._curStats = self._abc.xmgStats()
        # self._curStats_end2end = self.get_end2end_states()
        self._lastStats = self._curStats
        self._curStats = self._abc.xmgStats()
        return False
    '''def get_end2end_states(self):
        self.write_verilog(str(self.PID)+"temp.v")
        command_temp = "./converter -d --xor3 " + str(self.PID) + "temp.v"
        os.system(command_temp)
        command_temp2 = "python map_priority.py " + str(self.PID) + "temp_temp.txt > " + str(self.PID) + "temp_map.txt"
        # max_mem = map_priority(str(self.brief_name) +"_syn_out_opt_1_temp.txt", self.brief_name)
        os.system(command_temp2)
        return xmg_evaluation(str(self.PID) + "temp_map.txt")
    def get_end2end_map_txt(self, name_brief):
        self.write_verilog(str(self.PID)+"temp.v")
        command_temp = "./converter -d --xor3 " + str(self.PID) + "temp.v"
        os.system(command_temp)
        command_temp2 = "python map_priority.py " + str(self.PID) + "temp_temp.txt > " + "best_mapping/"+str(name_brief) + "_map_best.txt"
        # max_mem = map_priority(str(self.brief_name) +"_syn_out_opt_1_temp.txt", self.brief_name)
        os.system(command_temp2)
        return xmg_evaluation("best_mapping/" + str(name_brief) + "_map_best.txt")'''
    def state(self):
        """
        @brief current state
        """
        oneHotAct = np.zeros(self.numActions()) #self.numActions=5
        #print("self.lastAct:", self.lastAct)
        np.put(oneHotAct, self.lastAct, 1) #oneHotAct[self.lastAct]=1
        lastOneHotActs  = np.zeros(self.numActions())
        lastOneHotActs[self.lastAct2] += 1/4
        lastOneHotActs[self.lastAct3] += 1/4
        lastOneHotActs[self.lastAct] += 1/4
        lastOneHotActs[self.lastAct4] += 1/4

        stateArray = np.array([self._curStats.numXmgGates / self.initNumXmgGates,
            self._lastStats.numXmgGates / self.initNumXmgGates])
        stepArray = np.array([float(self.lenSeq) / length_of_command])
        combined = np.concatenate((stateArray, lastOneHotActs, stepArray), axis=-1)
        # print("combined Input state :", combined)
        # combined = np.expand_dims(combined, axis=0)
        # return stateArray.astype(np.float32)
        combined_torch = torch.from_numpy(combined.astype(np.float32)).float()
        # combine_torch share the same memary with combined
        # print("GE\n", datetime.now())
        '''for i in range(100):
            graph = GE.extract_dgl_graph(self._abc)
        print("endGE", datetime.now())'''
        graph = GE.extract_dgl_graph_xmg(self._abc)
        #print(graph)
        #print("input graph:", graph)
        #combined_torch.requires_grad = False

        return (combined_torch , graph)
    def write_verilog(self, filename):
        self._abc.write_verilog(filename)
    def reward(self):
        if self.lastAct == 8:  # term
            return 0

        # gate_num, latency, energy, row_usage,e*l = self.get_end2end_states()
        # weighted_reward = (self._lastStats_end2end[self.target_end2end_index] - self._curStats_end2end[self.target_end2end_index])/self.end_to_end_result[self.target_end2end_index] - self._rewardBaseline[self.target_end2end_index]
        # print("lastStats:", self.statValue(self._lastStats), "curStats:", self.statValue(self._curStats),"rewardBaseline:", self._rewardBaseline)
        # print("final rewards:", self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline)
        # old_reward = self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline
        # note that statValue is normalized
        #return weighted_reward
        #return self._lastStats.numAnd + self._curStats.numAnd - 1
        return self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline
        # return self._lastStats.numAnd + self._curStats.numAnd - 1
        if (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.xmg_lev < self._lastStats.xmg_lev):
            return 2
        elif (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.xmg_lev == self._lastStats.xmg_lev):
            return 0
        elif (self._curStats.numAnd == self._lastStats.numAnd and self._curStats.xmg_lev < self._lastStats.xmg_lev):
            return 1
        else:
            return -2
    def numActions(self):
        return 8
    def dimState(self):
        return 2 + self.numActions() * 1 + 1
    def returns(self):
        return self._curStats.numXmgGates
    def statValue(self, stat):
        #return float(stat.xmg_lev)  / float(self.initLev)
        return float(stat.numXmgGates) / float(self.initNumXmgGates) #  + float(stat.xmg_lev)  / float(self.initLev)
        #return stat.numAnd + stat.xmg_lev * 10
    def curStatsValue(self):
        return self.statValue(self._curStats)
    def seed(self, sd):
        pass
    def compress2rs(self):
        self._abc.compress2rs()
