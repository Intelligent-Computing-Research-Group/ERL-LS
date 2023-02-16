#!/usr/bin/python
# -*- coding: UTF-8 -*-
from convert_list2actions import get_metrics, make_abc_command
from datetime import datetime
import os
from subprocess import check_output
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from torch.utils.tensorboard import SummaryWriter

import sys, getopt

import torch
import time
import reinforce_aig as RF
from env_aig import EnvGraph_aig as Env  # change this to switch abc/mtl/mtl_xmg
from env_forLargeFile import EnvGraph_mtl_xmg as Env_largefile  # Remove some time-consuming instructions
import torch.multiprocessing as mp

import statistics

MAX_EP = 400

def prRed(prt): print("\033[91m {}\033[00m".format(prt))


def prGreen(prt): print("\033[92m {}\033[00m".format(prt))


def prYellow(prt): print("\033[93m {}\033[00m".format(prt))


def prLightPurple(prt): print("\033[94m {}\033[00m".format(prt))


def prPurple(prt): print("\033[95m {}\033[00m".format(prt))


def prCyan(prt): print("\033[96m {}\033[00m".format(prt))


def prLightGray(prt): print("\033[97m {}\033[00m".format(prt))


def prBlack(prt): print("\033[98m {}\033[00m".format(prt))


class MtlReturn:
    def __init__(self, returns):
        self.numGates = float(returns[0])  # in fact this is numGates
        self.level = float(returns[1])

    def __lt__(self, other):
        if (int(self.level) == int(other.level)):
            return self.numGates < other.numGates
        else:
            return self.level < other.level

    def __eq__(self, other):
        return int(self.level) == int(other.level) and int(self.numGates) == int(self.numGates)


def takeSecond(elem):
    return elem[0]


class Trajectory(object):
    """
    @brief The experience of a trajectory
    """

    def __init__(self, states, rewards, actions, value):
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.value = value
        # self.env_temp = env_temp

    def __lt__(self, other):
        return self.value < other.value
def is_equal(a, b):
    if -1e-5 < a-b < 1e-5:
        return True
    else:
        return False
class worker(mp.Process):
    def __init__(self, res_queue, time_queue, area_queue, delay_queue, name, circuit_file_name, gamma, pi, baseline, seed, FMG, FM, global_ep,
                 global_best_reward, global_best_gates, global_heuristic_result, processes, start_time, largefile, warmup, error_of_v_appro):
        super(worker, self).__init__()
        self.name = 'w%i' % name
        print("init for env:", self.name)
        self.global_heuristic_result = global_heuristic_result
        # if largefile:
        #     self.env = Env_largefile(circuit_file_name, self.global_heuristic_result)
        # else:
        #     self.env = Env(circuit_file_name, self.global_heuristic_result)
        self.circuit_file_name = circuit_file_name
        self.env = Env(circuit_file_name, self.global_heuristic_result)
        self._gamma = gamma
        self._global_FMG = FMG
        self._global_FM = FM
        self.global_epoch = global_ep
        self.global_best_reward = global_best_reward
        self._global_baseline = baseline
        self.local_FMG = RF.FcModelGraph(self.env.dimState(), self.env.numActions())
        self.local_FM = RF.FcModel(self.env.dimState(), 1)
        self._pi = RF.PiApprox(self.env.dimState(), self.env.numActions(), 8e-4, self.local_FMG)  # local
        self._baseline = RF.BaselineVApprox(self.env.dimState(), 3e-3, FM)  # use global FM
        self.error_of_v_appro = error_of_v_appro
        self.res_queue = res_queue
        self.time_queue = time_queue
        self.delay_queue = delay_queue
        self.area_queue = area_queue

        self.memTrajectory = []
        self.seed = seed
        self.warmup = warmup
        self.sumRewards = []
        self.epoch_count = 0
        self.max_num_worker = processes
        self.start_time = start_time
        self.local_best_reward = -9999.9
        self.local_best_gates = self.env.initNumAnd
        self.global_best_gates = global_best_gates
        self.cur_length_of_command = 20
        self.time_queue.put(0.0)
        self.res_queue.put(self.env.initNumAnd)
        actions = []
        self.post_mapping = False
        if self.post_mapping:
            abc_command = make_abc_command(actions, self.circuit_file_name)
            proc = check_output(['yosys-abc', '-c', abc_command])
            # print(proc)
            delay, area = get_metrics(proc)
            self.area_queue.put(area)
            self.delay_queue.put(delay)

        #self.ttl = 200/processes  # time to live
        self.ttl = 200  # time to live
        self.epoch_limit = 200

    def run(self):
        while self.global_epoch.value < self.epoch_limit:

            # print(self.name + "generate Trajectory...")
            do_reset = True
            if do_reset:
                self.env.reset()

            state = self.env.state()  # generate input state tensor, including state vector and ABC graph
            term = False
            states, rewards, actions = [], [0], []
            temp_increase = 0
            min_area = 9999999
            while not term:
                # print("num of states:", len(states))
                action = self._pi(state[0], state[1], True)  # _pi is PiApprox actually

                term = self.env.takeAction(action)

                nextState = self.env.state()
                # return state[0] and state[1]
                nextReward = self.env.reward()

                states.append(state)
                rewards.append(nextReward)
                actions.append(action)

                if self.post_mapping:
                    abc_command = make_abc_command(actions, self.circuit_file_name)
                    proc = check_output(['yosys-abc', '-c', abc_command])
                    # print(proc)
                    delay, area = get_metrics(proc)
                    print("area", area)
                    #print("delay", delay)
                    if area < min_area:
                        min_area=area

                state = nextState
                # print("\n")

                if len(states) >= self.cur_length_of_command:  # 9 is lenth of command
                    term = True
            # G_temp = sum(self._gamma ** (k - tIdx - 1) * rewards[k] for k in range(tIdx + 1, self.lenSeq + 1))
            # self.cur_length_of_command -= temp_increase
            # self.env.length_of_command -= temp_increase
            self.memTrajectory.append(actions)
            print('\n')
            print(self.name, "in global epoch:", self.global_epoch.value)
            print(self.name, "in local epoch:", self.epoch_count)
            cur_gates = self.env.returns()
            cur_sum_of_rewards = sum(rewards)
            cur_sum_of_rewards /= self.cur_length_of_command
            print("cur_sum_of_rewards:", cur_sum_of_rewards)

            print("local best record:", self.local_best_reward)
            print("global best record:", self.global_best_reward.value)
            update_global = False
            pull_from_global = False

            if is_equal(self.local_best_gates, cur_gates):
                # self.cur_length_of_command += 1
                # self.env.length_of_command += 1
                prCyan("SOTA policy but found no better results")  # update local policy

            if cur_gates < self.local_best_gates:
                prYellow("new local best record")
                self.local_best_reward = cur_sum_of_rewards
                self.local_best_gates = cur_gates
                # prYellow(self.local_best_reward)
                prYellow(cur_gates)

            if is_equal(self.local_best_gates, self.global_best_gates.value):
                prCyan("SOTA policy but found no better results")  # update local policy
                self.ttl -= 1
                if self.ttl <= 0:
                    pull_from_global = True
                    self.ttl += 20
                    prRed("Aged")

            elif self.local_best_gates > self.global_best_gates.value:
                if self.epoch_count >= self.warmup:
                    pull_from_global = True
                    prRed("loser")
                    self.local_best_reward = self.global_best_reward.value
                    self.local_best_gates = self.global_best_gates.value
                    # self.cur_length_of_command += 1
                    # self.env.length_of_command += 1
                    self.ttl += 20
            else:
                update_global = True
                self.global_best_reward.value = self.local_best_reward
                if cur_gates < self.global_best_gates.value:
                    self.global_best_gates.value = cur_gates
                prGreen("new global best record")
                prGreen(self.global_best_reward.value)
                prGreen("new best gates")
                prGreen(self.global_best_gates.value)
                # self.cur_length_of_command += 1
                # self.env.length_of_command += 1
                #self.ttl *= 2
                self.ttl += 20
            print("cur_gates:", cur_gates)
            print("global_best_gates:", self.global_best_gates.value)
            print("actions:", actions)

            #####
            # abc_command = make_abc_command(actions, self.circuit_file_name)
            # proc = check_output(['yosys-abc', '-c', abc_command])
            # # print(proc)
            # delay, area = get_metrics(proc)
            # print("area", area)
            # print("delay", delay)
            #####

            prPurple("ttl:" + str(self.ttl))
            # print("current returns:", self.env.returns())
            print('After updated -------------------')
            print("global best record:", self.global_best_reward.value)
            print("local best record:", self.local_best_reward)
            during = time.time() - self.start_time
            print("cur_time:", during)
                # self.global_best_gates.value = self.global_best_gates
            # with open(log, 'a') as outLog:
            #     line ="\n"
            #     outLog.write(line)

            # print(self.name + " End of genTraj------------------------------------------------\n")

            # T = Trajectory(states, rewards, actions, self.env.curStatsValue())
            # states = T.states
            # # print("states:")
            # # print(states)
            # rewards = T.rewards
            # # print("rewards:")
            # # print(rewards)
            # actions = T.actions
            self.lenSeq = len(states)  # Length of the episode
            # print("count:", len(states))
            # update--------------------------------------------------
            if self.epoch_count < self.warmup:
                pull_from_global = False
            for tIdx in range(self.lenSeq):
                G = sum(self._gamma ** (k - tIdx - 1) * rewards[k] for k in
                        range(tIdx + 1, self.lenSeq + 1))  # it is a nice format
                # print("tIdx:", tIdx)
                # print("G:", G)
                state = states[tIdx]
                action = actions[tIdx]
                baseline = self._baseline(
                    state[0])  # get an approximation with an FC model and combined tensor, using BaselineVApprox
                # print("baseline:", baseline)
                delta = G - baseline
                self._baseline.update(state[0], G)
                # print("ok to update baseline")
                self._pi.update(state[0], state[1], action, self._gamma ** tIdx, delta, self._global_FMG,
                                is_update=update_global, pull_from_global=pull_from_global)


            # self.sumRewards.append(sum_of_rewards)
            # end = time.time()

            self.global_epoch.value += 1
            self.epoch_count += 1

            self.time_queue.put(during)
            self.res_queue.put(cur_gates)
            if self.post_mapping:
                self.area_queue.put(min_area)
                self.delay_queue.put(delay)
            if self.global_epoch.value > self.epoch_limit:
                prLightPurple('Worker reached its limit')
                #return
                break
            if self.epoch_count > self.epoch_limit:
                prLightPurple('Worker reached its limit')
                #return
                break
            # end while
        self.res_queue.put(None)
def testReinforce(filename, ben, process, brief_name, batch_idx):
    resultName = "end2end_result/" + ben + "basic-info.csv"
    TestRecordName = "end2end_result/" + ben + "detailed-TestRecord.csv"
    #logfile = "asap7/runs_aig-age-v7a-" + str(brief_name) + '-'+str(process) + "worker"

    logfile = "mp-abcRL-200epoch" + str(batch_idx) + "/runs_aig-age-v7a-" + str(brief_name) + '-' + str(process) + "worker"
    global_heuristic_result = mp.Value('f', 0)
    # if largeFile:
    #     temp_env = Env_largefile(filename, global_heuristic_result)
    # else:
    #     temp_env = Env(filename, global_heuristic_result)
    temp_env = Env(filename, global_heuristic_result)
    # global_heuristic_result = mp.Value('f', 0)
    writer = SummaryWriter(log_dir=logfile)
    numActions = temp_env.numActions()
    dimState = temp_env.dimState()
    global_FMG = RF.FcModelGraph(dimState, numActions)
    global_FMG.share_memory()
    vApprox = RF.PiApprox(dimState, numActions, 8e-4, global_FMG)  # dimStates, numActs, alpha, network
    global_FM = RF.FcModel(dimState, 1)
    global_FM.share_memory()
    vbaseline = RF.BaselineVApprox(dimState, 3e-3, global_FM)

    # reinforce = RF.Reinforce(env, 0.9, vApprox, vbaseline, ben, filename, brief_name) #env, gamma, pi, baseline
    global_epoch = mp.Value('i', 0)
    global_warmup = mp.Value('i', 10)
    global_best_reward = mp.Value('f', -9999.9)
    global_best_gates = mp.Value('f', 99999.9)
    error_of_v_appro = mp.Value('f', 0)

    res = []
    timestamp = []
    res_queue = mp.Queue()
    time_queue = mp.Queue()
    area_queue = mp.Queue()
    delay_queue = mp.Queue()


    record_starttime = time.time()
    workers = [
        worker(pi=vApprox, baseline=vbaseline, circuit_file_name=filename, gamma=0.9, name=i, res_queue=res_queue,
               time_queue=time_queue, area_queue=area_queue, delay_queue=delay_queue, seed=i * i, FMG=global_FMG, FM=global_FM, global_ep=global_epoch,
               global_best_reward=global_best_reward, global_best_gates=global_best_gates, global_heuristic_result=global_heuristic_result, processes=process,
               start_time=record_starttime, largefile=largeFile, warmup=0, error_of_v_appro=error_of_v_appro) for i in range(process)]
    now = datetime.now()
    StartTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    print("\033[0;32;40m" + StartTime + "\033[0m")

    #[w.start() for w in workers]
    for w in workers:
        w.start()
        time.sleep(5)
    num_explored_spaces = 0
    best_area = 9999999
    post_mapping = False
    while True:
        r = res_queue.get()
        t = time_queue.get()
        if post_mapping:
            area = area_queue.get()
            delay = delay_queue.get()
            if area < best_area:
                best_area = area
        if r is not None:
            res.append(r)
            timestamp.append(t)
            writer.add_scalars("runtime/#gates in "+str(brief_name), {'global_best_gates': global_best_gates.value, 'current gates': r}, t)
            writer.add_scalars("explored spaces/#gates in "+str(brief_name), {'global_best_gates': global_best_gates.value, 'current gates': r}, num_explored_spaces)
            if post_mapping:
                writer.add_scalars("runtime/area in " + str(brief_name), {'global_best_area': best_area, 'current area': area}, t)
                writer.add_scalars("explored spaces/area in " + str(brief_name), {'global_best_area': best_area, 'current area': area}, num_explored_spaces)
            num_explored_spaces += 1
        else:
            break
    [w.join() for w in workers]
    # with open(resultName, 'a') as andLog:
    #     line = ""
    #     line += str(res)
    #     line += " "
    #     line += str(timestamp)
    #     line += "\n"
    #     andLog.write(line)
    # import matplotlib.pyplot as plt
    # plt.plot(res)
    # plt.ylabel('reward')
    # plt.xlabel("epoch")
    # plt.show()
    # input()
    for i in range(4):
        res.append(res_queue.get())
    now2 = datetime.now()
    EndTime = now2.strftime("%m/%d/%Y, %H:%M:%S") + "\n"

    TimeConsume = (now2 - now)

    print("\033[0;31;40m" + EndTime + "\033[0m")

    print("\n\n")
    # reinforce.replay()

    # lastfive.sort(key=lambda x : x.level)

    # print("lastfive:", lastfive)
    random_test = 0
    print("rewards:\n")
    print(res)

    now = datetime.now()
    EndTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    print("EndTime ", EndTime)

    with open(resultName, 'a') as andLog:
        line = ""
        line += str(EndTime)
        line += "\n"
        line += " "
        andLog.write(line)

if __name__ == "__main__":
    '''
    env = Env("./bench/i10.aig")
    vbaseline = RF.BaselineVApprox(4, 3e-3, RF.FcModel)
    for i in range(10000000):
        with open('log', 'a', 0) as outLog:
            line = "iter  "+ str(i) + "\n"
            outLog.write(line)
        vbaseline.update(np.array([2675.0 / 2675, 50.0 / 50, 2675. / 2675, 50.0 / 50]), 422.5518 / 2675)
        vbaseline.update(np.array([2282. / 2675,   47. / 50, 2675. / 2675,   47. / 50]), 29.8503 / 2675)
        vbaseline.update(np.array([2264. / 2675,   45. / 50, 2282. / 2675,   45. / 50]), 11.97 / 2675)
        vbaseline.update(np.array([2255. / 2675,   44. / 50, 2264. / 2675,   44. / 50]), 3 / 2675)
    '''

    argv_ = sys.argv[1:]
    #inputfile = './bench/sin.aig'
    #inputfile = './bench/c1355.aig'
    #briefname = 'c1355'
    process = '1'
    target = '0'
    largeFile = '0'  # modify into batch_idx

    try:
        opts, args = getopt.getopt(argv_, "hi:n:p:l:", ["ifile=", "name=", "process=", "largeFile="])
    except getopt.GetoptError:
        print('testReinforce.py -i <inputfile> -n <name> -p <process> -l <batch_idx>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -n <name> -p <process> -l <batch_idx>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-n", "--name"):
            briefname = arg
        elif opt in ("-p", "--process"):
            process = arg
        # elif opt in ("-t", "--target"):
        #     target = arg
        elif opt in ("-l", "--largeFile"):
            largeFile = arg
    print("input file:", inputfile)
    print("briefname:", briefname)
    # print("process", process)
    target_list = ["gate_num-", "latency-", "energy-", "row_usage-", "energy_latency_prodcut-"]
    #testReinforce(inputfile, briefname+"_xmg_9steps_"+str(process)+"-in-1-"+target_list[int(target)], process=int(process), brief_name=briefname, largeFile=largeFile)

    testReinforce(inputfile, briefname+"_aig_20steps_"+str(process)+"-agent", process=int(process), brief_name=briefname, batch_idx=largeFile)

    # testReinforce(inputfile, name+"_xmg_9steps_"+str(process)+"-in-1-"+target_list[int(target)], int(process), name, int(target), largeFile)

    # i10 c1355 c7552 c6288 c5315 dalu k2 mainpla apex1 bc0
    # testReinforce("./bench/MCNC/Combinational/blif/dalu.blif", "dalu")
    # testReinforce("./bench/MCNC/Combinational/blif/prom1.blif", "prom1")
    # testReinforce("./bench/MCNC/Combinational/blif/mainpla.blif", "mainpla")
    # testReinforce("./bench/MCNC/Combinational/blif/k2.blif", "k2")
    # testReinforce("./bench/MCNC/c1355_syn_out_opt_1.v", "c1355_syn_out_opt_1")

    # testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/mult_4_syn_out_opt_1.v", "mult_4")
    # testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/mult_64_syn_out_opt_1.v", "mult_64")
    # testReinforce("/home/lcy/Downloads/MIG_project-main_4-16/MIG_project-main/mult_8_syn_out_opt_1.v", "mult_8_xmg_10steps-4-18")

    # testReinforce("/home/lcy/Downloads/MIG_project-main/epfl_sin_syn_out_opt_1.v", "epfl_max_xmg_9steps_7-2 4-in-1", process=1, largeFile=False, brief_name='sin')

    #testReinforce("/home/lcy/Downloads/MIG_project-main/epfl_priority_syn_out_opt_1.v", "epfl_priority_xmg_9steps_7-14", process=4, largeFile=False, brief_name='epfl_priority1')

    # testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/div_32_syn_out_opt_1.v", "div_32_xmg_10steps-4-21")
    # testReinforce("/home/lcy/Downloads/MIG_project-main-4-19/MIG_project-main/div_16_syn_out_opt_1.v", "div_16_xmg_9steps-5-3-Release4.0 serial4-in-1")
    # testReinforce("/home/lcy/Downloads/MIG_project-main-4-19/MIG_project-main/add_64_syn_out_opt_1.v", "add_64_xmg_9steps-4-23")
    # testReinforce("/home/lcy/Downloads/MIG_project-main-4-19/MIG_project-main/epfl_priority_syn_out_opt_1.v", "epfl_priority_xmg_9steps-4-23")
    # testReinforce("/home/lcy/Downloads/MIG_project-main-4-19/MIG_project-main/div_32_syn_out_opt_1.v", "div32_xmg_9steps-4-26-mp")
    # testReinforce("/home/lcy/Downloads/MIG_project-main-4-19/MIG_project-main/epfl_voter_syn_out_opt_1.v", "epfl_voter_xmg_9steps-4-23")
    # testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/epfl_log2_syn_out_opt_1.v", "epfl_log2_xmg")
    # testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/epfl_sin_syn_out_opt_1.v", "epfl_sin_xmg_9steps-4-22")
    # testReinforce("/home/lcy/Downloads/MIG_project-main_3_21/MIG_project-main/epfl_sqrt_syn_out_opt_1.v", "epfl_sqrt_xmg")
    # testReinforce("./bench/MCNC/add_64_syn_out_opt_1.v", "add_64")

    # testReinforce("./bench/MCNC/add_64.v", "add_4_syn_out")
    # testReinforce("./bench/i10.aig", "i10-" + str('test'))
    # testReinforce("./bench/ISCAS/Verilog/c2670.v", "c2670.v")
    # testReinforce("./bench/MCNC/add_4_syn_out.v", "add_4_syn_out")

    # testReinforce("./bench/ISCAS/blif/c5315.blif", "c5315-mig")
    # testReinforce("./bench/ISCAS/blif/c6288.blif", "c6288-mig")
    # testReinforce("./bench/MCNC/Combinational/blif/apex1.blif", "apex1")
    # testReinforce("./bench/MCNC/Combinational/blif/bc0.blif", "bc0")
    '''for i in range(7):
        testReinforce("./bench/i10.aig", "i10-"+str(i+4))'''
    #
    # testReinforce("./bench/ISCAS/blif/c1355.blif", "c1355-mig")
    # testReinforce("./bench/ISCAS/blif/c7552.blif", "c7552-mig")
