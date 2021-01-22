import os
import sys
import time
import numpy as np
import math
from math import log
from math import exp

import torch
import torch.distributed as dist
from torch.autograd import Variable

from cjltest.utils_model import MySGD


# output: Here is only suitable for those without byzantine 
def mean(g_list, workers, dev, size):
    g_mean = []
    for p_idx, g_layer in enumerate(g_list[0]):
        global_update_layer = torch.zeros_like(g_layer.data).cuda(dev)
        for w in workers:
            global_update_layer += g_list[w][p_idx]
        g_mean.append(global_update_layer / len(workers))
    return g_mean

def IDA(g_list, workers, dev, size, average_k, k_list):
    g_mean = []
    for p_idx, g_layer in enumerate(g_list[0]):
        global_update_layer = torch.zeros_like(g_layer.data).cuda(dev)
        for w in range(size):
            if k_list[w] != 0:
                global_update_layer += g_list[w][p_idx] * float(k_list[w]) / average_k
        g_mean.append(global_update_layer / len(workers))
    return g_mean


def test_model(rank, model, test_data, dev):
    correct = 0
    total = 0
    # model.eval()
    with torch.no_grad():
        for data, target in test_data:
            data, target = Variable(data).cuda(dev), Variable(target).cuda(dev)
            output = model(data)
            # get the index of the max log-probability
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # pred = output.data.max(1)[1]
            # correct += pred.eq(target.data).sum().item()

    acc = format(correct / total, '.4%')
    # print('Rank {}: Test set: Accuracy: {}/{} ({})'
    #       .format(rank, correct, len(test_data.dataset), acc))
    return acc

def run(rank, size, model, args, iterationPerEpoch, test_data):
    cpu = torch.device('cpu')
    gpu = torch.device('cuda:{}'.format(rank%args.num_gpu))

    start = time.time()
    model = model.cuda(gpu)

    workers = [v+1 for v in range(size-1)]
    _group = [w for w in workers].append(rank)
    group = dist.new_group(_group)

    for p in model.parameters():
        tmp_p = torch.tensor(p.data, device=cpu)
        scatter_p_list = [tmp_p for _ in range(size)]
        dist.scatter(tensor=tmp_p, scatter_list=scatter_p_list, group=group)
    
    print('Model has sent to all nodes! ')

    print('Begin!')

    # naming rules: title + model_name
    str_lr = str(args.lr).replace('.', '')
    if args.iid:
        trainloss_file = './result/' + 'FedNovaSGD' + '_' + args.model  + str(size-1) + str(args.local_iteration) + '_iid_lr' + str(str_lr) + '_bsz' + str(args.train_bsz) + '.txt'
    else:
        trainloss_file = './result/' + 'FedNovaSGD' + '_' + args.model  + str(size-1) + str(args.local_iteration) + '_noniid_lr' + str(str_lr) + '_bsz' + str(args.train_bsz) + '.txt'
    if(os.path.isfile(trainloss_file)):
        os.remove(trainloss_file)
    f_trainloss = open(trainloss_file, 'w')

    param_list = [torch.tensor(param.data, device=gpu) for param in model.parameters()]
    momentum_list = [torch.zeros_like(param.data, device=gpu) for param in model.parameters()]
    
    for epoch in range(args.epochs):
        # receive the list of train loss from workers
        info_list = [torch.tensor([0.0]) for _ in range(len(workers) + 1)]
        dist.gather(tensor=torch.tensor([0.0]), gather_list=info_list, group=group)
        epoch_train_loss = sum(info_list).item()

        # receive the value of weighted tau
        k_list = [torch.tensor([0.0]) for _ in range(len(workers) + 1)]
        dist.gather(tensor=torch.tensor([0.0]), gather_list=k_list, group=group)
        tau_eff = sum(k_list).item()

##        # receive a_i*p_i
##        a_list = [torch.tensor([0.0]) for _ in range(size)]
##        dist.gather(tensor=torch.tensor([0.0]), gather_list=a_list, group=group)
##        sum_a = sum(a_list).item()

        # receive normalized gradients*p_i
        sum_d = [torch.zeros_like(param.data, device=gpu) for param in model.parameters()]
        for idx, param in enumerate(model.parameters()):
            tensor = torch.zeros_like(param.data, device=cpu)
            gather_list = [torch.zeros_like(param.data, device=cpu) for _ in range(size)]
            dist.gather(tensor=tensor, gather_list=gather_list, group=group)
            for w in range(size):
                sum_d[idx].data = sum_d[idx].data + torch.tensor(gather_list[w].data, device=gpu)

##        # # receive momentum
##        for idx, param in enumerate(model.parameters()):#receive momentums
##            tensor = torch.zeros_like(param.data, device=cpu)
##            gather_list = [torch.zeros_like(param.data, device=cpu) for _ in range(size)]
##            dist.gather(tensor=tensor, gather_list=gather_list, group=group)
##            # print("Epoch {} Iteration {} Received all nodes' gradient differentiation! ".format(epoch, iteration))
##            all_momentum[0].append(torch.tensor(gather_list[0].data, device=gpu))
##            for w in workers:
##                all_momentum[w].append(torch.tensor(gather_list[w].data, device=gpu))

        # Compute in server
##        # send latest parameters and momentums back to the workers
##        if args.average_method == 'FedNova':
##            new_param = mean(all_param, workers, gpu, size)
##            new_d = mean(all_d, workers, gpu, size)
##        elif args.average_method == 'IDA':
##            new_param = IDA(all_param, workers, gpu, size, average_k, k_list)
##            new_momentum = IDA(all_momentum, workers, gpu, size, average_k, k_list)
##        else:
##            print('No matched average method!')
##            sys.exit(-1)

        # send parameter to workers
        for idx, param in enumerate(model.parameters()):
            param.data = param.data + tau_eff*sum_d[idx].data
            scatter_list = [torch.tensor(param.data, device=cpu) for _ in range(size)]
            param_cpu = torch.zeros_like(param.data, device=cpu)
            dist.scatter(tensor=param_cpu, scatter_list=scatter_list, group=group)

##        # send normalized gradients to workers
##        for idx, param in enumerate(model.parameters()):
##            scatter_list = [torch.tensor(sum_d[idx].data, device=cpu) for _ in range(size)]
##            sum_d_cpu = torch.zeros_like(param.data, device=cpu)
##            dist.scatter(tensor=sum_d_cpu, scatter_list=scatter_list, group=group)

        print("Epoch: {}\t\tLoss: {}\t".format(epoch, epoch_train_loss))
        
        timestamp = time.time() - start
        # timestamp /= 60
        # print('begin test')
        test_acc = test_model(0, model, test_data, gpu)
        # test_acc = 0
        # print('end test')
        f_trainloss.write(str(epoch) + "\t" + str(timestamp) + "\t" + str(epoch_train_loss) + "\t" + str(test_acc) + "\n")
        # print('write into file')
        f_trainloss.flush()
    
    f_trainloss.close()

def init_processes(rank, size, model, args, iterationPerEpoch, test_data, backend='mpi'):
    dist.init_process_group(backend, rank=rank, world_size=size)
    run(rank, size, model, args, iterationPerEpoch, test_data)



