# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import numpy as np
import random
from threading import Thread
from math import exp
from math import log

import torch
import torch.distributed as dist
from torch.autograd import Variable

from cjltest.utils_model import MySGD, test_model

def fixed_update(rank, size, args, time_length, model, momentum_buffers, correction, norm_gradients, param_storage, epochs, loss_record, status, group, cpu, gpu, ):
    for epoch in range(epochs):
        status.append(False)
        time.sleep(time_length-5)
        status[-1] = True
        time.sleep(5)

        # if rank == 1:
        #     print("Rank 1 (Thread-1) parameters (Before updates): ")
        #     print(model.parameters())
        #     print(momentum_buffers)


        # calculate the loss and iterations
        loss = sum(loss_record).item() / len(loss_record)
        print("Rank: {}\t\tEpoch: {}\t\tLocal Updates: {}\t\tLoss: {}".format(rank, t, len(loss_record), loss))

        # Synchronization 
        # send epoch train loss to PS
        loss_cpu = torch.tensor(loss, device=cpu)
        dist.gather(tensor=loss_cpu, dst=0, group=group)

        # send K to PS
        tau = float(len(loss_record))
        k_cpu = torch.tensor(tau, device=cpu)
        dist.gather(tensor=k_cpu, dst=0, group=group)

        # Compute a_i
        a = (tau - args.beta*(1-args.beta**tau) / (1-args.beta)) / (1 - args.beta)
        a *= 1/(size-1)
        a_cpu = torch.tensor(a, device=cpu)
    
        # send a_i to server 
        dist.gather(tensor=a_cpu, dst=0, group=group)

        # # send normalized gradients to server
        for idx, param in enumerate(model.parameters()):
            norm_gradients[idx] = param.data - param_storage[idx].data
            norm_gradients[idx] /= args.lr*a*(size-1)
            norm_g_cpu = torch.tensor(data=norm_gradients[idx].data, device=cpu)
            dist.gather(tensor=norm_g_cpu, dst=0, group=group)

        # receive the parameters
        for idx, param in enumerate(model.parameters()):
            recv = torch.zeros_like(param.data, device=cpu)
            dist.scatter(tensor=recv, src=0, group=group)
            param.data = torch.tensor(recv, device=gpu)
            param_storage[idx].data = torch.zeros_like(param.data, device=gpu) + param.data
        
        del(recv)
        # # receive the normalized gradients d_i
        for idx, param in enumerate(model.parameters()):
            recv = torch.zeros_like(param.data, device=cpu)
            dist.scatter(tensor=recv, src=0, group=group)
            recv_d = torch.tensor(recv.data, device=gpu)
            correction[idx].data = recv_d - norm_gradients[idx]
        del(recv, recv_d)
        # Set the momentums to zeros, after each synchronization
        momentum_buffers[idx] = torch.zeros_like(param.data, device=gpu)
        # print("Rank {} threshold: {}".format(rank, threshold))
        print("Rank: {}\t\tEpoch: {}\t\tReceive the new gradient!".format(rank, epoch))

        # if rank == 1:
        #     print("Rank 1 (Thread-1) parameters (end updates): ")
        #     print(model.parameters())
        #     print(momentum_buffers)

        loss_record.clear()

        if epoch % args.lr_decay == 0:
            args.lr /= 10


# noinspection PyTypeChecker
# Notice: transferring requires cpu, calculation requires gpu
def run(rank, size, model, args, train_data, test_data, weight):
    cpu = torch.device('cpu')
    gpu = torch.device('cuda:{}'.format(rank%args.num_gpu))

    model = model.cuda(gpu)

    workers = [v+1 for v in range(size-1)]
    _group = [w for w in workers].append(rank)
    group = dist.new_group(_group)

    param_storage = [torch.zeros_like(param.data, device=gpu) for param in model.parameters()]

    # print('Rank {}: Waiting for receiving the model! '.format(rank))

    # Receive initial model from server
    for idx, p in enumerate(model.parameters()):
        tmp_p = torch.zeros_like(p, device=cpu)
        dist.scatter(tensor=tmp_p, src=0, group=group)
        p.data = torch.tensor(tmp_p, device=gpu)
        param_storage[idx].data += p.data

    print('Rank {} successfully received the model. '.format(rank))

##    gradients = [torch.zeros_like(param.data, device=gpu) for param in model.parameters()]
    norm_gradients = [torch.zeros_like(param.data, device=gpu) for param in model.parameters()]
   
    if args.local_iteration == 'linear':
        local_iteration = (100+(50*rank))
    elif args.local_iteration == 'SL':
        local_iteration = (100+(10*rank))
    elif args.local_iteration == 'LL':
        local_iteration = (100+(100*rank))
    elif args.local_iteration == 'exp':
        local_iteration = (2**(rank-1))
    else:
        print('No matched local iteration!')
        sys.exit(-1)
 
    loss_record, status = [], []
##    sync = Thread(target=fixed_update, args=(rank, size,args,  args.time_length, model, momentum_buffers, correction, norm_gradients, param_storage, args.epochs, loss_record, status, group, cpu, gpu,  ), daemon=True)
##    sync.start()

    optimizer = MySGD(model.parameters(), lr=args.lr)

    if args.model in ['MnistCNN', 'AlexNet']:
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()


    print('Rank {} begins!'.format(rank))

    model.train()
    batch_iter = iter(train_data)

    for t in range(args.epochs):
        for it in range(local_iteration):
            try:
                data, target = next(batch_iter)
            except:
                batch_iter = iter(train_data)
                data, target = next(batch_iter)
            
            data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            delta_ws = optimizer.get_delta_w()
            loss_record.append(loss.data)
            print('Rank: {}\t\tEpoch: {}\t\tIteration: {}\t\tLoss: {}'.format(rank, t, len(loss_record)-1, loss_record[-1]))
            
            for idx, param in enumerate(model.parameters()):
    ##            gradients[idx] = delta_ws[idx] / args.lr
                # if args.method == 'RemSGD':            
                    # param.data = param.data - args.lr * gradients[idx] * (args.gamma ** math.log2(len(loss_record))) + args.beta * momentum_buffers[idx]
                    # momentum_buffers[idx] = - args.lr * gradients[idx] * (args.gamma ** math.log2(len(loss_record))) + args.beta * momentum_buffers[idx]
##                if args.method == 'FedAvg':
##                    inter_grad = delta_ws[idx]/args.lr
                    #delta_ws[idx].cuda(gpu)
##                    momentum_buffers[idx] = args.beta * momentum_buffers[idx] + inter_grad
                param.data = param.data - delta_ws[idx]
##                else:
##                    print('No matched method! Need FedAvg.')
##                    sys.exit(-1)

        # Synchronization

        # calculate the loss and iterations
        loss = sum(loss_record).item() / len(loss_record)
        print("Rank: {}\t\tEpoch: {}\t\tLocal Updates: {}\t\tLoss: {}".format(rank, t, len(loss_record), loss))
        # send epoch train loss to PS
        loss_cpu = torch.tensor(loss*weight, device=cpu)
        dist.gather(tensor=loss_cpu, dst=0, group=group)

        # send weighted tau to PS
        tau = float(len(loss_record))
        k = tau * weight
        # print(tau)
        k_cpu = torch.tensor(k, device=cpu)
        dist.gather(tensor=k_cpu, dst=0, group=group)

##        # Compute a_i
##        if args.local_solver == 'FedAvg':
##            a = (tau - args.beta*(1-math.pow(args.beta, tau)) / (1-args.beta)) / (1 - args.beta)
##            a_cpu = torch.tensor(tau*weight, device=cpu)
##        elif args.local_solver == 'FedProx':
##            
##        # send a_i*p_i to server 
##        dist.gather(tensor=a_cpu, dst=0, group=group)

        # # send normalized gradients*p_i to server
        for idx, param in enumerate(model.parameters()):
            norm_gradients[idx] = param.data - param_storage[idx].data
            #print(norm_gradients[idx].data)
            norm_gradients[idx] = norm_gradients[idx] / tau
            norm_g_cpu = torch.tensor(data=norm_gradients[idx].data*weight, device=cpu)
            dist.gather(tensor=norm_g_cpu, dst=0, group=group)
        
        # receive the parameters
        for idx, param in enumerate(model.parameters()):
            recv = torch.zeros_like(param.data, device=cpu)
            dist.scatter(tensor=recv, src=0, group=group)
            param.data = torch.tensor(recv.data, device=gpu)
            param_storage[idx].data = torch.zeros_like(param.data, device=gpu) + param.data
        #print(param_storage[5].data)
        #del(recv)

##        # # receive the normalized gradients p_i*d_i
##        for idx, param in enumerate(model.parameters()):
##            recv = torch.zeros_like(param.data, device=cpu)
##            dist.scatter(tensor=recv, src=0, group=group)
##            recv_d = torch.tensor(recv.data, device=gpu)
##            correction[idx].data = recv_d - norm_gradients[idx]
        #del(recv, recv_d)
        # Set the momentums to zeros, after each synchronization
        # momentum_buffers[idx] = torch.zeros_like(param.data, device=gpu)
        # print("Rank {} threshold: {}".format(rank, threshold))
        print("Rank: {}\t\tEpoch: {}\t\tReceive the new gradient!".format(rank, t))
        loss_record.clear()


##        time.sleep(1/(2**(rank))) #### still need revised, to enlarge the difference between each workers ####

def init_processes(rank, size, model, args, train_data, test_data, weight, backend='mpi'):
    dist.init_process_group(backend, rank=rank, world_size=size)
    run(rank, size, model, args, train_data, test_data, weight)

