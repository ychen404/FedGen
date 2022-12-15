from FLAlgorithms.users.userDF import UserDF
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model, read_public_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
import copy
import time
import pdb
from torch.autograd import Variable as V


MIN_SAMPLES_PER_LABEL=1

class FedDF(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args)
        # data contains: clients, groups, train_data, test_data, proxy_data
        clients = data[0]
        total_users = len(clients)
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower()
        self.use_adam = 'adam' in self.algorithm.lower()

        self.early_stop = 20  # stop using generated samples after 20 local epochs
        self.student_model = copy.deepcopy(self.model)

        public_data = read_public_data(data, args.dataset)
        self.publicloader = DataLoader(public_data, self.batch_size, shuffle=True, drop_last=True)
        self.iter_publicloader = iter(self.publicloader)
        
        if not args.train:
            # print('number of generator parameteres: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters())) # param: 26434
        
        # self.latent_layer_idx = self.generative_model.latent_layer_idx
        
        self.init_ensemble_configs()
        # print("latent_layer_idx: {}".format(self.latent_layer_idx))
        # print("label embedding {}".format(self.generative_model.embedding))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta, self.ensemble_eta))
        # print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))
        self.init_loss_fn()
        self.train_data_loader, self.train_iter, self.available_labels = aggregate_user_data(data, args.dataset, self.ensemble_batch_size)
        
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)

        #### creating users ####
        self.users = []
        for i in range(total_users):
            id, train_data , test_data = read_user_data(i, data, dataset=args.dataset)
            user = UserDF(args, id, model, train_data, test_data, self.writer, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        
        print("Number of users / total users:",args.num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def train(self, args):
        #### pretraining
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users, self.user_idxs=self.select_users(glob_iter, self.num_users, return_idx=True)
            if not self.local:
                self.send_parameters(mode=self.mode)# broadcast averaged prediction model
            
            # self.evaluate()
            self.timestamp = time.time() # log user-training start time
            
            for user_id, user in zip(self.user_idxs, self.selected_users): # allow selected users to train
                # perform regularization using generated samples after the first communication round
                user.train(
                    glob_iter,
                    personalized=self.personalized)

            curr_timestamp = time.time() # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            if self.personalized:
                self.evaluate_personalized_model()

            self.timestamp = time.time() # log server-agg start time


            if args.distill_init == 'averaged':
                print("Init distill from averaged")
                self.aggregate_parameters()
            else:
                print("Init distill from prev")

            # model operations
            self.distill(args, 10, self.student_model)

            exit()
            #########################################
            # self.evaluate(glob_iter=glob_iter)

            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)

        self.save_results(args)
        self.save_model()
        
    def get_next_public_batch(self):
        try:
            # Samples a new batch of public data
            (X, y) = next(self.iter_publicloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_publicloader = iter(self.publicloader)
            (X, y) = next(self.iter_publicloader)
        result = {'X': X, 'y': y}
        return result

    def distill(self, args, distill_epoch, student_model):
        # self.generative_model.train()
        
        if 'EMnist' in args.dataset :
            num_class = 26
        else:
            NotImplementedError("How many classes are there?")
        
        student_model.train()
        outputs = []
        for de in range(distill_epoch):
            self.optimizer.zero_grad()
        
            ######### get teacher loss ############
            teacher_logit = 0
            distill_loss = 0
            correct = 0
            total = 0

            
            for batch_idx, (inputs, targets) in enumerate(self.publicloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                logit_buffer = torch.zeros(targets.shape[0],num_class).cuda() 
                for user_idx, user in enumerate(self.selected_users):
                    user.model.eval()

                    user_result = user.model(inputs, logit=True) 

                    # Be careful here, the '+' sign changes the graph and you will run into the following error:
                    # RuntimeError: Trying to backward through the graph a second time, 
                    # but the saved intermediate results have already been freed. 
                    # Specify retain_graph=True when calling .backward() or autograd.grad() the first time. 

                    # teacher_logit += user_result['logit'] * torch.tensor( 1 / len(self.selected_users) )
                    logit_buffer += user_result['logit'] * torch.tensor( 1 / len(self.selected_users) )
                    teacher_logit = V(logit_buffer)
                    # teacher_logits.append(user_result['logit'] * torch.tensor( 1 / len(self.selected_users)))
                    
                    outputs.append(user_result['output'] * torch.tensor( 1 / len(self.selected_users) ))
                
                # ######### get student loss ############
                student_output = student_model(inputs, logit=True)
                loss = F.kl_div(F.log_softmax(student_output['logit'], dim=1), F.softmax(teacher_logit, dim=1))

                loss.backward()
                self.optimizer.step()                

                outputs_tensors = torch.stack(outputs).sum(dim=0)
                distill_loss += loss.item()
                _, predicted = outputs_tensors.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if batch_idx % 20 == 0:
                    print(f"Distill Epoch: {de}, batch index: {batch_idx}, Loss: {distill_loss/(batch_idx+1)}, Acc: {100.*correct/total}")

        return loss