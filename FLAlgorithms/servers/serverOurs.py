from FLAlgorithms.users.userOurs import UserOurs
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


MIN_SAMPLES_PER_LABEL=1

class FedOurs(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args.dataset)
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
            user = UserOurs(args, id, model, train_data, test_data, use_adam=False)
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
            
            self.evaluate()
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

            ### need to change the aggregator ###
            self.aggregate_parameters()
            # model operations
            self.distill(args, 10, self.student_model)
            #########################################
            
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

    def distill(self, args, n_iters, student_model):
        # self.generative_model.train()
        student_model.train()
        
        for i in range(n_iters):
            self.optimizer.zero_grad()
            
            # y=np.random.choice(self.qualified_labels, batch_size)
            # y_input=torch.LongTensor(y)
            ## feed to generator

            # gen_result=self.generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
            # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
            # gen_output, eps=gen_result['output'], gen_result['eps'] # gen_output is 32x32
            
            ##### get losses ####
            # decoded = self.generative_regularizer(gen_output)
            # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
            # diversity_loss=self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

            ######### get teacher loss ############
            teacher_loss=0
            teacher_logit=0
            
            result = self.get_next_public_batch()
            X, y = result['X'], result['y']

            for user_idx, user in enumerate(self.selected_users):
                user.model.eval()

                # weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
                # expand_weight=np.tile(weight, (1, self.unique_labels))
                
                # the public dataset here
                X, y = result['X'], result['y']
                user_result=user.model(X, logit=True) 

                # user_output_logp_= F.log_softmax(user_result['logit'], dim=1)
                
                # teacher_loss_=torch.mean( \
                #     self.student_model.crossentropy_loss(user_output_logp_, y_input) * \
                #     torch.tensor(weight, dtype=torch.float32))
                # teacher_loss+=teacher_loss_
                # teacher_logit += user_result['logit'] * torch.tensor(expand_weight, dtype=torch.float32)
                teacher_logit += user_result['logit'] * torch.tensor( 1 / len(self.selected_users) )
                
            ######### get student loss ############
            student_output=student_model(X, logit=True)
            student_loss=F.kl_div(F.log_softmax(student_output['logit'], dim=1), F.softmax(teacher_logit, dim=1))

            # if self.ensemble_beta > 0:
                # loss=self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss
            # else:
            #     loss=self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss

            loss = student_loss 
            loss.backward()
            self.optimizer.step()

            # TEACHER_LOSS += self.ensemble_alpha * teacher_loss#(torch.mean(TEACHER_LOSS.double())).item()
            # STUDENT_LOSS += self.ensemble_beta * student_loss#(torch.mean(student_loss.double())).item()
            # DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#(torch.mean(diversity_loss.double())).item()
        return loss