import torch
from FLAlgorithms.users.userbase import User
import pdb

class UserDF(User):
    def __init__(self,  args, id, model, train_data, test_data, writer, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.writer = writer
        self.id = id.split('_')[1][-1] # keep track of each user for tensorboard

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):

        self.clean_up_counts()
        self.model.train()


        for epoch in range(1, self.local_epochs + 1):
            
            train_loss = 0
            correct = 0
            total = 0
            self.model.train()
            for i in range(self.K):
                for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)['output']
                    # pdb.set_trace()

                    loss = self.loss(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    if batch_idx % 500 == 0:
                        print(f"Epoch: {epoch}, batch index: {batch_idx}, Loss: {train_loss/(batch_idx+1)}, Acc: {100.*correct/total}")
                                
                # using only one batch of data
                # # result =self.get_next_train_batch(count_labels=count_labels)
                # # X, y = result['X'], result['y']
                # # if count_labels:
                # #     # result['labels'] are the unique labels in this batch
                # #     # result['counts'] are the number of samples belonging
                # #     # to each unique label
                # #     self.update_label_counts(result['labels'], result['counts'])

                # # self.optimizer.zero_grad()
                # # output = self.model(X)['output']
                # # loss = self.loss(output, y)

                # # # test_acc, test_loss, _ = self.test()
                # # # test_acc, test_loss = self.eval_test_acc()

                # # if epoch % 20 == 0:
                # #     print(f"Epoch: {epoch}, test acc: {test_acc}, test loss: {test_loss}")
                # # self.writer.add_scalar(f"user {str(self.id)}/Total loss", loss, glob_iter * self.local_epochs + epoch)
                # # self.writer.add_scalar(f"user {str(self.id)}/Test acc", test_acc, glob_iter * self.local_epochs + epoch)
                # # self.writer.add_scalar(f"user {str(self.id)}/Test loss", test_loss, glob_iter * self.local_epochs + epoch)

                # loss.backward()
                # self.optimizer.step()#self.plot_Celeb)

            # local-model <=== self.model
            # this function is not even using its return value
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
            if personalized:
                self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
            # local-model ===> self.model
            #self.clone_model_paramenter(self.local_model, self.model.parameters())
        if lr_decay:
            # self.lr_scheduler.step(glob_iter)
            self.lr_scheduler.step()