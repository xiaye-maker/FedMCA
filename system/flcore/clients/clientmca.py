# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import copy
import random
import torch.nn.functional as F


class clientMCA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # self.tau = args.tau
        self.mu = args.mu
        self.lamda = args.lamda

        self.MSE = nn.MSELoss()
        self.KL = nn.KLDivLoss()
        self.model_s = copy.deepcopy(self.model.base)
        for param in self.model_s.parameters():
            param.requires_grad = False

        self.cmodel = copy.deepcopy(self.model.head)
        self.received_models = []
        self.other_models = []

        trainloader = self.load_train_data()
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = self.model.base(x).detach()
            break
        self.feature_dim = rep.shape[1]
        self.sample_per_class = torch.zeros(self.num_classes)

        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        self.classes_index = []
        self.index_classes = torch.zeros(self.num_classes, dtype=torch.int64)
        for idx, c in enumerate(self.sample_per_class):
            if c > 0:
                self.classes_index.append(idx)
                self.index_classes[idx] += len(self.classes_index) - 1
        self.classes_index = torch.tensor(self.classes_index, device=self.device)
        self.cnum_classes = torch.sum(self.sample_per_class > 0).item()
        print(f'Client {self.id} has {self.cnum_classes} classes.')

        # 将全连接改造为只有客户端类的神经元节点（减少计算资源）
        self.cmodel = nn.Linear(self.feature_dim, self.cnum_classes, bias=False).to(self.device)

        self.feature_dim = list(args.model.head.parameters())[0].shape[1]
        self.W_h = nn.Linear(self.feature_dim, self.feature_dim, bias=False).to(self.device)
        self.optimizer_W = torch.optim.SGD(self.W_h.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_W = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_W,
            gamma=args.learning_rate_decay_gamma
        )

        self.optimizer_base = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_base,
            gamma=args.learning_rate_decay_gamma
        )

        self.optimizer_head = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_head,
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        # self.model.to(self.device)
        trainloader = self.load_train_data()
        self.model.train()
        #
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = self.index_classes[y].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                dissimilar_loss = 0.0
                if len(self.other_models) > 0:
                    for dissimilar_model in self.other_models:
                        dissimilar_output = dissimilar_model(rep)
                        dissimilar_loss += self.KL(F.log_softmax(output, dim=1), F.softmax(dissimilar_output, dim=1))
                    dissimilar_loss = dissimilar_loss / len(self.other_models)
                    dissimilar_loss = torch.tensor(dissimilar_loss)  # 转换为Tensor类型

                if len(self.received_models) > 0:
                    similar_loss = 0.0
                    for similar_model in self.received_models:
                        similar_output = similar_model(rep)
                        similar_loss += self.KL(F.log_softmax(output, dim=1), F.softmax(similar_output, dim=1))
                    similar_loss = similar_loss / len(self.received_models)
                    similar_loss = torch.tensor(similar_loss)  # 转换为Tensor类型
                    kl_loss = torch.exp(dissimilar_loss - similar_loss)
                    loss += (kl_loss * self.mu)

                self.optimizer_head.zero_grad()
                loss.backward()
                self.optimizer_head.step()

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = self.index_classes[y].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                # rep_g = self.model_s(x)
                # L_h = self.MSE(rep, self.W_h(rep_g))
                # loss += self.lamda * L_h

                self.optimizer_base.zero_grad()
                # self.optimizer_W.zero_grad()
                loss.backward()
                # self.optimizer_W.step()
                self.optimizer_base.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_base(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()
        self.model_s = base

    def set_head(self, rec, oth):
        self.received_models = rec
        self.other_models = oth

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = self.index_classes[y].to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                if len(set(y)) > 1:
                    y_prob.append(output.detach().cpu().numpy())
                    nc = self.num_classes
                    if self.num_classes == 2:
                        nc += 1
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        lb = lb[:, :2]
                    y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    # @property
    def train_metrics(self):

        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = self.index_classes[y].to(self.device)

                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                # rep_g = self.model_s(x)
                # L_h = self.MSE(rep, self.W_h(rep_g))
                # loss += L_h * self.lamda

                dissimilar_loss = 0.0
                if len(self.other_models) > 0:
                    for dissimilar_model in self.other_models:
                        dissimilar_output = dissimilar_model(rep)
                        dissimilar_loss += self.KL(F.log_softmax(output, dim=1), F.softmax(dissimilar_output, dim=1))
                    dissimilar_loss = dissimilar_loss / len(self.other_models)
                    dissimilar_loss = torch.tensor(dissimilar_loss)  # 转换为Tensor类型

                if len(self.received_models) > 0:
                    similar_loss = 0.0
                    for similar_model in self.received_models:
                        similar_output = similar_model(rep)
                        similar_loss += self.KL(F.log_softmax(output, dim=1), F.softmax(similar_output, dim=1))
                    similar_loss = similar_loss / len(self.received_models)
                    similar_loss = torch.tensor(similar_loss)  # 转换为Tensor类型

                    kl_loss = torch.exp(dissimilar_loss - similar_loss)
                    loss += (kl_loss * self.mu)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def tsne(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        embeddings = []
        labels = []

        with torch.no_grad():
            for x, y in testloaderfull:
                embeddings.append(self.model(x).numpy())
                labels.append(y.numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        return  embeddings, labels