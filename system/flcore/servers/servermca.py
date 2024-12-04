# -*- coding: utf-8 -*-
import copy
import random
import time
import torch
from flcore.clients.clientmca import clientMCA
from flcore.servers.serverbase import Server
from collections import defaultdict
# import matplotlib.pyplot as plt
# import seaborn as sns
import torch

class FedMCA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientMCA)

        self.global_heads = []
        self.uploaded_head_models = []
        self.K = args.K

        self.global_model = copy.deepcopy(args.model.base)
        copy_m = copy.deepcopy(args.model.head)
        for _ in range(self.num_clients):
            self.uploaded_head_models.append(copy_m)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.client_heads = [copy.deepcopy(c.cmodel) for c in self.clients]
        self.classes_indexs = [copy.deepcopy(c.classes_index) for c in self.clients]
        self.client_inlude_cla = defaultdict(list)
        for cid, (head, classes_index) in enumerate(zip(self.client_heads, self.classes_indexs)):
            for idx, h in enumerate(head.weight.data):
                cla = classes_index[idx].item()
                self.client_inlude_cla[cla].append(cid)

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):
            torch.manual_seed(0)
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models(i)

            if i % self.eval_gap == 0:
                print(f"\n-----------------Round number: {i}---------------------------")
                print("\nEvaluate personal model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()

            self.global_heads = []
            for cla in range(self.num_classes):
                self.global_heads.append(self.aggregate_pre_parameters(cla))
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()


    def send_models(self, R):
        assert (len(self.clients) > 0)

        r = R
        for client in self.clients:
            start_time = time.time()
            indices = []
            send_models = []
            other_models = []
            if len(self.global_heads) > 0:
                for idx in client.classes_index.cpu().numpy():
                    send_models.append(self.global_heads[idx])
                    indices.append(idx)

                remaining_ids = [id for id in range(self.num_classes) if id not in indices]

                for _ in range(min(len(remaining_ids), self.K)):
                    other_models.append(self.global_heads[remaining_ids[r % len(remaining_ids)]])
                    r = r + 1

            client.set_base(self.global_model)
            client.set_head(send_models, other_models)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        self.tot_per_class_samples = torch.zeros(self.num_classes)
        self.uploaded_per_class_samples = [torch.zeros(self.num_classes) for i in range(self.num_clients)]
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                #
                self.uploaded_per_class_samples[client.id] = client.sample_per_class
                self.tot_per_class_samples += client.sample_per_class

                self.uploaded_models.append(client.model.base)
                self.uploaded_head_models[client.id] = copy.deepcopy(client.model.head)

        for i, w in enumerate(self.uploaded_per_class_samples):
            self.uploaded_per_class_samples[i] = torch.div(w, self.tot_per_class_samples)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


    def aggregate_pre_parameters(self, cla):
        assert (len(self.uploaded_head_models) > 0)

        model = copy.deepcopy(self.uploaded_head_models[0])
        for param in model.parameters():
            param.data.zero_()

        for cid in self.client_inlude_cla[cla]:
            weight = self.uploaded_per_class_samples[cid].tolist()
            self.add_pre_parameters(model, weight[cla], self.uploaded_head_models[cid])
        return model

    def add_pre_parameters(self, model, w, client_model):
        for server_param, client_param in zip(model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w
