from .abstract import AbstractDataset, AbstractPLDataModule
import numpy as np
from torch.utils.data import random_split, Dataset, Subset
import torch
import random
import networkx as nx
from pynini import Fst, Arc
from tqdm import tqdm
from collections import Counter
import torch
import hydra
from datasets import Dataset as HFDataset

class SFSTDatamodule(AbstractPLDataModule):

    def __init__(self, dataset_parameters, **kwargs):
        super().__init__(**kwargs)
        self.dataset_parameters = dataset_parameters
        

        self.data_train = hydra.utils.instantiate(self.params['datasets']['train'], self.dataset_parameters)
        self.data_val = hydra.utils.instantiate(self.params['datasets']['val'], self.dataset_parameters)
        self.data_test = hydra.utils.instantiate(self.params['datasets']['test'], self.dataset_parameters)


class SFSTDataset(AbstractDataset):
    has_split = False # class variable to track whether train-val split has been performed

    def __init__(self, dataset_parameters, **kwargs):
       
        super().__init__(dataset_parameters, **kwargs)
        self.seed = dataset_parameters['seed']
        self.dataset_parameters = dataset_parameters
        self.split = self.params['split']
        assert self.split in {"train", "val", "test"}, "Unexpected split reference"
        

        if not SFSTDataset.has_split:  # perform split if it has not been done yet
            SFSTDataset.overfit_batch = dataset_parameters['overfit_batch']
            self.loaded_dataset = self._load_data(self.dataset_parameters)
            self.train_ratio = dataset_parameters['train_ratio']
            
            # overfit batch, using small batch of same data for training and validation
            if SFSTDataset.overfit_batch:
                self.train_dataset, self.val_dataset = self.set_same_batch(self.loaded_dataset, SFSTDataset.overfit_batch)
                self.test_dataset = self.val_dataset
            else:
                self.train_dataset, self.val_dataset = self.split_train_val(self.loaded_dataset)
                self.test_dataset = self.loaded_dataset['test']

            SFSTDataset.datum = {}
            SFSTDataset.datum['train'] = self.train_dataset
            SFSTDataset.train_len = len(self.train_dataset)
            SFSTDataset.datum['val'] = self.val_dataset
            SFSTDataset.datum['test'] = self.test_dataset

            SFSTDataset.has_split = True  # mark that the split has been performed

            self.assign_data_type(SFSTDataset.train_len)

        self.data = SFSTDataset.datum[self.split]
        self.data_type = SFSTDataset.data_type[self.split]


    def split_train_val(self, loaded_dataset):
        assert (self.train_ratio > 0 and self.train_ratio < 1), "Unexpected train_test ratio" 
        train_size = int(self.train_ratio * len(loaded_dataset['train']))
        lengths = [train_size, len(loaded_dataset['train']) - train_size]
        train_dataset, val_dataset = random_split(loaded_dataset['train'], lengths, 
                                                  generator=torch.Generator().manual_seed(self.seed))
        return train_dataset, val_dataset 

    def set_same_batch(self, dataset, batchsize):
        return Subset(dataset['train'], range(batchsize)), Subset(dataset['train'], range(batchsize))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"id": idx, "X": self.data[idx]['X'], 
                "Z": self.data[idx]['Z'], 'data_type': self.data_type[idx]}

    def _load_data(self, load_dataset_params):
        dataset_size = load_dataset_params['dataset_size']
        node_size = load_dataset_params['node_size']
        max_length = load_dataset_params['max_length']
        output_alphabet_size = load_dataset_params['output_alphabet_size']
        maximum_input_alphabet_delta = load_dataset_params['maximum_input_alphabet_delta']
        p_empty_emission = load_dataset_params['p_empty_emission']
        fst, FST_dic, input_alphabet, output_alphabet, nodes, edges, adjacency = self.generate_FST(node_size, output_alphabet_size=output_alphabet_size, 
                                                                                                   maximum_input_alphabet_delta=maximum_input_alphabet_delta, p_empty_emission=p_empty_emission)

        in_al = []
        out_al = []
        for v in FST_dic.values():
            for i in v.values():
                in_al.append(i[0])
                out_al.append(i[1])

        inputs, outputs = self.generate_dataset(FST_dic, max_length)
        assert len(inputs) == len(outputs)

        # Check if the lenght of dataset matches SCAN
        if len(inputs) >= dataset_size:
            print("Dataset")
            inputs = inputs[:dataset_size]
            outputs = outputs[:dataset_size]

            train_size = int(len(inputs) / 100 * 80)
            train_inputs = inputs[:train_size]
            test_inputs = inputs[train_size:]
            train_outputs = outputs[:train_size]
            test_outputs = outputs[train_size:]
            return {'train': SFSTDataset._create_dataset(train_inputs, train_outputs), 'test': SFSTDataset._create_dataset(test_inputs, test_outputs)}

    def _create_dataset(inputs, outputs):
        return HFDataset.from_dict({'X': inputs, 'Z': outputs})
    
    def generate_FST(self, node_size, output_alphabet_size=8, maximum_input_alphabet_delta=-1, p_empty_emission=0.5):
        while True:
            n = node_size
            adjacency = np.random.randint(0, 2, (n, n))
            G = nx.DiGraph(adjacency)

            labels = dict()
            for i in range(len(adjacency)):
                labels[i] = str(i)

            strong_index = []
            for i in nx.strongly_connected_components(G):
                #     print(i)
                if list(i)[-1] != 0:
                    gen = nx.all_simple_paths(G, source=0, target=list(i)[-1])
                    # if sum(1 for _ in gen) > 0:
                    try:
                        next(gen)
                        strong_index.append(1)
                    except:
                        strong_index.append(0)

            fst = True
            if 0 in strong_index:
                fst = False

            if fst:
                generate = False
                print(adjacency)
                #  nx.draw(G, labels=labels)

                branches = [i[0] for i in G.edges]
                minimum_input_alphabet = max(list(Counter(branches).values()))
                # if minimum_input_alphabet < 10:
                #     minimum_input_alphabet = 10

                if maximum_input_alphabet_delta != -1:
                    maximum_input_alphabet = minimum_input_alphabet + maximum_input_alphabet_delta
                    maximum_input_alphabet = min(maximum_input_alphabet, len(G.edges))
                    input_alphabet_size = np.random.randint(minimum_input_alphabet, maximum_input_alphabet)
                else:
                    input_alphabet_size = minimum_input_alphabet
                # output_alphabet_size = np.random.randint(1, len(G.edges))
                # input_alphabet_size = 30

                output_alphabet = random.sample(range(output_alphabet_size, output_alphabet_size*2), output_alphabet_size)

                input_alphabet = random.sample(range(0, input_alphabet_size), input_alphabet_size)
                input_alphabet_cpy = input_alphabet.copy()

                # empty_emmission = random.sample([0,1], 1)
                # empty_emmission = np.random.randint(2, size=1)

                print("Graph_nodes:", len(adjacency), "edges:", len(G.edges))
                print("input_alpha", input_alphabet, len(input_alphabet))
                print("output_alpha", output_alphabet, len(output_alphabet))

                fst = Fst(arc_type='standard')
                fst.add_state()
                fst.set_start(0)
                cnt = 0

                FST_dic = dict()
                node_dic = dict()
                for i in G.edges:
                    FST_dic[cnt] = node_dic
                    while i[0] > cnt:
                        fst.add_state()
                        cnt += 1
                        input_alphabet_cpy = input_alphabet.copy()
                        node_dic = dict()

                    in_index = random.randint(0, len(input_alphabet_cpy) - 1)
                    in_label = input_alphabet_cpy.pop(in_index)

                    empty_out = np.random.choice([0, 1], size=(1,), p=[1 - p_empty_emission, p_empty_emission])[0]
                    if empty_out == 1:
                        out_label = -1
                    else:
                        out_index = random.randint(0, len(output_alphabet) - 1)
                        out_label = output_alphabet[out_index]

                    node_dic[i[1]] = (in_label, out_label)

                    fst.add_arc(i[0], Arc(in_label, out_label, 0, i[1]))

                fst.minimize(allow_nondet=False)

                return fst, FST_dic, input_alphabet, output_alphabet, node_size, len(G.edges), adjacency


    def generate_dataset(self, FST_dic, max_len):
        #  print(FST_dic)
        inputs = []
        outputs = []
        node_cvr = [0]
        for _ in tqdm(range(100000)):
            max_size = random.randint(1, max_len)
            input_path = []
            output_path = []
            run = True
            i = 0
            cnt = 0
            stopper = False
            while run and cnt < max_size and stopper == False:
                try:
                    index = random.randint(0, (len(FST_dic[i])) - 1)
                    path = list(FST_dic[i].keys())[index]
                    input_path.append(FST_dic[i][path][0])
                    if FST_dic[i][path][1] != -1:
                        output_path.append(FST_dic[i][path][1])
                    i = path
                    node_cvr.append(i)
                    cnt += 1
                    if random.randint(0, 10) == 1:
                        stopper = True
                except:
                    run = False
            inputs.append(("".join([str(i) + " " for i in input_path]))[:-1])
            outputs.append(("".join([str(i)+ " " for i in output_path]))[:-1])

        test = [str(i) for i in inputs]
        samples = list(set(test))
        print("size of exp:", len(samples))

        dic_flat = {}
        for i, j in zip(inputs, outputs):
            dic_flat[str(i)] = j

        # import ast
        # inputs = [ast.literal_eval(i) for i in list(dic_flat.keys())]
        # outputs = list(dic_flat.values())
        print("size of exp:", len(inputs))
        print("node_covered:", len(list(set(node_cvr))))

        for i, j in zip(inputs[:10], outputs[:10]):
            print(i, "transduces:", j)
        return inputs, outputs


