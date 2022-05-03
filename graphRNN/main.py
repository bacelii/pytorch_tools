import sys
import os
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import networkx as nx
import argparse
from tqdm import tqdm
from mlp import MLPClassifier
import time



cmd_args = None

class Graph(object):
    def __init__(self, g, node_tags, label):
        
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        
        #label of the graph
        self.label = label

        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        
        #flattened edge_index
        self.edge_pairs = self.edge_pairs.flatten()

        ## Add neighbor info
        self.neighbor1 = []
        self.neighbor1_tag = []
        self.neighbor2 = []
        self.neighbor2_tag = []

        for i in range(self.num_nodes):
            self.neighbor1.append(g.neighbors(i))
            self.neighbor1_tag.append([node_tags[w] for w in g.neighbors(i)])
        for i in range(self.num_nodes):
            tmp = []
            for j in self.neighbor1[i]:
                for k in g.neighbors(j):
                    if k != i:
                        tmp.append(k)
            self.neighbor2.append(tmp)
            self.neighbor2_tag.append([node_tags[w] for w in tmp])



def load_data(
    data_name = None,
    fold=None,
    data_dir = "data/",
    verbose = False,
    
    ):
    
    if verbose:
        print('loading data')


    g_list = []
    g_neighbor_list = []
    label_dict = {} #will store the classes
    feat_dict = {}
    
    if data_name is None:
        data_name = f"{cmd_args.data}"
    if fold is None:
        fold = cmd_args.fold
    filename = (f'{data_dir}%s/%s.txt' % (data_name,data_name))
    if verbose:
        print(f"filename = {filename}")
    with open(filename, 'r') as f:
        curr_data = (f.readline().strip())
        n_g = int(curr_data)
        for i in range(n_g):
            curr_data = (f.readline().strip())
            row = curr_data.split()
            
            #gets the number of nodes and the label
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            n_edges = 0
            
            #iterates through the number of nodes
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                row = [int(w) for w in row]
                if not row[0] in feat_dict: #gives a class to the type of node 
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])
            assert len(g.edges()) * 2 == n_edges
            assert len(g) == n
            g_list.append(Graph(g, node_tags, l))
    for g in g_list:
        g.label = label_dict[g.label]
        
    num_class = len(label_dict)
    feat_dim = len(feat_dict)
    
    try:
        cmd_args.num_class =  num_class
        cmd_args.feat_dim =  feat_dim
    except:
        pass
    
    print('# classes: %d' % num_class)
    print('# node features: %d' % feat_dim)

    train_idxes = np.loadtxt(f'{data_dir}/%s/10fold_idx/train_idx-%d.txt' % (data_name, fold), dtype=np.int32).tolist()
    test_idxes = np.loadtxt(f'{data_dir}/%s/10fold_idx/test_idx-%d.txt' % (data_name, fold), dtype=np.int32).tolist()

    return [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes], g_list

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(LSTMClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(cmd_args.feat_dim, embedding_size)
        
        #the LSTM model (input size,hidden size)
        self.model = nn.LSTM(embedding_size*2, hidden_size)

        # 2 layer MLP with 1 activation layer(learns to predict the classes)
        self.mlp = MLPClassifier(input_size=hidden_size, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class)

    def forward(self, batch_graph):
        
        #gets the node features (AKA just the labels of the nodes)
        node_tags = batch_graph[0].node_tags
        node_tags = torch.LongTensor(node_tags).view(-1, 1)
        node_tags = node_tags#.cuda()
        
        # gets the label of the entire graph
        label = [batch_graph[0].label]
        label =  torch.LongTensor(label)
        label = label#.cuda()
        
        
        num_nodes = batch_graph[0].num_nodes
        
        #one hot encoding the node features (so could have other way of extracting)
        node_feat = torch.zeros(num_nodes, cmd_args.feat_dim)
        node_feat = node_feat#.cuda()
        
        #one hot enocdes the features
        node_feat.scatter_(1, node_tags, 1)
        node_feat = Variable(node_feat)
        
        #embeds the features with a linear layer
        node_feat = self.embedding(node_feat)
        # Prepare neighbor features
        neighbor1_tags = batch_graph[0].neighbor1_tag
        neighbor1_feat = Variable(torch.zeros(num_nodes, cmd_args.feat_dim))
        for i in range(num_nodes):
            for j in neighbor1_tags[i]:
                neighbor1_feat[i, j] = neighbor1_feat[i, j] + 1.
        neighbor1_feat = neighbor1_feat#.cuda()
        
        #embeds the neighborhood featrues
        neighbor1_feat = self.embedding(neighbor1_feat)
        
        #concatenates the node and neighborhood features
        input_feat = torch.cat((node_feat, neighbor1_feat), 1)
        input_feat = input_feat.view(num_nodes, 1, -1)
        
        #feeds ino the lstm
        out, hidden = self.model(input_feat)
        
        #sums the output to combine the sequence
        embed = torch.sum(out, dim = 0)
        embed = embed.view(1, -1)
        
        #will return the loss along with the logis because given the label
        #convert to the appropriate size
        class_output = self.mlp(embed, label)
        
        return class_output

    def init_hidden(self):
        h_t = Variable(torch.zeros(1, self.hidden_size))#.cuda()
        c_t = Variable(torch.zeros(1, self.hidden_size))#.cuda()
        return h_t, c_t

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

def loop_dataset(g_list, classifier, sample_idxes, optimizer=None):
    bsize = 1
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        _, loss, acc = classifier(batch_graph)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )

        total_loss.append( np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    return avg_loss


if __name__ == '__main__':
    cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
    cmd_opt.add_argument('-data', default='MUTAG', help='data folder name')
    cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
    cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of node feature')
    cmd_opt.add_argument('-embedding_dim', type=int, default=64, help='dimension of node embedding')
    cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
    cmd_opt.add_argument('-num_epochs', type=int, default=500, help='number of epochs')
    cmd_opt.add_argument('-rnn_hidden_dim', type=int, default=64, help='dimension of rnn hidden dimension')
    cmd_opt.add_argument('-hidden', type=int, default=64, help='dimension of classification')
    cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')

    cmd_args_curr, _ = cmd_opt.parse_known_args()
    cmd_args = cmd_args_curr
    
    
    train_graphs, test_graphs, all_graphs = load_data()
    print('# train: %d, # test: %d , #total: %d' % (len(train_graphs), len(test_graphs), len(all_graphs)))
    loss_function = nn.NLLLoss()
    input_size = cmd_args.feat_dim
    hidden_size = cmd_args.rnn_hidden_dim
    embedding_size = cmd_args.embedding_dim
    loss_function = nn.NLLLoss()
    classifier = LSTMClassifier(input_size, hidden_size, embedding_size)
    classifier = classifier#.cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)
    train_idxes = list(range(len(train_graphs)))
    start = time.time()
    best_loss = None
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
        print('Training of epoch %d: average loss %.5f acc %.5f' % (epoch, avg_loss[0], avg_loss[1]))

        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        print('Test of epoch %d: loss %.5f acc %.5f' % (epoch, test_loss[0], test_loss[1]))
    end = time.time()
    print('Time for %d epochs is %.f' %(cmd_args.num_epochs, end - start))
    
    
    

"""
python3 main.py \
        -seed 1 \
        -data ENZYMES \
        -learning_rate 0.001 \
        -num_epochs 200 \
        -hidden 64 \
        -fold 10 \
        -embedding_dim 64 \
        -rnn_hidden_dim 10

data: MUTAG, NCI1, NCI109, DD, ENZYMES
feat_dim: Number of node labels
embedding_dim: Dimension of node embedding
num_class: Number of graph classes
rnn_hidden_dim: Hidden unit size of RNN
learning_rate: initial learning_rate

"""