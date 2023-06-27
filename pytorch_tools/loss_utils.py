'''

Cross Entropy Loss (Log Likelihood):

Overall Idea: 
1) Given the estimated probabilities
2) Takes the log (so hihger probabilites get less of an absolute number)
    They are all negative numbers
3) Picks the probability of the right class
4) Flips the sign so becomes positive (so the less prediction probability the more loss)
5) Applies the weight to magnify the loss felt for the class it should have been

Effect of weight: When the clasifier is wrong on a highly weighted class
- IT REALLY POORS ON THE LOSS MAGNITUDE (unless perfect)


Makes everything a probability and then just sums up
the probabilities of the things that should be right 

What it does in regular terms: 
1) Has model compute the probabilities of each class
2) Takes a log of all the probabilities (log probabilies)
3) Taks the log probability of the class it should be
- so the higher the probability the lower the absolute value of the number
4) Adds up them all in batch
-- at this point the lower the absolute then the better performance
5) Takes the negative so the higher actual number the worse, and the
lower the actual number the better (so can just minimize)

NLLLoss: just sums the negative weighted number
- requires softmax and log to already be computed

CrossEntropy: 
1) applies softmax
2) applies log
3) multiples by weight and turns negative


All 3 of these models would produce the same output

--- Example 1: Outputs probabilities and then uss log with NLL Loss ----

    from torch.nn import Linear
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.nn import global_mean_pool

    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            torch.manual_seed(12345)
            self.conv1 = GCNConv(dataset.num_node_features, hidden_channels,bias=True)
            self.conv2 = GCNConv(hidden_channels,  hidden_channels,bias=True)
            self.conv3 = GCNConv(hidden_channels,  dataset.num_classes,bias=True)

        def forward(self, data):
            # 1. Obtain node embeddings 
            x, edge_index = data.x, data.edge_index
            batch = data.batch
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv3(x, edge_index)
    #         x = self.conv3(x, edge_index)

            # 2. Readout layer
            #x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

            # 3. Apply a final classifier
            # x = F.dropout(x, p=0.5, training=self.training)
            #x = self.lin(x)
            return F.softmax(x, dim=1)

    model = GCN(hidden_channels=7).to(device)
    out = model(curr_data)
    print(out)
    # criterion = F.nll_loss(
    #     weight=class_weights,
    # )
    F.nll_loss(torch.log(out),curr_data.y)

------ Example 2: LogSoftmax Output and then usses NLL

    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            torch.manual_seed(12345)
            self.conv1 = GCNConv(dataset.num_node_features, hidden_channels,bias=True)
            self.conv2 = GCNConv(hidden_channels,  hidden_channels,bias=True)
            self.conv3 = GCNConv(hidden_channels,  dataset.num_classes,bias=True)

        def forward(self, data):
            # 1. Obtain node embeddings 
            x, edge_index = data.x, data.edge_index
            batch = data.batch
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv3(x, edge_index)
    #         x = self.conv3(x, edge_index)

            # 2. Readout layer
            #x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

            # 3. Apply a final classifier
            # x = F.dropout(x, p=0.5, training=self.training)
            #x = self.lin(x)
            return F.log_softmax(x, dim=1)

    model = GCN(hidden_channels=7).to(device)
    out = model(curr_data)
    print(out)
    # criterion = F.nll_loss(
    #     weight=class_weights,
    # )
    F.nll_loss(out,curr_data.y)
    
# ---- Example 3: Raw Output and then uses Cross Entropy----

    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            torch.manual_seed(12345)
            self.conv1 = GCNConv(dataset.num_node_features, hidden_channels,bias=True)
            self.conv2 = GCNConv(hidden_channels,  hidden_channels,bias=True)
            self.conv3 = GCNConv(hidden_channels,  dataset.num_classes,bias=True)

        def forward(self, data):
            # 1. Obtain node embeddings 
            x, edge_index = data.x, data.edge_index
            batch = data.batch
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv3(x, edge_index)
    #         x = self.conv3(x, edge_index)

            # 2. Readout layer
            #x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

            # 3. Apply a final classifier
            # x = F.dropout(x, p=0.5, training=self.training)
            #x = self.lin(x)
            return x

    model = GCN(hidden_channels=7).to(device)
    out = model(curr_data)
    print(out)
    # criterion = F.nll_loss(
    #     weight=class_weights,
    # )
    criterion = torch.nn.CrossEntropyLoss()
    criterion(out,curr_data.y)



'''
