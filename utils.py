import pickle
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy
import dgl
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
"""
Utility functions to read data, handle data and early stop training model.
"""


def load_data(dataset=None,test_size=0.6):
    """
    Load graph, feature, and label given dataset name
    :param dataset: the dataset name
    :param test_size: the size of test set
    :returns: feature, label, graph, category features
    """

    prefix = 'data/'
    if dataset=="FFSD":
        df=pd.read_csv(prefix + "STRADneofull.csv")
        cat_features = ["Target","Location","Type"]
        data=df[df["Labels"]<=2]
        data=data.reset_index(drop=True)
        out=[]
        alls=[]
        allt=[]
        pair=["Source","Target","Location","Type"]
        for column in pair:
            src,tgt=[],[]
            edge_per_trans = 3
            for c_id, c_df in data.groupby(column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len) for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len) for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))
        graph_path = prefix+"graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        cal_list = ["Source","Target","Location","Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        feat_data=data.drop("Labels",axis=1)
        labels=data["Labels"]
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size/2,
                                                            random_state=2, shuffle=True)

    elif dataset=="yelp":
        cat_features = []
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                            random_state=2, shuffle=True)
        src=[]
        tgt=[]
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        graph_path = prefix+ "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
    elif dataset=="amazon":
        cat_features = []
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(3305, len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                            test_size=test_size, random_state=2, shuffle=True)
        src=[]
        tgt=[]
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        graph_path = prefix+ "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

    return feat_data, labels, train_idx, test_idx, g, cat_features


def featmap_gen(tmp_df=None):
    """
    Handle FFSD dataset and do some feature engineering
    :param tmp_df: the feature of input dataset
    """
    time_span = [2,5,12,20,60,120,300, 600, 1500, 3600, 10800, 32400, 64800, 129600, 259200] # Increase in the number of time windows to increase the characteristics.
    time_name = [str(i) for i in time_span]
    time_list = tmp_df['Time']
    post_fe = []
    for trans_idx, trans_feat in tmp_df.iterrows():
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.Amount
        for length, tname in zip(time_span, time_name):
            lowbound = (time_list >= temp_time - length)
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]
            new_df['trans_at_avg_{}'.format(tname)] = correct_data['Amount'].mean()
            new_df['trans_at_totl_{}'.format(tname)] = correct_data['Amount'].sum()
            new_df['trans_at_std_{}'.format(tname)] = correct_data['Amount'].std()
            new_df['trans_at_bias_{}'.format(tname)] = temp_amt - correct_data['Amount'].mean()
            new_df['trans_at_num_{}'.format(tname)] = len(correct_data)
            new_df['trans_target_num_{}'.format(tname)] = len(correct_data.Target.unique())
            new_df['trans_location_num_{}'.format(tname)] = len(correct_data.Location.unique())
            new_df['trans_type_num_{}'.format(tname)] = len(correct_data.Type.unique())
        post_fe.append(new_df)
    return pd.DataFrame(post_fe)


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


class early_stopper(object):
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initialize the early stopper
        :param patience: the maximum number of rounds tolerated
        :param verbose: whether to stop early
        :param delta: the regularization factor
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_value = None
        self.best_cv = None
        self.is_earlystop = False
        self.count = 0
        self.best_model = None
        #self.val_preds = []
        #self.val_logits = []

    def earlystop(self, loss, model=None):#, preds, logits):
        """
        :param loss: the loss score on validation set
        :param model: the model
        """
        value = -loss
        cv = loss
        # value = ap

        if self.best_value is None:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
            #self.val_preds = preds
            #self.val_logits = logits
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print('EarlyStoper count: {:02d}'.format(self.count))
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
            #self.val_preds = preds
            #self.val_logits = logits
            self.count = 0


def load_lpa_subtensor(node_feat, work_node_feat, labels, seeds, input_nodes, device):
    """
    Put the input data into the device
    :param node_feat: the feature of input nodes
    :param work_node_feat: the feature of work nodes
    :param labels: the labels of nodes
    :param seeds: the index of one batch data
    :param input_nodes: the index of batch input nodes
    :param device: where to train model
    """
    batch_inputs = node_feat[input_nodes].to(device)
    batch_work_inputs = {i: work_node_feat[i][input_nodes].to(device) for i in work_node_feat if i not in {"labels"}}
    batch_labels = labels[seeds].to(device)
    train_labels = copy.deepcopy(labels)
    propagate_labels = train_labels[input_nodes]
    propagate_labels[:seeds.shape[0]] = 2
    return batch_inputs, batch_work_inputs, batch_labels, propagate_labels.to(device)

