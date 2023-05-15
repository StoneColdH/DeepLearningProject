
# %%
import os
from dgl.dataloading import  MultiLayerFullNeighborSampler
from dgl.dataloading import NodeDataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils import *
from model import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score,f1_score
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
	Training TGTN-GNN 
    
"""

def train_gnn(feat_df, graph, train_idx, test_idx, labels, params, cat_features):
    device = params['device']
    graph=graph.to(device)
    oof_predictions = torch.from_numpy(np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(np.zeros([len(feat_df), 2])).float().to(device)
    kfold = StratifiedKFold(n_splits=params['n_fold'], shuffle=True, random_state=params['seed'])
    
    y_target = labels.iloc[train_idx].values
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(device) for col in cat_features}
    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
        print(f'Training fold {fold + 1}')
        trn_ind, val_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(device) , torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device) 

        train_sampler = MultiLayerFullNeighborSampler(params['n_layers'])
        train_dataloader = NodeDataLoader(graph,
                                          trn_ind,
                                          train_sampler,
                                          device=device,
                                          use_ddp=False,
                                          batch_size=params['batch_size'],
                                          shuffle=True,
                                          drop_last=False,
                                          num_workers=0
                                          )
        val_sampler = MultiLayerFullNeighborSampler(params['n_layers'])
        val_dataloader = NodeDataLoader(graph,
                                        val_ind,
                                        val_sampler,
                                        use_ddp=False,
                                        device=device,
                                        batch_size=params['batch_size'],
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=0,
                                        )
        model = eval(params['model'])(in_feats=feat_df.shape[1], 
                                      hidden_dim=params['hid_dim']//4,
                                      n_classes=2, 
                                      heads=[4]*params['n_layers'],
                                      activation=nn.PReLU(), 
                                      n_layers=params['n_layers'], 
                                      drop=params['dropout'],
                                      device=device,
                                      gated=params['gated'],
                                      ref_df=feat_df.iloc[train_idx],
                                      cat_features=cat_feat).to(device)
        lr = params['lr'] * np.sqrt(params['batch_size']/1024)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=params['wd'])
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[4000, 12000], gamma=0.3)

        earlystoper = early_stopper(patience=params['early_stopping'], verbose=True)
        start_epoch, max_epochs = 0, 2000
        for epoch in range(start_epoch, params['max_epochs']):
            train_loss_list = []
            # train_acc_list = []
            model.train()
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels, 
                                                                                               seeds, input_nodes, device)

                blocks = [block.to(device) for block in blocks]
                train_batch_logits = model(blocks, batch_inputs, lpa_labels, batch_work_inputs)
                mask = batch_labels == 2
                train_batch_logits = train_batch_logits[~mask]
                batch_labels = batch_labels[~mask]
                # batch_labels[mask] = 0

                train_loss = loss_fn(train_batch_logits, batch_labels)
                # backward
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss_list.append(train_loss.cpu().detach().numpy())

                if step % 10 == 0:
                    tr_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone().detach(), dim=1) == batch_labels) / batch_labels.shape[0]
                    score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[:, 1].cpu().numpy()
                    try:
                        print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                          'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(epoch,step,
                                                                                       np.mean(train_loss_list),
                                                                                       average_precision_score(batch_labels.cpu().numpy(), score), 
                                                                                       tr_batch_pred.detach(),
                                                                                       roc_auc_score(batch_labels.cpu().numpy(), score)))
                    except:
                        pass
        
            # mini-batch for validation
            val_loss_list = 0
            val_acc_list = 0
            val_all_list = 0
            model.eval()
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels, 
                                                                                                   seeds, input_nodes, device)
    
                    blocks = [block.to(device) for block in blocks]
                    val_batch_logits = model(blocks, batch_inputs, lpa_labels, batch_work_inputs)
                    oof_predictions[seeds] = val_batch_logits
                    mask = batch_labels == 2
                    val_batch_logits = val_batch_logits[~mask]
                    batch_labels = batch_labels[~mask]
                    # batch_labels[mask] = 0
                    val_loss_list = val_loss_list + loss_fn(val_batch_logits, batch_labels)
                    # val_all_list += 1
                    val_batch_pred = torch.sum(torch.argmax(val_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                    val_acc_list = val_acc_list + val_batch_pred * torch.tensor(batch_labels.shape[0])
                    val_all_list = val_all_list + batch_labels.shape[0]
                    if step % 10 == 0:
                        score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[:, 1].cpu().numpy()
                        try:
                            print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                              'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch,
                                                                      step,
                                                                      val_loss_list/val_all_list,
                                                                      average_precision_score(batch_labels.cpu().numpy(), score), 
                                                                      val_batch_pred.detach(),
                                                                      roc_auc_score(batch_labels.cpu().numpy(), score)))
                        except:
                            pass

            earlystoper.earlystop(val_loss_list/val_all_list, model)#val_acc_list/val_all_list, model)
            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break
        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))
        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(params['n_layers'])
        test_dataloader = NodeDataLoader(graph,
                                         test_ind,
                                         test_sampler,
                                         use_ddp=False,
                                         device=device,
                                         batch_size=params['batch_size'],
                                         shuffle=True,
                                         drop_last=False,
                                         num_workers=0,
                                         )
        b_model = earlystoper.best_model.to(device)
        b_model.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                # print(input_nodes)
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels, 
                                                                                               seeds, input_nodes, device)

                blocks = [block.to(device) for block in blocks]
                test_batch_logits = b_model(blocks, batch_inputs, lpa_labels, batch_work_inputs)
                test_predictions[seeds] = test_batch_logits
                test_batch_pred = torch.sum(torch.argmax(test_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                if step % 10 == 0:
                    print('In test batch:{:04d}'.format(step))
    mask = y_target == 2
    y_target[mask] = 0
    my_ap = average_precision_score(y_target, torch.softmax(oof_predictions, dim=1).cpu()[train_idx, 1])
    print("NN out of fold AP is:", my_ap)
    return earlystoper.best_model.to('cpu'), oof_predictions, test_predictions

# %%
# params = {
#     'model': 'GraphAttnModel',
#     'batch_size': 64,
#     'n_layers': 3,
#     'hid_dim': 256,
#     'lr': 0.003,
#     'wd': 1e-4,
#     'dropout': [0.2, 0.1],
#     'device': 'cuda:0',
#     'early_stopping': 10,
#     'n_fold': 5,
#     'seed': 2021,
#     'max_epochs': 15,
#     'gated': True,
#     'dataset':"amazon",
#     'test_size':0.6
# }
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GraphAttnModel', help='Use Attention Model')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--hid_dim', type=int, default=256,
                        help='number of hidden units')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight dacay')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='number of GNN layers')
    parser.add_argument('--dropout', type=list, default=[0.2,0.1],
                        help='dropout rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='model train device')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='early stopping number')
    parser.add_argument('--n_fold', type=int, default=5,
                        help='fold number')
    parser.add_argument('--seed', type=int, default=2021,
                        help='random seed')  
    parser.add_argument('--max_epochs', type=int, default=15,
                        help='number of max epochs to train')    
    parser.add_argument('--gated', type=bool, default=True,
                        help='whether to use gate')
    parser.add_argument('--dataset', type=str, default='yelp', help='dataset name')
    parser.add_argument('--test_size', type=float, default=0.6,
                        help='testing set percentage')                                                
    params = vars(parser.parse_args())
    
    feat_data, labels, train_idx, test_idx, g, cat_features=load_data(params['dataset'],params['test_size'])
    
    b_models, val_gnn_0, test_gnn_0 = train_gnn(feat_data, g, train_idx, test_idx,labels, params=params,cat_features=cat_features)
    
    test_score = torch.softmax(test_gnn_0, dim=1)[test_idx, 1].cpu().numpy()
    y_target = labels.iloc[test_idx].values
    test_score1 = torch.argmax(test_gnn_0, dim=1)[test_idx].cpu().numpy()
    
    print("test AUC:", roc_auc_score(y_target, test_score))
    print("test f1:", f1_score(y_target, test_score1,average="macro"))
    print("test AP:", average_precision_score(y_target, test_score))


# %%
