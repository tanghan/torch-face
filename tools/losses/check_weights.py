import torch
import torch.nn.functional as F
import numpy as np

local_arc_w_path = "local_arc_init_w.pt"
dist_softmax_w_path = "dist_softmax_init_"

local_fea_path = "p0_fea_"

def compare_init_fc_weights(rank=4):
    local_fc_path = "local_arc_init_w.pt"
    local_w = torch.load(local_fc_path)

    dist_w_list = []
    for rank_i in range(rank):
        fw_w_path = "dist_softmax_init_{}_w.pt".format(rank_i)
        dist_w_list.append(torch.load(fw_w_path))
    dist_w = torch.cat(dist_w_list, 0)

    diff = torch.sum(torch.abs(local_w - dist_w))
    print("init fc compare: {}".format(diff))
    return local_w

def compare_fea(rank=-1, data_idx=0):
    fea_path = "p0_fea_{}.pt".format(data_idx)
    features = torch.load(fea_path) 
    num_samples = features.size()[0]
    emb_size = features.size()[1]

    dist_fea_list = []
    reorder_list = []
    for i in range(rank):
        reorder = np.arange(i, num_samples, rank)
        reorder_list.extend(reorder)
    reorder_list = np.array(reorder_list)

    for rank_i in range(rank):
        fea_path = "p1_fea_{}_{}.pt".format(data_idx, rank_i)
        dist_fea_list.append(torch.load(fea_path))
    dist_features = torch.cat(dist_fea_list, 1)
    dist_features = dist_features.reshape((num_samples, emb_size))
    diff = torch.sum(torch.abs(features - dist_features[reorder_list]))
    print(features[:5, :5])
    print(dist_features[reorder_list][:5, :5])
    #diff = torch.sum(torch.abs(features - dist_features))

    print("compare fea: --- \ndata idx: {} diff: {}\n------".format(data_idx, diff))
    return features

def compare_fea_grad(rank=-1, data_idx=0):
    fea_path = "p0_fea_grad_{}.pt".format(data_idx)
    features = torch.load(fea_path) 
    num_samples = features.size()[0]
    emb_size = features.size()[1]

    dist_fea_list = []
    reorder_list = []
    for i in range(rank):
        reorder = np.arange(i, num_samples, rank)
        reorder_list.extend(reorder)
    reorder_list = np.array(reorder_list)

    for rank_i in range(rank):
        fea_path = "p1_fea_grad_{}_{}.pt".format(data_idx, rank_i)
        dist_fea_list.append(torch.load(fea_path))
    dist_features = torch.cat(dist_fea_list, 1)
    dist_features = dist_features.reshape((num_samples, emb_size))
    diff = torch.sum(torch.abs(features - dist_features[reorder_list]))
    #diff = torch.sum(torch.abs(features - dist_features))

    print("compare fea grad: --- \ndata idx: {} diff: {}\n------".format(data_idx, diff))
    return features


def compare_theta(rank=-1, data_idx=0):
    
    fea_path = "circle_theta_{}.pt".format(data_idx)
    features = torch.load(fea_path) 
    num_samples, emb_size = features.size()
    print(features.size())
    #print(features)
    dist_list = []
    reorder_list = []
    for i in range(rank):
        reorder = np.arange(i, num_samples, rank)
        reorder_list.extend(reorder)
    reorder_list = np.array(reorder_list)

    for rank_i in range(rank):
        fea_path = "dist_circle_theta_{}_{}.pt".format(data_idx, rank_i)
        fea = torch.load(fea_path)
        #print(fea)
        #print(fea.shape)
        dist_list.append(fea)
    dist_features = torch.cat(dist_list, 1)
    dist_features = dist_features.reshape((num_samples, emb_size))
    diff = torch.sum(torch.abs(features - dist_features[reorder_list]))

    print("compare theta: --- \ndata idx: {} diff: {}".format(data_idx, diff))
    return dist_features

def compare_logits(rank=-1, data_idx=0):
    
    fea_path = "p0_logits_{}.pt".format(data_idx)
    features = torch.load(fea_path) 
    num_samples, emb_size = features.size()

    dist_list = []
    reorder_list = []
    for i in range(rank):
        reorder = np.arange(i, num_samples, rank)
        reorder_list.extend(reorder)
    reorder_list = np.array(reorder_list)

    for rank_i in range(rank):
        fea_path = "dist_logits_{}_{}.pt".format(data_idx, rank_i)
        dist_list.append(torch.load(fea_path))
    dist_features = torch.cat(dist_list, 1)
    print(features[5:10, :10])
    print(dist_features[reorder_list][5:10, :10] / 2)
    #dist_features = dist_features.reshape((num_samples, emb_size))
    diff = torch.sum(torch.abs(features - (dist_features[reorder_list])))

    print("compare logits ----- \ndata idx: {} diff: {}".format(data_idx, diff))
    return features

def compare_opt_weights(rank=-1, data_idx=0):
    path = "local_fc_{}.pt".format(data_idx)
    local_w = torch.load(path)
    dist_list = []
    for rank_i in range(rank):
        fw_w_path = "softmax_fc_{}_{}.pt".format(data_idx, rank_i)
        dist_list.append(torch.load(fw_w_path))
    dist_w = torch.cat(dist_list, 0)

    diff = torch.sum(torch.abs(dist_w - local_w))

    print("compare opt weights------ \n data idx: {} diff: {}".format(data_idx, diff))

            
def compare_opt_backbone(rank=-1, data_idx=0):
    path = "local_backbone_{}.pt".format(data_idx)
    local_backbone_w = torch.load(path)
    path = "softmax_backbone_{}.pt".format(data_idx)
    dist_backbone_w = torch.load(path)
    diff = torch.sum(torch.abs(local_backbone_w - dist_backbone_w))

    print("compare opt weights------ \n data idx: {} diff: {}".format(data_idx, diff))
    print(local_backbone_w[0, 0, :, :])
    print(dist_backbone_w[0, 0, :, :])
            
    return local_backbone_w



def main():
    #fc_w = compare_init_fc_weights(rank=4)
    fea = compare_fea(rank=4, data_idx=0)
    compare_fea_grad(rank=4, data_idx=0)
    #logits = compare_logits(rank=4, data_idx=0)
    compare_opt_weights(rank=4, data_idx=0)
    compare_opt_backbone(rank=4, data_idx=0)
    #theta = compare_theta(rank=4, data_idx=0)
    #norm_w = F.normalize(fc_w)
    #out = F.linear(F.normalize(fea), norm_w)
    #print(out - theta)
    '''

    local_fea0 = load_fea(-1, 0)
    dist_softmax_fea0 = load_fea(4, 0)

    fea_diff0 = torch.sum(torch.abs(local_fea0 - dist_softmax_fea0))
    print("fea_diff0: ", fea_diff0)

    local_theta = load_theta(-1, 0)
    dist_softmax_theta = load_theta(4, 0)

    theta_diff0  = torch.sum(torch.abs(local_theta - dist_softmax_theta))
    print("theta_diff0: ", theta_diff0)
    print("local logits: ", local_theta[:5, :10])
    print("dist_softmax logits: ", dist_softmax_theta[:5, :10])

    local_logits0 = load_logits(-1, 0)
    dist_softmax_logits0 = load_logits(4, 0)

    logits_diff0  = torch.sum(torch.abs(local_logits0 - dist_softmax_logits0))
    print("logits_diff0: ", logits_diff0)
    print("local logits: ", local_logits0[:5, :10])
    print("dist_softmax logits: ", dist_softmax_logits0[:5, :10])


    local_opt_w = load_opt_weights(-1, 0)
    softmax_opt_w = load_opt_weights(4, 0)
    opt_fc_diff0  = torch.sum(torch.abs(softmax_opt_w - local_opt_w))
    print("opt fc diff0: ", opt_fc_diff0)
    '''

    '''
    local_opt_w = load_opt_weights(-1, 1)
    softmax_opt_w = load_opt_weights(4, 1)
    opt_fc_diff1  = torch.sum(torch.abs(softmax_opt_w - local_opt_w))
    print("opt fc diff1: ", opt_fc_diff1)

    local_opt_backbone = load_opt_backbone(-1, 0)
    softmax_opt_backbone = load_opt_backbone(4, 0)
    opt_backbone_diff0  = torch.sum(torch.abs(softmax_opt_backbone - local_opt_backbone))
    print("opt backbone diff0: ", opt_backbone_diff0)

    local_opt_backbone = load_opt_backbone(-1, 1)
    softmax_opt_backbone = load_opt_backbone(4, 1)
    opt_backbone_diff1  = torch.sum(torch.abs(softmax_opt_backbone - local_opt_backbone))
    print("opt backbone diff1: ", opt_backbone_diff1)
    '''



if __name__ == "__main__":
    main()
    
