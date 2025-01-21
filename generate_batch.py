import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchinfo import summary
import esm
import torch.nn.functional as F
import random
import tqdm
import pandas as pd
import gc
import argparse



batch_size = 26
pro_len = 2000  # Encoder max sequence length
n_fold = 5  # Number of folds for cross-validation


import pandas as pd


def load_if_add_dict(if_add_path):
    if_add_dict = {}
    if_add_path_list = os.listdir(if_add_path)
    for if_add_file in if_add_path_list:
        if_add_list=np.load(os.path.join(if_add_path, if_add_file), allow_pickle=True)
        for if_add_dict_single in if_add_list:
            if_add_dict.update(if_add_dict_single)
    return if_add_dict

def process_train_data(train_data, pro_len, batch_size=1000):
    protein_names = []
    seqs = []
    chain_id_res = []
    enc_tokens = []
    seq_features = []
    coor_features = []
    interface_atoms = []
    affinity = []
    interaction_type = []
    interaction_matrix = []
    res_mass_centor = []
    hetatm_features = []


    for idx, item in tqdm.tqdm(enumerate(train_data)):
        # pdbbind_name_list=np.load("./pdbbind_name_list.npy",allow_pickle=True)
        # if item["protein_name"].lower() not in pdbbind_name_list:
        #     continue
        protein_names.append(item["protein_name"])
        seq_temp=""
        for i in item["sequence"]:
            seq_temp+=i
        if len(seq_temp)>pro_len :
            continue
        # 对于pdbbind和skempi的不同处理
        seqs.append(seq_temp)
        chain_id_res.append(item["chain_id_res"])
        enc_tokens_temp=torch.cat(if_add_dict[item["protein_name"].replace(".pdb","")][2],dim=0).type(torch.int16)
        enc_tokens.append(F.pad(enc_tokens_temp,(0,pro_len-enc_tokens_temp.shape[0])))
        seq_feat_temp=torch.cat(if_add_dict[item["protein_name"].replace(".pdb","")][0],dim=1).squeeze()
        seq_features.append(F.pad(seq_feat_temp,(0,0,0,pro_len-seq_feat_temp.shape[0])))

        coor_feat_temp=torch.cat(if_add_dict[item["protein_name"].replace(".pdb","")][1],dim=0)
        coor_features.append(F.pad(coor_feat_temp,(0,0,0,pro_len-enc_tokens_temp.shape[0])))
        interface_res_matrix=torch.ones((pro_len,pro_len), dtype=torch.bool)
        # 将非界面氨基酸全部遮蔽
        for i in range(len(item["interface_res"])):
            for j in range(len(item["interface_res"][i])):
                if item["interface_res"][i][j]!=-1:
                    interface_res_matrix[i][item["interface_res"][i][j]]=False
        # 将自己同一条链的氨基酸全部遮蔽
        # chain_id_array = np.array(item["chain_id_res"])
        # i_grid, j_grid = np.meshgrid(np.arange(len(item["chain_id_res"])), np.arange(len(item["chain_id_res"])))
        # comparison_matrix = (chain_id_array[i_grid] == chain_id_array[j_grid])
        # interface_res_matrix[:len(item["chain_id_res"]), :len(item["chain_id_res"])] = torch.from_numpy(comparison_matrix)
        interface_atoms.append(interface_res_matrix)
        # 注意后来的系数
        # affinity.append(torch.tensor(iptm_dict[item["protein_name"].replace(".pdb","")[0:-7]]).float())
        affinity.append(torch.tensor(item["affinity"]))
        if_type=torch.tensor(item["interaction_type_matrix"]).type(torch.int16)
        interaction_type.append(F.pad(if_type,(0,pro_len-if_type.shape[0],0,pro_len-if_type.shape[0])))
        if_matrix=torch.tensor(item["interaction_matrix"]).type(torch.int16)
        interaction_matrix.append(F.pad(if_matrix,(0,0,0,pro_len-if_matrix.shape[0],0,pro_len-if_matrix.shape[0])))
        mass_centor=torch.tensor(item["res_mass_centor"])
        res_mass_centor.append(F.pad(mass_centor,(0,0,0,pro_len-mass_centor.shape[0])))
        hetatm_features_single=torch.tensor(item["hetatm_features"]).type(torch.float32)
        hetatm_features.append(F.pad(hetatm_features_single,(0,0,0,pro_len-hetatm_features_single.shape[0])))

        if (idx + 1) % batch_size == 0:
            yield {
                "protein_names": protein_names,
                "seqs": seqs,
                "chain_id_res": chain_id_res,
                "enc_tokens": enc_tokens,
                "seq_features": seq_features,
                "coor_features": coor_features,
                "interface_atoms": interface_atoms,
                "affinity": affinity,
                "interaction_type": interaction_type,
                "interaction_matrix": interaction_matrix,
                "res_mass_centor": res_mass_centor,
                "hetatm_features":hetatm_features
            }
            protein_names = []
            seqs = []
            chain_id_res = []
            enc_tokens = []
            seq_features = []
            coor_features = []
            interface_atoms = []
            affinity = []
            interaction_type = []
            interaction_matrix = []
            res_mass_centor = []
            hetatm_features = []
    if protein_names:
        yield {
            "protein_names": protein_names,
            "seqs": seqs,
            "chain_id_res": chain_id_res,
            "enc_tokens": enc_tokens,
            "seq_features": seq_features,
            "coor_features": coor_features,
            "interface_atoms": interface_atoms,
            "affinity": affinity,
            "interaction_type": interaction_type,
            "interaction_matrix": interaction_matrix,
            "res_mass_centor": res_mass_centor,
            "hetatm_features":hetatm_features
        }



if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument("--data", '-d', default="./data/checked_data/default/", type=str, help="checked data path")
    parser.add_argument("--gpu_path", '-g', default="./data/preprocess/gpu/default/", type=str, help="gpu data path")
    parser.add_argument("--batch_path", '-b', default="./data/batch/default/", type=str, help="batch data path")

    # assign your pdbs here
    # pdb_folder = "./data_final/pdb/Accepted"
    args = parser.parse_args()

    data_path=args.data

    data=np.load(data_path+"/checked_cpu_data.npy",allow_pickle=True)
    # data=np.load("/public/mxp/xiejun/py_project/PPI_affinity/data_final/sum_cpu/supplement/sum_cpu.npy",allow_pickle=True)

    random.seed(42)
    random.shuffle(data)
    print(len(data))
    for fold in range(n_fold):
        if fold!=0:
            continue
        if_add_path = args.gpu_path
        if_add_dict = load_if_add_dict(if_add_path)


        print("loading train data")

        output_path = args.batch_path

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for i, batch in enumerate(process_train_data(data, pro_len, batch_size)):
            torch.save(batch, os.path.join(output_path, f"batch_{i}.pt"))
            print(f"Saved batch {i}")
        print("save all")




    