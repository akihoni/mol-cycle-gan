import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras.backend as K
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
import torch
from models.net import Net
import os

from jtvae import (Vocab,
                   JTNNVAE)

main_label_path = './pred_model/fruity_stat_new.pkl'
# penl1_label_path = './pred_model/sweet_stat.pkl'
decode_path = './210627_data/model.iter-6'
vocab_path = './210627_data/vocab.txt'

def load_reward_model(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path))
    return model.cuda().eval()


def load_decode_model(hidden_size=450, latent_size=56, depth=3):
    model_path = decode_path
    vocab_path = './210627_data/vocab.txt'
    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)

    hidden_size = int(hidden_size)
    latent_size = int(latent_size)
    depth = int(depth)

    model = JTNNVAE(vocab, hidden_size, latent_size, depth)
    model.load_state_dict(torch.load(model_path))

    return model


def cal_pred(mol, model):
    # mol = smiles
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        X = torch.FloatTensor(fp).cuda()
        X = X.unsqueeze(0)
        pred = model(X).sigmoid_().cpu().detach().numpy()[0][0]
        return pred


def decode_from_vector(latent_vector, model):
    tree_dims = 56 // 2
    tree_vec = np.expand_dims(latent_vector[0:tree_dims], 0)
    mol_vec = np.expand_dims(latent_vector[tree_dims:], 0)
    tree_vec = torch.autograd.Variable(torch.from_numpy(tree_vec).float())
    mol_vec = torch.autograd.Variable(torch.from_numpy(mol_vec).float())
    # print(tree_vec, mol_vec)
    smiles = model.decode(tree_vec, mol_vec, prob_decode=False)
    return smiles


def reward_fn(vector, defult = 1):
    model_decode = load_decode_model(hidden_size=450, latent_size=56, depth=3)
    smiles = decode_from_vector(vector, model_decode)
    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        return defult

    model_main = load_reward_model(main_label_path)
    # model_penl1 = load_reward_model(penl1_label_path)

    score_main = cal_pred(mol, model_main)
    # score_penl1 = cal_pred(smiles, model_penl1)

    # score = K.abs(score_main - score_penl1)
    return 1 - score_main

