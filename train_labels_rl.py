import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import tqdm

from models.generator import resnet_generator
from train_labels_loss import reward_fn
from jtvae import (Vocab,
                   JTNNVAE)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


def create_networks(network_type, generator_params, model_path):
    netG_A, real_A, fake_B = resnet_generator(network_type=network_type, **generator_params)
    # netG_A.load_weights(os.path.join(save_path, 'Generator_A_weights.h5'))
    netG_A.load_weights(model_path)

    return netG_A, real_A, fake_B


def generate_data(data_path, file_path, input_shape, data_size=0, for_train=False):
    path = os.path.join(data_path, file_path)
    data = pd.read_csv(path, index_col=0)
    if for_train:
        data = data.sample(n=data_size)

    return data.values.reshape((-1,) + input_shape)


def train_as_rl(model, data_path, train_x, train_y, num_iterations=500, verbose_step=10, data_size=20, input_shape=(56,)):
    print('start training...')
    print('using model from', data_path)
    def loss(y_true, y_pred):
        diff = y_true - y_pred
        diff = diff ** 2
        return diff
        # print(y_true)
        # vector = y_true.astype(float)
        # return reward_fn(vector)
    model.compile(loss=loss, optimizer=Adam(lr=0.01))

    history = {'episode': [], 'reward': [], 'loss': []}

    for cur_iteration in range(num_iterations):
        mols = []

        real_A = generate_data(data_path, train_x, input_shape, data_size, for_train=True)
        real_B = generate_data(data_path, train_y, input_shape, data_size, for_train=True)
        fake_B = model.predict(real_A)
        r_list = [reward_fn(s) * 10 for s in fake_B]
        rewards = np.array(r_list)
      
        X = np.array(real_A)
        y = np.array(real_B)

        loss_cur = model.train_on_batch(X, y, sample_weight=rewards)

        history['episode'].append(cur_iteration)
        history['reward'].append(r_list)
        history['loss'].append(loss_cur)

        if cur_iteration and cur_iteration % verbose_step == 0:
            print('Episode: {} | reward: {} | loss: {:.3f}'.format(cur_iteration, r_list, loss_cur))

        mols = []

    model.save_weights(os.path.join(data_path, 'rl_fruity.h5'))
    print('finish training!')
    return history, model


def transform(model, data_path, test_file, save_path, input_shape=(56,)):
    print('start transforming...')
    inputs = generate_data(data_path, test_file, input_shape)
    vectors = model.predict(inputs)
    vectors_df = pd.DataFrame(vectors)
    vectors_df.to_csv(os.path.join(save_path, 'vector_A_to_B.csv'))
    mols = vectors_df.values
    smiles = []

    tree_dims = 56 // 2
    vocab_path = './210524_data/vocab.txt'
    decode_path = './210524_data/model.iter-6'
    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)

    hidden_size = int(450)
    latent_size = int(56)
    depth = int(3)

    decode_model = JTNNVAE(vocab, hidden_size, latent_size, depth)
    decode_model.load_state_dict(torch.load(decode_path))
    
    for i in tqdm.tqdm(range(mols.shape[0])):
        tree_vec = np.expand_dims(mols[i, 0:tree_dims], 0)
        mol_vec = np.expand_dims(mols[i, tree_dims:], 0)
        tree_vec = torch.autograd.Variable(torch.from_numpy(tree_vec).float())
        mol_vec = torch.autograd.Variable(torch.from_numpy(mol_vec).float())
        smi = decode_model.decode(tree_vec, mol_vec, prob_decode=False)
        smiles.append(smi)

    save_file = os.path.join(save_path, 'smiles_rl.csv')
    smiles_df = pd.DataFrame(smiles, columns=['smiles'])
    smiles_df.to_csv(save_file, index=False)
    print('finish transforming!')


def main():
    network_type = 'FC_smallest'
    generator_params = {
        'input_shape': (int(56),),
        'use_dropout': False,
        'use_batch_norm': True,
        'use_leaky_relu': True,
    }
    data_path = './mine_3/'
    save_path = os.path.join(data_path, 'res/')
    model_path = os.path.join(save_path, 'Generator_A_weights.h5')
    test_file = 'X_JTVAE_gs_test_A.csv'
    train_x = 'X_JTVAE_gs_train_A.csv'
    train_y = 'X_JTVAE_gs_train_B.csv'

    netG_A, real_A, fake_B = \
        create_networks(network_type, generator_params, model_path)
    history, new_model = train_as_rl(netG_A, data_path, train_x, train_y)
    transform(new_model, data_path, test_file, save_path)

if __name__ == "__main__":
    main()
