import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import os
import glob
import re
import numpy as np
from model import build_SND
# from Transformer_based import build_SND

import json
from torchvision import transforms
from PIL import Image
import h5py
import pickle
import jpegio as jio
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device_ids = [0]


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # for train
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')  # batch_size
    parser.add_argument('--model', default='SNP', type=str, help='social network provenance')
    parser.add_argument('--t', default='Test', type=str, help='test validation train')
    parser.add_argument('--extra_features', default='HEADER+META', type=str, help='[DCT,META,HEADER]')  
    parser.add_argument('--data_dir', default='/xxx/xxx/xxx/F-4OSN-SC/', type=str,
                        help='path of train data') 
    parser.add_argument('--train_mode', default='xxx', type=str, help='aim of train')
    parser.add_argument('--epoch', default=150, type=int, help='number of train epoches')  # epoch
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate for SGD')  
    parser.add_argument('--num_networks', default=4, type=int, help='number of networks') 
    parser.add_argument('--feature_dim', default=529, type=int, help='DCT: 369, META: 152, HEADER:8') 
    # parser.add_argument('--modes', default=[3, 4, 5], type=int, help='mode of manipulation')
    parser.add_argument('--mode', default=7, type=int,
                        help='number of manipulation, init : P: 0, S: 1, E: 2, Facebook: 3, Telegram: 4, WeChat: 5, Whatsapp: 6, orig: 7')  
    # for model
    parser.add_argument('--input_channel', default=9, type=int, help='1:gray image 3:color image') 
    parser.add_argument('--num_words', default=64, type=int, help='number of words stand for the feature map') 
    parser.add_argument('--d_model', default=512, type=int, help='Embedding Size') 
    parser.add_argument('--d_ff', default=2048, type=int, help='FeedForward dimension')  
    parser.add_argument('--d_k', default=64, type=int, help='dimension of K(=Q)')  
    parser.add_argument('--d_v', default=64, type=int, help='dimension of V')  
    parser.add_argument('--n_layers', default=6, type=int, help='number of Encoder of Decoder Layer')  
    parser.add_argument('--n_heads', default=8, type=int, help='number of heads in Multi-Head Attention')  
    parser.add_argument('--dropout', default=0.1, type=int, help='dropout in multihead') 
    return parser


def read_features(t):
    """
    Read features and label from h5py given
    the dataset, its configuration and the split t (train, test or validation)
    """
    hf = h5py.File('/xxx/xxx/xxx/F-SMUD.hdf5') 
    DCT = np.asarray(hf[t+'/features/dct'])
    META = np.asarray(hf[t+'/features/meta'])
    Header = np.asarray(hf[t+'/features/header'])

    labels = np.asarray(hf[t+'/labels'])
    labels = [l.decode("utf-8") for l in labels]

    return DCT, META, Header, labels


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def get_features(DCT, META, Header, m='HEADER+META'):
    """
    Return the correct set of features to use given the method m
    """
    if m == 'DCT':
        features = DCT
    elif m == 'META':
        features = META
    elif m == 'DCT+META':
        features = np.concatenate((DCT, META), axis=1)
    elif m == 'HEADER':
        features = Header
    elif m == 'HEADER+META':
        features = np.concatenate((Header, META), axis=1)
    elif m == 'HEADER+DCT+META':
        features = np.concatenate((Header, DCT, META), axis=1)
    return features


def dct_quant_table1(img_path, channel=0, T=20, transform=None):
    jpeg_struct = jio.read(img_path)

    dct_array = jpeg_struct.coef_arrays[channel] 
    quant_table = jpeg_struct.quant_tables[channel]

    if transform is not None:
        dct_array = transform(dct_array)

    t_DCT_vol = torch.zeros(size=(T + 1, dct_array.shape[0], dct_array.shape[1]))
    # t_DCT_vol[0] += (dct_array == 0).float().squeeze()
    t_DCT_vol[0] += (dct_array == 0).astype(np.float32).squeeze()
    for i in range(1, T):
        t_DCT_vol[i] += (dct_array == i).astype(np.float32).squeeze()
        t_DCT_vol[i] += (dct_array == -i).astype(np.float32).squeeze()
    t_DCT_vol[T] += (dct_array >= T).astype(np.float32).squeeze()
    t_DCT_vol[T] += (dct_array <= -T).astype(np.float32).squeeze()
    import copy
    return copy.deepcopy(dct_array), copy.deepcopy(t_DCT_vol), copy.deepcopy(quant_table)


class MyDataSet(Data.Dataset):
    def __init__(self, img_path: list, images_class: list, features, transform=None, seed=None):
        super(MyDataSet, self).__init__()
        self.img_path = img_path
        self.features = features
        self.org_to_idx = {'P': 0, 'S': 1, 'E': 2, 'Facebook': 3, 'Telegram': 4, 'WeChat': 5, 'Whatsapp': 6, 'orig': 7}
        self.l = len(self.org_to_idx)
        self.transform = transform
        self.images_class = images_class
        self.dec_in = []
        self.label = []
        for i in images_class:
            # if i not in self.org_to_idx:
            #     self.org_to_idx[i] = len(self.org_to_idx)
            for ii in i.split('_'):
                if ii not in self.org_to_idx:
                    self.org_to_idx[ii] = len(self.org_to_idx)
            tmp = []
            
            if i == 'orig':
                pad = self.l - 4
                for k in range(pad):
                    tmp.append(0)
                self.dec_in.append([1] + tmp) 
                self.label.append(tmp + [2])  
           
            elif i in self.org_to_idx and i != 'orig':
                tmp.append(self.org_to_idx[i]) 
                pad = self.l - 5 
                
                for k in range(pad):
                    tmp.append(0)
                self.dec_in.append([1] + tmp)  
                self.label.append(tmp + [2])  
           
            else:
                for j in i.split('_'):
                    tmp.append(self.org_to_idx[j])
                pad = self.l - 4 - len(i.split('_'))  
               
                tmp = tmp[::-1]
               
                for k in range(pad):
                    tmp.append(0)
                self.dec_in.append([1] + tmp)  
                self.label.append(tmp + [2])  
        json_str = json.dumps(dict((key, val) for key, val in self.org_to_idx.items()), indent=4)
        with open('org_to_idx.json', 'w') as json_file:
            json_file.write(json_str)
        self.seed = seed
        self.tansformer_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_path)
    # return 128

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        if self.seed is not None:
            self.seed += 1

            torch.manual_seed(self.seed)

        if self.transform is not None:
            img = self.transform(img)
        path = self.img_path[idx]
        dct_array, t_DCT_vol, t_quant_table = dct_quant_table1(path, channel=0, T=20, transform=self.transform)
        img = self.tansformer_tensor(img)
        dec_in = np.int64(np.array(self.dec_in[idx])) 
        label = np.int64(np.array(self.label[idx]))  
        features = np.float32(self.features[idx])  
        # features = np.float32(np.zeros(529))
        return img, dct_array, t_DCT_vol, t_quant_table, dec_in, label, features


def greedy_decoder(model, x, t_DCT_vol, t_quant_table, features, len_dec_in, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    batch_size = x.shape[0]
    if torch.cuda.is_available():
        dec_input = torch.zeros([1, len_dec_in]).long().cuda()  
    else:
        dec_input = torch.zeros([1, len_dec_in]).long()
    dec_input = dec_input.expand(batch_size, len_dec_in) 
    next_symbol = torch.tensor(start_symbol)  
    next_symbol = next_symbol.expand(batch_size, 1).cuda()
    dec_inputs = dec_input
    for i in range(0, dec_input.shape[1]):
        idx = torch.tensor(i)
        idx = idx.expand(batch_size, 1).cuda()
        dec_inputs = dec_inputs.scatter(1, idx, next_symbol)
        projected, _ = model(x, t_DCT_vol, t_quant_table, dec_inputs, features)
        projected = nn.Softmax(dim=-1)(projected)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]  
        prob = prob.view(batch_size, len_dec_in)
        next_word = prob[:, i]
        next_symbol = next_word.unsqueeze(1)
    return dec_inputs


def n_match(candidate, references, n):
    match_n = 0
    c = len(candidate)
    r = len(references)
    for i in range(c - n + 1):
        for j in range(r - n + 1):
            t1 = candidate[i:i + n]
            t2 = references[j:j + n]
            if t1 == t2:
                match_n += 1
    return match_n


def longestCommonSubsequence(str1, str2) -> int:
    str1 = str1.cpu().numpy().tolist()
    str2 = str2.cpu().squeeze().numpy().tolist()
    s1 = []
    s2 = []
    for i in str1:
        if i != 0:
            if i != 2:
                s1.append(i)
    for j in str2:
        if j != 0:
            if j != 2:
                s2.append(j)
    m, n = len(s1), len(s2)
    n_manipulation = n
    matches_n = []
    for i in range(m):
        if i > 4:
            break
        match_n = n_match(s1, s2, i + 1)
        if match_n != 0:
            matches_n.append(match_n)
    n_correct = len(matches_n)
    return n_correct, n_manipulation, m


def len_select(path, label, feature, c):
    images_path = []
    images_label = []
    features = []
    for i in range(len(label)):
        chain = label[i]
        if len(chain.split('-')) <= c:  
            images_path.append(
                path[i])  
            images_label.append(label[i])  # lable[0]='FB'
            features.append(feature[i])  

    return images_path, images_label, features


class Sequential8x8BlockCrop(object):
    def __init__(self, block_size, output_size, random_crop=True):
        self.block_size = block_size  
        self.output_size = output_size  
        self.random_crop = random_crop  

    def __call__(self, img):
        if isinstance(img, Image.Image):
            
            num_blocks_x = img.size[0] // self.block_size
            num_blocks_y = img.size[1] // self.block_size
        else:
            num_blocks_x = img.shape[0] // self.block_size
            num_blocks_y = img.shape[1] // self.block_size

        
        max_start_block_x = num_blocks_x - self.output_size // self.block_size
        max_start_block_y = num_blocks_y - self.output_size // self.block_size

        
        if max_start_block_x < 0 or max_start_block_y < 0:
            raise ValueError("Image is too small to get a 256x256 crop from 8x8 blocks.")

        if self.random_crop:
            
            start_block_x = random.randint(0, max_start_block_x)
            start_block_y = random.randint(0, max_start_block_y)

           
            start_x = start_block_x * self.block_size
            start_y = start_block_y * self.block_size
            end_x = start_x + self.output_size
            end_y = start_y + self.output_size
        else:
            
            start_block_x = max_start_block_x // 2
            start_block_y = max_start_block_y // 2

           
            start_x = start_block_x * self.block_size
            start_y = start_block_y * self.block_size
            end_x = start_x + self.output_size
            end_y = start_y + self.output_size
      
        # return img.crop((start_x, start_y, end_x, end_y))
        if isinstance(img, Image.Image):
           
            return img.crop((start_x, start_y, end_x, end_y))
        else:
            
            return img[start_x:end_x, start_y:end_y]


def load(root, Batch_size, t: str, shuffle=True, seed=None, m='HEADER+META'):
    DCT, META, Header, labels = read_features(t)  # labels = "Train/apple/facebook/001.jpg"
    images_path = [root + i for i in
                   labels]  # images_path="/data/yjx/Proj/SocialNetwork/FODB_OSN_Dataset/Train/apple/facebook/001.jpg"
    images_label = [i.split('/')[2] for i in labels]  
    features = get_features(DCT, META, Header, m)
    data_transform = {
        "Train": transforms.Compose([Sequential8x8BlockCrop(8, 256),
                                     # Sequential8x8BlockCrop(8, 256),
                                     ]),
        "Val": transforms.Compose([Sequential8x8BlockCrop(8, 256, random_crop=False),
                                   # Sequential8x8BlockCrop(8, 256, random_crop=False),
                                   ]),
        "Test": transforms.Compose([Sequential8x8BlockCrop(8, 256, random_crop=False),
                                    # Sequential8x8BlockCrop(8, 256, random_crop=False),
                                    ])
    }
    images_path, images_label, features = len_select(images_path, images_label, features, c=4)
    set = MyDataSet(images_path, images_label, features, data_transform[t], seed=seed)  # img, dec_in, label, features
    loader = Data.DataLoader(set, batch_size=Batch_size, shuffle=shuffle, drop_last=True)

    return loader


def predict2chain(p):
    with open('org_to_idx.json', 'r', encoding='utf8') as fp:
        import json
        org_to_idx = json.load(fp)
    new_dict = {v:k for k, v in org_to_idx.item()}
    encoded = p.cpu().numpy().tolist()
    chains = []
    for i in encoded:
        chain = [new_dict[j].split('_')[-1] for j in i if j != 0]
        chains.append(chain)

    return chains


class Chain2Labels():
    def __init__(self):
        with open('org_to_idx.json', 'r', encoding='UTF-8') as f:
            self.SN_idx = json.load(f)
        self.SN_idx = {v:k for k, v in self.SN_idx.items()}

        # for k, v in self.SN_idx.items():
        #     if v =='FaceBook':
        #         self.SN_idx[k] = 'FB'
        #     if v =='Flickr':
        #         self.SN_idx[k] = 'FL'
        #     if v =='WeChat':
        #         self.SN_idx[k] = 'WC'
        with open('chain_label.json', 'r', encoding='UTF-8') as f:
            self.Chain_idx = json.load(f)
        # self.Chain_idx = {v:k for k, v in self.Chain_idx.items()}


    def forward(self, Chain):
        Chain = Chain.cpu().numpy().tolist()
        Chain_list = [i for i in Chain if i not in [0, 2]]
        Chain_list = Chain_list[::-1]
        if Chain_list != []:
            chain_str = [self.SN_idx[i] for i in Chain_list]
            chain_str = '-'.join(chain_str)
            label = self.Chain_idx[chain_str]
        else:
            chain_str = 'Original'
            label = self.Chain_idx[chain_str]
        return int(label), chain_str


def main(args):
    # parameters
    Batch_Size = args.batch_size
    root = args.data_dir
    m = args.extra_features
    # data load
    t = args.t
    test_loaders = load(root, Batch_size=args.batch_size, t='Test', shuffle=False, seed=1, m=m)

    cuda = torch.cuda.is_available()
    save_dir = os.path.join('/xxx/xxx/xxx/', args.model + '_' + args.train_mode)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open('org_to_idx.json', 'r', encoding='utf8') as fp:
        import json
        org_to_idx = json.load(fp)
    args.mode = len(org_to_idx)
    model = build_SND(args)
    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

    #
    model_weight = torch.load(os.path.join(save_dir, 'model.pth'), map_location='cpu').state_dict()

    model.load_state_dict(model_weight, strict=True)
    # torch.save(model, os.path.join(save_dir, 'val_train_model_150.pth'))

    # data load
    print('===> Building model')
    # criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        correct_all = 0.0
        n = 0.0
        v_ALMS = 0.0
        pre = []
        gt = []
        # c2l = Chain2Labels()
        for n_count, Batch_xyz in enumerate(test_loaders):
            if cuda:
                img, dct_array, t_DCT_vol, t_quant_table, dec_inputs, labels, features = Batch_xyz[0].cuda(), Batch_xyz[1].cuda(), \
                                                                                         Batch_xyz[2].cuda(), Batch_xyz[3].cuda(), \
                                                                                         Batch_xyz[4].cuda(),  Batch_xyz[5].cuda(), \
                                                                                         Batch_xyz[6].cuda()
            else:
                img, dct_array, t_DCT_vol, t_quant_table, dec_inputs, labels, features = Batch_xyz[0], Batch_xyz[1], \
                                                                                         Batch_xyz[2], Batch_xyz[3], \
                                                                                         Batch_xyz[4], Batch_xyz[5], \
                                                                                         Batch_xyz[6]
            dec_input = greedy_decoder(model, img, t_DCT_vol, t_quant_table, features, args.num_networks + 1, start_symbol=1)
            predict, _,  = model(img, t_DCT_vol, t_quant_table, dec_input, features)
            predict_y = predict.squeeze(0).max(dim=-1, keepdim=False)[1]
            predict_y = predict_y.view(dec_input.shape)
            # print('predict_y:', predict_y)
            # print('labes:', labels)

            for vc in range(img.shape[0]):
                correct_all += torch.equal(predict_y[vc], labels[vc].squeeze())
                n +=1
            
        acc_all = correct_all / n * 100
        print('ACC:%.2f%%' % acc_all)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Multi-Manipulation-Detection', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
