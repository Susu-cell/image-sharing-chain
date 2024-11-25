import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import os
import glob
import re
import numpy as np
from model1 import build_SND
# from Transformer_based import build_SND

import json
from torchvision import transforms
from PIL import Image
import h5py
import pickle
import jpegio as jio
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
# device_ids = [0,1]
os.environ["CUDA_VISIBLE_DEVICES"] = '2,1,0'
device_ids = [0]


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # for train
    parser.add_argument('--batch_size', default=90, type=int, help='batch size')  # batch_size
    parser.add_argument('--model', default='SNP', type=str, help='social network provenance')
    parser.add_argument('--t', default='Test', type=str, help='test validation train')
    parser.add_argument('--extra_features', default='HEADER+META', type=str, help='[DCT,META,HEADER]')  # 添加额外特征的种类
    parser.add_argument('--data_dir', default='xxx', type=str,
                        help='path of train data')  # 数据集路径，有V和R两种
    parser.add_argument('--train_mode', default='xxx', type=str, help='aim of train')  # 预训练权重
    parser.add_argument('--epoch', default=150, type=int, help='number of train epoches')  # epoch
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate for SGD')  # 学习率
    parser.add_argument('--num_networks', default=4, type=int, help='number of networks')  # decoder传入的数(三个平台)???
    parser.add_argument('--feature_dim', default=529, type=int, help='DCT: 369, META: 152, HEADER:8')  # 特征维度
    # parser.add_argument('--modes', default=[3, 4, 5], type=int, help='mode of manipulation')
    parser.add_argument('--mode', default=6, type=int,
                        help='number of manipulation, init : P: 0, S: 1, E: 2, Facebook: 3, Telegram: 4, WeChat: 5, Whatsapp: 6, orig: 7')  # 操作的字典数
    # for model
    parser.add_argument('--input_channel', default=9, type=int, help='1:gray image 3:color image')  # 图像的通道数
    parser.add_argument('--num_words', default=64, type=int, help='number of words stand for the feature map')  # encoder输入模型的单词（按通道展平）个数
    parser.add_argument('--d_model', default=512, type=int, help='Embedding Size')  # 特征维度
    parser.add_argument('--d_ff', default=2048, type=int, help='FeedForward dimension')  # 前馈网络的维度
    parser.add_argument('--d_k', default=64, type=int, help='dimension of K(=Q)')  # k/q的维度???
    parser.add_argument('--d_v', default=64, type=int, help='dimension of V')  # v的维度???
    parser.add_argument('--n_layers', default=6, type=int, help='number of Encoder of Decoder Layer')  # 几层gmlb模块
    parser.add_argument('--n_heads', default=8, type=int, help='number of heads in Multi-Head Attention')  # 多投注意力的头数???
    parser.add_argument('--dropout', default=0.1, type=int, help='dropout in multihead')  # 正则化系数
    return parser


def read_features(t):
    """
    Read features and label from h5py given
    the dataset, its configuration and the split t (train, test or validation)
    """
    hf = h5py.File('xxx/F-SMUD.hdf5')  # 读取的是h5py模式的文件
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

    dct_array = jpeg_struct.coef_arrays[channel]  # (1536, 2048)为图像的像素块个数
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
            # 无链情况
            if i == 'orig':
                pad = self.l - 4
                for k in range(pad):
                    tmp.append(0)
                self.dec_in.append([1] + tmp)  # 1是开始符([[1,0,0,0,0]])
                self.label.append(tmp + [2])  # 2是结束符([0,0,0,0,2])
            # 单链情况
            elif i in self.org_to_idx and i != 'orig':
                tmp.append(self.org_to_idx[i])  # tmp放的是链的索引
                pad = self.l - 5  # 为了让tmp是四个
                # 为了确保tmp是4个(因为4条链)
                for k in range(pad):
                    tmp.append(0)
                self.dec_in.append([1] + tmp)  # 1是开始符([[1,3,0,0,0]])
                self.label.append(tmp + [2])  # 2是结束符([3,0,0,0,2])
            # 多链情况
            else:
                for j in i.split('_'):
                    tmp.append(self.org_to_idx[j])
                pad = self.l - 4 - len(i.split('_'))  # pad就是填充符
                # 用于将列表 tmp 中的元素进行逆序排列,因为上面把顺序搞反了
                tmp = tmp[::-1]
                # 为了确保tmp是4个
                for k in range(pad):
                    tmp.append(0)
                self.dec_in.append([1] + tmp)  # 1是开始符([[1,3,0,0,0]])
                self.label.append(tmp + [2])  # 2是结束符([3,0,0,0,2])
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

            # 是用于设置随机数生成器的种子值，以确保在每次运行时生成的随机数序列是可复现的。通过设置相同的种子值，可以在每次运行代码时得到相同的随机数序列，从而使结果具有可重现性
            torch.manual_seed(self.seed)

        if self.transform is not None:
            img = self.transform(img)
        path = self.img_path[idx]
        dct_array, t_DCT_vol, t_quant_table = dct_quant_table1(path, channel=0, T=20, transform=self.transform)
        img = self.tansformer_tensor(img)
        dec_in = np.int64(np.array(self.dec_in[idx]))  # dec_in 是解码器输入的传播链路径
        label = np.int64(np.array(self.label[idx]))  # # 解码器的标签
        features = np.float32(self.features[idx])  # 额外特征
        # features = np.float32(np.zeros(529))
        return img, dct_array, t_DCT_vol, t_quant_table, dec_in, label, features


def greedy_decoder(model, x, t_DCT_vol, t_quant_table, features, memory_context, need_update_memory_context, len_dec_in, start_symbol):
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
        dec_input = torch.zeros([1, len_dec_in]).long().cuda()  # len_dec_in为目标输入序列的长度, 1表示初始时刻的输入
    else:
        dec_input = torch.zeros([1, len_dec_in]).long()
    dec_input = dec_input.expand(batch_size, len_dec_in)  # 扩展解码器的输入dec_input的维度为[batch_size, len_dec_in]，使其适应批处理的数据(64,4)
    next_symbol = torch.tensor(start_symbol)  # 初始化next_symbol为起始符号，形状为[batch_size, 1]，并将其复制到整个批次中(起始为s=1)
    next_symbol = next_symbol.expand(batch_size, 1).cuda()
    dec_inputs = dec_input
    for i in range(0, dec_input.shape[1]):
        idx = torch.tensor(i)
        idx = idx.expand(batch_size, 1).cuda()
        dec_inputs = dec_inputs.scatter(1, idx, next_symbol)
        projected, _, _ = model(x, t_DCT_vol, t_quant_table, dec_inputs, features, memory_context, need_update_memory_context)
        projected = nn.Softmax(dim=-1)(projected)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]  # 从概率分布中选择概率最大的单词作为下一个符号next_word
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
        if len(chain.split('-')) <= c:  # 分割后的链长度是否小于或等于c
            images_path.append(
                path[i])  # path[0]='/data/yjx/Proj/SocialNetwork/FODB_OSN_Dataset/Train/apple/facebook/001.jpg'
            images_label.append(label[i])  # lable[0]='FB'
            features.append(feature[i])  # feature[0]=第一个图像的额外特征

    return images_path, images_label, features


class Sequential8x8BlockCrop(object):
    def __init__(self, block_size, output_size, random_crop=True):
        self.block_size = block_size  # 8x8块的尺寸
        self.output_size = output_size  # 输出尺寸，例如256x256
        self.random_crop = random_crop  # 是否随机裁剪

    def __call__(self, img):
        if isinstance(img, Image.Image):
            # 计算可能的8x8块数量
            num_blocks_x = img.size[0] // self.block_size
            num_blocks_y = img.size[1] // self.block_size
        else:
            num_blocks_x = img.shape[0] // self.block_size
            num_blocks_y = img.shape[1] // self.block_size

        # 计算能够生成256x256完整块的最大起始块索引范围
        max_start_block_x = num_blocks_x - self.output_size // self.block_size
        max_start_block_y = num_blocks_y - self.output_size // self.block_size

        # 检查是否可以生成256x256完整块
        if max_start_block_x < 0 or max_start_block_y < 0:
            raise ValueError("Image is too small to get a 256x256 crop from 8x8 blocks.")

        if self.random_crop:
            # 随机选择起始块
            start_block_x = random.randint(0, max_start_block_x)
            start_block_y = random.randint(0, max_start_block_y)

            # 计算起始点和结束点
            start_x = start_block_x * self.block_size
            start_y = start_block_y * self.block_size
            end_x = start_x + self.output_size
            end_y = start_y + self.output_size
        else:
            # 计算起始块的中间值
            start_block_x = max_start_block_x // 2
            start_block_y = max_start_block_y // 2

            # 计算起始点和结束点
            start_x = start_block_x * self.block_size
            start_y = start_block_y * self.block_size
            end_x = start_x + self.output_size
            end_y = start_y + self.output_size
        # 返回裁剪的图像
        # return img.crop((start_x, start_y, end_x, end_y))
        if isinstance(img, Image.Image):
            # 如果img是PIL图像，使用img.crop方法
            return img.crop((start_x, start_y, end_x, end_y))
        else:
            # 如果img是numpy.ndarray，使用NumPy来截取子数组
            return img[start_x:end_x, start_y:end_y]


def load(root, Batch_size, t: str, shuffle=True, seed=None, m='HEADER+META'):
    DCT, META, Header, labels = read_features(t)  # labels = "Train/apple/facebook/001.jpg"
    images_path = [root + i for i in
                   labels]  # images_path="/data/yjx/Proj/SocialNetwork/FODB_OSN_Dataset/Train/apple/facebook/001.jpg"
    images_label = [i.split('/')[2] for i in labels]  # 只有传播链
    features = get_features(DCT, META, Header, m)
    data_transform = {
        "Train": transforms.Compose([Sequential8x8BlockCrop(8, 128),
                                    # Sequential8x8BlockCrop(8, 256),
                                     ]),
        "Val": transforms.Compose([Sequential8x8BlockCrop(8, 128, random_crop=False),
                                    # Sequential8x8BlockCrop(8, 256, random_crop=False),
                                   ]),
        "Test": transforms.Compose([Sequential8x8BlockCrop(8, 128, random_crop=False),
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
    save_dir = os.path.join('xxx', args.model + '_' + args.train_mode)
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

    # data load
    print('===> Building model')
    # criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        # class_dicts = Classes_dict().dicts
        # cm = np.zeros([len(class_dicts), len(class_dicts)])
        # org_to_idx_ver = {v:k for k, v in org_to_idx.items()}
        #
        # cmname = t + 'Confusion_Matrix' + args.train_mode + '.csv'
        # h = []
        # idx = []
        # for key, value in class_dicts.items():
        #     tmp = []
        #     for item in key:
        #         if item in ['3', '4', '5']:
        #             tmp.append(org_to_idx_ver[int(item)])
        #     h.append('-'.join(tmp))
        #     idx.append('-'.join(tmp))
        # Confusion_Matrix = pd.DataFrame(columns=h, index=idx)
        # Confusion_Matrix.to_csv(cmname, index=True)
        # loss_hat = 0.0
        correct_all = 0.0
        n = 0.0
        pre = []
        gt = []
        # c2l = Chain2Labels()
        data_root = 'chuanbolian_memory_context/xxx'
        save_path = os.path.join(data_root, "train_memory_context.pkl")
        with open(save_path, 'rb') as f:
            memory_context = pickle.load(f)
        need_update_memory_context = False
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
            dec_input = greedy_decoder(model, img, t_DCT_vol, t_quant_table, features, memory_context, need_update_memory_context, args.num_networks + 1, start_symbol=1)
            predict, _, _ = model(img, t_DCT_vol, t_quant_table, dec_input, features, memory_context, need_update_memory_context)
            predict_y = predict.squeeze(0).max(dim=-1, keepdim=False)[1]
            predict_y = predict_y.view(dec_input.shape)
            # print('predict_y:', predict_y)
            # print('labes:', labels)

            for vc in range(img.shape[0]):
                correct_all += torch.equal(predict_y[vc], labels[vc].squeeze())
                n +=1
                # label_p, p = c2l.forward(predict_y[vc])
                # label_t, t = c2l.forward(labels[vc])
                # if not torch.equal(predict_y[vc], labels[vc].squeeze()):
                #     label_p, p = c2l.forward(predict_y[vc])
                #     label_t, t = c2l.forward(labels[vc])
                #     cv2.imwrite('./test_out/'+str(num)+'__'+t+'->'+p+'.png', img)
                # pre.append(label_p)
                # gt.append(label_t)
            # P_Chains = predict2chain(predict_y[:,:3])
            # gt_Chains = predict2chain(labels[:,:3])
            # for vcc in range(Batch_Size):
            #     if P_Chains[vcc] == gt_Chains[vcc]:
            #         correct_all_chain += 1
            # if n_count >= 78:
            #     break
        acc_all = correct_all / n * 100
        print('ACC:%.2f%%' % acc_all)
        # conf_matrix = confusion_matrix(gt, pre, normalize='true')
        # with open('chain_label.json', 'r', encoding='UTF-8') as f:
        #     Chain_idx = json.load(f)
        # dsp_labels = [k for k, v in Chain_idx.items()]
        # dsp_labels = dsp_labels[:12]
        # conf_matrix = np.around(conf_matrix * 100, 1)
        # # disp = ConfusionMatrixDisplay(conf_matrix, display_labels=dsp_labels)
        # # disp.plot(xticks_rotation="vertical")
        # plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        # for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        #     plt.text(j, i, conf_matrix[i, j],
        #              horizontalalignment="center",
        #              color="white" if conf_matrix[i, j] > (conf_matrix.max() / 2) else "black")
        # tick_marks = np.arange(len(dsp_labels))
        # plt.xticks(tick_marks, dsp_labels, rotation=90)
        # plt.yticks(tick_marks, dsp_labels)
        # plt.tight_layout()
        # plt.ylabel('True label', fontdict={'weight': 'bold', 'size': 13})
        # plt.xlabel('Predicted label', fontdict={'weight': 'bold', 'size': 13})
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Multi-Manipulation-Detection', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)