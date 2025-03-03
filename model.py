from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
import math
import os
import pickle


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked

    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]  # (64,4,4)
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    if torch.cuda.is_available():
        subsequence_mask = subsequence_mask.cuda()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x, mask=None):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        if mask is not None:
            # out = u * torch.matmul(mask, v)
            out = torch.matmul(mask, u * v) 
            # out = torch.matmul(mask, u) * torch.matmul(mask, v)

        else:
            out = u * v
        return out


class Mask_SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.Feed_Forward = nn.Sequential(
            nn.Linear(d_ffn, d_ffn*2),
            nn.ReLU(),
            nn.Linear(d_ffn*2, d_ffn)
        )

    def forward(self, x, mask=None):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.Feed_Forward(v)

        out = torch.matmul(mask, u * v)

        return out


class Corss_SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len, mode):  # seq_len=8, mode=4
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn * 2)
        self.spatial_proj = nn.Conv1d(seq_len, mode, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x, y):  # x=(64,5,4096)
        y = self.norm(y)  # (64,10,4096)
        y = self.spatial_proj(y)  # (64,5,4096)
        out = x * y  # out=(64,4,4096)
        return out


class Mask_gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = Mask_SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x, mask)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)  # (64,8,512)
        x = F.gelu(self.channel_proj1(x))  # (64,8,4096)
        x = self.sgu(x, mask)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class Cross_gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj1_y = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn * 2, d_model)
        self.sgu = Corss_SpatialGatingUnit(d_ffn, seq_len=num_words, mode=seq_len)

    def forward(self, x, y):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        y = F.gelu(self.channel_proj1_y(y))
        x = self.sgu(x, y)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class Encoder(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, num_layers=6):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [gMLPBlock(d_model, d_ffn, seq_len) for _ in range(num_layers)]
        )

    def forward(self, sentence):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = sentence
        for layer in self.layers:
            enc_outputs = layer(enc_outputs)
        return enc_outputs


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_mask_gMLP = Mask_gMLPBlock(d_model=d_model, d_ffn=d_ff, seq_len=num_networks+1)
        self.dec_enc_gMLP = Cross_gMLPBlock(d_model=d_model, d_ffn=d_ff, seq_len=num_networks+1)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs = self.dec_mask_gMLP(x=dec_inputs, mask=dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.dec_enc_gMLP(x=dec_outputs, y=enc_outputs)
        return dec_outputs


class CrossAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CrossAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.query = nn.Linear(in_dim, out_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, y):  # (enc_outputs_average(b,512), dec_inputs_pos(b,5,512))
        q = self.query(x.unsqueeze(1))
        k = self.key(y)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.out_dim ** 0.5)  # (b,1,5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        v = self.value(y)
        output = torch.bmm(attn_weights, v)
        output = output.squeeze(1)
        return output


class Memory_fingerprint(nn.Module):
    def __init__(self, num_words):
        super(Memory_fingerprint, self).__init__()
        memory_fingerprint_init = torch.zeros(197*num_words, 512) 
        nn.init.kaiming_uniform_(memory_fingerprint_init)
        self.memory_fingerprint = nn.Parameter(memory_fingerprint_init)

        data_root = 'xxx/xxx/xxx/'
        save_path3 = os.path.join(data_root, f"train_memory_context_init.pkl")
        with open(save_path3, 'rb') as f:
            init_memory_context = pickle.load(f)

        self.memory_context = nn.Parameter(init_memory_context.clone())

    def forward(self, enc_outputs, calculate_memory_context, k=50):  
        cos_sim = torch.nn.functional.cosine_similarity(calculate_memory_context.unsqueeze(1), self.memory_context.unsqueeze(0), dim=2)
        top_k_values, top_k_indices = torch.topk(cos_sim, k, dim=1)
        softmax_weights = F.softmax(top_k_values, dim=1)  
        indices = top_k_indices.flatten()  
        start_indices = 64 * indices 
        end_indices = start_indices + 63  
        indices = torch.cat([torch.arange(start, end + 1) for start, end in zip(start_indices, end_indices)]).long().to("cuda")
        values = torch.index_select(self.memory_fingerprint, dim=0, index=indices)
        num_samples = top_k_indices.size(0)  
        num_indices = top_k_indices.size(1) 
        
        values_grouped = values.view(num_samples, num_indices, 64, 512)
       
        memory_fingerprint_total = torch.sum(values_grouped * softmax_weights.unsqueeze(-1).unsqueeze(-1), dim=1) 
       
        weight = nn.Parameter(torch.randn(1)).to("cuda")  
        weight = weight.expand_as(enc_outputs)
       
        enc_output = memory_fingerprint_total * weight + enc_outputs * (1 - weight)
        return enc_output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(mode, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.memory_bank = Memory_fingerprint(num_words)
        self.cross_attention = CrossAttention(in_dim=512, out_dim=512)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len] (64,4)
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model] (64,8,512)
        '''
        enc_outputs_average = torch.mean(enc_outputs, dim=1)  # (b,512)
        # word embeding + position
        dec_inputs_emb = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]  (64,4,512)
        dec_inputs_pos = self.pos_emb(dec_inputs_emb.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        
        calculate_memory_context = self.cross_attention(enc_outputs_average, dec_inputs_pos)  # (b,512)
        
        enc_output = self.memory_bank(enc_outputs, calculate_memory_context)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # [batch_size, tgt_len, tgt_len]  
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)  # [batch_size, tgt_len, tgt_len]  
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0)  # [batch_size, tgt_len, tgt_len]  
        dec_self_attn_mask = abs(dec_self_attn_mask.float() - 1) 
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_output = layer(dec_inputs_pos, enc_output, dec_self_attn_mask)
        return dec_output


class gMLPTransformer(nn.Module):
    def __init__(self):
        super(gMLPTransformer, self).__init__()
        self.encoder = Encoder(d_model=d_model, d_ffn=d_ff, seq_len=num_words, num_layers=n_layers)
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, mode, bias=False)  # mode=4

   
    def forward(self, sentence, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        enc_outputs = self.encoder(sentence)
        dec_outputs = self.decoder(dec_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, mode] (64,4,6)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_outputs


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel=64, out_channel=64):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        # nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class DCT_stream(nn.Module):
    def __init__(self, batchsize):
        super(DCT_stream, self).__init__()
        self.dct_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=21, out_channels=64, kernel_size=3,
                      stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.dct_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True))
        self.batchsize = batchsize

    def forward(self, t_DCT_vol, t_quant_table):
        global dct64
        dct1 = self.dct_layer0(t_DCT_vol)
        dct2 = self.dct_layer1(dct1)
        B, C, H, W = dct2.shape
        dct3 = dct2.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4).reshape(B, 64 * C, H // 8,
                                                                                          W // 8)  # [B, 256,
        x_temp = dct2.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4)  # [B, C, 8, 8, 32, 32]
        q_temp = t_quant_table.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 8, 8, 1, 1]
        xq_temp = x_temp * q_temp  # [B, C, 8, 8, 32, 32]
        dct4 = xq_temp.reshape(B, 64 * C, H // 8, W // 8)  # [B, 256, 32, 32]
        x = torch.cat([dct3, dct4], dim=1)  # [B, 512, 32, 32]
        # B1, C1, H1, W1 = x.shape
        # x1 = x.reshape(B1, C1, H1 // 4, 4, W1// 4, 4).permute(0, 1, 3, 5, 2, 4).reshape(B1, 16 * C1, H1 // 4, W1 // 4)  # [B,512*16,8,8]
        # return x1
        return x

class SRMConv2D(nn.Module):
    def _get_srm_list(self):
        # srm kernel 1dct_noise_memory_M=197_64word_6layer_update_context
        srm1 = np.zeros([5, 5]).astype('float32')
        srm1[1:-1, 1:-1] = np.array([[-1, 2, -1],
                                     [2, -4, 2],
                                     [-1, 2, -1]])
        srm1 /= 4.
        # srm kernel 2
        srm2 = np.array([[-1, 2, -2, 2, -1],
                         [2, -6, 8, -6, 2],
                         [-2, 8, -12, 8, -2],
                         [2, -6, 8, -6, 2],
                         [-1, 2, -2, 2, -1]]).astype('float32')
        srm2 /= 12.
        # srm kernel 3
        srm3 = np.zeros([5, 5]).astype('float32')
        srm3[2, 1:-1] = np.array([1, -2, 1])
        srm3 /= 2.
        return [srm1, srm2, srm3]

    def _build_SRM_kernel(self):
        kernel = []
        srm_list = self._get_srm_list()
        for idx, srm in enumerate(srm_list):
            for ch in range(3):
                this_ch_kernel = np.zeros([5, 5, 3]).astype('float32')
                this_ch_kernel[:, :, ch] = srm
                kernel.append(this_ch_kernel)
        kernel = np.stack(kernel, axis=-1)
        # srm_kernel = K.variable( kernel, dtype='float32', name='srm' )
        '''
        Keras kernel form   (kernel_width, kernel_height, inputChanels, outputChanels)
        pytorch Kernal form (inputChanels, outputChanel, kernel_size, kernel_size)

        There is a need to switch the dim to fit in pytorch with the Mantra-Net source code writting in keras.
        '''
        kernel = np.swapaxes(kernel, 1, 2)
        # kernel = np.swapaxes(kernel,1,2)
        kernel = np.swapaxes(kernel, 0, 3)
        return kernel

    def __init__(self):
        super(SRMConv2D, self).__init__()
        self.weight = torch.tensor(self._build_SRM_kernel())

    def forward(self, x):
        with torch.no_grad():
            self.weight = self.weight.to(x.device)

            return torch.nn.functional.conv2d(x, weight=self.weight, padding=2)


class SND(nn.Module):
    def __init__(self, input_channel, num_words, batchsize, feature_dim=529):
        super(SND, self).__init__()
        self.num_words = num_words
        self.feature_extractor_dct = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature_extractor_noise = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.feature_extractor_total_feature = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.resblock1 = BasicBlock(64, 64)
        self.resblock2 = BasicBlock(64, 64)
        self.resblock3 = BasicBlock(64, 64)
        self.resblock4 = BasicBlock(64, 64)
        self.resblock5 = BasicBlock(64, 64)
        self.resblock6 = BasicBlock(64, 64)
        self.resblock7 = BasicBlock(64, 64)
        self.resblock8 = BasicBlock(64, 64)
        self.get_Features_words_noise = nn.Sequential(
            nn.Conv2d(64 * 5, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_words, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_words),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((32, 32))
        )
        self.SE_Layer = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(num_words, 2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(2, num_words, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.feature_words = nn.Sequential(
            nn.Conv2d(1184, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_words, 512))
        
        self.transformer = gMLPTransformer()
        self.dct_stream = DCT_stream(batchsize)
    
        self.srm = SRMConv2D()

    def forward(self, img, t_DCT_vol, t_quant_table, y, meta):  # x :(batch_size*C*H*W)->[64, 1, 256, 256]     y:[64, N]    meta(3,529)

        noise = self.srm(img)

        dct_feature = self.dct_stream(t_DCT_vol, t_quant_table)  #(22,512,32,32)
        dct_features = self.feature_extractor_dct(dct_feature)  # (batch_size, C, H, W)->(b, 128, 32, 32)

        Noise = self.feature_extractor_noise(noise)  # (b,64,256,256)
        skip1 = F.adaptive_max_pool2d(Noise, (32, 32))  # (b,64,32,32)
        Noise = self.resblock1(Noise)  # (b,64,256,256)
        skip2 = F.adaptive_max_pool2d(Noise, (32, 32))  # (b, 64, 32, 32)
        Noise = F.max_pool2d(Noise, kernel_size=(2, 2), stride=2, ceil_mode=True)  # (b, 64,128, 128)
        Noise = self.resblock2(Noise)  # (b, 64,128, 128)
        skip3 = F.adaptive_max_pool2d(Noise, (32, 32))  # (b, 64, 32, 32)
        Noise = self.resblock3(Noise)  # (b, 64,128, 128)
        Noise = F.max_pool2d(Noise, kernel_size=(2, 2), stride=2, ceil_mode=True)  # (b,64,64,64)
        skip4 = F.adaptive_max_pool2d(Noise, (32, 32))  # (b,64,32,32)
        Noise = self.resblock4(Noise)  # (b, 64,32, 32)
        Noise = F.max_pool2d(Noise, kernel_size=(2, 2), stride=2, ceil_mode=True)  # (b,64,32,32)
        Noise = torch.cat((Noise, skip1, skip2, skip3, skip4), dim=1)  # (b,320,32,32)
        Noise = self.get_Features_words_noise(Noise)  # (b, 64, 32, 32)
        Noise_se_weight = self.SE_Layer(Noise)  # (b,num_words,1,1)
        Noise = Noise * Noise_se_weight

        total_feature = torch.cat([dct_features, Noise], dim=1) 
        total_feature = self.feature_extractor_total_feature(total_feature)
        total_feature = total_feature.view(total_feature.shape[0], 64, 1024)  #(22,64,1024)


        meta = torch.unsqueeze(meta, 1).repeat(1, 64, 1)  # (22,64,160)
        total_feature = torch.cat([meta, total_feature], dim=2)  # (22,64,1184)
        total_feature = torch.unsqueeze(total_feature.transpose(1, 2), 3)  
        total_feature = total_feature.squeeze(3)  # (64,512,64)
        outputs, enc_outputs = self.transformer(total_feature.transpose(1, 2) + self.pos_embedding, y)  

        return outputs, enc_outputs


def build_SND(args):
    # Transformer Parameters
    global d_model, d_ff, d_k, d_v, n_layers, n_heads, mode, dropout, num_words, out_num, feature_dim, num_networks
    mode=8
    d_model = 512  # Embedding Size
    d_ff = args.d_ff  # FeedForward dimension
    d_k = args.d_k  # dimension of K(=Q), V
    d_v = args.d_v
    feature_dim = args.feature_dim
    n_layers = args.n_layers  # number of Encoder of Decoder Layer
    n_heads = args.n_heads  # number of heads in Multi-Head Attention
    dropout = args.dropout
    # mode = args.mode
    num_words = args.num_words
    num_networks = args.num_networks
    model = SND(args.input_channel, num_words, feature_dim)
    return model
