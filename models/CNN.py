__all__ = ['CNN']

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pack_sequence,pad_sequence
#from crf import CRF
import torch
#from dilated_conv import *
from torch import Tensor, add, xlogy_
import torch.nn.functional as F
from torch.nn import init
import itertools
import math
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        dim_feedforward=d_model*4
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
class TransformerEncoder(nn.Module):
    def __init__(self, d_model,num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=3)
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        #print(output.shape)

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class ChannelNorm(nn.Module):
    '''https://github.com/facebookresearch/CPC_audio/blob/master/cpc/model.py
    '''
    def __init__(self,
                 numFeatures,
                 epsilon=1e-05,
                 affine=True):

        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1,
                                                              numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean)*torch.rsqrt(cumVar + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x
def pad_after_conv(inputs,seq_size,device):
  if inputs.shape[-1]!=seq_size:
    pad = torch.zeros((inputs.shape[0],inputs.shape[1],seq_size-inputs.shape[-1])).to(device)
    inputs = torch.cat((inputs,pad),-1) 
  return inputs        
#Always edit this version!
class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool1d(1)
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        self.se=nn.Sequential(
            nn.Conv1d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv1d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class ResidualAttention(nn.Module):

    def __init__(self, channel=256 ,num_class=2,la=0.1):
        super().__init__()
        self.la=la
        self.fc=nn.Conv1d(in_channels=channel,out_channels=num_class,kernel_size=1,stride=1,bias=False)

    def forward(self, x):
        b,c,h=x.shape
        #print("inside resiudal attention:",x.shape)
        y_raw=self.fc(x) #Bx 2 x L
        #print(y_raw.shape)
        y_avg=torch.mean(y_raw,dim=1) #b,num_class
        y_max=torch.max(y_raw,dim=1)[0] #b,num_class
        score=y_avg+self.la*y_max
        #print(score.shape)
        score = nn.Sigmoid()(score.unsqueeze(1))
        return score


class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=1):
        super().__init__()
        self.conv=nn.Conv1d(2,1,kernel_size=1,padding=0)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,inds=torch.max(x,dim=1,keepdim=True)
        max_result = (max_result*2)**2
        avg_result=torch.mean(x,dim=1,keepdim=True)
        #min_result,_=torch.min(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        
        output=self.conv(result)
        #print(output.shape)
        output=self.sigmoid(output)
        return output
class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=2,kernel_size=1):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)
        #self.sa = ResidualAttention(channel = channel, num_class=2,la=0.2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _ = x.size()
        residual=x
        out=x*self.sa(x)
        out=out*self.ca(out)
        return out+residual

class Conv_Layer(nn.Module):
    def __init__(self,
                 input_c: int,
                 output_c: int,
                 seq_size: int,
                 drop_rate: float,
                 ):
        super(Conv_Layer, self).__init__()
        
        self.drop_rate = drop_rate
        
        #self.cbam = SpatialAttention(1)
        self.cbam = ResidualAttention(output_c)
        #self.cbam = CBAMBlock(channel = output_c,reduction = 4,kernel_size =1)
        self.add_module("norm1", ChannelNorm(input_c))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv1d(in_channels=input_c,
                                           out_channels=output_c,
                                           kernel_size=1,
                                           stride=1,
                                           padding = int((1-1)/2),
                                           bias=False))
        #out1_size = seq_size-4+1
        self.add_module("norm2", ChannelNorm(output_c))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv1d(output_c,
                                           output_c,
                                           kernel_size=2,
                                           stride=1,
                                           padding=int((2-1)/2),
                                           bias=False))
        #self.add_module("norm3", nn.BatchNorm1d(output_c*2))
        self.add_module("norm3", ChannelNorm(output_c))
        self.add_module("relu3", nn.ReLU(inplace=True))

        self.add_module("conv3", nn.Conv1d(output_c,
                                           output_c,
                                           kernel_size=3,
                                           stride=1,
                                           padding=int((3-1)/2),
                                           bias=False))
        #self.add_module("norm4", nn.BatchNorm1d(output_c*3))
        self.add_module("norm4", ChannelNorm(input_c))
        self.add_module("relu4", nn.ReLU(inplace=True))
        self.add_module("conv4", nn.Conv1d(input_c,
                                           output_c,
                                           kernel_size=4,
                                           stride=1,
                                           padding=int((4-1)/2),
                                           bias=False))
        #self.add_module("norm5", nn.BatchNorm1d(output_c*4))
        self.add_module("norm5", ChannelNorm(output_c))
        self.add_module("relu5", nn.ReLU(inplace=True))
        self.add_module("conv5", nn.Conv1d(output_c,
                                           output_c,
                                           kernel_size=5,
                                           stride=1,
                                           padding=int((5-1)/2),
                                          bias=False))

        
        
        #self.add_module("norm6", nn.BatchNorm1d(output_c*5))
        self.add_module("norm6", ChannelNorm(output_c))
        self.add_module("relu6", nn.ReLU(inplace=True))
        self.add_module("conv6", nn.Conv1d(output_c,
                                           output_c,
                                           kernel_size=6,
                                           stride=1,
                                           padding=int((6-1)/2),
                                           bias=False))
        #self.add_module("norm7", nn.BatchNorm1d(output_c*6))
        self.add_module("norm7", ChannelNorm(output_c))
        self.add_module("relu7", nn.ReLU(inplace=True))
        self.add_module("conv7", nn.Conv1d(output_c,
                                           output_c,
                                           kernel_size=7,
                                           stride=1,
                                           padding=int((7-1)/2),
                                           bias=False))
        #self.add_module("norm8", nn.BatchNorm1d(output_c*7))
        self.add_module("norm8", ChannelNorm(output_c))
        self.add_module("relu8", nn.ReLU(inplace=True))
        self.add_module("conv8", nn.Conv1d(output_c,
                                           output_c,
                                           kernel_size=8,
                                           stride=1,
                                           padding=int((8-1)/2),
                                           bias=False))
        #self.add_module("norm9", ChannelNorm(output_c))
        #self.add_module("relu9", nn.ReLU(inplace=True))
        #self.add_module("conv9", nn.Conv1d(output_c,
        #                                   output_c,
        #                                   kernel_size=9,
        #                                   stride=1,
        #                                   padding=int((9-1)/2),
        #                                   bias=False))
        
        
        
        self.seq_size = seq_size
        
        self.spatial_attention  = ResidualAttention(output_c)
        
        
    
    def attention(self,attention_list):
        #attention = torch.stack(attention_list).transpose(-2,-1)
        S=[]
        seq_length = self.seq_size
        #print("+++++++++++++++++",seq_length)
        device = attention_list[0].get_device()
        pad_attention_list = []
        for X_l in attention_list:
            #print(type(X_l))
            X_l = X_l.transpose(2,1)#B X L X E
            #print(X_l.shape)
            #S_l = X_l.sum(dim = -1).unsqueeze(-1)
            S_l = self.spatial_attention(X_l.transpose(2,1)).transpose(2,1).squeeze().unsqueeze(-1)
            #print("before padding sl:",S_l.shape,X_l.shape)
            if X_l.shape[1]!=seq_length:
                pad_size = seq_length-X_l.shape[1]
                pad_tensor = torch.zeros(X_l.shape[0],pad_size,X_l.shape[2]).to(device)
                
                X_l = torch.cat((X_l,pad_tensor),1)
                pad_l = torch.zeros(S_l.shape[0],pad_size,S_l.shape[2]).to(device)
                #print("pad_l",pad_l.shape)
                S_l = torch.cat((S_l,pad_l),1)
                #S_l=self.atten_conv(S_l)
                #print("after padding:",S_l.shape)
            pad_attention_list.append(X_l)
            #print("sl:",S_l.shape)#1xB X L X1
            S.append(S_l)
        ensem = torch.cat(S,-1)# BxLx8, 8=#dense layers
        
        alpha= F.softmax(ensem,dim=-1)
        #print(torch.sum(alpha,dim=-1))
        for i,X_l in enumerate(pad_attention_list):
            #print(X_l.shape,alpha[:,:,i].shape)#BXLXE
            if i==0:
                final_score = X_l*alpha[:,:,i][...,None] #BxLxE x BxLx1
            else:
                final_score+=X_l*alpha[:,:,i][...,None]
            #print(final_score.shape)    
        #print("return by attention layer:", final_score.shape)
        return final_score
    
    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs.transpose(2,1)
        seq_size = inputs.shape[-1]
        device = inputs.get_device()
        '''
        new_features1 = self.conv1(self.relu1(self.norm1(inputs)))
        #print(new_features1.shape)
        new_features1 = new_features1*(1+self.cbam(new_features1))
        new_features1 = pad_after_conv(new_features1,seq_size,device)
        #print("after pad",new_features1.shape)

        new_features2 = self.conv2(self.relu2(self.norm2(new_features1)))
        new_features2 = new_features2*(1+self.cbam(new_features2))
        new_features2 = pad_after_conv(new_features2,seq_size,device)

        
        # Dense connection
        #new_features = torch.cat([new_features1,new_features2],1)

        new_features3 = self.conv3(self.relu3(self.norm3(new_features2)))
        
        new_features3 = new_features3*(1+self.cbam(new_features3))
        new_features3 = pad_after_conv(new_features3,seq_size,device)
        '''
        # Dense connection
        #new_features = torch.cat([new_features1,new_features2,new_features3],1)
        #print(new_features3.shape,self.norm4)
        new_features4 = self.conv4(self.relu4(self.norm4(inputs)))
        new_features4 = new_features4*(1+self.cbam(new_features4))
        new_features4 = pad_after_conv(new_features4,seq_size,device)
        # Dense connection
        #new_features = torch.cat([new_features1,new_features2,new_features3,new_features4],1)

        new_features5 = self.conv5(self.relu5(self.norm5(new_features4)))
        new_features5 = new_features5*(1+self.cbam(new_features5))
        new_features5 = pad_after_conv(new_features5,seq_size,device)
        
        # Dense connection
        #new_features = torch.cat([new_features1,new_features2,new_features3,new_features4,new_features5],1)
        
        new_features6 = self.conv6(self.relu6(self.norm6(new_features5)))
        new_features6 = new_features6*(1+self.cbam(new_features6))
        new_features6 = pad_after_conv(new_features6,seq_size,device)
        # Dense connection
        #new_features = torch.cat([new_features1,new_features2,new_features3,new_features4,new_features5,new_features6],1)

        new_features7 = self.conv7(self.relu7(self.norm7(new_features6)))
        new_features7 = new_features7*(1+self.cbam(new_features7))
        new_features7 = pad_after_conv(new_features7,seq_size,device)
        # Dense connection
        #new_features = torch.cat([new_features1,new_features2,new_features3,new_features4,new_features5,new_features6,new_features7],1)

        new_features8 = self.conv8(self.relu8(self.norm8(new_features7)))
        new_features8 = new_features8*(1+self.cbam(new_features8))
        new_features8 = pad_after_conv(new_features8,seq_size,device)
        
        #new_features9 = self.conv9(self.relu9(self.norm9(new_features8)))
        #new_features9 = new_features9*(1+self.cbam(new_features9))
        #new_features9 = pad_after_conv(new_features9,seq_size,device)


        
        new_features = new_features8
        #print(new_features8.shape)
        #new_features = [new_features4.unsqueeze(0),new_features5.unsqueeze(0),new_features6.unsqueeze(0),new_features7.unsqueeze(0),new_features8.unsqueeze(0)]
        
        #new_features = torch.cat(new_features, 0)
        #print("5:",new_features.shape)
        #new_features =self.lin_attention(new_features)
        #print(score.shape,torch.cat(pad_list,1).shape)
        #new_features = score
        #print(new_features.shape)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                      p=self.drop_rate,
                                      training=self.training)

        return new_features.transpose(2,1) #growthrate*3


class Inception(nn.Module):
    def __init__(self, in_channels=12, num_init_features=32, ks1 = 3, ks2 = 5, ks3 = 7, ks4 = 9):
        super(Inception, self).__init__()

        self.branch1 = nn.Conv1d(in_channels, num_init_features, kernel_size=ks1, stride=1, padding=int((ks1-1)/2))
        self.branch2 = nn.Conv1d(in_channels, num_init_features, kernel_size=ks2, stride=1, padding=int((ks2-1)/2))
        self.branch3 = nn.Conv1d(in_channels, num_init_features, kernel_size=ks3, stride=1, padding=int((ks3-1)/2))
        #self.branch4 = nn.Conv1d(in_channels, num_init_features, kernel_size=ks4, stride=1, padding=int((ks4-1)/2))
        #self.maxpool = nn.MaxPool1d(3,stride = 2,padding =1)
        self.relu = nn.ReLU(inplace=True)
        #self.cbam = SpatialAttention(num_init_features,3)
        self.cbam = ResidualAttention(num_init_features)
        #self.cbam = CBAMBlock(num_init_features,4,1)
    def forward(self, x):
        seq_size = x.shape[-1]
        device = x.get_device()
        branch1 = self.branch1(x)
        branch1 = self.relu(branch1)
        branch1 = branch1*(1+self.cbam(branch1))
        #branch1 = self.cbam(branch1)
        #branch1 = self.maxpool(branch1.transpose(2,1)).transpose(2,1)
        
        if branch1.shape[-1]!=seq_size:
          pad = torch.zeros((branch1.shape[0],branch1.shape[1],seq_size-branch1.shape[-1])).to(device)
          #print(pad.size,branch1.size)
          branch1 = torch.cat((branch1,pad),-1)
          
        branch2 = self.branch2(x)
        branch2 = self.relu(branch2)
        branch2 = branch2*(1+self.cbam(branch2))
        #branch2 = self.cbam(branch2)
        #branch2 = self.maxpool(branch2.transpose(2,1)).transpose(2,1)
        if branch2.shape[-1]!=seq_size:
          pad = torch.zeros((branch2.shape[0],branch2.shape[1],seq_size-branch2.shape[-1])).to(device)
          branch2 = torch.cat((branch2,pad),-1)

        branch3 = self.branch3(x)
        branch3 = self.relu(branch3)
        branch3 = branch3*(1+self.cbam(branch3))
        #branch3 = self.cbam(branch3)
        #branch3 = self.maxpool(branch3.transpose(2,1)).transpose(2,1)
        if branch3.shape[-1]!=seq_size:
          pad = torch.zeros((branch3.shape[0],branch3.shape[1],seq_size-branch3.shape[-1])).to(device)
          branch3 = torch.cat((branch3,pad),-1)
        '''
        branch4 = self.branch4(x)
        branch4 = self.relu(branch4)
        #branch4 = branch4*(1+self.cbam(branch4))
        #branch4 = self.cbam(branch4)
        #branch4 = self.maxpool(branch4.transpose(2,1)).transpose(2,1)
        if branch4.shape[-1]!=seq_size:
          pad = torch.zeros((branch4.shape[0],branch4.shape[1],seq_size-branch4.shape[-1])).to(device)
          branch4 = torch.cat((branch4,pad),-1)
        '''
        #print(branch1.shape,branch2.shape,branch3.shape,branch4.shape)
        outputs = [branch1, branch2, branch3]
        outputs = F.max_pool1d(torch.cat(outputs, 1).transpose(2,1),1).transpose(2,1)

        return outputs #return num_init_features*3
    

class Model(nn.Module):
    def __init__(self,configs,weight = False):
        super(Model, self).__init__()
       
        self.conv_layer_seq = Conv_Layer(3,64,310,0.1)#96,128
        #num_patch = 1
        self.transformer = TransformerEncoder(3,num_layers=2)
        #self.causal = TSEncoder(input_dims=12,output_dims=64)
        #self.transformer = BertForGenomicSequenceClassification(num_classes=2)
        
        self.BiLSTM = nn.Sequential(nn.LSTM(input_size=64,hidden_size=32,num_layers=2,batch_first=True,bidirectional=True,bias=True,dropout=0.1))
        #https://github.com/xiaobaicxy/text-classification-BiLSTM-Attention-pytorch/blob/master/bilstm_attention.py
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(32, 32),#hidden size =64
            nn.ReLU(inplace=True)
        )
        
        #self.squeeze = nn.Conv1d(103,1,kernel_size=1)
        self.conv_last1 = nn.Conv1d(310,310,kernel_size=1)
        
        self.conv_embed = nn.Conv1d(68,68,kernel_size=1)
        
        self.classifier = nn.Linear(64, 2)
        #self.classifier = nn.Linear(256, 2)
        #self.att_bais=nn.Parameter(torch.rand(1))
        self.init_weights()
        #self.layers = nn.ModuleList([Self_Attention(hidden_size=64) for _ in range(3)])
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight,mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, onehot ,mask = False):
        
        #ss =ss.unsqueeze(-1)
        #onehot = onehot.transpose(0,1)
        #embeddings = self.reduction(embeddings.transpose(2,1)).transpose(2,1)
        
        if mask ==True:
          cnn_output = generate_binomial_mask(jitter(cnn_output)).to(onehot.device)
        #cnn_output = onehot.transpose(0,1)
        #print(onehot.shape)
        cnn_output = self.conv_layer_seq(onehot)
        #cnn_output = self.transformer(onehot)
        
        
        #print("after conv layer: ", cnn_output.shape)
        
        
        bilstm_out, (h_n,c_n) = self.BiLSTM(cnn_output)
        #print("bilstm output: ",bilstm_out.shape)
        
        
        (forward_out, backward_out) = torch.chunk(bilstm_out, 2, dim = 2)
        out = forward_out + backward_out  #[seq_len, batch, hidden_size]
        #out = out.permute(1, 0, 2)  #[batch, seq_len, hidden_size]
        #print("after transpose",out.shape)
        h_n = h_n.permute(1, 0, 2)  #[batch, num_layers * num_directions,  hidden_size]
        h_n = torch.sum(h_n, dim=1) #[batch, 1,  hidden_size]
        h_n = h_n.squeeze(dim=1)  #[batch, hidden_size]
        
        attention_w = self.attention_weights_layer(h_n) #h_n #[batch, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1) #[batch, 1, hidden_size]
        attention_context = torch.bmm(attention_w, out.transpose(2,1))  #[batch, 1, seq_len]
        softmax_w = F.softmax(attention_context, dim=-1)  #[batch, 1, seq_len],权重归一化
        
        x = torch.bmm(softmax_w, bilstm_out)#out  #[batch, 1, hidden_size]
        x = x.squeeze(dim=1)#128x64
        
        #x = torch.cat([x,all_freqs],-1)
        
        
        #x = x.transpose(-1,-2)
        #x = torch.cat((x,freqs),-1)
        
        #x = F.relu(cnn_output, inplace=True)
        #x = torch.flatten(F.adaptive_avg_pool1d(x, 64),1)
        #print(x.shape)
        
        #embeddings = self.transformer(embeddings)
        #print(embeddings.shape,cnn_output.shape)
        #cnn_output = torch.cat([cnn_output,embeddings],dim = -1)
        #cnn_output = self.conv_embed(cnn_output.transpose(2,1)).transpose(2,1)
        '''
        x = F.adaptive_avg_pool1d(cnn_output, 1).squeeze()
        x=x.transpose(-2,-1)
        #print(x.shape)
        x = self.conv_last1(x)
        #x = self.conv_last2(x)
        #print(x.shape)
        x = x.transpose(-2,-1)
        '''
        #print(pred.shape)
        #x = torch.sum(cnn_output,dim = -1
        #x = x.transpose(-1,-2)
        #x = cnn_output[:,-1,:]
        fea = F.normalize(x,dim=1)
        #x = torch.flatten(x, 1)
        #print("before classifier",x.shape)#182x82x128,128x84
        pred = self.classifier(x)
        #pred = F.sigmoid(pred)
        #print("after classifier",pred.shape)
        
        #return fea,pred
        
        return pred