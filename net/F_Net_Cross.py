import torch
import torch.nn as nn
from einops import rearrange
import numbers
import torch.nn.functional as F
from crossatt import BidirectionalCrossAttention

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[2, 2],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])


    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        return  out_enc_level1


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)


class INN(nn.Module):
    def __init__(self):
        super(INN , self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2


    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2



class Restormer_Block(nn.Module):
    def __init__(self,
                 fea_channels,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Restormer_Block , self).__init__()

        self.channels = fea_channels
        self.restormer_model =  nn.Sequential(
            *[TransformerBlock(dim=self.channels, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)])
    def forward(self , x):
        x_fea = self.restormer_model(x)

        return x_fea





class Restormer_Blocks(nn.Module):
    def __init__(self,
                 fea_channels,
                 dim=64,
                 num_blocks=[2, 2],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Restormer_Blocks , self).__init__()

        self.channels = fea_channels
        self.restormer_model =  nn.Sequential(
            *[TransformerBlock(dim=self.channels, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)for i in range(num_blocks[0])])
    def forward(self , x):
        x_fea = self.restormer_model(x)

        return x_fea





class Fusion_INN_Block(nn.Module):
    def __init__(self, fea_channels, hsi_channels):
        super(Fusion_INN_Block, self).__init__()
        self.dim = fea_channels
        self.inn_1 = INN()
        self.inn_2 = INN()
        self.inn_3 = INN()
        self.hsi_even_msi = nn.Conv2d(fea_channels , self.dim , kernel_size=1 ,padding=0, bias=True)
        self.hsi_odd_msi = nn.Conv2d(fea_channels , self.dim , kernel_size=1 ,padding=0, bias=True)
        self.conv_1 = nn.Conv2d(fea_channels * 2 , self.dim , kernel_size=1 , padding=0 , bias=True)

        self.conv_2 = nn.Conv2d(self.dim  , self.dim , kernel_size=3 , padding=1 , bias=True)
        self.lrelu = nn.LeakyReLU(0.1 , inplace=True)
        self.conv_3 = nn.Conv2d(self.dim , hsi_channels , kernel_size=1 , padding=0 , bias=True)

    def forward(self, hsi_fea , msi_fea):

        hsi_even , hsi_odd = hsi_fea[: , : : 2 , : , :] , hsi_fea[: , 1 : : 2 , : , :]
        # msi_even , msi_odd = msi_fea[: , : : 2 , : , :] , msi_fea[: , 1 : : 2 , : , :]
        msi_1 , msi_2 = msi_fea[: , : msi_fea.shape[1] // 2 , : , :] , msi_fea[: , msi_fea.shape[1] // 2 : msi_fea.shape[1] , : , :]

        #hsi_even + msi_1
        inn_1_1 , inn_1_2 = self.inn_1(hsi_even , msi_1)
        inn_1_fea = torch.cat((inn_1_1 , inn_1_2) , dim=1)
        inn_1_fea = self.hsi_even_msi(inn_1_fea)
        inn_1_fea_hsi = inn_1_fea + hsi_fea   # res

        #hsi_odd + msi_2
        inn_2_1 , inn_2_2 = self.inn_2(hsi_odd , msi_2)
        inn_2_fea = torch.cat((inn_2_1 , inn_2_2) , dim=1)
        inn_2_fea = self.hsi_odd_msi(inn_2_fea)
        inn_2_fea_hsi = inn_2_fea + hsi_fea

        #inn_1_fea_hsi  + inn_2_fea_hsi
        inn_1_cat_2 = torch.cat((inn_1_fea_hsi , inn_2_fea_hsi) , dim=1)


        inn_1_cat_2 = self.conv_1(inn_1_cat_2)

        #最后一个INN

        z1 , z2 = inn_1_cat_2[: , : inn_1_cat_2.shape[1] // 2] , inn_1_cat_2[: ,  inn_1_cat_2.shape[1] // 2 : inn_1_cat_2.shape[1]]

        inn_3_1 , inn_3_2 = self.inn_3(z1 , z2)

        inn_3_fea = torch.cat((inn_3_1 , inn_3_2) , dim=1)
        # print('inn: ', inn_3_fea.shape)
        fusion_fea = self.lrelu(self.conv_2(inn_3_fea))

        res = self.conv_3(fusion_fea)

        return res

class Conv_Block(nn.Module):
    def __init__(self , fea_channels):
        super(Conv_Block , self).__init__()
        # self.conv = nn.Conv2d(fea_channels, msi_channels, 3, padding=1)
        # self.leaky = nn.LeakyReLU()
        self.conv_block = nn.Sequential(
            nn.Conv2d(fea_channels, fea_channels, 3, padding=1 , bias=True),
            nn.LeakyReLU()
        )
    def forward(self , msi_fea):
        msi_ = self.conv_block(msi_fea)

        return msi_




class Fusion_Net(nn.Module):
    def __init__(self , hsi_channels , msi_channels):
        super(Fusion_Net, self).__init__()
        fea_channels = 64 ##
        dim = 32  ## CAVE Harvard
        # dim = 64  ## Chikusei


        self.hsi_init = Restormer_Encoder(hsi_channels)
        self.msi_init = Restormer_Encoder(msi_channels)

        ## CAVE HAVARD
        
        self.restormer_block_1 = Restormer_Blocks(fea_channels)
        self.restormer_block_2 = Restormer_Blocks(dim * 2)
        self.restormer_block_3 = Restormer_Blocks(dim * 2)


        

        self.hsi_conv_1x1 = nn.Conv2d(fea_channels * 2 , fea_channels , kernel_size=1 , bias=True)
        self.hsi_conv_1x1_2 = nn.Conv2d(dim * 2  , fea_channels  , kernel_size=1 , bias=True)
        self.hsi_conv_1x1_3 = nn.Conv2d(fea_channels * 2  , fea_channels  , kernel_size=1 , bias=True)
        self.hsi_conv_1x1_4 = nn.Conv2d(dim * 2  , fea_channels  , kernel_size=1 , bias=True)

        self.conv_64_to_32 = nn.Conv2d(fea_channels , dim , kernel_size=1, padding=0 , bias=True)
        self.conv_block_1 = Conv_Block(dim)
        self.conv_block_2 = Conv_Block(dim * 2)
        self.conv_block_3 = Conv_Block(dim * 2)

        self.msi_conv_1x1 = nn.Conv2d(dim * 3 , dim * 2 , kernel_size=1 , bias=True)
        self.msi_conv_1x1_2 = nn.Conv2d(dim * 3 , dim * 2  , kernel_size=1 , bias=True)
        self.msi_conv_1x1_3 = nn.Conv2d(dim * 2 , dim * 2  , kernel_size=1 , bias=True)

        ##Cross Att
        self.Dual_Att_1 = BidirectionalCrossAttention(dim * 2 , dim * 2 , 4)
        self.Dual_Att_2 = BidirectionalCrossAttention(dim * 2 , dim * 2 , 4)
        self.Dual_Att_3 = BidirectionalCrossAttention(dim * 2 , dim * 2 , 4)

        self.hsi_to_64 = nn.Conv2d(fea_channels * 3 , fea_channels , kernel_size=1 , padding=0 , bias=True)
        self.msi_to_64 = nn.Conv2d(fea_channels * 3 , fea_channels , kernel_size=1 , padding=0 , bias=True)
        ############


        


        self.fusion_inn_block = Fusion_INN_Block(fea_channels , hsi_channels)


    def forward(self , hsi , msi):
        hsi_init_fea = self.hsi_init(hsi)
        msi_init_fea = self.msi_init(msi)


        hsi_fea_1 = self.restormer_block_1(hsi_init_fea)
        # hsi_fea_1 = self.CA_1(hsi_fea_1)



        hsi_init_cat_1 = torch.cat((hsi_init_fea , hsi_fea_1) , dim=1)
        hsi_init_cat_1 = self.hsi_conv_1x1(hsi_init_cat_1) ##  C: 64


        msi_fea_1 = self.conv_64_to_32(msi_init_fea)
        msi_fea_1 = self.conv_block_1(msi_fea_1) ## C: 32

        msi_init_cat_1 = torch.cat((msi_init_fea , msi_fea_1) , dim=1)   ## C:96
        msi_init_cat_1 = self.msi_conv_1x1(msi_init_cat_1)  ##  C:64


        # hsi_cat_msi_1 = torch.cat((hsi_init_cat_1 , msi_init_cat_1) , dim=1)
        hsi_cat_msi_1 = self.Dual_Att_1(hsi_init_cat_1 , msi_init_cat_1)
        hsi_cat_msi_1 = self.hsi_conv_1x1_2(hsi_cat_msi_1)   ##  C:64

        hsi_fea_2 = self.restormer_block_2(hsi_cat_msi_1)
        # hsi_fea_2 = self.CA_2(hsi_fea_2)
        hsi_1_cat_2 = torch.cat((hsi_fea_1 , hsi_fea_2) , dim=1)
        hsi_1_cat_2 = self.hsi_conv_1x1_3(hsi_1_cat_2)  ##  C:64

        msi_fea_2 = self.conv_block_2(msi_init_cat_1)   ## C:32
        msi_1_cat_2 = torch.cat((msi_fea_1 , msi_fea_2) , dim=1)   ## C : 96

        msi_1_cat_2 = self.msi_conv_1x1_2(msi_1_cat_2) ## C:32

        # hsi_cat_msi_2 = torch.cat((hsi_1_cat_2 , msi_1_cat_2) , dim=1)
        hsi_cat_msi_2 = self.Dual_Att_2(hsi_1_cat_2 , msi_1_cat_2)
        hsi_cat_msi_2 = self.hsi_conv_1x1_4(hsi_cat_msi_2)   ##    C：64


        hsi_fea_3 = self.restormer_block_3(hsi_cat_msi_2)   ## C: 64
        # hsi_fea_3 = self.CA_3(hsi_fea_3)
        hsi_2_cat_3 = torch.cat((hsi_fea_2 , hsi_fea_3) , dim=1)   ## C:128
        hsi_2_cat_3_cat_init = torch.cat((hsi_2_cat_3 , hsi_init_fea) , dim=1)   ## C:192


        msi_fea_3 = self.conv_block_3(msi_1_cat_2)
        msi_2_cat_3 = torch.cat((msi_fea_2 , msi_fea_3) , dim=1)   # C : 64
        msi_2_cat_3_cat_init = torch.cat((msi_2_cat_3 , msi_init_fea) , dim=1)  ## C: 128


        # print('hsi_2_cat_3_cat_init , msi_2_cat_3_cat_init: ' , hsi_2_cat_3_cat_init.shape , msi_2_cat_3_cat_init.shape)
        # hsi_2_cat_3_cat_init = self.hsi_to_64(hsi_2_cat_3_cat_init)
        hsi_2_cat_3_cat_init = self.hsi_to_64(hsi_2_cat_3_cat_init)
        msi_2_cat_3_cat_init = self.msi_to_64(msi_2_cat_3_cat_init)
        hsi_2_cat_3_cat_init = self.Dual_Att_3(hsi_2_cat_3_cat_init, msi_2_cat_3_cat_init)
        # hsi_cat_msi_3 = torch.cat((hsi_2_cat_3_cat_init , msi_2_cat_3_cat_init) , dim=1)
        hsi_inn_fea = hsi_2_cat_3_cat_init
        msi_inn_fea = msi_2_cat_3_cat_init
        # print('reduce: ', hsi_inn_fea.shape, msi_inn_fea.shape)

        fusion_result = self.fusion_inn_block(hsi_inn_fea , msi_inn_fea)
        output = fusion_result + hsi

        return output

if __name__ == '__main__':
    hsi = torch.randn((4, 31, 128, 128))
    msi = torch.randn((4, 3, 128, 128))
    f_net = Fusion_Net(31, 3)
    out = f_net(hsi, msi)
    print(out.shape)
    # input = torch.randn((2 , 64 , 128 , 128))
    # z1 , z2 = input[: , : input.shape[1] // 2] , input[: , input.shape[1] // 2 : input.shape[1]]
    # print(z1.shape)
    # inn = INN()
    # # print(out[1][0].shape, out[1][1].shape)
    # out = inn(input)
    # print(out[0].shape , out[1].shape)
