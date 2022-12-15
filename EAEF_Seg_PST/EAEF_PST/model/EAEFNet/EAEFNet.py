import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class Atttion_avg_pool(nn.Module):
    def __init__(self, dim, reduction):
        super(Atttion_avg_pool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Feature_Pool, self).__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Linear(dim, dim // ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim // ratio, dim)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.up(self.act(self.down(self.gap_pool(x).permute(0,2,3,1)))).permute(0,3,1,2).view(b,c)
        return y

class Channel_Attention(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention, self).__init__()
        self.gap_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim//ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim//ratio, dim)
    def forward(self, x):
        max_out = self.up(self.act(self.down(self.gap_pool(x).permute(0,2,3,1)))).permute(0,3,1,2)
        return max_out

class Spatial_Attention(nn.Module):
    def __init__(self, dim):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1,bias=True)
    def forward(self, x):
        x1 = self.conv1(x)
        return x1

class EAEF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_pool_rgb = Feature_Pool(dim)
        self.mlp_pool_t = Feature_Pool(dim)
        self.dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=7,padding=3,groups=dim)
        self.cse = Channel_Attention(dim*2)
        self.sse_r = Spatial_Attention(dim)
        self.sse_t = Spatial_Attention(dim)
    def forward(self, RGB, T):
        ############################################################################
        b, c, h, w = RGB.size()
        rgb_y = self.mlp_pool_rgb(RGB)
        t_y = self.mlp_pool_t(T)
        rgb_y = rgb_y / rgb_y.norm(dim=1, keepdim=True)
        t_y = t_y / t_y.norm(dim=1, keepdim=True)
        rgb_y = rgb_y.view(b, c, 1)
        t_y = t_y.view(b, 1, c)
        logits_per = c * rgb_y @ t_y
        cross_gate = torch.diagonal(torch.sigmoid(logits_per)).reshape(b, c, 1, 1)
        add_gate = torch.ones(cross_gate.shape).cuda() - cross_gate
        ##########################################################################
        New_RGB_A = RGB * cross_gate
        New_T_A = T * cross_gate
        x_cat = torch.cat((New_RGB_A,New_T_A),dim=1)
        ##########################################################################
        fuse_gate = torch.sigmoid(self.cse(self.dwconv(x_cat)))
        rgb_gate, t_gate = fuse_gate[:, 0:c, :], fuse_gate[:, c:c * 2, :]
        ##########################################################################
        New_RGB = RGB * add_gate + New_RGB_A * rgb_gate
        New_T = T * add_gate + New_T_A * t_gate
        ##########################################################################
        New_fuse_RGB = self.sse_r(New_RGB)
        New_fuse_T = self.sse_t(New_T)
        attention_vector = torch.cat([New_fuse_RGB, New_fuse_T], dim=1)
        attention_vector = torch.softmax(attention_vector,dim=1)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        New_RGB = New_RGB * attention_vector_l + New_T * attention_vector_r
        New_T = New_T * attention_vector_r
        ##########################################################################
        return New_RGB, New_T

class MCDropout(nn.Module):
    def __init__(self, p: float = 0.5, force_dropout: bool = False):
        super().__init__()
        self.force_dropout = force_dropout
        self.p = p

    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=self.training or self.force_dropout)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

#Global Contextual module
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = BasicConv2d(in_channel, out_channel, 1)
        self.branch1_1 = BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))
        self.branch1_2 = BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0))

        self.branch2 = BasicConv2d(in_channel, out_channel, 1)
        self.branch2_1 = BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2))
        self.branch2_2 = BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0))

        self.branch3 = BasicConv2d(in_channel, out_channel, 1)
        self.branch3_1 = BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3))
        self.branch3_2 = BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0))

        self.se = Atttion_avg_pool(out_channel,4)
        self.conv_res = nn.Conv2d(in_channel,out_channel,kernel_size=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x0 = self.se(x0)
        x1 = self.branch1_2(self.branch1_1(self.branch1(x)))
        x1 = self.se(x1)
        x2 = self.branch2_2(self.branch2_1(self.branch2(x)))
        x2 = self.se(x2)
        x3 = self.branch3_2(self.branch3_1(self.branch3(x)))
        x3 = self.se(x3)
        x_add = x0 + x1 + x2 + x3
        x = self.relu(x_add + self.conv_res(x))
        return x

#aggregation of the high-level(teacher) features
class aggregation_init(nn.Module):
    def __init__(self, channel,mode='None'):
        super(aggregation_init, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        #########################################################################################################
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.se = Atttion_avg_pool(3 * channel,4)
        if mode == 'out':
            self.conv = nn.Conv2d(3*channel, 5, 1)
        if mode == 'splat':
            self.conv = nn.Conv2d(3 * channel, 1, 1)
    def forward(self, x1, x2, x3):
        ##########################################################################################################
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3
        ##########################################################################################################
        x2_2 = self.conv_concat2(torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1))
        x3_2 = self.conv_concat3(torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1))
        x = self.conv(self.se(self.conv4(x3_2)))
        return x

#Refinement flow
class Refine(nn.Module):
    def __init__(self):
        super(Refine,self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, attention,x1,x2,x3):
        x1 = x1 + torch.mul(x1, self.upsample2(attention))
        x2 = x2 + torch.mul(x2, self.upsample2(attention))
        x3 = x3 + torch.mul(x3, attention)
        return x1, x2, x3
####################################################################################################
class FA_encoder(nn.Module):
    def __init__(self,dropout_rate: float = .09):
        super(FA_encoder, self).__init__()
        self.num_resnet_layers = 152
        if self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)
            self.inplanes = 2048
        ########  Thermal ENCODER  ########
        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),dim=1)
        self.encode_dropout = MCDropout(dropout_rate)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4
        ######## EAEF ################
        self.idam_0 = EAEF(64)
        self.idam_1 = EAEF(256)
        self.idam_2 = EAEF(512)
        self.idam_3 = EAEF(1024)
        self.idam_4 = EAEF(2048)
        ########  RGB ENCODER  ########
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        #增加MCD机制
        self.encode_dropout = MCDropout(dropout_rate)
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4
    def forward(self, input):
        rgb = input[:, :3]
        thermal = input[:, 3:]
        verbose = False
        ######################################################################
        if verbose: print("rgb.size() original: ", rgb.size())  # (480, 640)
        if verbose: print("thermal.size() original: ", thermal.size())  # (480, 640)
        ######################################################################
        # layer0
        ######################################################################
        rgb = self.encoder_rgb_conv1(rgb)
        if verbose: print("rgb.size() after conv1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_bn1(rgb)
        if verbose: print("rgb.size() after bn1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_relu(rgb)
        if verbose: print("rgb.size() after relu: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_maxpool(rgb)
        if verbose: print("rgb.size() after maxpool: ", rgb.size())  # (120, 160)
        thermal = self.encoder_thermal_conv1(thermal)
        if verbose: print("thermal.size() after conv1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_bn1(thermal)
        if verbose: print("thermal.size() after bn1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_relu(thermal)
        if verbose: print("thermal.size() after relu: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_maxpool(thermal)
        if verbose: print("thermal.size() after maxpool: ", thermal.size())  # (120, 160)
        ######################################################################
        rgb, thermal = self.idam_0(rgb, thermal)
        ######################################################################
        # layer1
        ######################################################################
        rgb_1 = self.encoder_rgb_layer1(rgb)
        if verbose: print("rgb.size() after layer1: ", rgb_1.size())  # (120, 160)
        thermal_1 = self.encoder_thermal_layer1(thermal)
        if verbose: print("thermal.size() after layer1: ", thermal_1.size())  # (120, 160)
        rgb_1, thermal_1= self.idam_1(rgb_1, thermal_1)
        ######################################################################
        # layer2
        ######################################################################
        rgb_2_1 = self.encoder_rgb_layer2(rgb_1)
        if verbose: print("rgb.size() after layer2: ", rgb_2_1.size())  # (60, 80)
        thermal_2 = self.encoder_thermal_layer2(thermal_1)
        if verbose: print("thermal.size() after layer2: ", thermal_2.size())  # (60, 80)
        rgb_2_1, thermal_2= self.idam_2(rgb_2_1, thermal_2)
        ######################################################################
        # layer3
        ######################################################################
        rgb_3_1 = self.encoder_rgb_layer3(rgb_2_1)
        if verbose: print("rgb.size() after layer3: ", rgb_3_1.size())  # (30, 40)
        thermal_3_1 = self.encoder_thermal_layer3(thermal_2)
        if verbose: print("thermal.size() after layer3: ", thermal_3_1.size())  # (30, 40)
        rgb_3_1, thermal_3_1= self.idam_3(rgb_3_1, thermal_3_1)
        ######################################################################
        # layer4
        ######################################################################
        rgb_4_1 = self.encoder_rgb_layer4(rgb_3_1)
        if verbose: print("rgb.size() after layer4: ", rgb_4_1.size())  # (15, 20)
        thermal_4_1 = self.encoder_thermal_layer4(thermal_3_1)
        if verbose: print("thermal.size() after layer4: ", thermal_4_1.size())  # (15, 20)
        rgb_4_1, thermal_4_1 = self.idam_4(rgb_4_1, thermal_4_1)
        ######################################################################
        return [rgb_4_1,rgb_3_1,rgb_2_1,rgb_1,rgb]

class TransBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out

class Turbo_decoder(nn.Module):
    def __init__(self,n_class=5, channel=64):
        super(Turbo_decoder, self).__init__()
        ########  DECODER  ########
        self.rfb2_1 = GCM(512, channel)
        self.rfb3_1 = GCM(1024, channel)
        self.rfb4_1 = GCM(2048, channel)
        self.rfb0_2 = GCM(64, channel)
        self.rfb1_2 = GCM(256, channel)
        self.rfb5_2 = GCM(512, channel)
        self.agg1 = aggregation_init(channel,mode='out')
        self.agg1_splat = aggregation_init(channel,mode='splat')
        self.HA = Refine()
        ######################################################################
        # upsample function
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ######################################################################
        # Components of PTM module
        self.inplanes = channel
        self.agant1 = self._make_agant_layer(channel, channel)
        self.deconv1 = self._make_transpose_layer(TransBottleneck, channel, 3, stride=2)
        self.inplanes = channel
        self.agant2 = self._make_agant_layer(channel, channel)
        self.deconv2 = self._make_transpose_layer(TransBottleneck, channel, 3, stride=2)
        ######################################################################
        # test3
        ######################################################################
        self.out2_conv = nn.Conv2d(channel, n_class, kernel_size=1)
        ######################################################################

    def _make_transpose_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes))
        for m in upsample.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))
        layers.append(block(self.inplanes, planes, stride, upsample))

        self.inplanes = planes
        return nn.Sequential(*layers)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def forward(self, x):
        ######################################################################
        rgb, rgb_1, rgb_2_1, rgb_3_1, rgb_4_1 = x[4],x[3],x[2],x[1],x[0]
        ######################################################################
        # produce initial saliency map by decoder1
        ######################################################################
        x2_1 = self.rfb2_1(rgb_2_1)
        ux2_1 = self.upsample2(x2_1)
        x3_1 = self.rfb3_1(rgb_3_1)
        ux3_1 = self.upsample4(x3_1)
        x4_1 = self.rfb4_1(rgb_4_1)
        ux4_1 = self.upsample(x4_1)
        attention_gate = torch.sigmoid(self.agg1_splat(x4_1, x3_1, x2_1))
        ##############################################################################
        x, x1, x5 = self.HA(attention_gate, rgb, rgb_1, rgb_2_1)
        x0_2 = self.rfb0_2(x)
        ux0_2 = x0_2
        x1_2 = self.rfb1_2(x1)
        ux1_2 = x1_2
        x5_2 = self.rfb5_2(x5)
        ux5_2 = self.upsample2(x5_2)
        ##############################################################################
        feature_map = ux5_2 + ux1_2 + ux2_1 + ux3_1 + ux0_2 + ux4_1
        ##############################################################################
        hight_output = self.upsample(self.agg1(x4_1, x3_1, x2_1))
        ##############################################################################
        # Refine low-layer features by initial map
        ##############################################################################
        # PTM module
        ##############################################################################
        y = feature_map
        y = self.agant1(y)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        y = self.out2_conv(y)
        ######################################################################
        return hight_output, y

class EAEFNet(nn.Module):
    def __init__(self,n_class):
        super(EAEFNet, self).__init__()
        self.FA_encoder = FA_encoder()
        self.Turbo_decoder = Turbo_decoder(n_class)
    def forward(self,x):
        encoder_input = self.FA_encoder(x)
        out,out_1 = self.Turbo_decoder(encoder_input)
        return out,out_1


def unit_test():
    num_minibatch = 1
    rgb = torch.randn(num_minibatch, 3, 480, 640).cuda(0)
    thermal = torch.randn(num_minibatch, 1, 480, 640).cuda(0)
    rtf_net = EAEFNet(5).cuda(0)
    input = torch.cat((rgb, thermal), dim=1)
    rtf_net(input)


if __name__ == '__main__':
    unit_test()