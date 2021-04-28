def to_np(ten):
    return ten.cpu().detach().numpy()

class CBA2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,act=nn.LeakyReLU(0.2,inplace=True)):
        super(CBA2d,self).__init__()
        assert kernel_size%2==1,"Only odd kernels allowed"
        padding=kernel_size//2
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
        self.norm=nn.BatchNorm2d(out_channels)
        self.act=act
    def forward(self,x):
        return self.act(self.norm(self.conv(x)))

class CBA3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,act=nn.LeakyReLU(0.2,inplace=True)):
        super(CBA3d,self).__init__()
        assert kernel_size%2==1,"Only odd kernels allowed"
        padding=kernel_size//2
        self.conv=nn.Conv3d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
        self.norm=nn.BatchNorm3d(out_channels)
        self.act=act
    def forward(self,x):
        return self.act(self.norm(self.conv(x)))

class ConvUNet32(nn.Module):
    def __init__(self,in_channels,out_channels,features=(16,48,80,112),out_act=lambda x:x):
        super(ConvUNet32,self).__init__()
        self.down=nn.MaxPool2d(kernel_size=2)
        self.up=nn.Upsample(scale_factor=2)
        self.cba_in1=CBA2d(in_channels,features[0],act=act)
        self.cba_in2=CBA2d(features[0],features[0],act=act)
        self.cba_d1_1=CBA2d(features[0],features[1],act=act)
        self.cba_d1_2=CBA2d(features[1],features[1],act=act)
        self.cba_d2_1=CBA2d(features[1],features[2],act=act)
        self.cba_d2_2=CBA2d(features[2],features[2],act=act)
        self.cba_bn_1=CBA2d(features[2],features[3],act=act)
        self.cba_bn_2=CBA2d(features[3],features[3],act=act)
        self.cba_u1_1=CBA2d(features[3]+features[2],features[2],act=act)
        self.cba_u1_2=CBA2d(features[2],features[2],act=act)
        self.cba_u2_1=CBA2d(features[2]+features[1],features[1],act=act)
        self.cba_u2_2=CBA2d(features[1],features[1],act=act)
        self.cba_out_1=CBA2d(features[1]+features[0]+in_channels,features[0],act=act)
        self.cba_out_2=CBA2d(features[0],features[0],act=act)
        self.conv_out=nn.Conv2d(features[0],out_channels,kernel_size=3,padding=1)
        self.out_act=out_act

    def forward(self,x):
        saves=[x]
        x=self.cba_in2(self.cba_in1(x))
        saves.append(x)
        self.down(x)
        x=self.cba_d1_2(self.cba_d1_1(x))
        saves.append(x)
        x=self.down(x)
        x=self.cba_d2_2(self.cba_d2_1(x))
        saves.append(x)
        x=self.down(x)
        x=self.cba_bn_2(self.cba_bn_1(x))
        x=self.up(x)
        x=torch.cat([x,saves[-1]],dim=1)
        x=self.cba_u1_2(self.cba_u1_1(x))
        x=self.up(x)
        x=torch.cat([x,saves[-2]],dim=1)
        x=self.cba_u2_2(self.cba_u2_1(x))
        x=torch.cat([x,saves[-3],saves[-4]],dim=1)
        x=self.cba_out_2(self.cba_out_1(x))
        return self.out_act(self.conv_out(x))

class ConvClassifier3d_smallz(nn.Module):
    def __init__(self,sh3d,in_channels,outdim,features=(16,48,80,112,144,176),act=nn.LeakyReLU(0.2,inplace=True)):
        super(ConvClassifier3d,self).__init__()

        assert np.prod(sh3d)%4096==0

        self.convsize=np.prod(sh3d)//4096

        self.down=nn.MaxPool3d(kernel_size=2)
        self.down_noz=nn.MaxPool3d(kernel_size=(2,2,1))

        self.act=act

        self.cba_in1=CBA3d(in_channels,features[0],act=act)
        self.cba_in2=CBA3d(features[0],features[0],act=act)
        self.cba_d1_1=CBA3d(features[0],features[1],act=act)
        self.cba_d1_2=CBA3d(features[1],features[1],act=act)
        self.cba_d2_1=CBA3d(features[1],features[2],act=act)
        self.cba_d2_2=CBA3d(features[2],features[2],act=act)
        self.cba_d3_1=CBA3d(features[2],features[3],act=act)
        self.cba_d3_2=CBA3d(features[3],features[3],act=act)
        self.cba_d4_1=CBA3d(features[3],features[4],act=act)
        self.cba_d4_2=CBA3d(features[4],features[4],act=act)
        self.cba_d5_1=CBA3d(features[4],features[5],act=act)
        self.cba_d5_2=CBA3d(features[5],features[5],act=act)

        self.lin_1=nn.Linear(self.convsize*features[5],256)
        self.lin_2=nn.Linear(256,128)
        self.lin_out=nn.Linear(128,outdim)


    def forward(self,x):
        b=x.size(0)
        x=self.cba_in2(self.cba_in1(x))
        x=self.down_noz(x)
        x=self.cba_d1_2(self.cba_d1_1(x))
        x=self.down(x)
        x=self.cba_d2_2(self.cba_d2_1(x))
        x=self.down_noz(x)
        x=self.cba_d3_2(self.cba_d3_1(x))
        x=self.down(x)
        x=self.cba_d4_2(self.cba_d4_1(x))
        x=self.down_noz(x)
        x=self.cba_d5_2(self.cba_d5_1(x))
        x=x.reshape(b,-1)
        x=self.act(self.lin_1(x))
        x=self.act(self.lin_2(x))
        return self.lin_out(x)

class ConvClassifier2d(nn.Module):
    def __init__(self,sh2d,in_channels,outdim,features=(16,48,80,112,144,176),act=nn.LeakyReLU(0.2,inplace=True)):
        super(ConvClassifier2d,self).__init__()

        assert np.prod(sh2d)%1024==0

        self.convsize=np.prod(sh2d)//1024

        self.down=nn.MaxPool2d(kernel_size=2)

        self.act=act

        self.cba_in1=CBA2d(in_channels,features[0],act=act)
        self.cba_in2=CBA2d(features[0],features[0],act=act)
        self.cba_d1_1=CBA2d(features[0],features[1],act=act)
        self.cba_d1_2=CBA2d(features[1],features[1],act=act)
        self.cba_d2_1=CBA2d(features[1],features[2],act=act)
        self.cba_d2_2=CBA2d(features[2],features[2],act=act)
        self.cba_d3_1=CBA2d(features[2],features[3],act=act)
        self.cba_d3_2=CBA2d(features[3],features[3],act=act)
        self.cba_d4_1=CBA2d(features[3],features[4],act=act)
        self.cba_d4_2=CBA2d(features[4],features[4],act=act)
        self.cba_d5_1=CBA2d(features[4],features[5],act=act)
        self.cba_d5_2=CBA2d(features[5],features[5],act=act)

        self.lin_1=nn.Linear(self.convsize*features[5],256)
        self.lin_2=nn.Linear(256,128)
        self.lin_out=nn.Linear(128,outdim)


    def forward(self,x):
        b=x.size(0)
        x=self.cba_in2(self.cba_in1(x))
        x=self.down(x)
        x=self.cba_d1_2(self.cba_d1_1(x))
        x=self.down(x)
        x=self.cba_d2_2(self.cba_d2_1(x))
        x=self.down(x)
        x=self.cba_d3_2(self.cba_d3_1(x))
        x=self.down(x)
        x=self.cba_d4_2(self.cba_d4_1(x))
        x=self.down(x)
        x=self.cba_d5_2(self.cba_d5_1(x))
        x=x.reshape(b,-1)
        x=self.act(self.lin_1(x))
        x=self.act(self.lin_2(x))
        return self.lin_out(x)
