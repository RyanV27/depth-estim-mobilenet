import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        # print(f"up_x shape: {up_x.shape}")
        result1 = torch.cat([up_x, concat_with], dim=1)
        # print(f"result1 shape: {result1.shape}")
        result2 = self.convA(result1)
        # print(f"result2 shape: {result2.shape}")
        result3 = self.leakyreluA(result2)
        # print(f"result3 shape: {result3.shape}")
        result4 = self.convB(result3)
        # print(f"result4 shape: {result4.shape}")
        result5 = self.leakyreluB(result4)
        # print(f"result5 shape: {result5.shape}")
        return result5

class Decoder(nn.Module):
    def __init__(self, num_features=1280, decoder_width = .6):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)
        
        self.up0 = UpSample(skip_input=features//1 + 320, output_features=features//2)
        self.up1 = UpSample(skip_input=features//2 + 160, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 64, output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 32, output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 +  24, output_features=features//8)
        self.up5 = UpSample(skip_input=features//8 +  16, output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        # print("Start of decoder")
        x_block0, x_block1, x_block2, x_block3, x_block4, x_block5, x_block6 = features[2], features[4], features[6], features[9], features[15],features[18],features[19]
        x_d0 = self.conv2(x_block6)
        x_d1 = self.up0(x_d0, x_block5)
        x_d2 = self.up1(x_d1, x_block4)
        x_d3 = self.up2(x_d2, x_block3)
        # print(f"Shape of x_d3: {x_d3.shape}")
        # print(f"Shape of x_block2: {x_block2.shape}")
        x_d4 = self.up3(x_d3, x_block2)
        # print(f"Shape of x_d4: {x_d4.shape}")
        # print(f"Shape of x_block1: {x_block1.shape}")
        x_d5 = self.up4(x_d4, x_block1)
        # print(f"Shape of x_d5: {x_d5.shape}")
        # print(f"Shape of x_block0: {x_block0.shape}")
        x_d6 = self.up5(x_d5, x_block0)
        # print(f"Shape of x_d6: {x_d6.shape}")
        result = self.conv3(x_d6)
        # print("After running decoder.")
        return result

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()     
        import torchvision.models as models
        self.original_model = models.mobilenet_v2( weights='MobileNet_V2_Weights.DEFAULT' )      # pretrained=True

    def forward(self, x):
        # print("Start of encoder")
        features = [x]
        # for k, v in self.original_model.features._modules.items(): 
        #     # features.append( v(features[-1]) )
       
        for k, v in self.original_model.features._modules.items(): 
            # print(f"{k}: \n{v}")
            int_out = v(features[-1])
            # print(f"\nSize of output: {int_out.shape}\n\n\n")
            features.append(int_out)
        
        # print("End of encoder")
        return features

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        # print("Before starting model")
        output = self.decoder( self.encoder(x) )
        # print("After running model")
        return output