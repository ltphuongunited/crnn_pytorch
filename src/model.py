import torch.nn as nn
from backbone.convnextv2 import *

class CRNN(nn.Module):

    def __init__(self, img_channel=3, img_height=64, img_width=1024, num_class=20,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)
    
        # self.cnn = convnextv2_femto(pretrained=True)
        # output_channel, output_height, output_width = 384, img_height//32, img_width//32

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)


        # self.attention = nn.MultiheadAttention(embed_dim=map_to_seq_hidden, num_heads=8)

        # self.dense = nn.Linear(map_to_seq_hidden, num_class)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)
        self.dense = nn.Linear(2 * rnn_hidden, num_class)
        

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)
        batch, channel, height, width = conv.size()
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)
        
        
        # seq = seq.permute(1, 0, 2)  # (batch, width, feature)
        # attn_output, _ = self.attention(seq, seq, seq)  # (batch, width, feature)
        # attn_output = attn_output.permute(1, 0, 2)  # (width, batch, feature)
        # output = self.dense(attn_output)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)
        output = self.dense(recurrent)

        
        return output  # shape: (seq_len, batch, num_class)


# x = torch.Tensor(2,1,64, 1024)

# model = CRNN()

# out = model(x)
# print(out.shape)
# from torchsummary import summary
# summary(model, (3,32, 512))