#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from copy import deepcopy
from nnunet.network_architecture.custom_modules.helperModules import Identity
from torch import nn



class DenseDownBlock_first(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        """
        This is the conv bn nonlin conv bn nonlin kind of block
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()

        # if props['dropout_op_kwargs'] is None and props['dropout_op_kwargs'] > 0:
        #     raise NotImplementedError("ResidualBottleneckBlock does not yet support dropout!")

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.bottleneck_planes = out_planes // 4

        if stride is not None:
            kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
            kwargs_conv1['stride'] = (1,1,1)
        else:
            kwargs_conv1 = props['conv_op_kwargs']

        self.pool_op = nn.MaxPool3d(2,stride=2)


        # small version
        self.conv1 = props['conv_op'](in_planes, in_planes, kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      **kwargs_conv1)
        self.norm1 = props['norm_op'](in_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv2 = props['conv_op'](in_planes, in_planes, kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm2 = props['norm_op'](in_planes, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])




    def forward(self, x):


        # small version
        residual_1 = x  # 32

        out_1 = self.nonlin1(self.norm1(self.conv1(x)))  # 32
        residual_2 = out_1
        concat_1 = out_1 + residual_1 # 32

        residual_out = self.nonlin2(self.norm2(self.conv2(concat_1)))  # 32

        concat_2 = concat_1 + residual_1


        out = self.pool_op(concat_2)

        return out, concat_2








# 여기서 DenseBlock 구현

class DenseDownBlock_2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        """
        This is the conv bn nonlin conv bn nonlin kind of block
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()

        # if props['dropout_op_kwargs'] is None and props['dropout_op_kwargs'] > 0:
        #     raise NotImplementedError("ResidualBottleneckBlock does not yet support dropout!")

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.bottleneck_planes = out_planes // 4

        if stride is not None:
            kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
            kwargs_conv1['stride'] = (1,1,1)
        else:
            kwargs_conv1 = props['conv_op_kwargs']


        # maxpooling 구현
        self.pool_op = nn.MaxPool3d(2,stride=2)


        self.conv1 = props['conv_op'](in_planes, in_planes, kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      **kwargs_conv1)
        self.norm1 = props['norm_op'](in_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv2 = props['conv_op'](in_planes, in_planes*2, kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm2 = props['norm_op'](in_planes*2, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])

        # self.conv3 = props['conv_op'](in_planes, in_planes , [1 for _ in kernel_size],
        #                               padding=[0 for i in kernel_size],
        #                               **props['conv_op_kwargs'])
        # self.norm3 = props['norm_op'](in_planes , **props['norm_op_kwargs'])
        # self.nonlin3 = props['nonlin'](**props['nonlin_kwargs'])



    def forward(self, x):




        out_1 = self.nonlin1(self.norm1(self.conv1(x)))  # 32

        concat_1 = out_1 + x # 32

        out = self.nonlin2(self.norm2(self.conv2(concat_1)))  # 32



        concat_2 = out

        # residual_out = self.nonlin3(self.norm3(self.conv3(concat_2)))
        out = self.pool_op(concat_2)




        return out, concat_2





class DenseDownLayer_2(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None, block=DenseDownBlock_2):
        super().__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.convs = nn.Sequential(
            block(input_channels, output_channels, kernel_size, network_props, first_stride),
            *[block(output_channels, output_channels, kernel_size, network_props) for _ in
              range(num_blocks - 1)]
        )

    def forward(self, x):
        return self.convs(x)






class DenseDownLayer_first(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None, block=DenseDownBlock_first):
        super().__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.convs = nn.Sequential(
            block(input_channels, output_channels, kernel_size, network_props, first_stride),
            *[block(output_channels, output_channels, kernel_size, network_props) for _ in
              range(num_blocks - 1)]
        )

    def forward(self, x):
        return self.convs(x)











# 여기는 Dense_Up_Block 구현하기



class DenseUpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        """
        This is the conv bn nonlin conv bn nonlin kind of block
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()

        # if props['dropout_op_kwargs'] is None and props['dropout_op_kwargs'] > 0:
        #     raise NotImplementedError("ResidualBottleneckBlock does not yet support dropout!")

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes
        # self.bottleneck_planes = out_planes // 4

        if stride is not None:
            kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
            kwargs_conv1['stride'] = (1,1,1)
        else:
            kwargs_conv1 = props['conv_op_kwargs']







        # small version

        aim_planes = in_planes // 2  # 256

        self.conv0 = props['conv_op'](in_planes, aim_planes, [1 for _ in kernel_size],
                                      padding=[0 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm0 = props['norm_op'](aim_planes, **props['norm_op_kwargs'])
        self.nonlin0 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv1 = props['conv_op'](aim_planes, aim_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                      **kwargs_conv1)
        self.norm1 = props['norm_op'](aim_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv2 = props['conv_op'](aim_planes, aim_planes, kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm2 = props['norm_op'](aim_planes, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv3 = props['conv_op'](aim_planes, aim_planes, [1 for _ in kernel_size],
                                      padding=[0 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm3 = props['norm_op'](aim_planes, **props['norm_op_kwargs'])
        self.nonlin3 = props['nonlin'](**props['nonlin_kwargs'])








    def forward(self, x):


        x = self.nonlin0(self.norm0(self.conv0(x)))  # 512
        residual_1 = x  # 256



        x = self.nonlin1(self.norm1(self.conv1(x)))  # 256
        x = x + residual_1

        residual_2 = x
        x = self.nonlin2(self.norm2(self.conv2(x)))  # 256
        out = x + residual_2

        # out = self.norm3(self.conv3(x))
        # out = self.nonlin3(out)
        return out




class DenseUpLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None, block=DenseUpBlock):
        super().__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.convs = nn.Sequential(
            block(input_channels, output_channels, kernel_size, network_props, first_stride),
            *[block(output_channels, output_channels, kernel_size, network_props) for _ in
              range(num_blocks - 1)]
        )

    def forward(self, x):
        return self.convs(x)



