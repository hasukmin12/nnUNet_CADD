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
from nnunet.network_architecture.SDDense_UNet import CA_MDD_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.loss_functions.dice_loss import get_tp_fp_fn_tn
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.network_architecture.custom_modules.conv_block_for_Double_Dense import DenseUpBlock, DenseUpLayer, DenseDownBlock_2, DenseDownLayer_2
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.for_DTC import compute_sdf, boundary_loss
from torch.nn import MSELoss
from torch import nn
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss

class SDDenseUNet_DP(CA_MDD_UNet):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None,
                 block=DenseDownBlock_2):


        super(SDDenseUNet_DP, self).__init__(input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None,
                 block=DenseDownBlock_2)   # , lambda x: x)



        self.ce_loss = RobustCrossEntropyLoss()
        # self.loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
        # self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)


# ????????? x??? input(CT ??????), y??? label??? GT(Label GT)??????.

# ????????? res[i], y[i] ?????? sampling?????? Layer ????????? output??? ?????????.
# ????????? i ?????? ???????????? ???????????? ??????.
# ????????? ????????? res ??? y??? ?????? ????????? (40*40)??????.
# ????????? ?????? ??? ????????? ???????????? ????????? ???????????? (40*40)??? ??????????????? ???????????? ??????.

    def forward(self, x, y=None, return_hard_tp_fp_fn=False):
        res, res_tanh = super(SDDenseUNet_DP, self).forward(x)  # regular Generic_UNet forward pass

        if y is None:
            return res, res_tanh

        # ????????? ??????????????? ?????? ??????.
        else:
            if self._deep_supervision and self.do_ds:

                if y[0].max() == 0:
                # label??? ?????? ?????? ????????? loss ????????????
                # res_tanh??? res??? T^-1?????? MSE??? loss??? ??????
                # print(x, "no label is here")
                    dis_to_mask = torch.sigmoid(-1500 * res_tanh)
                    outputs_soft = torch.sigmoid(res[0])
                    # consistency_loss = torch.mean((dis_to_mask - outputs_soft) ** 2)
                    consistency_loss = self.ce_loss(dis_to_mask, outputs_soft)
                    consistency_loss = torch.unsqueeze(consistency_loss, 0)

                    return consistency_loss

                else:

                    ce_losses = [self.ce_loss(res[0], y[0]).unsqueeze(0)]

                    # tp : True Positive
                    # fp : False Positive
                    # fn : False Negative
                    tps = []
                    fps = []
                    fns = []

                    res_softmax = softmax_helper(res[0])
                   # print("res_softmax.shape : ", res_softmax.shape)
                    tp, fp, fn, _ = get_tp_fp_fn_tn(res_softmax, y[0])
                    tps.append(tp)
                    fps.append(fp)
                    fns.append(fn)
                    for i in range(1, len(y)):
                        ce_losses.append(self.ce_loss(res[i], y[i]).unsqueeze(0))
                        res_softmax = softmax_helper(res[i])
                        tp, fp, fn, _ = get_tp_fp_fn_tn(res_softmax, y[i])
                        tps.append(tp)
                        fps.append(fp)
                        fns.append(fn)



                    # loss between output, output_tanh
                    dis_to_mask = torch.sigmoid(-1500 * res_tanh)
                    outputs_soft = torch.sigmoid(res[0])
                    # consistency_loss = torch.mean((dis_to_mask - outputs_soft) ** 2)
                    consistency_loss = self.ce_loss(dis_to_mask, outputs_soft)
                    consistency_loss = torch.unsqueeze(consistency_loss, 0)


                    # loss between output_tanh, GT_tanh
                    # gt_dis = compute_sdf(y[0].cpu().numpy(), res[0].shape)
                    gt_dis = compute_sdf(y.cpu().numpy(), res[0].shape)
                    gt_dis = torch.from_numpy(gt_dis).float().cuda()
                    loss_sdf = self.ce_loss(res_tanh, gt_dis)  # torch.mean((res_tanh-gt_dis) ** 2)
                    # loss_sdf = boundary_loss(res_tanh, gt_dis)


                    consistency_loss = consistency_loss * 0.1
                    loss_sdf = loss_sdf * 0.2


                    # # loss between output_tanh, GT_tanh
                    # gt_dis = compute_sdf(y[0].cpu().numpy(), res[0].shape)
                    # gt_dis = torch.from_numpy(gt_dis).float().cuda()
                    # # loss_sdf = torch.mean((res_tanh - gt_dis) ** 2)
                    # loss_sdf = boundary_loss(res_tanh, gt_dis)


                    ret = ce_losses, tps, fps, fns# , consistency_loss, loss_sdf
                    return ret




            # ?????? ??????????????? ????????????.
            else:
                # print("deep_supervision is False")
                ce_loss = self.ce_loss(res, y).unsqueeze(0)

                # tp fp and fn need the output to be softmax
                res_softmax = softmax_helper(res)

                tp, fp, fn, _ = get_tp_fp_fn_tn(res_softmax, y)

                ret = ce_loss, tp, fp, fn


            # validation ??? ????????? ????????????.
            if return_hard_tp_fp_fn:

                # print("return_hard_tp_fp_fn is True")
                if self._deep_supervision and self.do_ds:
                    output = res[0]
                    target = y[0]
                else:
                    target = y
                    output = res

                with torch.no_grad():
                    num_classes = output.shape[1]
                    output_softmax = softmax_helper(output)
                    output_seg = output_softmax.argmax(1)
                    target = target[:, 0]
                    axes = tuple(range(1, len(target.shape)))
                    tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
                    fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
                    fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
                    for c in range(1, num_classes):
                        tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                        fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                        fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

                    tp_hard = tp_hard.sum(0, keepdim=False)[None]
                    fp_hard = fp_hard.sum(0, keepdim=False)[None]
                    fn_hard = fn_hard.sum(0, keepdim=False)[None]

                    ret = *ret, tp_hard, fp_hard, fn_hard
            return ret