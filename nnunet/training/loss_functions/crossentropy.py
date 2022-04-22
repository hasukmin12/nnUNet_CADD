from torch import nn, Tensor


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]

        # if input[0].shape[0] > 2:
        #     if 1 not in target:
        #         for i in range(0,3):
        #             input[i][1] = 0
        #     if 2 not in target:
        #         # net_output = net_output.detach().cpu().numpy()
        #         # net_output =  np.delete(net_output, 2, 1)
        #         # net_output = torch.from_numpy(net_output).cuda()
        #         for i in range(0,3):
        #             input[i][2] = 0
        #
        #     if 3 not in target:
        #         # net_output = net_output.detach().cpu().numpy()
        #         # net_output = np.delete(net_output, 3, 1)
        #         # net_output = torch.from_numpy(net_output).cuda()
        #         for i in range(0,3):
        #             input[i][3] = 0
        return super().forward(input, target.long())