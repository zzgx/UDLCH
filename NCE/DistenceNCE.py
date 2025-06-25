import pdb
import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math
import torch.nn.functional as F

class DistenceNCE(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=True):

        print(f'NCE=DistenceNCE')
        print(f'K={K}')

        super(DistenceNCE, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T * math.sqrt(inputSize), -1, -1, momentum]))


        stdv = 1. / math.sqrt(inputSize / 3)


        rnd = torch.randn(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)


        self.register_buffer('memory', F.normalize(rnd.sign(), dim=1))

    def forward(self, l, ab, y, idx=None, epoch=None):

        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()
        momentum = self.params[4].item() if (epoch is None) else (0 if epoch < 0 else self.params[4].item())
        batchSize = l.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        if idx is None:

            batchSize = l.size(0)
            idx_list = []


            with torch.no_grad():


                query = (l + ab) / 2.
                query.div_(query.norm(dim=1, keepdim=True))



                similarities = torch.mm(query, self.memory.t())



                for i in range(batchSize):
                    sim_i = similarities[i]




                    sim_i[y[i]] = float('inf')


                    sorted_indices = torch.argsort(sim_i, descending=True)
                    total = sorted_indices.size(0)


                    pos_indices = sorted_indices[:1]

                    low = int(total * 0.01)
                    high = int(total * 0.90)

                    # low = 200
                    # high = total
                    candidates = sorted_indices[low:high]


                    selected = torch.randperm(len(candidates))[:self.K]
                    neg_indices = candidates[selected]

                    idx_i = torch.cat([pos_indices, neg_indices])


                    idx_list.append(idx_i)
            idx = torch.stack(idx_list).to(l.device)


        if momentum <= 0:
            weight = (l + ab) / 2.
            inx = torch.stack([torch.arange(batchSize)] * batchSize)
            inx = torch.cat([torch.arange(batchSize).view([-1, 1]), inx[torch.eye(batchSize) == 0].view([batchSize, -1])], dim=1).to(weight.device).view([-1])
            weight = weight[inx].view([batchSize, batchSize, -1])
        else:

            weight = torch.index_select(self.memory, 0, idx.view(-1)).detach().view(batchSize, len(idx_i), inputSize)


        weight = weight.sign_()


        out_ab = torch.bmm(weight, ab.view(batchSize, inputSize, 1))
        out_l = torch.bmm(weight, l.view(batchSize, inputSize, 1))

        if self.use_softmax:

            out_ab = torch.div(out_ab, T)
            out_ab = out_ab.contiguous()
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
        else:

            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))

            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))

            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()


        with torch.no_grad():
            l = (l + ab) / 2.
            l.div_(l.norm(dim=1, keepdim=True))  #
            l_pos = torch.index_select(self.memory, 0, y.view(-1))
            l_pos.mul_(momentum)  #
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_pos = l_pos.div_(l_pos.norm(dim=1, keepdim=True))

            self.memory.index_copy_(0, y, l_pos)

        return out_l, out_ab
