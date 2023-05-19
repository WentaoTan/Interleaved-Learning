from __future__ import print_function, absolute_import, division

import time
import torch
from .utils.meters import AverageMeter


class Trainer(object):
    def __init__(self, args, model, memory):
        super(Trainer, self).__init__()
        self.model = model
        self.memory = memory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args

    def trainFB(self, epoch, data_loaders, optimizer, print_freq=10, train_iters=400):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        source_count = len(data_loaders)

        end = time.time()
        for i in range(train_iters):

            batch_data = [data_loaders[i].next() for i in range(source_count)]
            data_time.update(time.time() - end)
            inputs_list = []
            targets_list = []

            for ith in range(len(batch_data)):
                inputs = batch_data[ith][0].cuda()
                targets = batch_data[ith][2].cuda()
                inputs_list.append(inputs)
                targets_list.append(targets)
            inputs = torch.cat(inputs_list)
            targets = torch.cat(targets_list)

            loss_id = 0.
            output_list = []
            for j in range(source_count):
                true_bn_x = self.model(inputs_list[j], style=False)
                loss_id += self.memory[j](true_bn_x, targets_list[j]).mean()
                output_list.append(true_bn_x)
            loss_final = loss_id / source_count

            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()

            losses.update(loss_final.item())

            with torch.no_grad():
                for m_ind in range(source_count):
                    self.memory[m_ind].module.MomentumUpdate(output_list[m_ind], targets_list[m_ind])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'loss {:.3f} ({:.3f})'.format(epoch, i + 1, train_iters,
                                                    batch_time.val, batch_time.avg,
                                                    losses.val, losses.avg))
    def train(self, epoch, data_loaders, optimizer, print_freq=10, train_iters=400):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        source_count = len(data_loaders)

        end = time.time()
        for i in range(train_iters):

            batch_data = [data_loaders[i].next() for i in range(source_count)]
            data_time.update(time.time() - end)
            inputs_list = []
            targets_list = []

            for ith in range(len(batch_data)):
                inputs = batch_data[ith][0].cuda()
                targets = batch_data[ith][2].cuda()
                inputs_list.append(inputs)
                targets_list.append(targets)
            inputs = torch.cat(inputs_list)
            targets = torch.cat(targets_list)

            loss_id = 0.
            for j in range(source_count):
                true_bn_x = self.model(inputs_list[j], style=False)
                loss_id += self.memory[j](true_bn_x, targets_list[j]).mean()

            loss_final = loss_id / source_count

            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()

            losses.update(loss_final.item())

            with torch.no_grad():
                each_source_index = []
                for j in range(source_count):
                    each_source_index.append(torch.full([self.args.batch_size], j))
                each_source_index = torch.cat(each_source_index, dim=0)
                random_index = torch.randperm(inputs.size(0))
                inputs = inputs[random_index]
                targets = targets[random_index]
                each_source_index_shuffle = each_source_index[random_index]
                f_new = self.model(inputs, style=self.args.updateStyle)
                for m_ind in range(source_count):
                    current_index = (each_source_index_shuffle == m_ind).nonzero().view(-1).cuda()
                    cur_f_new = torch.index_select(f_new, index=current_index, dim=0)
                    cur_tar = torch.index_select(targets, index=current_index, dim=0)
                    self.memory[m_ind].module.MomentumUpdate(cur_f_new, cur_tar)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'loss {:.3f} ({:.3f})'.format(epoch, i + 1, train_iters,
                                                    batch_time.val, batch_time.avg,
                                                    losses.val, losses.avg))
