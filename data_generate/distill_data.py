
import os
import json
import torch
import torch.nn as nn
import copy
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import random
import pickle
import sys
import torchvision.transforms as transforms

class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class output_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer. 
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None


class DistillData(object):
    def __init__(self):
        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []

    def hook_fn_forward(self, module, input, output):
        input = input[0]
        mean = input.mean([0, 2, 3])
        # use biased var in train
        var = input.var([0, 2, 3], unbiased=False)

        self.mean_list.append(mean)
        self.var_list.append(var)
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)

    def getDistilData_hardsample(self,
                                model_name="resnet18",
                                teacher_model=None,
                                batch_size=256,
                                num_batch=1,
                                group=1,
                                augMargin=0.4,
                                beta=1.0,
                                gamma=0,
                                save_path_head=""
                                ):

        data_path = os.path.join(save_path_head, model_name+"_refined_gaussian_hardsample_" \
                    + "beta"+ str(beta) +"_gamma" + str(gamma) + "_group" + str(group) + ".pickle")
        label_path = os.path.join(save_path_head, model_name+"_labels_hardsample_" \
                    + "beta"+ str(beta) +"_gamma" + str(gamma) + "_group" + str(group) + ".pickle")

        print(data_path, label_path)

        if model_name in ['resnet20_cifar10','resnet20_cifar100']:
            shape = (batch_size, 3, 32, 32)
        else:
            shape = (batch_size, 3, 224, 224)

        # initialize hooks and single-precision model
        teacher_model = teacher_model.cuda()
        teacher_model = teacher_model.eval()

        refined_gaussian = []
        labels_list = []

        CE_loss = nn.CrossEntropyLoss(reduction='none').cuda()
        MSE_loss = nn.MSELoss().cuda()

        # hooks, hook_handles = [], []
        for n, m in teacher_model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_hook(self.hook_fn_forward)

        for i in range(1280//batch_size):
            # initialize the criterion, optimizer, and scheduler

            if model_name in ['resnet20_cifar10', 'resnet20_cifar100']:
                RRC = transforms.RandomResizedCrop(size=32,scale=(augMargin, 1.0))
            else:
                RRC = transforms.RandomResizedCrop(size=224,scale=(augMargin, 1.0))
            RHF = transforms.RandomHorizontalFlip()

            gaussian_data = torch.randn(shape).cuda()
            gaussian_data.requires_grad = True
            optimizer = optim.Adam([gaussian_data], lr=0.5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             min_lr=1e-4,
                                                             verbose=False,
                                                             patience=50)

            if model_name == 'resnet20_cifar10':
                labels = torch.randint(0, 10, (len(gaussian_data),)).cuda()
                labels_mask = F.one_hot(labels, num_classes=10).float()
            elif model_name == 'resnet20_cifar100':
                labels = torch.randint(0, 100, (len(gaussian_data),)).cuda()
                labels_mask = F.one_hot(labels, num_classes=100).float()
            else:
                labels = torch.randint(0, 1000, (len(gaussian_data),)).cuda()
                labels_mask = F.one_hot(labels, num_classes=1000).float()
            gt = labels.data.cpu().numpy()

            for it in range(500*2):
                if model_name in ['resnet20_cifar10', 'resnet20_cifar100']:
                    new_gaussian_data = []
                    for j in range(len(gaussian_data)):
                        new_gaussian_data.append(gaussian_data[j])
                    new_gaussian_data = torch.stack(new_gaussian_data).cuda()
                else:
                    if random.random() < 0.5:
                        new_gaussian_data = []
                        for j in range(len(gaussian_data)):
                            new_gaussian_data.append(RHF(RRC(gaussian_data[j])))
                        new_gaussian_data = torch.stack(new_gaussian_data).cuda()
                    else:
                        new_gaussian_data = []
                        for j in range(len(gaussian_data)):
                            new_gaussian_data.append(gaussian_data[j])
                        new_gaussian_data = torch.stack(new_gaussian_data).cuda()

                self.mean_list.clear()
                self.var_list.clear()
                self.teacher_running_mean.clear()
                self.teacher_running_var.clear()

                output = teacher_model(new_gaussian_data)
                d_acc = np.mean(np.argmax(output.data.cpu().numpy(), axis=1) == gt)
                a = F.softmax(output, dim=1)
                mask = torch.zeros_like(a)
                b=labels.unsqueeze(1)
                mask=mask.scatter_(1,b,torch.ones_like(b).float())
                p=a[mask.bool()]

                loss_target = beta * ((1-p).pow(gamma) * CE_loss(output, labels)).mean()

                mean_loss = torch.zeros(1).cuda()
                var_loss = torch.zeros(1).cuda()
                for num in range(len(self.mean_list)):
                    mean_loss += MSE_loss(self.mean_list[num], self.teacher_running_mean[num].detach())
                    var_loss += MSE_loss(self.var_list[num], self.teacher_running_var[num].detach())

                mean_loss = mean_loss / len(self.mean_list)
                var_loss = var_loss / len(self.mean_list)

                total_loss = mean_loss + var_loss + loss_target
                print(i, it, 'lr', optimizer.state_dict()['param_groups'][0]['lr'],
                      'd_acc', d_acc, 'mean_loss', mean_loss.item(), 'var_loss',
                      var_loss.item(), 'loss_target', loss_target.item())

                optimizer.zero_grad()
                # update the distilled data
                total_loss.backward()
                optimizer.step()
                scheduler.step(total_loss.item())

            with torch.no_grad():
                output = teacher_model(gaussian_data.detach())
                d_acc = np.mean(np.argmax(output.data.cpu().numpy(), axis=1) == gt)
                print('d_acc', d_acc)

            refined_gaussian.append(gaussian_data.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())

            gaussian_data = gaussian_data.cpu()
            del gaussian_data
            del optimizer
            del scheduler
            del labels
            torch.cuda.empty_cache()

        with open(data_path, "wb") as fp:  # Pickling
            pickle.dump(refined_gaussian, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(label_path, "wb") as fp:  # Pickling
            pickle.dump(labels_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
        sys.exit()
        # return refined_gaussian


