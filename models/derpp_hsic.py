
from datasets import get_dataset
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from models.derpp import Derpp
from utils.args import *
import torch
import numpy as np
from collections import OrderedDict

from utils.distributed import make_dp
from utils.hsic import *

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' CL with HBaR.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)

    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--lambda_x', type=float, default=0.0005,
                        help='Penalty weight.')
    parser.add_argument('--lambda_y', type=float, default=0.005,
                        help='Penalty weight.')
    parser.add_argument('--hsic_layer_decay', type=float, default=1.0,
                        help='Penalty weight.')
    parser.add_argument('--sigma', type=float, default=5.0,
                        help='Penalty weight.')
    parser.add_argument('--k_type_y', type=str, default='linear', choices=['gaussian', 'linear'],
                        help='Penalty weight.')
    parser.add_argument('--buffer_hsic', type=float, required=False, default=1.0,
                        help='Lambda parameter for lipschitz budget distribution loss')
    parser.add_argument('--hsic_features_to_include', type=str, default="0_1_2_3_4", help='list of features to use in HSIC, for example "0 1 2"')
    parser.add_argument('--new_batch_for_buffer_hsic', action='store_true',
                        help='Lambda parameter for lipschitz budget distribution loss')
    parser.add_argument('--current_hsic', type=float, required=False, default=0.0,
                        help='Lambda parameter for lipschitz budget distribution loss')
    parser.add_argument('--interact_hsic', type=float, required=False, default=0.0,
                        help='Lambda parameter for lipschitz budget distribution loss')
    parser.add_argument('--debug_x_hsic', action='store_true',
                        help='Lambda parameter for lipschitz budget distribution loss')
    parser.add_argument('--no_projector', action='store_true',
                        help='Lambda parameter for lipschitz budget distribution loss')
    return parser


class DerppHSIC(Derpp):
    NAME = 'derpp_hsic'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DerppHSIC, self).__init__(backbone, loss, args, transform)
        self.ntasks = get_dataset(self.args).N_TASKS

    def calculate_hbar(self, x, y, z_list):
        x = x.view(-1, np.prod(x.size()[1:]))
        y = to_categorical(y, num_classes=self.cpt*self.ntasks, device=self.device).float()
        z_list = self.get_hsic_features_list(z_list)
        total_lx = 0
        total_ly = 0
        total_hbar = 0
        lx, ly, ld = self.args.lambda_x, self.args.lambda_y, self.args.hsic_layer_decay
        if ld > 0:
            lx, ly = lx * (ld ** len(z_list)), ly * (ld ** len(z_list))
        for idx, z in enumerate(z_list):
            if len(z.size()) > 2:
                z = z.view(-1, np.prod(z.size()[1:])) 
            hx_l, hy_l = hsic_objective(
                z,
                y,
                x,
                sigma=self.args.sigma,
                k_type_y=self.args.k_type_y
            )
            if ld > 0:
                lx, ly = lx/ld, ly/ld
            # total_lx and total_ly are not for backprop
            total_lx += hx_l.item()
            total_ly += hy_l.item()
            if self.args.debug_x_hsic:
                if idx == 0:
                    total_hbar += lx * hx_l - ly * hy_l
                else:
                    total_hbar += -ly * hy_l
            else:
                total_hbar += lx * hx_l - ly * hy_l
        return total_hbar, total_lx, total_ly
    
    def calculate_interact_hsic(self, z_list1, z_list2):
        assert len(z_list1) == len(z_list2)
        total_interact_hsic = 0
        # TODO: make another argument for lx here
        lx, ld = self.args.lambda_x, self.args.hsic_layer_decay
        if ld > 0:
            lx = lx * (ld ** len(z_list1))
        for z1, z2 in zip(z_list1, z_list2):
            if len(z1.size()) > 2: z1 = z1.view(-1, np.prod(z1.size()[1:]))
            if len(z2.size()) > 2: z2 = z2.view(-1, np.prod(z2.size()[1:]))
            if ld > 0: lx = lx / ld
            # use Gaussian kernel for default
            # TODO: make the kernel changeable?
            total_interact_hsic += lx * hsic_normalized_cca(z1, z2, sigma=self.args.sigma)
        return total_interact_hsic
    
    def get_hsic_features_list(self, feature_list):
        hsic_layers_str = self.args.hsic_features_to_include
        res = list(range(0, len(feature_list)))
        if hsic_layers_str:
            res_str = hsic_layers_str.split('_')
            res = [int(x) for x in res_str]
            assert max(res) < len(feature_list)
        res_feature_list = []
        for idx, feature in enumerate(feature_list):
            if idx in res:
                res_feature_list.append(feature)
        return res_feature_list

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, epoch=None):
        total_loss = 0
        original_loss = 0
        interact_hsic_loss = None
        buf_total_lx, buf_total_ly = 0, 0
        cur_total_lx, cur_total_ly = 0, 0
        p_cur, z_cur = None, None
        labels = labels.long()
        self.opt.zero_grad()
        # if calculating hsic for data from current batch
        if self.args.current_hsic > 0 or abs(self.args.interact_hsic) > 0.0001:
            if self.args.use_siam:
                outputs, output_feature, p_cur, z_cur = self.net(inputs, 'use_siam')
            else:
                outputs, outputs_feature_list = self.net(inputs, 'full')
                outputs = outputs.float()
                outputs_feature_list = [x.float() for x in outputs_feature_list]
                # Fix small bug here!!
                # get all features excepted the final output
                outputs_feature_list = outputs_feature_list[:-1]
        else:
            outputs = self.net(inputs).float()
        
        # add cl mask to only the first batch, if specified
        if self.args.use_cl_mask:
            masked_outputs = self.mask_output(outputs)
            cur_xent_loss = self.loss(masked_outputs, labels)
        else:
            cur_xent_loss = self.loss(outputs, labels)

        total_loss += cur_xent_loss
        original_loss += cur_xent_loss.item()

        if self.args.current_hsic > 0:
            cur_total_hbar, cur_total_lx, cur_total_ly = self.calculate_hbar(x=inputs, y=labels, z_list=outputs_feature_list)
            total_loss += self.args.current_hsic * cur_total_hbar.to(self.device)
        
        # add self.current_task > 0 to make sure only replay after 1st task
        if not self.buffer.is_empty() and self.current_task > 0:
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.setting.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs).float()
            derpp_loss1 = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            total_loss += derpp_loss1
            original_loss += derpp_loss1.item()

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.setting.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs).float()
            derpp_loss2 = self.args.beta * self.loss(buf_outputs, buf_labels)
            total_loss += derpp_loss2
            original_loss += derpp_loss2.item()

            if self.args.buffer_hsic > 0:
                if self.args.new_batch_for_buffer_hsic:
                    buf_inputs, buf_labels, _ = self.buffer.get_data(self.setting.minibatch_size, transform=self.transform)
                buf_outputs, buf_feature_list = self.net(buf_inputs, 'full')
                buf_outputs = buf_outputs.float()
                buf_feature_list = [x.float() for x in buf_feature_list]
                # get all features excepted the final output
                buf_feature_list = buf_feature_list[:-1]
                # calculate HSIC loss
                buf_total_hbar, buf_total_lx, buf_total_ly = self.calculate_hbar(x=buf_inputs, y=buf_labels, z_list=buf_feature_list)
                total_loss += self.args.buffer_hsic * buf_total_hbar.to(self.device)
                # add interaction term if needed
                if abs(self.args.interact_hsic) > 0.0001:
                    if self.args.use_siam:
                        assert p_cur is not None
                        assert z_cur is not None
                        _, _, p_buf, z_buf = self.net(buf_inputs, 'use_siam')
                        if self.args.no_projector:
                            interact_hsic_loss = self.calculate_interact_hsic([z_buf], [z_cur])
                        else:
                            interact_hsic_loss = self.calculate_interact_hsic([p_buf], [z_cur]) + self.calculate_interact_hsic([p_cur], [z_buf])
                        total_loss += self.args.interact_hsic * interact_hsic_loss
                    else:
                        interact_hsic_loss = self.calculate_interact_hsic(outputs_feature_list, buf_feature_list)
                        total_loss += self.args.interact_hsic * interact_hsic_loss

        total_loss.backward()
        self.opt.step()

        if self.args.update_buffer_at_task_end:
            self.buffer_backup.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)
        else:
            self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels,
                                logits=outputs.data)
        # get losses
        aux_loss = OrderedDict()
        if self.args.current_hsic > 0 or self.args.buffer_hsic > 0:
            aux_loss['loss_ori'] = original_loss
            if interact_hsic_loss:
                aux_loss['loss_hbar'] = total_loss.item() - interact_hsic_loss.item() - original_loss
                aux_loss['loss_inter'] = interact_hsic_loss.item()
            else:
                aux_loss['loss_hbar'] = total_loss.item() - original_loss
        if self.args.current_hsic > 0:
            aux_loss['c_hx'] = cur_total_lx
            aux_loss['c_hy'] = cur_total_ly
        if self.args.buffer_hsic > 0:
            aux_loss['b_hx'] = buf_total_lx
            aux_loss['b_hy'] = buf_total_ly

        return total_loss.item(), aux_loss
