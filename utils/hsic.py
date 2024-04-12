import torch
import numpy as np
from torch.autograd import Variable, grad

def add_hsic_args(parser):
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
    parser.add_argument('--use_bufferCE', action='store_true', default=False, help='use Siam to calculate interact loss or not')
    return parser

def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X,Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med=np.mean(Tri)
    if med<1E-2:
        med=1E-2
    return med

def distmat(X):
    """ distance matrix
    """
    r = torch.sum(X*X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X,0,1))
    D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
    D = torch.abs(D)
    return D

def kernelmat(X, sigma, k_type="gaussian"):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    dim = int(X.size()[1]) * 1.0
    H = torch.eye(m) - (1./m) * torch.ones([m,m])

    if k_type == "gaussian":
        Dxx = distmat(X)
        
        if sigma:
            variance = 2.*sigma*sigma*X.size()[1]            
            Kx = torch.exp( -Dxx / variance).type(torch.FloatTensor)   # kernel matrices        
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X,X)
                Kx = torch.exp( -Dxx / (2.*sx*sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))

    ## Adding linear kernel
    elif k_type == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)

    Kxc = torch.mm(Kx,H)

    return Kxc

def distcorr(X, sigma=1.0):
    X = distmat(X)
    X = torch.exp( -X / (2.*sigma*sigma))
    return torch.mean(X)

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def mmd(x, y, sigma=None, use_cuda=True, to_numpy=False):
    m = int(x.size()[0])
    H = torch.eye(m) - (1./m) * torch.ones([m,m])
    # H = Variable(H)
    Dxx = distmat(x)
    Dyy = distmat(y)

    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
        sxy = sigma
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    # Kxc = torch.mm(Kx,H)            # centered kernel matrices
    # Kyc = torch.mm(Ky,H)
    Dxy = distmat(torch.cat([x,y]))
    Dxy = Dxy[:x.size()[0], x.size()[0]:]
    Kxy = torch.exp( -Dxy / (1.*sxy*sxy))

    mmdval = torch.mean(Kx) + torch.mean(Ky) - 2*torch.mean(Kxy)

    return mmdval

def mmd_pxpy_pxy(x,y,sigma=None,use_cuda=True, to_numpy=False):
    """
    """
    if use_cuda:
        x = x.cuda()
        y = y.cuda()
    m = int(x.size()[0])

    Dxx = distmat(x)
    Dyy = distmat(y)
    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    A = torch.mean(Kx*Ky)
    B = torch.mean(torch.mean(Kx,dim=0)*torch.mean(Ky, dim=0))
    C = torch.mean(Kx)*torch.mean(Ky)
    mmd_pxpy_pxy_val = A - 2*B + C 
    return mmd_pxpy_pxy_val

def hsic_regular(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """
    """
    Kxc = kernelmat(x, sigma)
    Kyc = kernelmat(y, sigma)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy

def hsic_normalized(x, y, sigma=None, use_cuda=True, to_numpy=True):
    """
    """
    m = int(x.size()[0])
    Pxy = hsic_regular(x, y, sigma, use_cuda)
    Px = torch.sqrt(hsic_regular(x, x, sigma, use_cuda))
    Py = torch.sqrt(hsic_regular(y, y, sigma, use_cuda))
    thehsic = Pxy/(Px*Py)
    return thehsic

def hsic_normalized_cca(x, y, sigma, use_cuda=True, to_numpy=True, k_type_y='gaussian'):
    """
    """
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma, k_type=k_type_y)

    epsilon = 1E-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon*m*K_I)
    Kyc_i = torch.inverse(Kyc + epsilon*m*K_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy

def hsic_objective(hidden, h_target, h_data, sigma, k_type_y='gaussian'):


    hsic_hy_val = hsic_normalized_cca( hidden, h_target, sigma=sigma, k_type_y=k_type_y)
    hsic_hx_val = hsic_normalized_cca( hidden, h_data,   sigma=sigma)

    return hsic_hx_val, hsic_hy_val

def to_categorical(y, num_classes, device):
    """ 1-hot encodes a tensor """
    return torch.squeeze(torch.eye(num_classes, device=device)[y])


def get_hsic_features_list(feature_list, hsic_features_to_include="0_1_2_3_4"):
    # hsic_layers_str = self.args.hsic_features_to_include
    hsic_layers_str = hsic_features_to_include
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

def calculate_hbar(x, y, z_list, cpt, ntasks, device, args):
    x = x.view(-1, np.prod(x.size()[1:]))
    # y = to_categorical(y, num_classes=self.cpt*self.ntasks, device=self.device).float()
    y = to_categorical(y, num_classes=cpt*ntasks, device=device).float()
    # z_list = self.get_hsic_features_list(z_list)
    z_list = get_hsic_features_list(z_list, args.hsic_features_to_include)
    total_lx = 0
    total_ly = 0
    total_hbar = 0
    # lx, ly, ld = self.args.lambda_x, self.args.lambda_y, self.args.hsic_layer_decay
    lx, ly, ld = args.lambda_x, args.lambda_y, args.hsic_layer_decay
    if ld > 0:
        lx, ly = lx * (ld ** len(z_list)), ly * (ld ** len(z_list))
    for idx, z in enumerate(z_list):
        if len(z.size()) > 2:
            z = z.view(-1, np.prod(z.size()[1:])) 
        hx_l, hy_l = hsic_objective(
            z,
            y,
            x,
            sigma=args.sigma, # self.args.sigma,
            k_type_y=args.k_type_y # self.args.k_type_y
        )
        if ld > 0:
            lx, ly = lx/ld, ly/ld
        # total_lx and total_ly are not for backprop
        total_lx += hx_l.item()
        total_ly += hy_l.item()
        # if self.args.debug_x_hsic:
        #     if idx == 0:
        #         total_hbar += lx * hx_l - ly * hy_l
        #     else:
        #         total_hbar += -ly * hy_l
        # else:
        total_hbar += lx * hx_l - ly * hy_l
    return total_hbar, total_lx, total_ly
    
def calculate_interact_hsic(z_list1, z_list2, args):
    assert len(z_list1) == len(z_list2)
    total_interact_hsic = 0
    # TODO: make another argument for lx here
    # lx, ld = self.args.lambda_x, self.args.hsic_layer_decay
    lx, ld = args.lambda_x, args.hsic_layer_decay
    if ld > 0:
        lx = lx * (ld ** len(z_list1))
    for z1, z2 in zip(z_list1, z_list2):
        if len(z1.size()) > 2: z1 = z1.view(-1, np.prod(z1.size()[1:]))
        if len(z2.size()) > 2: z2 = z2.view(-1, np.prod(z2.size()[1:]))
        if ld > 0: lx = lx / ld
        # use Gaussian kernel for default
        # TODO: make the kernel changeable?
        # total_interact_hsic += lx * hsic_normalized_cca(z1, z2, sigma=self.args.sigma)
        total_interact_hsic += lx * hsic_normalized_cca(z1, z2, sigma=args.sigma)
    return total_interact_hsic


if __name__ == "__main__":
    x = torch.randn(size=(2, 5))
    print(x)
    kx_l =  kernelmat(x, sigma=None, k_type='linear')
    kx_g =  kernelmat(x, sigma=None, k_type='gaussian')
    print(kx_l)
    print(kx_g)
