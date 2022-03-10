import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
#from depthset import train_dataset

'''
Just for debugging:

device = 'cpu'
model = 'model'
data = next(iter(train_dataset))
print(len(data))
print(type(data))
# Make tensor batches (to tensor and then unsqueeze)
data['image'] = torch.tensor(data['image'], dtype=torch.float).unsqueeze(0)
data['depth'] = torch.tensor(data['depth'], dtype=torch.float).unsqueeze(0)
data['mask'] = torch.tensor(data['mask'], dtype=torch.float).unsqueeze(0)
data['edges'] = torch.tensor(data['edges'], dtype=torch.float).unsqueeze(0)
data['range_min'] = torch.tensor(data['range_min'], dtype=torch.float).unsqueeze(0)
data['range_max'] = torch.tensor(data['range_max'], dtype=torch.float).unsqueeze(0)
'''

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    # L1 norm
    def forward(self, grad_fake, grad_real):
        return nn.L1Loss()(grad_real, grad_fake)

class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, grad_fake, grad_real):
        prod = (grad_fake[:, :, None, :] @ grad_real[:, :, :, None]).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt(torch.sum(grad_fake ** 2, dim=-1))
        real_norm = torch.sqrt(torch.sum(grad_real ** 2, dim=-1))
        return 1 - torch.mean(prod / ((fake_norm * real_norm) + 1e-10))#torch.abs(1 - prod / ((fake_norm * real_norm) + 1e-10)).mean()

#fehler torch.abs(1 - prod / ((fake_norm * real_norm) + 1e-10)).mean()
#file:///C:/Users/alexa/Downloads/wacv2019.pdf
#https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/a486b9a3fc708a4f320881ee3b31f94022e1c184/train.py#L112

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    # L1 norm
    def forward(self, pred, target, mask, device, factor=0.6):
        # assuming mask consists of 0, 1 values
        mask = (mask * factor).to(device, dtype=torch.float)
        ones = (torch.ones(mask.shape).to(device) * (1 - factor)).to(device)
        mask = torch.where(mask == 0, ones, mask).to(device, dtype=torch.float)
        losses = torch.abs((target - pred) * mask)
        return torch.mean(losses), losses

def imgrad(img, device):
    img = torch.mean(img, 1, True)
    # grad x
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0).to(device)
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)
    # grad y
    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0).to(device)
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x

def get_prediction(data, model, device, interpolate):
    orig_inp = data['image'].permute(0, 3, 1, 2)
    inp = Variable(orig_inp).to(device, dtype=torch.float)
    target = Variable(data['depth']).to(device, dtype=torch.float).unsqueeze(1)
    mask = data['mask'].to(device).unsqueeze(1)
    edges = Variable(data['edges']).to(device).unsqueeze(1)
    range_min = data['range_min'].to(device, dtype=torch.float)
    range_max = data['range_max'].to(device, dtype=torch.float)
    out = model(inp)
    if not interpolate:
        out = out * mask
        target = target * mask
    return inp, out, target, edges.detach(), orig_inp.detach(), range_min.detach(), range_max.detach()

def imgrad_yx(img, device):
    N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img, device)
    return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)

def calc_loss(data, model, criterion1, criterion_img, criterion_norm, device, interpolate, edge_factor, batch_idx, reg=False):
    inp, out, target, edges, orig_inp, _, _ = get_prediction(data, model, device, interpolate)
    imgrad_true = imgrad_yx(target, device)
    imgrad_out = imgrad_yx(out, device)
    l1_loss, l1_losses = criterion1(out, target, edges, device, factor=edge_factor)
    loss_grad = criterion_img(imgrad_out, imgrad_true)
    loss_normal = criterion_norm(imgrad_out, imgrad_true)
    total_loss = 0.2*l1_loss + 1.0 * loss_grad + 0.1 * loss_normal #1.0,0.5,0.5
    
    
    if reg:
        loss_reg = Variable(torch.tensor(0.)).to(device)
        for param in model.parameters():
            loss_reg = loss_reg + param.norm(2)
        total_loss = total_loss + 1e-20 * loss_reg
    '''
    if batch_idx % 10 == 0:
        print(f'L1-loss: {l1_loss.item()}, '
              f'Loss Grad: {loss_grad.item()},  '
              f'Loss Normal: {loss_normal.item()}')
    '''
    return total_loss, l1_losses, out, target, inp, orig_inp, edges

l1_criterion = MaskedL1Loss()
grad_criterion = GradLoss()
normal_criterion = NormalLoss()

'''
loss, l1_losses, out, target, _, orig_inp, edges = calc_loss(data, model,
                                                                 l1_criterion, grad_criterion, normal_criterion,
                                                                 device=device,
                                                                 interpolate=False,
                                                                 edge_factor=0.6,
                                                                 batch_idx=None)
'''