import math
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function


# https://github.com/DHUDBlab/scDSC/blob/1247a63aac17bdfb9cd833e3dbe175c4c92c26be/layers.py#L43
class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


# https://github.com/DHUDBlab/scDSC/blob/1247a63aac17bdfb9cd833e3dbe175c4c92c26be/layers.py#L43
class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def to_categorical(y, num_classes, dtype=torch.int):
    """ 1-hot encodes a tensor """
    one_hot = torch.eye(num_classes, dtype=dtype)[y]
    return one_hot


def grad_reverse(x):
    return ReverseLayerF()(x)


def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x.

    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)


def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x.

    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * torch.log(2 * torch.tensor(math.pi, requires_grad=True)) - log_var / 2 - (x - mu) ** 2 / (
        2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)


def log_Normal_diag(x, mean, log_var, average=False, dim=1):
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_Normal_standard(x, average=False, dim=1):
    log_normal = -0.5 * torch.pow(x, 2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def count_labels(arr):
    """
    Counts elements in array

    :param arr:
    :return:
    """
    elements_count = {}
    for element in arr:
        if element in elements_count:
            elements_count[element] += 1
        else:
            elements_count[element] = 1
    to_remove = []
    for key, value in elements_count.items():
        print(f"{key}: {value}")
        if value <= 2:
            to_remove += [key]

    return to_remove


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """

    def reparametrize(self, mu, log_var, train, beta=0):
        if train:
            epsilon = Variable(beta * torch.randn(mu.size()), requires_grad=False)
        else:
            epsilon = Variable(torch.zeros_like(mu), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        z = std * epsilon + mu
        # z = mu.addcmul(std, epsilon)

        return z


class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x, train=False, beta=1.0):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparametrize(mu, log_var, train, beta), mu, log_var


class Classifier(nn.Module):
    def __init__(self, in_shape=64, out_shape=9):
        super(Classifier, self).__init__()
        self.linear2 = nn.Sequential(
            nn.Linear(in_shape, out_shape),
        )
        self.random_init()

    def forward(self, x):
        x = self.linear2(x)
        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.975)
                nn.init.constant_(m.bias, 0.125)

    def predict_proba(self, x):
        return self.linear2(x).detach().cpu().numpy()

    def predict(self, x):
        return self.linear2(x).argmax(1).detach().cpu().numpy()


class Classifier2(nn.Module):
    def __init__(self, in_shape=64, out_shape=9, dropout=0.):
        super(Classifier2, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(in_shape, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(128, out_shape),
        )
        self.random_init()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.975)
                nn.init.constant_(m.bias, 0.125)


class Encoder(nn.Module):
    def __init__(self, in_shape, layer1, layer2, dropout):
        super(Encoder, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(in_shape, layer1),
            nn.BatchNorm1d(layer1),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(layer1, layer2),
            nn.BatchNorm1d(layer2),
            nn.LeakyReLU(),
        )
        self.random_init()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.975)
                nn.init.constant_(m.bias, 0.125)


class Decoder(nn.Module):
    def __init__(self, in_shape, n_batches, layer1, layer2, dropout):
        super(Decoder, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(layer1 + n_batches, layer2),
            nn.BatchNorm1d(layer2),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(layer2, in_shape),
            # nn.BatchNorm1d(layer2),
            # nn.Sigmoid(),
        )
        self.random_init()

    def forward(self, x, batches=None):
        if batches is not None:
            x = torch.cat((x, batches), 1)
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        return [x1, x2]

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.975)
                nn.init.constant_(m.bias, 0.125)


class AutoEncoder(nn.Module):
    def __init__(self, in_shape, n_batches, nb_classes, variational, layer1, layer2, dropout, zinb=False,
                 conditional=False, add_noise=False,
                 tied_weights=0):
        super(AutoEncoder, self).__init__()
        self.add_noise = add_noise
        self.zinb = zinb
        self.tied_weights = tied_weights
        self.flow_type = 'vanilla'
        self.enc = Encoder(in_shape, layer1, layer2, dropout)
        if conditional:
            self.dec = Decoder(in_shape, n_batches, layer2, layer1, dropout)
        else:
            self.dec = Decoder(in_shape, 0, layer2, layer1, dropout)
        self.classifier = Classifier(layer2, n_batches)

        if variational:
            self.gaussian_sampling = GaussianSample(layer2, layer2)
        else:
            self.gaussian_sampling = None
        self.dann_discriminator = nn.Linear(layer2, n_batches)
        self.classifier = nn.Linear(layer2, nb_classes)
        self._dec_mean = nn.Sequential(nn.Linear(layer1, in_shape), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(layer1, in_shape), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(layer1, in_shape), nn.Sigmoid())
        self.random_init(nn.init.kaiming_normal_)

    def forward(self, x, batches, alpha, sampling, beta=1.0):
        if self.add_noise:
            x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)

        enc = self.enc(x)
        if self.gaussian_sampling is not None:
            if sampling:
                enc, mu, log_var = self.gaussian_sampling(enc, train=True, beta=beta)
                # Kullback-Leibler Divergence
                # kl = self._kld(enc, (mu, log_var))
                mean_sq = mu * mu
                std = log_var.exp().sqrt()
                stddev_sq = std * std
                # kl = 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
                # https://arxiv.org/pdf/1312.6114.pdf equation 10, first part and
                # https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder
                kl = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), axis=1)
            else:
                enc, _, _ = self.gaussian_sampling(enc, train=False)
                kl = torch.Tensor([0])
        else:
            kl = torch.Tensor([0])
        if not self.tied_weights:
            rec = self.dec(enc, batches)
        else:
            rec = [F.relu(F.linear(enc, self.enc.linear2[0].weight.t()))]
            rec += [F.linear(rec[0], self.enc.linear1[0].weight.t())]
            rec[-1] = F.relu(rec[-1])

        if self.zinb:
            _mean = self._dec_mean(rec[0])
            _disp = self._dec_disp(rec[0])
            _pi = self._dec_pi(rec[0])
            zinb_loss = self.zinb_loss(x, _mean, _disp, _pi, scale_factor=1)
        else:
            zinb_loss = 0
        # reverse = ReverseLayerF.apply(enc, alpha)
        # b_preds = self.classifier(reverse)
        return enc, rec[1], zinb_loss, kl

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 0.975)
                nn.init.constant_(m.bias, 0.125)

    def predict_proba(self, x):
        return self.classifier(x).detach().cpu().numpy()

    def predict(self, x):
        return self.classifier(x).argmax(1).detach().cpu().numpy()

    def _kld(self, z, q_param, h_last=None, p_param=None):
        if len(z.shape) == 1:
            z = z.view(1, -1)
        if (self.flow_type == "nf") and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        elif (self.flow_type == "iaf") and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z, log_det_z = self.flow(z, h_last)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        elif (self.flow_type in ['hf', 'ccliniaf']) and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z = self.flow(z, h_last)
            qz = log_gaussian(z, mu, log_var)
            z = f_z
        elif self.flow_type in ["o-sylvester", "h-sylvester", "t-sylvester"] and self.n_flows > 0:
            mu, log_var, r1, r2, q_ortho, b = q_param
            f_z = self.flow(z, r1, r2, q_ortho, b)
            qz = log_gaussian(z, mu, log_var)
            z = f_z
        # vanilla
        else:
            (mu, log_var) = q_param
            qz = log_Normal_diag(z, mu, log_var)
        if p_param is None:
            pz = log_Normal_standard(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = -(pz - qz)

        return kl

    # based on https://github.com/DHUDBlab/scDSC/blob/master/layers.py
    def zinb_loss(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        # scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge
        result = torch.mean(result)
        return result
