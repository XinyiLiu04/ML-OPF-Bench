"""Published NGT equations; benchmark network physics and data interfaces retained."""
import math
import torch
from deepopf_ngt_common import DeepOPFNGT, VoltageDenormaliser, LossTerms


class PaperNetwork(DeepOPFNGT):
    def forward(self, x):
        return torch.sigmoid(self.net(x))


class PaperVoltageDenormaliser(VoltageDenormaliser):
    """Use a full-period angle chart; the paper does not specify its affine scale."""
    def __call__(self, output):
        n = self.n_nonzib
        voltage = self.v_min + output[:, :n] * (self.v_max - self.v_min)
        angles = (2 * output[:, n:] - 1) * math.pi
        angles = angles - angles[:, self.reference_position:self.reference_position + 1]
        return voltage, angles


class PaperLossTerms(LossTerms):
    def __call__(self, results):
        terms = super().__call__(results)
        if self.any_branch_limit:
            penalties = []
            for p, q in [('P_branch', 'Q_branch'), ('P_branch_to', 'Q_branch_to')]:
                flow = torch.linalg.vector_norm(torch.stack((results[p], results[q]), dim=-1), dim=-1)
                penalties.append(torch.relu(flow[:, self.br_mask] - self.rate_a).square())
            terms['L_Sl'] = sum(penalties).sum(dim=1).mean()
        return terms


def update_paper_coefficients(coeffs, losses, upper):
    """Eq.12 before the current batch gradient; coefficient ratios are detached."""
    objective = float(losses['L_obj'].detach())
    if not math.isfinite(objective):
        raise ValueError('Nonfinite paper objective')
    for weight, key in [('k_g', 'L_g'), ('k_Sl', 'L_Sl'), ('k_theta', 'L_theta'),
                        ('k_z', 'L_z'), ('k_d', 'L_d')]:
        value = float(losses[key].detach())
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid constraint loss: {key}')
        if value == 0:
            ratio = upper[weight]  # zero residual and gradient: finite zero-division convention
        else:
            ratio = coeffs['k_obj'] * objective / value
        coeffs[weight] = min(ratio, upper[weight])


def supervised_voltage_loss(v_pred, theta_pred, v_true, theta_true):
    return ((v_pred-v_true).square() + (theta_pred-theta_true).square()).sum(dim=1).mean()


def total_loss_supervised(voltage_loss, losses, k_v, coeffs):
    return k_v * voltage_loss + sum(coeffs[w] * losses[k] for w, k in [
        ('k_g', 'L_g'), ('k_Sl', 'L_Sl'), ('k_theta', 'L_theta'), ('k_z', 'L_z'), ('k_d', 'L_d')])
