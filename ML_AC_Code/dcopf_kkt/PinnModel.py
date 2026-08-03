import os
import sys
import torch
import torch.nn as nn

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入 Slack 版本的 PinnLayer
try:
    from PinnLayer_slack import PinnLayer
except ImportError:
    from PinnLayer import PinnLayer


class PinnModel(nn.Module):
    """
    PyTorch版本的PINN模型（集成 Slack Bus + Collocation Points）

    版本: v3.0 - Collocation Points Integration

    说明:
    ----
    v3.0 新增:
    - compute_collocation_loss(): 仅计算 KKT 物理损失（无监督标签）
    - compute_supervised_loss(): 计算完整监督损失（MAE_p + MAE_l + MAE_ε）
    - compute_loss(): 保持向后兼容（原始全监督模式）
    """

    def __init__(self, weight1, weight2, simulation_parameters, learning_rate=0.001, device='cuda'):
        super(PinnModel, self).__init__()

        self.device = device

        # 使用支持 Slack Bus 的 PinnLayer
        self.pinn_layer = PinnLayer(simulation_parameters=simulation_parameters, device=device)

        # 损失权重
        # [Pg, lambda, mu_g_min, mu_g_max, mu_line_pos, mu_line_neg, KKT_error]
        self.loss_weights = [1.0, weight1, weight1, weight1, weight1, weight1, weight2*1e-8]

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # 损失函数 (MAE)
        self.criterion = nn.L1Loss()

    def forward(self, inputs):
        """
        前向传播

        返回:
        ----
        outputs[0]: pg_non_slack  [batch, n_g_non_slack]
        outputs[1]: lambda        [batch, 1]
        outputs[2]: mu_g_up       [batch, n_g]
        outputs[3]: mu_g_down     [batch, n_g]
        outputs[4]: mu_line_up    [batch, n_line]
        outputs[5]: mu_line_down  [batch, n_line]
        outputs[6]: kkt_error     [batch]
        """
        return self.pinn_layer(inputs)

    def compute_loss(self, outputs, targets):
        """
        计算加权损失（向后兼容的全监督模式）

        所有 7 个输出都与目标计算 MAE，适用于原始全监督训练。
        """
        total_loss = 0.0
        losses = []

        for i, (output, target, weight) in enumerate(zip(outputs, targets, self.loss_weights)):
            loss = self.criterion(output, target)
            weighted_loss = weight * loss
            total_loss += weighted_loss
            losses.append(loss.item())

        return total_loss, losses

    def compute_supervised_loss(self, outputs, targets):
        """
        计算监督样本的损失（与 compute_loss 相同）

        Supervised 样本拥有完整标签:
          targets = (pg, lambda, mu_g_min, mu_g_max, mu_line_pos, mu_line_neg, physics_zeros)

        损失 = Λ_P·MAE_p + Λ_L·MAE_l + Λ_ε·MAE_ε
        """
        return self.compute_loss(outputs, targets)

    def compute_collocation_loss(self, outputs):
        """
        计算 Collocation 样本的损失（仅物理损失，无监督标签）

        论文公式(19)中，collocation points 不提供 Pg 或 Lm 的真实值，
        仅通过 KKT 误差（MAE_ε）进行自监督训练。

        Collocation 损失 = weight_kkt × mean(KKT_error)

        参数:
        ----
        outputs: 模型前向传播输出
            outputs[6] 是 KKT_error [batch]

        返回:
        ----
        total_loss: 仅包含 KKT 物理损失
        losses: 各项损失值列表（前6项为0，第7项为KKT损失）
        """
        kkt_error = outputs[6]  # [batch]
        kkt_loss = torch.mean(kkt_error)

        weight_kkt = self.loss_weights[6]  # weight2
        total_loss = weight_kkt * kkt_loss

        # 返回与 compute_loss 对齐的 losses 列表（方便日志记录）
        losses = [0.0] * 6 + [kkt_loss.item()]

        return total_loss, losses

    def predict(self, x):
        """预测（推理模式）"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
        return outputs