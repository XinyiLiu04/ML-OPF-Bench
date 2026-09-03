import torch
import torch.nn as nn


class DenseCoreNetwork(nn.Module):
    """
    PINN模型的核心神经网络 - PyTorch版本（集成 Slack Bus）

    版本: v3.0 - 去除懒加载 (Eager Init)

    v3.0 修订:
    --------
    彻底移除 v2.x 的懒加载机制 (_initialize_layers / _initialized)。
    根因: 隐藏层在首次 forward 时才创建, 而 PinnModel 在 __init__ 中
    forward 之前就用 Adam(self.parameters()) 锁定了参数快照, 导致懒加载
    出来的隐藏层永远不进优化器、永不更新 (实测 grad 存在但 step 不更新)。

    解决: 构造时直接传入 input_size, 在 __init__ 内一次性建好全部隐藏层。
    这样 PinnModel 创建优化器时所有参数都已就位, 无需 dummy forward。

    设计:
    --------
    1. Pg 输出维度: n_gbus_non_slack（只预测非 Slack）
    2. 对偶变量维度: n_gbus_all（所有发电机，包括 Slack）
    3. 隐藏层在 __init__ 内按 input_size 构建好
    """

    def __init__(self, input_size, n_gbus_non_slack, n_gbus_all, n_line,
                 neurons_in_hidden_layers_Pg, neurons_in_hidden_layers_Lm):
        super(DenseCoreNetwork, self).__init__()

        self.input_size = input_size
        self.n_gbus_non_slack = n_gbus_non_slack
        self.n_gbus_all = n_gbus_all
        self.n_line = n_line

        # ========== Pg 网络隐藏层 (eager 构建) ==========
        pg_layers = []
        prev_size = input_size
        for n_units in neurons_in_hidden_layers_Pg:
            pg_layers.append(nn.Linear(prev_size, n_units))
            pg_layers.append(nn.ReLU())
            prev_size = n_units
        self.pg_hidden = nn.Sequential(*pg_layers)
        pg_last = prev_size

        # Pg 输出层（只输出非 Slack）
        self.pg_output = nn.Linear(pg_last, n_gbus_non_slack)

        # ========== Lm 网络隐藏层 (eager 构建) ==========
        lm_layers = []
        prev_size = input_size
        for n_units in neurons_in_hidden_layers_Lm:
            lm_layers.append(nn.Linear(prev_size, n_units))
            lm_layers.append(nn.ReLU())
            prev_size = n_units
        self.lm_hidden = nn.Sequential(*lm_layers)
        lm_last = prev_size

        # Lm 输出层 (拉格朗日乘子) - 维度保持所有发电机
        self.lm_output = nn.Linear(lm_last, 1)                  # λ (系统级)
        self.mu_g_up_output = nn.Linear(lm_last, n_gbus_all)    # 所有发电机
        self.mu_g_down_output = nn.Linear(lm_last, n_gbus_all)  # 所有发电机
        self.mu_line_up_output = nn.Linear(lm_last, n_line)
        self.mu_line_down_output = nn.Linear(lm_last, n_line)

    def forward(self, inputs):
        """
        前向传播

        返回:
        ----
        pg_output     : Tensor [batch, n_gbus_non_slack]   非 Slack 出力
        lm_output     : Tensor [batch, 1]                  λ
        mu_g_up       : Tensor [batch, n_gbus_all]
        mu_g_down     : Tensor [batch, n_gbus_all]
        mu_line_up    : Tensor [batch, n_line]
        mu_line_down  : Tensor [batch, n_line]
        """
        # Pg 网络前向传播（输出非 Slack）
        x_pg = self.pg_hidden(inputs)
        pg_output = self.pg_output(x_pg)              # [batch, n_gbus_non_slack]

        # Lm 网络前向传播（输出所有发电机的对偶）
        x_lm = self.lm_hidden(inputs)
        lm_output = self.lm_output(x_lm)              # [batch, 1]
        mu_g_up = self.mu_g_up_output(x_lm)           # [batch, n_gbus_all]
        mu_g_down = self.mu_g_down_output(x_lm)       # [batch, n_gbus_all]
        mu_line_up = self.mu_line_up_output(x_lm)     # [batch, n_line]
        mu_line_down = self.mu_line_down_output(x_lm) # [batch, n_line]

        return pg_output, lm_output, mu_g_up, mu_g_down, mu_line_up, mu_line_down