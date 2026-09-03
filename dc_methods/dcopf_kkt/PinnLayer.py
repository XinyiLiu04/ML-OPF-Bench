import os
import sys
import torch
import torch.nn as nn
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from DenseCoreNetwork import DenseCoreNetwork
except ImportError:
    pass


class PinnLayer(nn.Module):
    """
    PINN Layer - 严格对齐 paper TF 实现 + 本工程 Slack Bus 集成

    版本: v3.2 — 回到 paper TF 约定的训练尺度

    关于 v3.0 / v3.1 的反思:
    ------------------------
    v3.0 完全按论文 Eq.(15)-(18) 的"求和"形式去掉了分母, 结果 KKT_error 量级
    达到 10^3-10^4, 把 MAE_p、MAE_l 训练信号完全压死, 造成 Non-Slack MAE
    从 ~1% 恶化到 ~15%。

    仔细读 paper 的 TF 参考实现 (`paper_pinnlayer`) 发现, 论文公式和代码之间
    存在两个 *用于训练数值稳定的* pragmatic 差别:
      (a) comp 项中 μ 用 **scaled** 值 (n_o_a_u) 直接乘 (P_Gens − Pg_max),
          而不是反缩放后的物理 μ。这让 comp 项保持 O(1)。
      (b) dual 项 π(μ) 对 scaled μ 取 relu(−·), 无分母。

    v3.2 严格遵循这两个约定, 同时保留 v3.0 所发现的真正的 bug 修复:

    --- 确定的 bug 修复 (必须保留) ---
    [FIX #1] ε_stat 中 λ 必须广播到 N_bus 并放进 abs() 内部。这是 paper
             Eq.(5)/(15) 的数学要求, paper 的 TF 代码把 λ 单独加在 abs 外
             (对应项:  `tf.reduce_sum(n_o_l, axis=1)*Lg_Max[0]/100 + ...`),
             这是 paper 代码的一个轻微偏差。v3.2 按数学定义修正。
    [FIX #2] Pg 与 Pg_max/Pg_min 在 primal/comp 中必须量纲一致 (均归一化空间)。
             注: paper TF 里 Pg_max 就是归一化的 (= 1.0), 所以这个问题其实
             是我们工程里 Pg_max = 物理 p.u. 导致的 — 本项修复才能正确。
    [FIX #3] 对偶变量反缩放: 在 stat 内部必须用物理 μ。当 MinMaxScaler 的
             data_min_ ≠ 0 时 `scaled * Lg_Max` 错, 应改为 full MinMax 反变换;
             fallback 到 Lg_Max 保持向后兼容。
    [FIX #4] 移除非 Slack Pg 的 [0, 1.2] clamp (让 ε_prim 上界项有梯度)。
    [FIX #5] 功率平衡用绝对值 (paper Eq.(18)), 但保留 /max(L_max) 归一化
             (paper TF 代码这一项确实也归一化了, 见第 34 行)。

    --- FIX #6 的处理 ---
    保留 paper TF 里原有的 per-unit 归一化分母 (/n_gbus, /(max·n_line), /100),
    让 KKT_error 维持 O(1) 量级, 不破坏 loss 平衡。这与 v3.1 的精神一致,
    但具体分母和 paper TF 完全一致, 不再用 `pg_scale * n_line` 等自造常数。

    另外:
    - KKT_error 输出从 [B] 改为 [B, 1], 修复 L1Loss broadcasting warning。
    """

    def __init__(self, simulation_parameters, device='cuda'):
        super(PinnLayer, self).__init__()
        self.device = device
        self.eps = 1e-8

        # ---------- P_d 反缩放参数 ----------
        self.pd_scale_type = simulation_parameters.get('pd_scale_type', None)
        if self.pd_scale_type == 'minmax':
            self.pd_min = torch.tensor(simulation_parameters['pd_min'], dtype=torch.float32, device=device)
            self.pd_max = torch.tensor(simulation_parameters['pd_max'], dtype=torch.float32, device=device)
        elif self.pd_scale_type == 'standard':
            self.pd_mean = torch.tensor(simulation_parameters['pd_mean'], dtype=torch.float32, device=device)
            self.pd_std = torch.tensor(simulation_parameters['pd_std'], dtype=torch.float32, device=device)

        # ---------- 系统参数 ----------
        self.n_buses = simulation_parameters['general']['n_buses']
        self.n_g = simulation_parameters['general']['n_g']
        self.n_g_non_slack = simulation_parameters['general']['n_g_non_slack']
        self.n_line = simulation_parameters['general']['n_line']

        slack_gen_indices = simulation_parameters['general']['slack_gen_indices']
        non_slack_gen_indices = simulation_parameters['general']['non_slack_gen_indices']
        self.slack_gen_indices = torch.tensor(slack_gen_indices, dtype=torch.long, device=device)
        self.non_slack_gen_indices = torch.tensor(non_slack_gen_indices, dtype=torch.long, device=device)
        self.n_slack_gens = len(slack_gen_indices)

        print(f"\n[PinnLayer v3.2 paper-TF-aligned] Initialization:")
        print(f"  Total generators: {self.n_g}")
        print(f"  Non-Slack generators: {self.n_g_non_slack}")
        print(f"  Slack generators: {self.n_slack_gens}")
        print(f"  Slack generator indices: {slack_gen_indices}")

        # ---------- 核心网络 ----------
        neurons_pg = simulation_parameters['training']['neurons_in_hidden_layers_Pg']
        neurons_lm = simulation_parameters['training']['neurons_in_hidden_layers_Lm']
        self.core_network = DenseCoreNetwork(
            input_size=self.n_buses,
            n_gbus_non_slack=self.n_g_non_slack,
            n_gbus_all=self.n_g,
            n_line=self.n_line,
            neurons_in_hidden_layers_Pg=neurons_pg,
            neurons_in_hidden_layers_Lm=neurons_lm
        ).to(device)

        # ---------- 约束参数 ----------
        self.C_Pg = torch.tensor(simulation_parameters['constraints']['C_Pg'],
                                 dtype=torch.float32, device=device)
        Pg_min_phys = torch.tensor(simulation_parameters['constraints']['Pg_min'],
                                   dtype=torch.float32, device=device)
        Pg_max_phys = torch.tensor(simulation_parameters['constraints']['Pg_max'],
                                   dtype=torch.float32, device=device)
        self.Pl_max = torch.tensor(simulation_parameters['constraints']['Pl_max'],
                                   dtype=torch.float32, device=device)
        self.Pg_max_real = torch.tensor(simulation_parameters['constraints']['Pg_max_real'],
                                        dtype=torch.float32, device=device)
        self.PTDF = torch.tensor(simulation_parameters['constraints']['PTDF'],
                                 dtype=torch.float32, device=device)
        self.Map_g = torch.tensor(simulation_parameters['constraints']['Map_g'],
                                  dtype=torch.float32, device=device)
        self.Map_L = torch.tensor(simulation_parameters['constraints']['Map_L'],
                                  dtype=torch.float32, device=device)

        if Pg_min_phys.ndim == 2: Pg_min_phys = Pg_min_phys.flatten()
        if Pg_max_phys.ndim == 2: Pg_max_phys = Pg_max_phys.flatten()
        if self.Pg_max_real.ndim == 2: self.Pg_max_real = self.Pg_max_real.flatten()

        # ============================================================
        # FIX #2: Pg_max / Pg_min 归一化空间
        # ============================================================
        self.Pg_max_norm = Pg_max_phys / (self.Pg_max_real + self.eps)   # [n_g], ≈ 1
        self.Pg_min_norm = Pg_min_phys / (self.Pg_max_real + self.eps)   # [n_g], ≈ 0
        self.Pg_max_phys = Pg_max_phys
        self.Pg_min_phys = Pg_min_phys

        # ---------- 对偶反缩放 (FIX #3) ----------
        self.Lg_Max = simulation_parameters['Lg_Max']
        dual_scalers = simulation_parameters.get('dual_scalers', None)
        self._has_full_dual_scalers = dual_scalers is not None
        if self._has_full_dual_scalers:
            def _to_t(arr):
                return torch.tensor(np.asarray(arr).flatten(),
                                    dtype=torch.float32, device=device)
            a, b = dual_scalers['lambda']
            self.lambda_min   = _to_t(a); self.lambda_range = _to_t(b) - self.lambda_min
            a, b = dual_scalers['mu_g_max']
            self.mu_g_max_min   = _to_t(a); self.mu_g_max_range = _to_t(b) - self.mu_g_max_min
            a, b = dual_scalers['mu_g_min']
            self.mu_g_min_min   = _to_t(a); self.mu_g_min_range = _to_t(b) - self.mu_g_min_min
            a, b = dual_scalers['mu_line_pos']
            self.mu_line_pos_min   = _to_t(a); self.mu_line_pos_range = _to_t(b) - self.mu_line_pos_min
            a, b = dual_scalers['mu_line_neg']
            self.mu_line_neg_min   = _to_t(a); self.mu_line_neg_range = _to_t(b) - self.mu_line_neg_min
            print(f"  Dual descaling: full MinMax params (FIX #3 active)")
        else:
            print(f"  Dual descaling: legacy Lg_Max fallback (assumes data_min_=0)")

        self.BASE_MVA = simulation_parameters['general'].get('BASE_MVA', 100.0)

        # 对齐作者源码的归一化分母 np.max(L_max) = 系统最大单点负荷。
        # 由 main 传入 simulation_parameters['load_scale'] = max(x_scaler.data_max_)。
        # 不设默认值: 缺失时直接 KeyError 暴露。
        self.load_scale = torch.tensor(simulation_parameters['load_scale'],
                                       dtype=torch.float32, device=device)
        print(f"  Load scale (max pd_max):    {self.load_scale.item():.4f} p.u.")

    # ------------------------------------------------------------------
    def _descale_dual(self, scaled, name):
        """FIX #3 helper. 仅 stat 项使用 (paper comp/dual 直接用 scaled)。"""
        if self._has_full_dual_scalers:
            if name == 'lambda':      return scaled * self.lambda_range + self.lambda_min
            if name == 'mu_g_max':    return scaled * self.mu_g_max_range + self.mu_g_max_min
            if name == 'mu_g_min':    return scaled * self.mu_g_min_range + self.mu_g_min_min
            if name == 'mu_line_pos': return scaled * self.mu_line_pos_range + self.mu_line_pos_min
            if name == 'mu_line_neg': return scaled * self.mu_line_neg_range + self.mu_line_neg_min
        if name == 'lambda':      return scaled * self.Lg_Max[0]
        if name == 'mu_g_max':    return scaled * self.Lg_Max[1]
        if name == 'mu_g_min':    return scaled * self.Lg_Max[2]
        if name == 'mu_line_pos': return scaled * self.Lg_Max[3]
        if name == 'mu_line_neg': return scaled * self.Lg_Max[4]
        raise KeyError(name)

    # ------------------------------------------------------------------
    # FIX #4: 重建完整 Pg, 非 Slack 不 clamp
    # ------------------------------------------------------------------
    def _reconstruct_full_pg(self, pg_non_slack, pd_total):
        batch_size = pg_non_slack.shape[0]
        device = pg_non_slack.device

        pg_full = torch.zeros(batch_size, self.n_g,
                              dtype=pg_non_slack.dtype, device=device)
        pg_full[:, self.non_slack_gen_indices] = pg_non_slack

        pg_non_slack_real = pg_non_slack * self.Pg_max_real[self.non_slack_gen_indices].unsqueeze(0)
        pg_non_slack_total = torch.sum(pg_non_slack_real, dim=1)

        pg_slack_total = pd_total - pg_non_slack_total

        if self.n_slack_gens > 0:
            pg_slack_per_gen = pg_slack_total / (self.n_slack_gens + self.eps)
            slack_pg_max_real = self.Pg_max_real[self.slack_gen_indices]
            pg_slack_normalized = pg_slack_per_gen.unsqueeze(1) / (slack_pg_max_real.unsqueeze(0) + self.eps)
            pg_full[:, self.slack_gen_indices] = pg_slack_normalized

        return pg_full

    # ==================================================================
    # KKT error — 严格对齐 paper TF 实现 + FIX #1/#2/#3/#4/#5
    # ==================================================================
    def get_kkt_error(self, P_Gens, P_Loads, n_o_l, n_o_a_u, n_o_a_d, n_o_b_u, n_o_b_d):
        """
        参数与 paper TF 对应关系:
          P_Gens  ↔ Pg (normalized, [B, n_g])
          P_Loads ↔ Pd (物理 p.u. [B, n_buses])
          n_o_l   ↔ scaled λ   [B, 1]
          n_o_a_u ↔ scaled μ̄_g [B, n_g]
          n_o_a_d ↔ scaled μ_g [B, n_g]
          n_o_b_u ↔ scaled μ̄_l [B, n_line]
          n_o_b_d ↔ scaled μ_l [B, n_line]

        与 paper TF 的唯一语义差异 (FIX #1): λ 项广播进 abs 内部。
        """

        # ========== 公共物理量 ==========
        Pg_phys = P_Gens * self.Pg_max_real              # [B, n_g]
        total_gen  = torch.sum(Pg_phys, dim=1)           # [B]
        total_load = torch.sum(P_Loads, dim=1)           # [B]

        P_gen_bus  = torch.matmul(Pg_phys, self.Map_g)   # [B, n_buses]
        P_load_bus = torch.matmul(P_Loads, self.Map_L)   # [B, n_buses]
        net_inj    = P_gen_bus - P_load_bus
        line_flows = torch.matmul(net_inj, self.PTDF)    # [B, n_line]

        # ============================================================
        # ε_prim — paper Eq.(18) & TF 行 34-38
        # FIX #5: 功率平衡绝对误差, 分母用 load_scale (= max pd_max, 对齐源码 max(L_max))
        # FIX #2: Pg_max_norm / Pg_min_norm (归一化空间)
        # ============================================================
        # 功率平衡
        eps_prim = torch.abs(total_gen - total_load) / self.load_scale

        # 发电机上/下限 (归一化)
        eps_prim = eps_prim + torch.sum(torch.relu(P_Gens - self.Pg_max_norm), dim=1) / self.n_g
        eps_prim = eps_prim + torch.sum(torch.relu(self.Pg_min_norm - P_Gens), dim=1) / self.n_g

        # 线路上/下限 (物理)
        eps_prim = eps_prim + torch.sum(torch.relu( line_flows - self.Pl_max), dim=1) / (self.load_scale * self.n_line)
        eps_prim = eps_prim + torch.sum(torch.relu(-line_flows - self.Pl_max), dim=1) / (self.load_scale * self.n_line)

        # ============================================================
        # ε_stat — paper Eq.(15) & TF 行 41
        # FIX #1: λ 广播进 abs (paper TF 这里是 bug, 我们修正)
        # FIX #3: 用 full MinMax 反缩放 (paper TF 假设 data_min_=0)
        # ============================================================
        lam_phys     = self._descale_dual(n_o_l,   'lambda')
        mu_g_up_phys = self._descale_dual(n_o_a_u, 'mu_g_max')
        mu_g_dn_phys = self._descale_dual(n_o_a_d, 'mu_g_min')
        mu_l_up_phys = self._descale_dual(n_o_b_u, 'mu_line_pos')
        mu_l_dn_phys = self._descale_dual(n_o_b_d, 'mu_line_neg')

        c_at_bus    = torch.matmul(self.C_Pg.unsqueeze(0), self.Map_g)   # [1, n_buses]
        lam_bcast   = lam_phys.expand(-1, self.n_buses)                  # [B, n_buses]
        mu_g_up_bus = torch.matmul(mu_g_up_phys, self.Map_g)
        mu_g_dn_bus = torch.matmul(mu_g_dn_phys, self.Map_g)
        mu_l_up_bus = torch.matmul(mu_l_up_phys, self.PTDF.t())
        mu_l_dn_bus = torch.matmul(mu_l_dn_phys, self.PTDF.t())

        # c + λ + μ̄_g − μ_g + μ̄_l·PTDF − μ_l·PTDF
        stationarity = (c_at_bus + lam_bcast
                        + mu_g_up_bus - mu_g_dn_bus
                        + mu_l_up_bus - mu_l_dn_bus)
        # paper TF 里的分母就是 /100, 这里保留
        eps_stat = torch.sum(torch.abs(stationarity), dim=1) / 100.0

        # ============================================================
        # ε_comp — paper Eq.(16) & TF 行 44-47
        # 严格按 paper TF: 使用 **scaled 对偶** 直接乘 violation, 不做反缩放
        # (这让 comp 项保持 O(1), 是训练稳定的关键)
        # FIX #2: Pg_max_norm / Pg_min_norm
        # ============================================================
        comp_up_g = torch.abs(n_o_a_u * (P_Gens - self.Pg_max_norm))
        comp_dn_g = torch.abs(n_o_a_d * (self.Pg_min_norm - P_Gens))
        eps_comp  = torch.sum(comp_up_g, dim=1) / self.n_g
        eps_comp = eps_comp + torch.sum(comp_dn_g, dim=1) / self.n_g

        comp_up_l = torch.abs(n_o_b_u * ( line_flows - self.Pl_max))
        comp_dn_l = torch.abs(n_o_b_d * (-line_flows - self.Pl_max))
        eps_comp = eps_comp + torch.sum(comp_up_l, dim=1) / (self.load_scale * self.n_line)
        eps_comp = eps_comp + torch.sum(comp_dn_l, dim=1) / (self.load_scale * self.n_line)

        # ============================================================
        # ε_dual — paper Eq.(17) & TF 行 50-53
        # 严格按 paper TF: relu(−scaled dual), 无分母
        # ============================================================
        eps_dual  = torch.sum(torch.relu(-n_o_a_u), dim=1)
        eps_dual = eps_dual + torch.sum(torch.relu(-n_o_a_d), dim=1)
        eps_dual = eps_dual + torch.sum(torch.relu(-n_o_b_u), dim=1)
        eps_dual = eps_dual + torch.sum(torch.relu(-n_o_b_d), dim=1)

        # ============================================================
        # 总 KKT 误差, 输出 [B, 1]
        # ============================================================
        KKT_error = eps_stat + eps_comp + eps_dual + eps_prim
        return KKT_error.unsqueeze(-1)   # [B, 1]

    # ==================================================================
    def forward(self, inputs):
        (g_ns, n_l, n_au, n_ad, n_bu, n_bd) = self.core_network(inputs)

        if self.pd_scale_type == 'minmax':
            P_Loads_unscaled = inputs * (self.pd_max - self.pd_min) + self.pd_min
        elif self.pd_scale_type == 'standard':
            P_Loads_unscaled = inputs * self.pd_std + self.pd_mean
        else:
            P_Loads_unscaled = inputs

        pd_total = torch.sum(P_Loads_unscaled, dim=1)
        g_full = self._reconstruct_full_pg(g_ns, pd_total)

        KKT_error = self.get_kkt_error(
            P_Gens=g_full, P_Loads=P_Loads_unscaled,
            n_o_l=n_l, n_o_a_u=n_au, n_o_a_d=n_ad, n_o_b_u=n_bu, n_o_b_d=n_bd
        )

        return (g_ns, n_l, n_au, n_ad, n_bu, n_bd, KKT_error)