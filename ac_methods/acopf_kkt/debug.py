# -*- coding: utf-8 -*-
"""
验证脚本: 检查 PinnModel 的隐藏层参数是否进了 Adam 优化器, 以及是否真的在训练。

放在与下面三个文件同一目录运行即可 (脚本里已自动处理 import):
  acopf_densecorenetwork.py
  acopf_pinnlayer.py
  acopf_pinnmodel.py

用法:
  python verify_optimizer_coverage.py

它用一个合成的小电网 (3 母线 / 2 发电机 / 2 支路) 构造真实的 PinnModel,
不需要你的数据集。如果你想用真实 params, 看文件末尾的说明。
"""

import numpy as np
import torch

from acopf_pinnmodel import PinnModel


# ---------------------------------------------------------------------
# 1. 合成一个最小可用的 simulation_parameters
# ---------------------------------------------------------------------
def build_synthetic_params():
    n_buses = 3
    bus_ids = np.array([1, 2, 3])
    bus_types = np.array([3, 2, 1])          # slack(3), PV(2), PQ(1)
    bus_id_to_idx = {1: 0, 2: 1, 3: 2}

    gen_bus_ids = np.array([1, 2])           # gen0 在 slack 母线, gen1 在 PV 母线
    n_gen = 2
    slack_gen_mask = np.array([True, False])
    non_slack_gen_idx = np.array([1])
    n_gen_non_slack = 1

    load_bus_ids = np.array([1, 2, 3])
    n_loads = 3

    f_bus = np.array([1, 2])
    t_bus = np.array([2, 3])
    n_branches = 2

    return {
        'general': {
            'n_buses': n_buses, 'n_gen': n_gen,
            'n_gen_non_slack': n_gen_non_slack, 'n_branches': n_branches,
            'n_loads': n_loads, 'gen_bus_ids': gen_bus_ids,
            'load_bus_ids': load_bus_ids, 'BASE_MVA': 100.0,
            'bus_ids': bus_ids, 'bus_types': bus_types,
            'bus_id_to_idx': bus_id_to_idx,
            'slack_gen_mask': slack_gen_mask,
            'non_slack_gen_idx': non_slack_gen_idx,
        },
        'generator': {
            'pg_min': np.array([[0.0, 0.0]], dtype=np.float32),
            'pg_max': np.array([[2.0, 2.0]], dtype=np.float32),
            'qg_min': np.array([[-1.0, -1.0]], dtype=np.float32),
            'qg_max': np.array([[1.0, 1.0]], dtype=np.float32),
            'cost_c1': np.array([10.0, 20.0], dtype=np.float32),
            'cost_c2': np.array([0.1, 0.1], dtype=np.float32),
            'cost_c0': np.array([0.0, 0.0], dtype=np.float32),
        },
        'bus': {
            'vm_min': np.array([0.9, 0.9, 0.9], dtype=np.float32),
            'vm_max': np.array([1.1, 1.1, 1.1], dtype=np.float32),
        },
        'branch': {
            'f_bus': f_bus, 't_bus': t_bus,
            'r_pu': np.array([0.01, 0.02], dtype=np.float64),
            'x_pu': np.array([0.10, 0.20], dtype=np.float64),
            'b_pu': np.array([0.02, 0.04], dtype=np.float64),
            'rate_a': np.array([1.0, 1.0], dtype=np.float64),
            'tap_ratio': np.array([1.0, 1.0], dtype=np.float64),
            'shift_deg': np.array([0.0, 0.0], dtype=np.float64),
        },
        'training': {
            'neurons_in_hidden_layers_V': [64, 32],
            'neurons_in_hidden_layers_G': [64, 32],
            'neurons_in_hidden_layers_Lg': [64, 32],
        },
    }


def is_hidden(name):
    """命名里含 'hidden' 的是隐藏层, 其余 (g_output/v_output/lg_*) 视为输出层。"""
    return 'hidden' in name


def make_dummy_targets(model, batch, device):
    """跑一次前向拿到各输出形状, 据此造随机监督目标 (供 compute_loss 用)。"""
    x = torch.randn(batch, 2 * model.pinn_layer.n_loads, device=device)
    with torch.no_grad():
        out = model(x)
    keys = ['pg_qg', 'v_rect', 'lambda_p', 'mu_g_u', 'mu_g_d',
            'mu_v_u', 'mu_v_d', 'mu_sm_fr', 'mu_sm_to']
    targets = {k: torch.rand_like(out[k]) for k in keys}
    return x, targets


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    print(f"PyTorch version: {torch.__version__}")

    params = build_synthetic_params()

    # 关键: 像 acopf_pinn_main.py 那样, 先建模型(优化器在 __init__ 里建好), 再 .to(device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device for main test: {device}")

    model = PinnModel(
        simulation_parameters=params,
        lambda_P=10.0, lambda_V=10.0, lambda_L=1e-3, lambda_eps=1e-4,
        collocation_ratio=0.5, learning_rate=1e-3, device=device,
    ).to(device)

    # ================================================================
    # 测试 1: 优化器是否覆盖了所有参数 (尤其隐藏层)
    # ================================================================
    banner("测试 1: 优化器参数覆盖")

    model_param_ids = {id(p): n for n, p in model.named_parameters()}
    opt_param_ids = set()
    for g in model.optimizer.param_groups:
        for p in g['params']:
            opt_param_ids.add(id(p))

    missing = [(n) for pid, n in model_param_ids.items() if pid not in opt_param_ids]
    hidden_total = sum(1 for n in model_param_ids.values() if is_hidden(n))
    hidden_missing = [n for n in missing if is_hidden(n)]

    print(f"  模型参数张量数: {len(model_param_ids)}")
    print(f"  优化器中参数数: {len(opt_param_ids)}")
    print(f"  其中隐藏层参数张量数: {hidden_total}")
    if missing:
        print(f"  ✗ 有 {len(missing)} 个参数不在优化器里:")
        for n in missing:
            print(f"      - {n}")
        print(f"  ✗ 其中隐藏层缺失: {len(hidden_missing)} 个 -> {hidden_missing}")
        verdict1 = "FAIL (隐藏层确实没进优化器)" if hidden_missing else "FAIL (有非隐藏层缺失)"
    else:
        print("  ✓ 所有参数 (含全部隐藏层) 都在优化器里")
        verdict1 = "PASS"
    print(f"  >>> 测试 1 结论: {verdict1}")

    # ================================================================
    # 测试 2: 训练若干步后, 隐藏层权重到底动没动
    # ================================================================
    banner("测试 2: 真实训练步后的参数变化")

    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    batch = 16
    x, targets = make_dummy_targets(model, batch, device)
    mask = torch.ones(batch, 1, device=device)   # 全监督, 让 mae_g/v/l 都生效

    model.train()
    for step in range(20):
        model.optimizer.zero_grad()
        out = model(x)
        loss, _ = model.compute_loss(out, targets, mask)
        loss.backward()
        model.optimizer.step()

    print(f"  跑了 20 步, 最终 loss = {loss.item():.6f}")
    print(f"\n  各参数训练前后最大绝对变化 (>0 表示在学):")
    hidden_changed = output_changed = 0
    hidden_dead = []
    for n, p in model.named_parameters():
        delta = (p.detach() - before[n]).abs().max().item()
        tag = "HIDDEN" if is_hidden(n) else "output"
        moved = delta > 1e-12
        if is_hidden(n):
            hidden_changed += int(moved)
            if not moved:
                hidden_dead.append(n)
        else:
            output_changed += int(moved)
        print(f"    [{tag:6s}] {n:42s} Δmax = {delta:.3e} {'(动)' if moved else '(没动!)'}")

    print(f"\n  隐藏层有变化的张量: {hidden_changed}/{hidden_total}")
    if hidden_dead:
        print(f"  ✗ 以下隐藏层没动 (疑似没被训练): {hidden_dead}")
        verdict2 = "FAIL (隐藏层没被训练)"
    else:
        print("  ✓ 所有隐藏层权重都发生了变化 -> 隐藏层确实在被训练")
        verdict2 = "PASS"
    print(f"  >>> 测试 2 结论: {verdict2}")

    # ================================================================
    # 测试 3: backward 后隐藏层是否有梯度
    # ================================================================
    banner("测试 3: 隐藏层梯度")

    model.optimizer.zero_grad()
    out = model(x)
    loss, _ = model.compute_loss(out, targets, mask)
    loss.backward()

    no_grad_hidden = []
    for n, p in model.named_parameters():
        if is_hidden(n):
            g = p.grad
            gn = 0.0 if g is None else g.abs().sum().item()
            if g is None or gn == 0.0:
                no_grad_hidden.append((n, gn))
    if no_grad_hidden:
        print(f"  ✗ 这些隐藏层梯度为 0 或 None: {no_grad_hidden}")
        verdict3 = "FAIL"
    else:
        print("  ✓ 所有隐藏层都有非零梯度")
        verdict3 = "PASS"
    print(f"  >>> 测试 3 结论: {verdict3}")

    # ================================================================
    # 测试 4: .to(device) 后参数对象身份是否保持, 优化器是否仍引用活参数
    # ================================================================
    banner("测试 4: .to(device) 是否破坏优化器引用 (复现 main 的调用顺序)")

    if not torch.cuda.is_available():
        print("  (无 CUDA, 仅在 CPU 上构造, 用 .to('cpu') 走同一代码路径检查身份)")
        target_dev = 'cpu'
    else:
        target_dev = 'cuda'

    m2 = PinnModel(simulation_parameters=params, device='cpu')  # 优化器在 CPU 上建
    ids_before = {n: id(p) for n, p in m2.named_parameters()}
    opt_ids_before = set(id(p) for g in m2.optimizer.param_groups for p in g['params'])

    m2 = m2.to(target_dev)  # <-- main 里就是这么写的

    ids_after = {n: id(p) for n, p in m2.named_parameters()}
    opt_ids_after = set(id(p) for g in m2.optimizer.param_groups for p in g['params'])
    live_ids_after = set(id(p) for p in m2.parameters())

    identity_preserved = all(ids_before[n] == ids_after[n] for n in ids_before)
    opt_refs_live = opt_ids_after.issubset(live_ids_after) and len(opt_ids_after) == len(live_ids_after)

    print(f"  .to('{target_dev}') 后参数对象 id 是否全部保持不变: {identity_preserved}")
    print(f"  优化器引用的参数是否 == 模型当前的活参数: {opt_refs_live}")
    if identity_preserved and opt_refs_live:
        print("  ✓ .to() 没有破坏优化器引用 (就地搬运, 优化器仍指向活参数)")
        verdict4 = "PASS"
    else:
        print("  ✗ .to() 后优化器引用了过期参数 -> 这才是真正会让某些层不更新的 bug")
        verdict4 = "FAIL (建议: 先 .to(device) 再建优化器)"
    print(f"  >>> 测试 4 结论: {verdict4}")

    # ================================================================
    banner("汇总")
    print(f"  测试1 优化器覆盖     : {verdict1}")
    print(f"  测试2 训练步参数变化 : {verdict2}")
    print(f"  测试3 隐藏层梯度     : {verdict3}")
    print(f"  测试4 .to(device)引用: {verdict4}")
    print("\n  解读:")
    print("   - 1/2/3 全 PASS  => 隐藏层进了优化器且确实在训练, '只有 output 在学' 不成立。")
    print("   - 若 2 FAIL 而 1 PASS => 参数在优化器里但没动, 要查梯度/学习率/掩码, 而非懒加载。")
    print("   - 若 4 FAIL        => 你的 PyTorch 版本下 .to() 重建了参数, 这才是真 bug。")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# 想用真实参数而不是合成系统? 把 build_synthetic_params() 换成:
#
#   from acopf_data_setup import load_parameters_from_csv
#   params = load_parameters_from_csv(case_name, params_path)
#   params['training'] = {
#       'neurons_in_hidden_layers_V':  [64, 32],
#       'neurons_in_hidden_layers_G':  [64, 32],
#       'neurons_in_hidden_layers_Lg': [64, 32],
#   }
#
# 其余测试逻辑完全一致。
# ---------------------------------------------------------------------