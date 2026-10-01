"""故障容错配方规划的冒烟测试（不依赖 pytest，可直接 python3 运行）。

重点：
  * 三态结论（optimal / infeasible / incomplete 的接口语义）正确；
  * 正常配方的最优性（最小正常总功率 + ID 升序字典序）用独立穷举核对：
    调整上限**只约束仍工作的通道**——熄灭路恒为 0，其归零落差不受
    max_adjust 限制，故正常驱动级可取 0～255 任意值，正常候选为完整
    256³ 空间。预言机不逐三元组枚举，而是先由颜色窗口把第三条通道的
    可行级差解成整数区间，再把每个故障场景存活通道的可行网格做 ±adj
    邻域膨胀（256 位整数位掩码），对 (d0_0, d0_1) 逐对用位运算求区间内
    最小可行 d0_2（3 路时完整精确，与 CP-SAT 模型相互独立）；
  * 所有备用配方逐项核对：故障路为 0（归零落差不受上限限制）、存活通道
    调整级差不超限、颜色窗口（整数分子精确判定，未裁剪）、0～1000、
    功率自洽；
  * 单场景预求解证明的不可行（某路熄灭后无法配色）能被识别；
  * schema 对非法参数一律 422。
"""

import random
import sys

sys.path.insert(0, "/workspace")

from pydantic import ValidationError

from app.robust_solver import solve_robust
from app.schemas import RobustSolveRequest
from app.solver import Channel, solve as solve_plain

SAMPLE = [
    Channel("R", (1000, 120, 40), 3200),
    Channel("G", (160, 1000, 90), 2800),
    Channel("B", (60, 80, 1000), 3400),
    Channel("W", (820, 800, 760), 6500),
]


def window_ok(scaled, target, tol):
    return all(
        255 * (target[c] - tol) <= scaled[c] <= 255 * (target[c] + tol)
        and scaled[c] <= 255 * 1000
        for c in range(3)
    )


def drives_scaled(channels, drives):
    return [
        sum(channels[i].rgb[c] * drives[i] for i in range(len(channels)))
        for c in range(3)
    ]


def check_result(name, channels, target, tol, adj, res):
    """对 optimal 结果逐项核对全部场景约束与自洽性。"""
    assert res.status == "optimal", f"{name}: 期望 optimal，实得 {res.status}（{res.message}）"
    assert res.normal is not None and len(res.backups) == len(channels)
    n = len(channels)
    variants = [res.normal] + res.backups

    for q, v in enumerate(variants):
        assert len(v.drives) == n and len(v.deltas) == n
        scaled = drives_scaled(channels, v.drives)
        assert scaled == v.scaled_rgb, (q, scaled, v.scaled_rgb)
        assert all(0 <= d <= 255 for d in v.drives)
        assert window_ok(scaled, target, tol), (q, scaled)
        assert v.in_range == [True, True, True]
        pnum = sum(channels[i].power_mw * v.drives[i] for i in range(n))
        assert pnum == v.power_num
        assert abs(v.power_mw - pnum / 255.0) < 1e-12
        assert all(abs(x - s / 255.0) < 1e-12
                   for x, s in zip(v.actual_rgb, scaled))
        if q == 0:
            # 正常场景：deltas 全 0
            assert v.deltas == [0] * n
        else:
            # 故障路必须为 0；其归零落差不受调整上限限制。
            # 仅仍工作的通道与正常值之差不超过上限。
            assert v.drives[q - 1] == 0
            for i in range(n):
                if i == q - 1:
                    continue
                assert abs(v.drives[i] - res.normal.drives[i]) <= adj, (
                    f"{name}: 故障{q} 存活通道{i} 调整 {v.deltas[i]} 超过上限 {adj}"
                )
            assert v.deltas == [
                v.drives[i] - res.normal.drives[i] for i in range(n)
            ]
    return variants


def _dilate_cols(cols, adj):
    """把“某 x 列可行 y 集合”的位掩码数组做 y 方向 ±adj 邻域膨胀。

    cols[x] 为 256 位整数：第 y 位为 1 表示备用对 (x, y) 落在颜色窗口内。
    膨胀后 cols[x] 的第 y 位为 1 ⇔ 原列在 [y−adj, y+adj]（裁剪到 0～255）
    内存在可行点，即给定存活驱动 y 时，另一存活驱动可在 ±adj 内补上。
    逐位累计移位（不能只对二进制置位步长各移一次——那只覆盖若干孤立
    半径的并集，并非整个 [0, adj] 区间）。
    """
    full = (1 << 256) - 1
    out = []
    for m in cols:
        dil = m
        for _ in range(adj):
            dil = ((dil << 1) | (dil >> 1) | m) & full
        out.append(dil)
    return out


def _dilate_rows(cols, adj):
    """对列掩码数组做 x 方向 ±adj 邻域膨胀（行方向由列下标承载）。"""
    out = [0] * 256
    for x, m in enumerate(cols):
        lo, hi = max(0, x - adj), min(255, x + adj)
        for xx in range(lo, hi + 1):
            out[xx] |= m
    return out


def brute_force_robust(channels, target, tol, adj):
    """3 路独立穷举预言机：返回 (最优正常驱动, 最优功率分子)；无解返回 None。

    关键语义（曾因实现与预言机同时搞错而误判）：调整上限**只约束仍工作
    的通道**。熄灭路备用驱动恒为 0，其从 d0 跌到 0 的落差不受限，因此
    正常驱动级 d0_i ∈ [0, 255] 完整枚举，不做 d0_i ≤ adj 的剪枝。

    枚举 d0=(a,b,c) 时按 (a,b) 逐对推进：先由正常颜色窗口（三条整数线性
    带）交出 c 的整数区间 [c_lo, c_hi]；每个故障场景的存活通道可行网格
    预先膨胀成位掩码（另一存活通道可在 ±adj 内补偿），于是给定 (a,b) 时
    “三个故障场景都备得出来”的 c 集合是三个 256 位掩码的按位与，在
    [c_lo, c_hi] 内取最低置位即该 (a,b) 下功率最小、C 级最小的候选。
    全程纯整数运算，与 CP-SAT 模型相互独立。
    """
    n = len(channels)
    assert n == 3
    order = sorted(range(n), key=lambda k: channels[k].id)

    def scaled_at(drives, c):
        return sum(channels[i].rgb[c] * drives[i] for i in range(n))

    lo_c = [255 * (target[c] - tol) for c in range(3)]
    hi_c = [min(255 * (target[c] + tol), 255 * 1000) for c in range(3)]

    # 每个故障场景：两条存活通道的 256×256 可行网格（位掩码按列存）。
    # 场景 f 熄灭通道 f，存活对为另外两路。
    alive = {f: tuple(i for i in range(n) if i != f) for f in range(n)}
    grids = {}
    for f, (i1, i2) in alive.items():
        cols = [0] * 256
        any_feas = False
        for x in range(256):
            m = 0
            for y in range(256):
                drives = [0] * n
                drives[i1] = x
                drives[i2] = y
                s = [scaled_at(drives, c) for c in range(3)]
                if all(lo_c[c] <= s[c] <= hi_c[c] for c in range(3)):
                    m |= 1 << y
                    any_feas = True
            cols[x] = m
        if not any_feas:
            return None  # 单场景本身不可行 → 整体必然不可行
        grids[f] = cols

    # 每个存活网格做二维 ±adj 膨胀：膨胀后 (u,v) 为 1 ⇔ 原网格在
    # [u−adj,u+adj]×[v−adj,v+adj] 内存在可行备用对。
    dil_grids = {
        f: _dilate_rows(_dilate_cols(cols, adj), adj)
        for f, cols in grids.items()
    }

    def cover_mask(f, fixed):
        """场景 f 下，固定通道正常级为 fixed 时，待选存活通道（其 ±adj
        邻域内能与固定通道的 ±adj 调整共同配出颜色）可行级差的位掩码。
        直接取自整张存活网格的二维 ±adj 膨胀结果（按列存的掩码数组）。
        """
        return dil_grids[f][fixed]

    ch0, ch1, ch2 = 0, 1, 2
    p0, p1, p2 = (channels[i].power_mw for i in (ch0, ch1, ch2))
    r0 = [channels[ch0].rgb[c] for c in range(3)]
    r1 = [channels[ch1].rgb[c] for c in range(3)]
    r2 = [channels[ch2].rgb[c] for c in range(3)]

    best = None  # ((功率分子, ID 序驱动序列), 驱动)
    for a in range(256):
        for b in range(256):
            # 正常窗口对 c 的整数区间：lo_c ≤ r0·a+r1·b+r2·c ≤ hi_c
            c_lo, c_hi = 0, 255
            feasible_window = True
            for c in range(3):
                base = r0[c] * a + r1[c] * b
                need_lo = lo_c[c] - base
                need_hi = hi_c[c] - base
                if r2[c] > 0:
                    # ceil(need_lo / r2) 与 floor(need_hi / r2)，
                    # 被除数可能为负：r2>0 时 ceil(a/b) = -((-a)//b) 恒成立
                    cl = -((-need_lo) // r2[c])
                    ch = need_hi // r2[c]
                    c_lo = max(c_lo, cl)
                    c_hi = min(c_hi, ch)
                elif not (need_lo <= 0 <= need_hi):
                    feasible_window = False
                    break
            if not feasible_window or c_lo > c_hi or c_hi < 0 or c_lo > 255:
                continue
            c_lo = max(c_lo, 0)
            c_hi = min(c_hi, 255)

            # 三个故障场景对 c 的允许集合
            # 场景 2（熄灭 C=通道2）：存活 (0,1)，c 是熄灭路，不受调整约束，
            # 只需 (a,b) 的 ±adj 邻域内存在可行备用对。
            if dil_grids[2][a] & (1 << b) == 0:
                continue
            # 场景 0（熄灭 A）：存活 (1,2)，固定 b，求允许的 c
            m0 = cover_mask(0, b)
            # 场景 1（熄灭 B）：存活 (0,2)，固定 a，求允许的 c
            m1 = cover_mask(1, a)
            cand = m0 & m1 & (((1 << (c_hi + 1)) - 1) ^ ((1 << c_lo) - 1))
            if cand == 0:
                continue
            c = (cand & -cand).bit_length() - 1  # 区间内最小可行 c
            d0 = [a, b, c]
            pnum = p0 * a + p1 * b + p2 * c
            key = (pnum, [d0[i] for i in order])
            if best is None or key < best[0]:
                best = (key, d0)
    return None if best is None else (best[1], best[0][0])


def check_against_brute(name, channels, target, tol, adj, expect_feasible):
    res = solve_robust(channels, target, tol, adj, time_limit_seconds=10)
    want = brute_force_robust(channels, target, tol, adj)
    if want is None:
        assert expect_feasible is False
        assert res.status == "infeasible", f"{name}: 穷举无解，求解器却 {res.status}"
        assert res.normal is None and res.backups == []
        print(f"[OK] {name}: 穷举与求解器一致判定无完整容错方案（{adj=}）")
        return
    assert expect_feasible is True
    assert res.status == "optimal", f"{name}: 期望 optimal，实得 {res.status}"
    check_result(name, channels, target, tol, adj, res)
    want_drives, want_power = want
    assert res.normal.power_num == want_power, (
        f"{name}: 正常功率分子 {res.normal.power_num} ≠ 穷举最优 {want_power}"
    )
    order = sorted(range(len(channels)), key=lambda k: channels[k].id)
    got_seq = [res.normal.drives[i] for i in order]
    assert got_seq == [want_drives[i] for i in order], (got_seq, want_drives)
    print(f"[OK] {name}: 正常 {res.normal.drives}（功率 {want_power}/255，"
          f"按ID序 {got_seq}）与穷举最优一致；备用 "
          f"{[b.drives for b in res.backups]}")


def main():
    # 1) 同色白光灯 ×3（满档读数相同、功率不同），目标 (31,31,31) 容差 1：
    #    实际色 = 1000·(驱动级之和)/255，窗口 [30,32] 恰要求驱动级之和 = 8
    #    （和 7 → 27.45 超下限；和 9 → 35.29 超上限），可手工 + 穷举核对
    white3 = [
        Channel("A", (1000, 1000, 1000), 3000),
        Channel("B", (1000, 1000, 1000), 2500),
        Channel("C", (1000, 1000, 1000), 4000),
    ]
    # adj=8：正常方案把 8 级都放在最便宜的 B 上（0,8,0）；
    # 任一故障下另外两路可在 ±8 内重新分配出和 8
    check_against_brute("同色白光-adj8", white3, (31, 31, 31), 1, 8, True)
    # 同例 ID 逆序输入，tie-break/最优性不变
    check_against_brute("同色白光-adj8-ID逆序", list(reversed(white3)),
                        (31, 31, 31), 1, 8, True)

    # 1b) 回归用例（曾被误判为 infeasible）：adj=2 时正常驱动不受
    # “d0_i ≤ adj”限制——熄灭路归零落差不算调整量，正常方案为 [4,4,0]
    # （功率分子 22000），三路熄灭各自都备得出和为 8 的存活两路组合：
    #   A 灭 → [0,6,2]，B 灭 → [6,0,2]，C 灭 → [2,6,0]，存活 Δ 均 ≤2。
    # 穷举预言机独立核对最优功率与 ID 升序字典序。
    res = solve_robust(white3, (31, 31, 31), 1, 2)
    assert res.status == "optimal", f"回归用例被误判：{res.status}（{res.message}）"
    assert res.normal.drives == [4, 4, 0], res.normal.drives
    assert res.normal.power_num == 22000, res.normal.power_num
    check_result("回归-白光adj2", white3, (31, 31, 31), 1, 2, res)
    # 备用只需可行：每条备用故障路归零、两路存活之和为 8（窗口恰好要求）、
    # 存活各路与正常值之差 ≤2；归零落差（A/B 为 4→0）不受上限限制。
    for f, b in enumerate(res.backups):
        assert b.drives[f] == 0
        assert sum(b.drives) == 8
        assert all(abs(b.drives[i] - res.normal.drives[i]) <= 2
                   for i in range(3) if i != f)
        assert b.deltas == [b.drives[i] - res.normal.drives[i] for i in range(3)]
    print("[OK] 回归：熄灭路归零落差不受上限限制，adj2 正常 [4,4,0]、"
          "功率 22000/255，逐路备用齐备")
    check_against_brute("同色白光-adj2-穷举核对", white3,
                        (31, 31, 31), 1, 2, True)

    # 1c) adj=1：两路存活在 ±1 内至多把正常和补到 S+1，正常和必须为 8，
    #    而每个故障场景下两条存活通道的正常级之和至少要 7（才补得到 8）；
    #    三条这样的下界相加得 2·S ≥ 21，与 S=8 矛盾 → 耦合不可行
    #    （单场景预求解都可行，不可行性确实来自调整上限耦合）。
    res = solve_robust(white3, (31, 31, 31), 1, 1)
    assert res.status == "infeasible" and res.normal is None and res.backups == []
    check_against_brute("同色白光-adj1-穷举核对", white3,
                        (31, 31, 31), 1, 1, False)
    print("[OK] 调整上限过小=无完整容错方案（耦合不可行，非单场景不可行）")

    # 2) 纯 RGB 三路：任一颜色路熄灭后无人能补该色 → 预求解即证明不可行
    rgb3 = [
        Channel("R", (1000, 0, 0), 1000),
        Channel("G", (0, 1000, 0), 1000),
        Channel("B", (0, 0, 1000), 1000),
    ]
    res = solve_robust(rgb3, (500, 500, 500), 12, 255)
    assert res.status == "infeasible", res.status
    assert res.normal is None and res.backups == []
    print("[OK] 纯RGB缺色=无完整容错方案，不返回任何正常/备用驱动级")

    # 3) 正常场景本身不可行
    res = solve_robust(rgb3, (500, 500, 500), 1, 255)
    assert res.status == "infeasible" and res.backups == []
    print("[OK] 正常场景无可行配方=容错方案不可行")

    # 4) 4 路示例通道（穷举 256^4 代价高，做全约束精确核对而非最优性穷举）
    res = solve_robust(SAMPLE, (500, 500, 500), 60, 80)
    assert res.status == "optimal", res.message
    check_result("示例4路容错-adj80", SAMPLE, (500, 500, 500), 60, 80, res)
    print(f"[OK] 示例4路容错-adj80: 正常 {res.normal.drives} "
          f"功率 {res.normal.power_num}/255；逐路备用 "
          f"{[b.drives for b in res.backups]}")

    # 4b) 同例 adj=255 可行；adj=30 不可行（补偿级差不够）
    res255 = solve_robust(SAMPLE, (500, 500, 500), 60, 255)
    assert res255.status == "optimal"
    check_result("示例4路容错-adj255", SAMPLE, (500, 500, 500), 60, 255, res255)
    res30 = solve_robust(SAMPLE, (500, 500, 500), 60, 30)
    assert res30.status == "infeasible" and res30.normal is None
    print("[OK] 示例4路: adj255 有完整方案，adj30 无完整方案")

    # 4c) adj=255 的正常最优必须与普通配方最优一致（备用无额外限制时）
    plain = solve_plain(SAMPLE, (500, 500, 500), 60)
    assert plain is not None
    assert res255.normal.drives == plain.drives, (
        res255.normal.drives, plain.drives
    )
    assert res255.normal.power_num == plain.power_num
    print("[OK] adj=255 时容错正常方案与普通最优配方（含字典序）完全一致")

    # 5) 手工构造的非平凡 3 路实例：通道接近白光但读数/功率各不相同。
    #    目标为低亮度白；存活通道只能在 ±adj 内补偿熄灭路（熄灭路自身的
    #    归零落差不受限）。目标/容差由穷举预言机确认可解，覆盖窄容差（2）
    #    与严格色阶。
    crafted = [
        (
            "近白光-读数各异",
            [Channel("A", (1000, 950, 900), 3000),
             Channel("B", (880, 1000, 920), 2500),
             Channel("C", (940, 900, 1000), 4000)],
            (26, 25, 27), 2, 12,
        ),
        (
            "近白光-ID逆序(X/Y/Z)",
            [Channel("Z", (1000, 800, 700), 2000),
             Channel("Y", (700, 1000, 800), 5000),
             Channel("X", (800, 700, 1000), 3500)],
            (20, 20, 20), 2, 12,
        ),
        (
            "同白光-大上限验证字典序",
            [Channel("A", (1000, 1000, 1000), 3000),
             Channel("B", (1000, 1000, 1000), 2500),
             Channel("C", (1000, 1000, 1000), 4000)],
            # 实际色 = 1000·驱动级之和/255；窗口 [54,56] 恰要求和为 14
            (55, 55, 55), 1, 20,
        ),
    ]
    for cname, chs, target, tol, adj in crafted:
        # 先确认预言机认为有解（窗口按整数分子判定，容差需真的覆盖离散色阶）
        want = brute_force_robust(chs, target, tol, adj)
        assert want is not None, f"{cname}: 构造实例在穷举下无解，需调整目标/容差"
        check_against_brute(cname, chs, target, tol, adj, True)

    # 5b) 随机 3 路实例用大上限 + 较宽容差提高命中率：仍用穷举预言机全面核对
    rng = random.Random(20260930)
    found = 0
    tried = 0
    over_adj_seen = False  # 至少一例正常驱动超过上限（旧实现会误判的形态）
    while found < 2 and tried < 120:
        tried += 1
        chs = [
            Channel(cid, tuple(rng.randint(300, 1000) for _ in range(3)),
                    rng.randint(100, 9000))
            for cid in ("A", "B", "C")
        ]
        target = (rng.randint(120, 300), rng.randint(120, 300),
                  rng.randint(120, 300))
        adj = rng.choice((40, 60, 90))
        tol = rng.choice((8, 12, 20))
        want = brute_force_robust(chs, target, tol, adj)
        if want is None:
            continue
        res = solve_robust(chs, target, tol, adj, time_limit_seconds=10)
        check_against_brute(f"随机3路实例#{found + 1}", chs, target, tol, adj, True)
        if max(res.normal.drives) > adj:
            over_adj_seen = True
        found += 1
    assert found == 2, f"随机搜索只找到 {found} 个可行实例（试了 {tried} 次）"
    assert over_adj_seen, "随机场未覆盖正常驱动超过调整上限的形态"

    # 6) 6 路可解且全部备用约束成立
    six = SAMPLE + [Channel("W2", (700, 720, 750), 5000),
                    Channel("Amber", (950, 600, 60), 2600)]
    res = solve_robust(six, (500, 500, 500), 60, 120)
    assert res.status == "optimal", res.message
    check_result("6路容错", six, (500, 500, 500), 60, 120, res)
    print(f"[OK] 6路容错: 正常 {res.normal.drives}，{len(res.backups)} 个备用配方齐备")

    # 7) schema 非法参数全部 422
    CH4 = [
        {"id": "R", "r": 1000, "g": 120, "b": 40, "power_mw": 3200},
        {"id": "G", "r": 160, "g": 1000, "b": 90, "power_mw": 2800},
        {"id": "B", "r": 60, "g": 80, "b": 1000, "power_mw": 3400},
        {"id": "W", "r": 820, "g": 800, "b": 760, "power_mw": 6500},
    ]
    BASE = dict(
        channels=CH4,
        target_r=500, target_g=500, target_b=500,
        tolerance=12, max_adjust=60,
    )
    RobustSolveRequest(**BASE)  # 基线合法

    def reject(name, **body):
        try:
            RobustSolveRequest(**body)
        except ValidationError:
            print(f"[OK] {name}: 422 拒绝")
            return
        raise AssertionError(f"{name}: 应拒绝却通过")

    reject("调整上限 0", **{**BASE, "max_adjust": 0})
    reject("调整上限 256", **{**BASE, "max_adjust": 256})
    reject("缺少调整上限", **{k: v for k, v in BASE.items() if k != "max_adjust"})
    reject("通道数 2", **{**BASE, "channels": CH4[:2]})
    reject("通道数 7", **{**BASE, "channels": CH4 + CH4[:3]})
    reject("ID 重复", **{**BASE,
                        "channels": [CH4[0]] + CH4[1:3] + [CH4[0]] + CH4[3:]})
    reject("目标 1001", **{**BASE, "target_r": 1001})
    reject("容差 21", **{**BASE, "tolerance": 21})
    reject("容差 -1", **{**BASE, "tolerance": -1})

    print("\n全部通过 ✔")


if __name__ == "__main__":
    main()
