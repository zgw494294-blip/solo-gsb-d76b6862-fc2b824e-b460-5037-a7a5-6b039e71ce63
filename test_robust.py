"""故障容错配方规划的冒烟测试（不依赖 pytest，可直接 python3 运行）。

重点：
  * 三态结论（optimal / infeasible / incomplete 的接口语义）正确；
  * 正常配方的最优性（最小正常总功率 + ID 升序字典序）用独立穷举核对：
    备用场景中故障路必须为 0，**其落差不受 max_adjust 限制**（上限只约束
    仍工作的两路），故正常驱动级仍取完整 0～255 域；先枚举 256² 个正常
    (x,y)，用颜色窗口对第三级求出可行整数区间，再逐故障场景用 256²
    存活对的二维前缀和做“d0 存活邻域盒内是否存在可行备用”的常数时间查询
    （3 路时完整精确，不依赖 CP-SAT）；
  * 所有备用配方逐项核对：故障路为 0、调整级差不超限、颜色窗口（整数
    分子精确判定，未裁剪）、0～1000、功率自洽；
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
            # 故障路必须为 0；其相对正常方案的落差 = −d0 不受上限约束
            fi = q - 1
            assert v.drives[fi] == 0
            assert v.deltas[fi] == -res.normal.drives[fi]
            for i in range(n):
                if i == fi:
                    continue
                # 仅仍工作的通道要求与正常值之差不超过上限
                assert abs(v.drives[i] - res.normal.drives[i]) <= adj, (
                    f"{name}: 故障{q} 存活通道{i} 调整 {v.deltas[i]} 超过上限 {adj}"
                )
            assert v.deltas == [
                v.drives[i] - res.normal.drives[i] for i in range(n)
            ]
    return variants


def _prefix2d(grid):
    """256×256 0/1 网格的二维前缀和。"""
    ps = [[0] * 257 for _ in range(257)]
    for x in range(256):
        row_sum = 0
        for y in range(256):
            row_sum += grid[x][y]
            ps[x + 1][y + 1] = ps[x][y + 1] + row_sum
    return ps


def _rect_has(ps, x0, y0, x1, y1):
    """矩形 [x0,x1]×[y0,y1] 内是否存在可行点（坐标裁剪到 0～255）。"""
    x0 = max(0, min(255, x0)); x1 = max(0, min(255, x1))
    y0 = max(0, min(255, y0)); y1 = max(0, min(255, y1))
    if x0 > x1 or y0 > y1:
        return False
    return (
        ps[x1 + 1][y1 + 1] - ps[x0][y1 + 1]
        - ps[x1 + 1][y0] + ps[x0][y0]
    ) > 0


def _ceil_div(a: int, b: int) -> int:
    """b > 0 时 ceil(a/b)，对负分子同样成立。"""
    return -((-a) // b)


def _z_span(channels, xy, target, tol):
    """固定通道 0/1 的驱动 (x,y) 后，通道 2 满足颜色窗口的整数级区间。

    每色窗口对 z 是线性不等式；rgb[2][c] = 0 时该色与 z 无关，直接核验
    固定项。返回含端点的 (lo, hi)；不存在可行整数时返回 None。
    """
    x, y = xy
    lo, hi = 0, 255
    for c in range(3):
        base = channels[0].rgb[c] * x + channels[1].rgb[c] * y
        w = channels[2].rgb[c]
        lower = 255 * (target[c] - tol)
        upper = min(255 * (target[c] + tol), 255 * 1000)
        if w == 0:
            if not lower <= base <= upper:
                return None
            continue
        clo = _ceil_div(lower - base, w)
        chi = (upper - base) // w
        lo, hi = max(lo, clo), min(hi, chi)
        if lo > hi:
            return None
    return lo, hi


def brute_force_robust(channels, target, tol, adj):
    """3 路独立穷举预言机：返回 (最优正常驱动, 最优功率分子)；无解返回 None。

    语义：故障路驱动恒为 0，其落差（d0 → 0）**不受** adj 限制；adj 只约束
    仍工作的通道，因此正常驱动级仍取完整 0～255 域，不能用 d0_i ≤ adj 剪枝。

    做法（完全独立于 CP-SAT）：
      1. 每个故障场景枚举 256² 个存活对，存可行网格并用二维前缀和把
         “正常存活对 (x,y) 的 ±adj 邻域盒内是否存在可行备用”膨胀成
         256×256 覆盖表（O(256²) 次矩形查询）；
      2. 正常方案枚举 (x,y) ∈ 256²，第三级 z 的可行集由三色窗口求出整数
         区间（窄容差下通常只有几个值），逐 z 查三张覆盖表即可。
    目标为正常功率分子最小，平手时按 ID 升序驱动序列字典序最小。
    """
    n = len(channels)
    assert n == 3
    order = sorted(range(n), key=lambda k: channels[k].id)
    alive = [[i for i in range(n) if i != f] for f in range(n)]

    # 每个故障场景：(存活下标 i1,i2, 覆盖表 cover[x][y])；
    # cover[x][y] ⇔ 正常存活对 (x,y) 的 ±adj 邻域盒内存在可行备用
    covers = []
    for f in range(n):
        i1, i2 = alive[f]
        grid = [[0] * 256 for _ in range(256)]
        any_feas = False
        for x in range(256):
            for y in range(256):
                drives = [0] * n
                drives[i1] = x
                drives[i2] = y
                if window_ok(drives_scaled(channels, drives), target, tol):
                    grid[x][y] = 1
                    any_feas = True
        if not any_feas:
            return None  # 单场景本身不可行 → 整体必然不可行
        ps = _prefix2d(grid)
        cover = [
            [
                1 if _rect_has(ps, x - adj, y - adj, x + adj, y + adj) else 0
                for y in range(256)
            ]
            for x in range(256)
        ]
        covers.append((i1, i2, cover))

    best_key = None  # (功率分子, ID 序驱动序列)
    best_drives = None
    for a in range(256):
        for b in range(256):
            span = _z_span(channels, (a, b), target, tol)
            if span is None:
                continue
            clo, chi = span
            for c in range(clo, chi + 1):
                d0 = (a, b, c)
                ok = True
                for i1, i2, cover in covers:
                    if not cover[d0[i1]][d0[i2]]:
                        ok = False
                        break
                if not ok:
                    continue
                pnum = sum(channels[i].power_mw * d0[i] for i in range(n))
                key = (pnum, [d0[i] for i in order])
                if best_key is None or key < best_key:
                    best_key = key
                    best_drives = list(d0)
    return None if best_key is None else (best_drives, best_key[0])


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
    # adj=2（回归：旧模型把熄灭路落差也限在 2 内 ⇒ 正常每路 ≤2、和至多 6，
    # 误判 infeasible）。熄灭路落差不受限，正常驱动可取完整 0～255：
    # 故障某路时另两路要在 ±2 内凑出和 8，故正常每路 ≤ 4；最便宜的 B 取满 4，
    # 其余 4 级放次便宜的 A：正常 (4,4,0)，功率分子 3000·4+2500·4 = 22000
    res = solve_robust(white3, (31, 31, 31), 1, 2)
    assert res.status == "optimal", res.message
    assert res.normal.drives == [4, 4, 0], res.normal.drives
    assert res.normal.power_num == 22000, res.normal.power_num
    check_result("同色白光-adj2", white3, (31, 31, 31), 1, 2, res)
    check_against_brute("同色白光-adj2-预言机", white3, (31, 31, 31), 1, 2, True)
    # 三个备用方案逐路核对：熄灭路为 0，存活两路在 ±2 内凑出和 8。
    # A、B 熄灭时存活对在 ±2 盒内只有唯一配法；C 熄灭时 (4,4,0) 本身即
    # 可行，存活对有多解，只核对结构（备用只需可行，不参与目标）
    assert res.backups[0].drives == [0, 6, 2]
    assert res.backups[1].drives == [6, 0, 2]
    b2 = res.backups[2].drives
    assert b2[2] == 0 and sum(b2) == 8
    assert all(abs(b2[i] - res.normal.drives[i]) <= 2 for i in (0, 1))
    print("[OK] adj=2 回归：合法容错方案 optimal，正常 [4,4,0]，功率分子 22000，"
          f"逐路备用 {[b.drives for b in res.backups]}")
    # adj=1：穷举预言机精确判定耦合不可行（单场景预求解仍可行：两路满档
    # 本可配出目标，但 ±1 的补偿级差无法覆盖正常方案），求解器须一致
    assert brute_force_robust(white3, (31, 31, 31), 1, 1) is None
    res = solve_robust(white3, (31, 31, 31), 1, 1)
    assert res.status == "infeasible" and res.normal is None and res.backups == []
    print("[OK] adj=1 仍为耦合不可行（上限真的不足），不返回任何正常/备用驱动级")

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
    #    目标为低亮度白；注意正常驱动级不再被压到 ≤ adj（熄灭路落差不受限）。
    #    目标/容差由穷举预言机确认可解，覆盖窄容差（2）与严格色阶。
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
        check_against_brute(f"随机3路实例#{found + 1}", chs, target, tol, adj, True)
        found += 1
    assert found == 2, f"随机搜索只找到 {found} 个可行实例（试了 {tried} 次）"

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
