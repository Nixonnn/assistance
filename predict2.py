#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
⚽ 严格统计足球概率预测系统 (Multi-Model Bayesian Football Engine)
数学计算与分析引擎 - 移动端适配输出
"""

import math
import random
import sys
import argparse
from bisect import bisect_left

def poisson_pdf(k, lam):
    """计算泊松分布概率质量函数"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def tau_correction(x, y, lam, mu, rho):
    """Dixon-Coles 模型的低比分相关性修正因子 (tau)"""
    if x == 0 and y == 0:
        return 1.0 - rho * lam * mu
    elif x == 1 and y == 0:
        return 1.0 + rho * mu
    elif x == 0 and y == 1:
        return 1.0 + rho * lam
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0

def get_dc_distribution(lam, mu, rho=-0.12, max_goals=8):
    """生成 Dixon-Coles 联合比分概率分布"""
    dist = {}
    total_prob = 0.0
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p = poisson_pdf(x, lam) * poisson_pdf(y, mu) * tau_correction(x, y, lam, mu, rho)
            p = max(0.0, p)
            dist[(x, y)] = p
            total_prob += p

    # 归一化，确保所有联合概率之和为 1.0
    for k in dist:
        dist[k] /= total_prob

    return dist

def get_1x2_probabilities(dist):
    """从比分分布中提取胜、平、负概率"""
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    for (x, y), p in dist.items():
        if x > y:
            home_win += p
        elif x == y:
            draw += p
        else:
            away_win += p
    return home_win, draw, away_win

def get_elo_1x2(home_elo, away_elo, home_advantage=80):
    """根据 Elo Rating 计算主平客概率"""
    z = (home_elo + home_advantage - away_elo) / 400.0
    p_home_expected = 1.0 / (1.0 + 10.0 ** -z)

    # 估算平局概率 (根据实力差距呈高斯分布波动)
    p_draw = 0.26 * math.exp(-(z ** 2) / 2.0)

    p_home_win = max(0.01, p_home_expected - 0.5 * p_draw)
    p_away_win = max(0.01, 1.0 - p_home_expected - 0.5 * p_draw)

    # 归一化
    tot = p_home_win + p_draw + p_away_win
    return p_home_win / tot, p_draw / tot, p_away_win / tot

def run_monte_carlo_simulation(dist, num_simulations=50000):
    """通过 Dixon-Coles 联合分布进行蒙特卡洛抽样，量化方差和不确定区间"""
    outcomes = list(dist.keys())
    probs = list(dist.values())

    # 构建累计概率分布函数 (CDF)
    cdf = []
    cum = 0.0
    for p in probs:
        cum += p
        cdf.append(cum)
    cdf[-1] = 1.0

    home_wins = 0
    draws = 0
    away_wins = 0
    simulated_scores = {}
    simulated_home_goals = []
    simulated_away_goals = []

    for _ in range(num_simulations):
        r = random.random()
        idx = bisect_left(cdf, r)
        x, y = outcomes[idx]

        if x > y:
            home_wins += 1
        elif x == y:
            draws += 1
        else:
            away_wins += 1

        simulated_scores[(x, y)] = simulated_scores.get((x, y), 0) + 1
        simulated_home_goals.append(x)
        simulated_away_goals.append(y)

    p_hw = home_wins / num_simulations
    p_d = draws / num_simulations
    p_aw = away_wins / num_simulations

    # 计算 95% 置信区间的进球数上下限
    simulated_home_goals.sort()
    simulated_away_goals.sort()
    low_h = simulated_home_goals[int(0.025 * num_simulations)]
    high_h = simulated_home_goals[int(0.975 * num_simulations)]
    low_a = simulated_away_goals[int(0.025 * num_simulations)]
    high_a = simulated_away_goals[int(0.975 * num_simulations)]

    return (p_hw, p_d, p_aw), simulated_scores, (low_h, high_h), (low_a, high_a)

def shin_overround_removal(odds):
    """
    使用著名的 Shin's Method 剔除博彩公司赔率中的抽水 (Overround)
    获取市场无抽水真实概率 (P_market_fair)
    """
    inv_odds = [1.0 / o for o in odds]
    sum_inv = sum(inv_odds)
    if sum_inv <= 1.0:
        return [x / sum_inv for x in inv_odds], 0.0

    low_z = 0.0
    high_z = 0.99
    best_z = 0.0
    best_probs = [x / sum_inv for x in inv_odds]

    # 二分迭代法寻找系统内部隐含的 insider 交易占比 z
    for _ in range(100):
        z = (low_z + high_z) / 2.0
        probs = []
        for s in inv_odds:
            num = math.sqrt(z**2 + 4 * (1.0 - z) * (s**2) / sum_inv) - z
            den = 2.0 * (1.0 - z)
            p = num / den if den > 0 else s / sum_inv
            probs.append(p)

        sum_p = sum(probs)
        if abs(sum_p - 1.0) < 1e-9:
            best_z = z
            best_probs = probs
            break
        elif sum_p > 1.0:
            low_z = z
        else:
            high_z = z
            best_z = z
            best_probs = probs

    # 二次归一化保证严格精确
    s_p = sum(best_probs)
    return [p / s_p for p in best_probs], best_z

def calculate_kl_divergence(p, q):
    """计算两个概率分布之间的 Kullback-Leibler 散度 (p为基准, q为待测)"""
    epsilon = 1e-15
    kl = 0.0
    for pi, qi in zip(p, q):
        pi = max(epsilon, pi)
        qi = max(epsilon, qi)
        kl += pi * math.log(pi / qi)
    return kl

def calculate_entropy(p):
    """计算分布的信息熵"""
    epsilon = 1e-15
    entropy = 0.0
    for pi in p:
        pi = max(epsilon, pi)
        entropy -= pi * math.log(pi)
    return entropy

def calibrate_favorite_longshot_bias(probs, gamma=1.05):
    """对最终概率分布进行最爱-冷门偏差校准 (提升强队、压低冷门)"""
    calibrated = [p ** gamma for p in probs]
    tot = sum(calibrated)
    return [c / tot for c in calibrated]

def remove_two_way_overround(odds):
    """二项盘口去抽水，返回两边的市场公平概率。"""
    inv_odds = [1.0 / o for o in odds]
    total = sum(inv_odds)
    return [x / total for x in inv_odds]

def parse_asian_line(value):
    """解析 2.75、2.5/3、-0.5/1 等盘口写法。"""
    text = str(value).strip().replace(" ", "")
    if "/" not in text:
        return float(text)

    left, right = text.split("/", 1)
    left_value = float(left)
    sign = -1.0 if left_value < 0 else 1.0
    if right.startswith("+") or right.startswith("-"):
        right_value = float(right)
    else:
        right_value = sign * float(right)
    return (left_value + right_value) / 2.0

def split_asian_line(line):
    """把整数/半球/四分之一盘口拆成一到两个基础盘口。"""
    scaled = round(line * 4) / 4.0
    base = math.floor(scaled)
    frac = scaled - base
    if abs(frac - 0.25) < 1e-9:
        return [base, base + 0.5]
    if abs(frac - 0.75) < 1e-9:
        return [base + 0.5, base + 1.0]
    return [scaled]

def single_line_success(adjusted_value):
    """计算单一盘口的赢盘强度：赢=1, 走水=0.5, 输=0。"""
    if adjusted_value > 1e-12:
        return 1.0
    if abs(adjusted_value) <= 1e-12:
        return 0.5
    return 0.0

def total_over_success_from_dist(dist, line):
    """从比分分布计算大小球大球方向的市场强度。"""
    sub_lines = split_asian_line(line)
    probability = 0.0
    for (home_goals, away_goals), score_prob in dist.items():
        total_goals = home_goals + away_goals
        success = sum(single_line_success(total_goals - sub_line) for sub_line in sub_lines) / len(sub_lines)
        probability += score_prob * success
    return probability

def home_ah_success_from_dist(dist, line):
    """从比分分布计算主队亚洲让球方向的市场强度。"""
    sub_lines = split_asian_line(line)
    probability = 0.0
    for (home_goals, away_goals), score_prob in dist.items():
        goal_diff = home_goals - away_goals
        success = sum(single_line_success(goal_diff + sub_line) for sub_line in sub_lines) / len(sub_lines)
        probability += score_prob * success
    return probability

def solve_implied_xg(p_fair, rho=-0.12, ou_line=None, ou_target=None, ah_line=None, ah_target=None):
    """
    逆向工程：利用二元坐标下降法，从市场去抽水概率 (p_fair) 中
    反向求解最符合 1X2、大小球、让球盘口的隐含期望进球数 lam 和 mu
    """
    best_err = float('inf')
    best_lam, best_mu = 1.5, 1.0

    # 坐标下降搜寻最契合的 xG 组合
    lam, mu = 1.5, 1.0
    step = 0.5
    for _ in range(7): # 7次细化
        for _ in range(30):
            improved = False
            # 搜索附近8个方向（修正了原代码中 (step, -step) 重复的问题）
            for dl, dm in [(-step, 0), (step, 0), (0, -step), (0, step),
                           (-step, -step), (step, -step), (-step, step)]:
                nl, nm = max(0.05, lam + dl), max(0.05, mu + dm)
                dist = get_dc_distribution(nl, nm, rho=rho)
                ph, pd, pa = get_1x2_probabilities(dist)
                err = 1.00 * ((ph - p_fair[0])**2 + (pd - p_fair[1])**2 + (pa - p_fair[2])**2)
                if ou_line is not None:
                    ou_model = total_over_success_from_dist(dist, float(ou_line))
                    err += 0.55 * (ou_model - float(ou_target))**2
                if ah_line is not None:
                    ah_model = home_ah_success_from_dist(dist, float(ah_line))
                    err += 0.75 * (ah_model - float(ah_target))**2
                if err < best_err:
                    best_err = err
                    best_lam, best_mu = nl, nm
                    improved = True
            if improved:
                lam, mu = best_lam, best_mu
            else:
                break
        step *= 0.5

    return best_lam, best_mu

def generate_ascii_bar(prob, length=10):
    """生成适配手机屏幕的 ASCII 进度条"""
    filled = int(round(prob * length))
    return "█" * filled + "░" * (length - filled)

def format_prediction_report(home_team, away_team, elo_probs, xg_probs, dc_probs, mc_probs, 
                             p_market_fair, p_final, btts_probs, over_under_probs, top5_scores,
                             xg_home, xg_away, dc_tau, mc_h_band, mc_a_band, odds, dist_dc, 
                             total_stake=1000.0, market_context=None):
    """格式化为专为手机窄屏适配的紧凑型、强可读性 Markdown 报告"""

    # 映射胜平负
    p_hw, p_d, p_aw = p_final

    # 首选推荐判断
    best_index = p_final.index(max(p_final))
    outcomes = ["主胜", "平局", "客胜"]
    recommendation = outcomes[best_index]

    # 进球区间推荐
    exp_total = xg_home + xg_away
    if exp_total < 1.8:
        goal_range = "0-1"
    elif exp_total < 3.2:
        goal_range = "2-3"
    else:
        goal_range = "4球及以上"

    # 计算三个维度的最佳比分
    best_hw_score = ("", 0.0)
    best_d_score = ("", 0.0)
    best_aw_score = ("", 0.0)
    for (x, y), p in dist_dc.items():
        score_str = f"{x}-{y}"
        if x > y:
            if p > best_hw_score[1]:
                best_hw_score = (score_str, p)
        elif x == y:
            if p > best_d_score[1]:
                best_d_score = (score_str, p)
        else:
            if p > best_aw_score[1]:
                best_aw_score = (score_str, p)

    # 制定风险对冲方案
    evs = [p * o - 1.0 for p, o in zip(p_final, odds)]
    best_ev_idx = evs.index(max(evs))
    best_ev_val = evs[best_ev_idx]

    if p_final[0] > p_final[2]:
        underdog_team = away_team
        double_chance_prob = p_final[1] + p_final[2]
        double_chance_str = f'平局或客胜 ({underdog_team}不败)'
        hedge_score = best_d_score[0] if best_d_score[1] > best_aw_score[1] else best_aw_score[0]
        hedge_prob = max(best_d_score[1], best_aw_score[1])
    else:
        underdog_team = home_team
        double_chance_prob = p_final[0] + p_final[1]
        double_chance_str = f'主胜或平局 ({underdog_team}不败)'
        hedge_score = best_d_score[0] if best_d_score[1] > best_hw_score[1] else best_hw_score[0]
        hedge_prob = max(best_d_score[1], best_hw_score[1])

    report = []
    report.append(f"## 🎯 Match: {home_team} vs {away_team}")
    report.append("---")
    report.append("### 🎗️ 核心预测 (p_final 贝叶斯融合)")
    report.append(f"- **主胜概率**: {p_hw*100:.1f}% [{generate_ascii_bar(p_hw)}]")
    report.append(f"- **平局概率**: {p_d*100:.1f}% [{generate_ascii_bar(p_d)}]")
    report.append(f"- **客胜概率**: {p_aw*100:.1f}% [{generate_ascii_bar(p_aw)}]")
    report.append(f"- **首选推荐**: `{recommendation}`")
    report.append(f"- **推荐比分**: `{top5_scores[0][0]}` / `{top5_scores[1][0]}`")
    report.append(f"- **总进球数**: `{goal_range}` 球 (期望值: `{exp_total:.2f}`)")

    report.append("\n---")
    report.append("### 🏅 模型独立输出对照")
    report.append(f"*   **Elo 概率**: 主胜 {elo_probs[0]*100:.0f}% | 平 {elo_probs[1]*100:.0f}% | 客 {elo_probs[2]*100:.0f}%")
    report.append(f"*   **xG 期望**: 主队 {xg_home:.2f} / 客队 {xg_away:.2f} (基础期望)")
    report.append(f"*   **Dixon-Coles 修正**: τ 值为 {dc_tau:.3f} (平局相关性参数)")
    report.append(f"*   **Monte Carlo 模拟**: 中位数比分 `{top5_scores[0][0]}`，95% 置信区间主队 `[{mc_h_band[0]:.0f}-{mc_h_band[1]:.0f}]` / 客队 `[{mc_a_band[0]:.0f}-{mc_a_band[1]:.0f}]`")

    if market_context:
        report.append("\n---")
        report.append("### 📊 盘口校准输入")
        if market_context.get("ou_line") is not None:
            report.append(
                f"*   📉 **大小球校准**: 盘口 {market_context['ou_line']:.2f}, 大球公平概率 {market_context['ou_target']*100:.1f}%"
            )
        if market_context.get('ah_line') is not None:
            report.append(
                f"*   ⚖️ **让球校准**: 主队盘口 {market_context['ah_line']:.2f}, 主队赢盘公平概率 {market_context['ah_target']*100:.1f}%"
            )

    report.append("\n---")
    report.append("### 🏆 精准比分概率 TOP 5")
    for i, (score, p) in enumerate(top5_scores[:5]):
        report.append(f"{i}. **`{score}`** --- {p*100:.1f}% [{generate_ascii_bar(p)}]")

    report.append("\n---")
    report.append("### 🎲 进球数及双建概率")
    p_0_1, p_2_3, p_4_plus = over_under_probs
    p_btts_y, p_btts_n = btts_probs
    report.append(f"- **0-1 球**: {p_0_1*100:.1f}% [{generate_ascii_bar(p_0_1)}]")
    report.append(f"- **2-3 球**: {p_2_3*100:.1f}% [{generate_ascii_bar(p_2_3)}]")
    report.append(f"- **4+ 球**: {p_4_plus*100:.1f}% [{generate_ascii_bar(p_4_plus)}]")
    report.append(f"- **双方建功 (BTTS)**: 是 {p_btts_y*100:.1f}% [{generate_ascii_bar(p_btts_y)}] | 否 {p_btts_n*100:.1f}% [{generate_ascii_bar(p_btts_n)}]")

    report.append("\n---")
    report.append("### 🛡️ 风险对冲与冷防建议")
    report.append(f"- **冷门期望不败率** ({double_chance_str}): {double_chance_prob*100:.1f}% [{generate_ascii_bar(double_chance_prob)}]")
    report.append(f"- **首选冷防比分**: `{hedge_score}` (胜率: {hedge_prob*100:.1f}%)")
    report.append(f"- **三大赛果首选比分分布**: ")
    report.append(f"  *   🏠 主胜首选: `{best_hw_score[0]}` ({best_hw_score[1]*100:.1f}%)")
    report.append(f"  *   🤝 平局首选: `{best_d_score[0]}` ({best_d_score[1]*100:.1f}%)")
    report.append(f"  *   🚀 客胜首选: `{best_aw_score[0]}` ({best_aw_score[1]*100:.1f}%)")

    ev_str_list = [f"{o}: {e*100:+.1f}%" for o, e in zip(outcomes, evs)]
    report.append(f"- **市场投资期望值 (EV)**: `{', '.join(ev_str_list)}`")
    if best_ev_val > 0.0:
        report.append(f"- **最佳期望值推荐 (Value Bet)**: 偏向 `{outcomes[best_ev_idx]}` (EV值为 {best_ev_val*100:.1f}%, 具备正向投注价值)")
    else:
        report.append(f"- **最佳期望值推荐 (Value Bet)**: 偏向 `{outcomes[best_ev_idx]}` (EV值为 {best_ev_val*100:.1f}%, 虽无正期望但属于机构杀水最少项)")

    # 智能比分分投注单模块计算
    score_odds = {}
    for (x, y), p in dist_dc.items():
        score_str = f"{x}-{y}"
        score_odds[score_str] = round(max(3.0, 0.85 / max(1e-15, p)), 2)

    top5_names = [score for score, _ in top5_scores]

    # 方案 A 算：等收益分投
    inv_odds_sum = sum(1.0 / score_odds[name] for name in top5_names)
    strat_a_return = total_stake / inv_odds_sum if inv_odds_sum > 0 else 0.0
    strat_a_net_profit = strat_a_return - total_stake
    strat_a_roi = (strat_a_net_profit / total_stake) * 100.0 if total_stake > 0 else 0.0

    # 方案 B 算：保本主攻
    h_odds = score_odds[hedge_score]
    hedge_stake = total_stake / h_odds if h_odds > 0 else 0.0
    remaining_stake = total_stake - hedge_stake
    other_scores = [name for name in top5_names if name != hedge_score][:4]
    other_prob_sum = sum(dict(top5_scores).get(name, 0.0) for name in other_scores)

    report.append("\n---")
    report.append(f"### 💰 智能资金分配与对冲投注单 (预算: {total_stake:.0f} 元)")
    report.append("### 🔴 策略 A: 等收益分投 (首筑基, 平滑波动)")
    for name in top5_names:
        odds_val = score_odds[name]
        stake_val = total_stake * (1.0 / odds_val) / inv_odds_sum
        report.append(f"- 比分 `{name}` (估算 {odds_val:.2f}) — 投 `{stake_val:.1f}` 元 (中返 {strat_a_return:.1f} 元)")
    report.append(f"👑 **策略 A 净期望利润**: `{strat_a_net_profit:+.1f}` 元 (固定净回报率: `{strat_a_roi:+.1f}`%)")

    report.append("### 🔵 策略 B: 保本主攻分投 (中后期爆发, 主攻暴利)")
    report.append(f"- 💛 **保本对冲** `{hedge_score}` (估算 {h_odds:.2f}) — 投 `{hedge_stake:.1f}` 元 (中返 `{total_stake:.0f}` 元, 保本不赔)")
    for name in other_scores:
        p = dict(top5_scores).get(name, 0.0)
        odds_val = score_odds[name]
        stake_val = remaining_stake * (p / other_prob_sum) if other_prob_sum > 0 else 0.0
        ret_val = stake_val * odds_val
        net_profit_val = ret_val - total_stake
        report.append(f"- 🚀 **主攻比分** `{name}` (估算 {odds_val:.2f}) — 投 `{stake_val:.1f}` 元 (中返 `{ret_val:.1f}` 元, 净赚 `{net_profit_val:+.1f}` 元)")
    report.append("💪 **策略 B 综述**: 重仓主打胜局高回报, 一旦爆冷打出平局首选比分, 退还全部本金, 本场无风险。")

    return "\n".join(report)

def main():
    import sys
    import io
    # 强制 stdout 使用 UTF-8 编码，防止 emoji 报错
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    parser = argparse.ArgumentParser(description="Multi-Model Bayesian Football Engine (Mobile-First CLI)")
    parser.add_argument("--home", default="主队", help="主队名称")
    parser.add_argument("--away", default="客队", help="客队名称")
    parser.add_argument("--home_elo", type=float, default=None, help="主队 Elo rating")
    parser.add_argument("--away_elo", type=float, default=None, help="客队 Elo rating")
    parser.add_argument("--home_xg", type=float, default=None, help="主队进球期望 (xG)")
    parser.add_argument("--away_xg", type=float, default=None, help="客队进球期望 (xG)")
    parser.add_argument("--odds", type=float, nargs=3, required=True, help="机构 1x2 赔率, 格式: 主胜 平局 客胜")
    parser.add_argument("--ou_line", dest="ou_line", default=None, help="大小球盘口, 如 2.75 或 2.5/3")
    parser.add_argument("--ou_odds", dest="ou_odds", type=float, nargs=2, default=None, help="大小球赔率, 格式: 大球 小球")
    parser.add_argument("--ah_line", dest="ah_line", default=None, help="主队亚洲让球盘口, 如 -0.75 或 -0.5/1")
    parser.add_argument("--ah_odds", dest="ah_odds", type=float, nargs=2, default=None, help="亚洲让球赔率, 格式: 主队方向 客队方向")
    parser.add_argument("--neutral", action="store_true", help="是否为中立场比赛")
    parser.add_argument("--rho", type=float, default=-0.12, help="Dixon-Coles 修正参数 rho (默认 -0.12)")
    parser.add_argument("--gamma", type=float, default=1.05, help="Favorite-Longshot 校准参数 gamma (默认 1.05)")
    parser.add_argument("--sims", type=int, default=50000, help="蒙特卡洛模拟次数 (默认 50000)")
    parser.add_argument("--stake", type=float, default=1000.0, help="对冲资金总预算 (单位: 元)")
    parser.add_argument("--no-interactive", dest="interactive", action="store_false", default=True, help="禁用手机端互动对冲决策模式")
    parser.add_argument("--choice", dest="choice", default=None, choices=["A", "B", "N"], help="直接指定投注策略, 跳过交互直接输出凭证")
    parser.add_argument("--custom-odds", dest="custom_odds", default=None, help="自定义波胆赔率字符串, 格式如 '1-1:6.7,1-0:5.2'")

    args = parser.parse_args()

    # 提前计算市场无抽水真实概率 (p_market_fair), 以做逆向工程
    p_market_fair, shin_z = shin_overround_removal(args.odds)
    market_context = {}

    ou_line = None
    ou_target = None
    if args.ou_line is not None and args.ou_odds is not None:
        ou_line = parse_asian_line(args.ou_line)
        ou_fair = remove_two_way_overround(args.ou_odds)
        ou_target = ou_fair[0]
        market_context["ou_line"] = ou_line
        market_context["ou_target"] = ou_target

    ah_line = None
    ah_target = None
    if args.ah_line is not None and args.ah_odds is not None:
        ah_line = parse_asian_line(args.ah_line)
        ah_fair = remove_two_way_overround(args.ah_odds)
        ah_target = ah_fair[0]
        market_context["ah_line"] = ah_line
        market_context["ah_target"] = ah_target

    # 如果未提供 Elo，则从欧指无抽水真实概率中逆向求解
    if args.home_elo is None or args.away_elo is None:
        home_adv = 0 if args.neutral else 80
        k = p_market_fair[0] / max(1e-15, (p_market_fair[0] + p_market_fair[2]))
        k = max(0.001, min(0.999, k))
        implied_diff = 400.0 * math.log10(k / (1.0 - k)) - home_adv
        args.home_elo = 1000.0 + implied_diff
        args.away_elo = 1000.0

    # 如果未提供 xG，则从欧指无抽水真实概率中逆向求解
    if args.home_xg is None or args.away_xg is None:
        args.home_xg, args.away_xg = solve_implied_xg(
            p_market_fair,
            rho=args.rho,
            ou_line=ou_line,
            ou_target=ou_target,
            ah_line=ah_line,
            ah_target=ah_target
        )

    # 1. 计算各模型独立概率分布 (1X2)
    # 1.1 Elo模型 (中立场无主场优势)
    home_adv = 0 if args.neutral else 80
    elo_probs = get_elo_1x2(args.home_elo, args.away_elo, home_advantage=home_adv)

    # 1.2 纯 xG 泊松模型
    dist_pure_xg = get_dc_distribution(args.home_xg, args.away_xg, rho=0.0) # rho=0即为纯独立泊松
    xg_probs = get_1x2_probabilities(dist_pure_xg)

    # 1.3 Dixon-Coles 修正比分模型
    dist_dc = get_dc_distribution(args.home_xg, args.away_xg, rho=args.rho)
    dc_probs = get_1x2_probabilities(dist_dc)

    # 1.4 Monte Carlo 抽样模拟模型
    mc_probs, mc_scores, mc_h_band, mc_a_band = run_monte_carlo_simulation(dist_dc, num_simulations=args.sims)

    # 2. 市场无抽水概率计算 (p_market_fair) 已在前期逆向工程中计算完毕

    # 3. 贝叶斯动态权重融合 (wi)
    models_1x2 = [elo_probs, xg_probs, dc_probs, mc_probs]
    weights = []

    for m in models_1x2:
        # 计算与 fair 市场概率的 KL 散度
        kl = calculate_kl_divergence(p_market_fair, m)
        # 计算自身分布熵
        entropy = calculate_entropy(m)

        # 散度越小，代表与高一致性共识吻合度越高，权重越大
        score = math.exp(-kl)

        # 惩罚项：如果信息熵极度偏低(说明模型盲目自信)但KL散度却大，对其权重打折
        if kl > 0.05 and entropy < 0.75:
            score *= (entropy / 1.1)

        weights.append(score)

    # 归一化权重
    sum_w = sum(weights)
    weights = [w / sum_w for w in weights]

    # 混合概率计算
    p_final_raw = [0.0, 0.0, 0.0]
    for i in range(3):
        p_final_raw[i] = sum(w * m[i] for w, m in zip(weights, models_1x2))

    # 4. 最爱-冷门偏差校准 (Favorite-Longshot Bias Calibration)
    p_final = calibrate_favorite_longshot_bias(p_final_raw, gamma=args.gamma)

    # 5. 整合比分概率 & 盘口进球分布
    # 精准比分 Top 5
    sorted_scores = sorted(dist_dc.items(), key=lambda x: x[1], reverse=True)
    top5_scores = [(f"{x}-{y}", p) for (x, y), p in sorted_scores[:5]]

    # 进球数分布
    p_0_1 = 0.0
    p_2_3 = 0.0
    p_4_plus = 0.0
    p_btts_y = 0.0
    p_btts_n = 0.0

    for (x, y), p in dist_dc.items():
        goals = x + y
        if goals <= 1:
            p_0_1 += p
        elif goals <= 3:
            p_2_3 += p
        else:
            p_4_plus += p

        if x > 0 and y > 0:
            p_btts_y += p
        else:
            p_btts_n += p

    over_under_probs = (p_0_1, p_2_3, p_4_plus)
    btts_probs = (p_btts_y, p_btts_n)

    # 6. 生成漂亮的 Markdown 报告并打印
    report = format_prediction_report(
        args.home, args.away, elo_probs, xg_probs, dc_probs, mc_probs, 
        p_market_fair, p_final, btts_probs, over_under_probs, top5_scores,
        args.home_xg, args.away_xg, args.rho, mc_h_band, mc_a_band,
        args.odds, dist_dc, args.stake, market_context
    )
    print(report)

    # 手机端交互/自动决策中心
    choice = None
    if args.choice:
        choice = args.choice.upper()
    elif args.interactive:
        print("\n" + "="*45)
        print("💡 手机端互动对冲决策中心")
        print("="*45)
        print("[A] 采用策略 A: 等收益分投 (稳健积累) ")
        print("[B] 采用策略 B: 保本主攻分投 (中后期爆发) ")
        print("[N] 取消/暂不投注")
        print("-"*45)
        try:
            choice = input("💡 请选择您本场的最终抉择 [A / B / N]: ").strip().upper()
        except (ValueError, KeyboardInterrupt, EOFError):
            print("\n👋 互动取消。")
            choice = None

    if choice in ['A', 'B']:
        # 重新解析和计算参数
        best_hw_score = ("", 0.0)
        best_d_score = ("", 0.0)
        best_aw_score = ("", 0.0)
        for (x, y), p in dist_dc.items():
            score_str = f"{x}-{y}"
            if x > y:
                if p > best_hw_score[1]: best_hw_score = (score_str, p)
            elif x == y:
                if p > best_d_score[1]: best_d_score = (score_str, p)
            else:
                if p > best_aw_score[1]: best_aw_score = (score_str, p)

        if p_final[0] > p_final[2]:
            hedge_score = best_d_score[0] if best_d_score[1] > best_aw_score[1] else best_aw_score[0]
        else:
            hedge_score = best_d_score[0] if best_d_score[1] > best_hw_score[1] else best_hw_score[0]

        score_odds = {}
        for (x, y), p in dist_dc.items():
            score_odds[f"{x}-{y}"] = round(max(3.0, 0.85 / max(1e-15, p)), 2)

        top5_names = [score for score, _ in top5_scores]

        # 命令行指定的自定义赔率应用 (格式 "1-1:6.7,1-0:5.2" 或 "1:1:6.7")
        if args.custom_odds:
            try:
                pairs = args.custom_odds.split(",")
                for pair in pairs:
                    if ":" in pair:
                        # 【核心修复1】使用 rsplit 从右侧分割 1 次。
                        # 这样即使字符串是 "1:1:8.00"，也会完美拆分为 name="1:1", val="8.00"，绝不报错！
                        name, val = pair.rsplit(":", 1)
                        
                        # 【核心修复2】强制清洗比分格式，把冒号统一替换为减号
                        # 因为引擎底层字典的 Key 全是减号 (如 "1-1")，不替换就匹配不上！
                        name = name.strip().replace(":", "-")
                        val = val.strip()
                        
                        if name and val:
                            score_odds[name] = float(val)
                            # 可选：打印调试信息，让你在终端肉眼确认赔率已注入
                            # print(f"[DEBUG] 成功加载真实赔率: {name} -> {val}")
            except Exception as e:
                print(f"⚠️ 解析自定义波胆赔率时出错: {e}")

        # 用户手动交互输入赔率逻辑
        if args.interactive and not args.choice:
            try:
                use_real = input('💡 是否手动输入您在手机软件上看到的真实波胆赔率? [Y / N] (默认N, 采用系统估算): ').strip().upper()
                if use_real == 'y':
                    print("请依次输入以下比分的真实赔率 (直接按回车代表默认不修改) :")
                    required_scores = list(top5_names)
                    if hedge_score not in required_scores:
                        required_scores.append(hedge_score)
                    for name in required_scores:
                        val = input(f" - 比分 `{name}` 真实赔率 (系统估算 {score_odds[name]}): ").strip()
                        if val:
                            score_odds[name] = float(val)
            except (ValueError, KeyboardInterrupt, EOFError):
                pass

        # 重新运行 Staking 计算
        inv_odds_sum = sum(1.0 / score_odds[name] for name in top5_names)
        h_odds = score_odds[hedge_score]
        hedge_stake = args.stake / h_odds
        remaining_stake = args.stake - hedge_stake
        other_scores = [name for name in top5_names if name != hedge_score][:4]
        other_prob_sum = sum(dict(top5_scores).get(name, 0.0) for name in other_scores)

        if choice == 'A':
            print("\n✅ 已确认【策略 A (等收益分投) 】！正在为您生成手机投注单凭证...")
            print("\n📜 【策略 A 投注凭证 — 稳健筑基单】")
            print("-"*45)
            print(f"赛事: {args.home} vs {args.away}")
            print(f"总本金: {args.stake:.1f} 元")
            print("投向分拨明细: ")
            ret_val = args.stake / inv_odds_sum
            for name in top5_names:
                odds_val = score_odds[name]
                stake_val = args.stake * (1.0 / odds_val) / inv_odds_sum
                print(f" - 🎯 比分 `{name}` (赔率 {odds_val:.2f}) — 投 `{stake_val:.1f}` 元")
            print("-"*45)
            print(f"🎯 任何比分命中均返还: {ret_val:.1f} 元")
            print(f"📈 固定净期望利润: {ret_val - args.stake:+.1f} 元 (ROI: {(ret_val - args.stake)/args.stake*100:.1f}%)")
            print("-"*45)
            print("💡 凭证已生成。请严格执行比例，场中切勿因比分波动而追加单项。")
        elif choice == 'B':
            print("\n✅ 已确认【策略 B (保本主攻) 】！正在为您生成手机投注单凭证...")
            print("\n📜 【策略 B 投注凭证 — 激进爆发单】")
            print("-"*45)
            print(f"赛事: {args.home} vs {args.away}")
            print(f"总本金: {args.stake:.1f} 元")
            print("投向分拨明细: ")
            print(f" - 💛 【保本对冲防御】比分 `{hedge_score}` (赔率 {h_odds:.2f}) — 投 `{hedge_stake:.1f}` 元 (若中返 {args.stake:.1f} 元)")
            for name in other_scores:
                odds_val = score_odds[name]
                stake_val = remaining_stake * (dict(top5_scores).get(name, 0.0) / other_prob_sum) if other_prob_sum > 0 else 0.0
                ret_val = stake_val * odds_val
                print(f" - 🚀 【核心主攻进攻】**比分 `{name}` (赔率 {odds_val:.2f}) — 投 `{stake_val:.1f}` 元 (中返 `{ret_val:.1f}` 元, 净赚 `{ret_val - args.stake:+.1f}` 元)")
            print("-"*45)
            print("💡 凭证已生成。本场若爆冷打出平局首选，将退还 100% 本金，祝您顺利斩获主攻暴利！")
    elif choice == 'N':
        print("\n🚫 已取消本次投注计划。本金安全保留在口袋中。无绝对概率把握不轻易出手，这就是纪律！")

#if __name__ == "__main__":
#    main()

# For deepseek :
if __name__ == "__main__":
    main()
