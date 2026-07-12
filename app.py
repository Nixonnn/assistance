import streamlit as st
import subprocess
import sys
import os
import json, base64, requests
from datetime import datetime
import time
import re
import pandas as pd

# ================= GitHub 读写函数 =================

def load_history_from_github():
    """从GitHub仓库读取历史记录（无缓存，确保每次拿到最新 SHA）"""
    try:
        url = f"https://api.github.com/repos/{st.secrets['github']['repo']}/contents/{st.secrets['github']['file_path']}"
        headers = {"Authorization": f"token {st.secrets['github']['token']}"}
        params = {"ref": st.secrets['github'].get('branch', 'main')}

        resp = requests.get(url, headers=headers, params=params, timeout=10)

        if resp.status_code == 404:
            return [], None

        resp.raise_for_status()
        data = resp.json()
        sha = data.get("sha")
        content = base64.b64decode(data["content"]).decode("utf-8")

        if not content.strip():
            return [], sha

        return json.loads(content), sha
    except Exception as e:
        st.error(f"读取历史记录失败: {e}")
        return [], None


def save_history_to_github(record: dict, max_retries: int = 3):
    """通过GitHub API追加保存分析记录（含冲突自动重试）"""
    for attempt in range(max_retries):
        history, old_sha = load_history_from_github()

        if not isinstance(history, list):
            history = []

        record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history.insert(0, record)
        history = history[:100]

        new_content = base64.b64encode(
            json.dumps(history, ensure_ascii=False, indent=2).encode("utf-8")
        ).decode("utf-8")

        url = f"https://api.github.com/repos/{st.secrets['github']['repo']}/contents/{st.secrets['github']['file_path']}"
        headers = {"Authorization": f"token {st.secrets['github']['token']}"}
        payload = {
            "message": f"📊 Add analysis: {record['params'].get('home_team','')} vs {record['params'].get('away_team','')} at {record['timestamp']}",
            "content": new_content,
            "branch": st.secrets['github'].get('branch', 'main'),
        }
        if old_sha:
            payload["sha"] = old_sha

        resp = requests.put(url, json=payload, headers=headers, timeout=15)

        if resp.status_code in (200, 201):
            return True
        elif resp.status_code in (409, 422) and attempt < max_retries - 1:
            wait_time = (attempt + 1) * 1.5
            st.warning(f"⚠️ 保存冲突或 SHA 过期，{wait_time}s 后自动重试 ({attempt+1}/{max_retries})...")
            time.sleep(wait_time)
            continue
        else:
            st.error(f"保存失败 ({resp.status_code}): {resp.text}")
            return False

    st.error("❌ 多次重试后仍无法保存，请稍后再试")
    return False


my_env = os.environ.copy()
my_env['PYTHONIOENCODING'] = 'utf-8'

# ================= 页面设置与初始化 =================

st.set_page_config(page_title="⚽ Win-assistance", layout="centered", page_icon="⚽")
st.title("⚽ Multi-Model Bayesian Football Engine")
st.caption("v1.2 — 基于 Elo / xG / Dixon-Coles / Monte Carlo 的贝叶斯融合预测")

DEFAULTS = {
    "home_team": "法国", "away_team": "摩洛哥",
    "home_elo": 0.0, "away_elo": 0.0,
    "home_xg": 0.0, "away_xg": 0.0,
    "odds_h": 1.55, "odds_d": 4.10, "odds_a": 5.80,
    "ou_line": "2.5", "ou_over": 2.04, "ou_under": 1.84,
    "ah_line": "-1", "ah_home": 2.01, "ah_away": 1.89,
    "neutral": True,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def fill_form_callback(params_dict):
    """点击历史记录时的回调：在输入框渲染前，将参数安全填入"""
    for field in DEFAULTS.keys():
        if field in params_dict:
            val = params_dict[field]
            expected_type = type(DEFAULTS[field])
            try:
                if expected_type == float:
                    val = float(val)
                elif expected_type == int:
                    val = int(val)
                elif expected_type == bool:
                    val = bool(val)
            except (ValueError, TypeError):
                val = DEFAULTS[field]
            st.session_state[field] = val


tab1, tab2 = st.tabs(["📊 第一阶段：生成预测报告", "💰 第二阶段：生成投注凭证"])

# ================= Tab 1: 预测报告 =================

with tab1:
    st.header("🎯 输入比赛数据")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("主队名称", key="home_team")
        st.number_input("主队 Elo (可选, 留空自动反推)", value=0.0, step=1.0, format="%.0f", key="home_elo")
        st.number_input("主队 xG (可选, 留空自动反推)", value=0.0, step=0.01, format="%.2f", key="home_xg")
    with col2:
        st.text_input("客队名称", key="away_team")
        st.number_input("客队 Elo (可选)", value=0.0, step=1.0, format="%.0f", key="away_elo")
        st.number_input("客队 xG (可选)", value=0.0, step=0.01, format="%.2f", key="away_xg")

    st.subheader("赔率与盘口")
    col3, col4 = st.columns(2)
    with col3:
        st.number_input("主胜赔率", value=0.0, step=0.01, format="%.2f", key="odds_h")
        st.number_input("平局赔率", value=0.0, step=0.01, format="%.2f", key="odds_d")
        st.number_input("客胜赔率", value=0.0, step=0.01, format="%.2f", key="odds_a")
    with col4:
        st.text_input("大小球盘口 (如 2.5)", key="ou_line")
        st.number_input("大球赔率", value=0.0, step=0.01, format="%.2f", key="ou_over")
        st.number_input("小球赔率", value=0.0, step=0.01, format="%.2f", key="ou_under")
        st.text_input("让球盘口 (如 -1)", key="ah_line")
        st.number_input("让球方赔率", value=0.0, step=0.01, format="%.2f", key="ah_home")
        st.number_input("受让方赔率", value=0.0, step=0.01, format="%.2f", key="ah_away")

    st.checkbox("中立场", key="neutral")

    if st.button("🚀 生成预测报告", type="primary"):
        cmd = [sys.executable, "predict2.py",
               "--home", st.session_state.home_team,
               "--away", st.session_state.away_team,
               "--odds", str(st.session_state.odds_h), str(st.session_state.odds_d), str(st.session_state.odds_a),
               "--ou_line", str(st.session_state.ou_line),
               "--ou_odds", str(st.session_state.ou_over), str(st.session_state.ou_under),
               "--ah_line", str(st.session_state.ah_line),
               "--ah_odds", str(st.session_state.ah_home), str(st.session_state.ah_away),
               "--no-interactive"]

        if st.session_state.neutral:
            cmd.append("--neutral")
        if st.session_state.home_elo > 0:
            cmd.extend(["--home_elo", str(st.session_state.home_elo)])
        if st.session_state.away_elo > 0:
            cmd.extend(["--away_elo", str(st.session_state.away_elo)])
        if st.session_state.home_xg > 0:
            cmd.extend(["--home_xg", str(st.session_state.home_xg)])
        if st.session_state.away_xg > 0:
            cmd.extend(["--away_xg", str(st.session_state.away_xg)])

        with st.spinner("⏳ 正在调用物理计算引擎 `predict2.py`..."):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30, env=my_env)
                if result.returncode == 0:
                    st.success("✅ 预测报告生成成功！")
                    st.markdown("---")
                    if result.stdout and result.stdout.strip():
                        st.code(f"调试：捕获到 {len(result.stdout)} 个字符", language="text")
                        st.markdown(result.stdout)

                        # 优化正则：只匹配 数字:数字
                        score_pattern = r'(?<!\d)(\d{1,2}):(\d{1,2})(?!\d)'
                        found_scores = re.findall(score_pattern, result.stdout)

                        top5_scores = []
                        for s in found_scores:
                            score_str = f"{s[0]}:{s[1]}"
                            if score_str not in top5_scores:
                                top5_scores.append(score_str)
                            if len(top5_scores) >= 5:
                                break

                        st.session_state.top5_cs_scores = top5_scores

                        record = {
                            "params": {
                                "home_team": st.session_state.home_team,
                                "away_team": st.session_state.away_team,
                                "home_elo": st.session_state.home_elo,
                                "away_elo": st.session_state.away_elo,
                                "home_xg": st.session_state.home_xg,
                                "away_xg": st.session_state.away_xg,
                                "odds_h": st.session_state.odds_h,
                                "odds_d": st.session_state.odds_d,
                                "odds_a": st.session_state.odds_a,
                                "ou_line": st.session_state.ou_line,
                                "ou_over": st.session_state.ou_over,
                                "ou_under": st.session_state.ou_under,
                                "ah_line": st.session_state.ah_line,
                                "ah_home": st.session_state.ah_home,
                                "ah_away": st.session_state.ah_away,
                                "neutral": st.session_state.neutral
                            },
                            "report": result.stdout
                        }
                        with st.spinner("正在保存历史记录到云端..."):
                            if save_history_to_github(record):
                                st.success("✅ 分析记录已同步至 GitHub 仓库")
                    else:
                        st.error("⚠️ 虽然脚本成功运行，但捕获到的标准输出为空！")
                else:
                    st.error("❌ 引擎运行出错！")
                    st.code(result.stderr)
            except FileNotFoundError:
                st.error("❌ 未找到 `predict2.py` 文件。")
            except subprocess.TimeoutExpired:
                st.error("⏰ 计算超时。")

# ================= Tab 2: 投注凭证 =================

with tab2:
    st.header("💰 波胆投注策略与凭证生成")
    st.info("请先在第一阶段生成预测报告，然后在下方录入波胆比分与赔率。")

    # 1. 基础设置
    col_stake, col_strat = st.columns(2)
    with col_stake:
        stake = st.number_input("投注本金 (元)", value=100, step=10, min_value=1, key="stake_input")
    with col_strat:
        strategy = st.radio("选择策略", ["A (等收益分投)", "B (保本主攻分投)"], index=0, key="strategy_input")
    strategy_map = {"A (等收益分投)": "A", "B (保本主攻分投)": "B"}

    st.divider()

    # 2. 快捷操作区
    col_quick, col_count = st.columns([2, 1])
    with col_quick:
        if "top5_cs_scores" in st.session_state and len(st.session_state.top5_cs_scores) > 0:
            if st.button("✨ 一键导入 Tab1 Top5 波胆", type="primary"):
                st.session_state.bet_count = len(st.session_state.top5_cs_scores)
                st.session_state.prefill_scores = list(st.session_state.top5_cs_scores)
                st.rerun()

    with col_count:
        default_count = st.session_state.get("bet_count", 5)
        bet_count = st.number_input(
            "投注行数", min_value=1, max_value=20, value=default_count, step=1, key="bet_count_input"
        )
        if bet_count != st.session_state.get("bet_count", 5):
            st.session_state.bet_count = bet_count
            st.session_state.prefill_scores = []
            st.rerun()

    st.divider()

    # 3. 核心输入区：st.form
    with st.form("bets_form"):
        st.subheader("📝 波胆投注明细")

        h_col1, h_col2, h_col3 = st.columns([1, 3, 2])
        h_col1.markdown("**序号**")
        h_col2.markdown("**比分 (如 1:0)**")
        h_col3.markdown("**赔率**")

        form_bets = []
        prefill = st.session_state.get("prefill_scores", [])

        for i in range(st.session_state.get("bet_count", 5)):
            c1, c2, c3 = st.columns([1, 3, 2])
            c1.markdown(f"<h3 style='text-align:center; margin-top:10px;'>{i+1}</h3>", unsafe_allow_html=True)

            default_score = prefill[i] if i < len(prefill) else ""
            score_val = c2.text_input(
                f"score_{i}",
                value=default_score,
                placeholder="如 2-1",
                key=f"form_score_{i}",
                label_visibility="collapsed"
            )

            odds_val = c3.number_input(
                f"odds_{i}",
                value=8.0,
                step=0.1,
                min_value=1.0,
                key=f"form_odds_{i}",
                label_visibility="collapsed"
            )
            form_bets.append({"score": score_val.strip(), "odds": odds_val})

        submitted = st.form_submit_button("💾 确认投注明细", type="primary", use_container_width=True)
        if submitted:
            valid_bets = [b for b in form_bets if b["score"]]
            st.session_state.confirmed_bets = valid_bets
            st.rerun()

    # 4. 显示已确认的投注 & 生成凭证
    confirmed = st.session_state.get("confirmed_bets", [])

    if confirmed:
        st.success(f"✅ 已确认 **{len(confirmed)}** 注波胆")

        display_data = [{"序号": i + 1, "比分": b["score"], "赔率": b["odds"]} for i, b in enumerate(confirmed)]
        st.table(pd.DataFrame(display_data))

        custom_odds = ",".join(f"{b['score']}:{b['odds']:.2f}" for b in confirmed)

        if st.button("📜 生成最终投注凭证", type="primary", use_container_width=True):
            cmd = [sys.executable, "predict2.py",
                   "--home", st.session_state.home_team,
                   "--away", st.session_state.away_team,
                   "--odds", str(st.session_state.odds_h), str(st.session_state.odds_d), str(st.session_state.odds_a),
                   "--ou_line", str(st.session_state.ou_line),
                   "--ou_odds", str(st.session_state.ou_over), str(st.session_state.ou_under),
                   "--ah_line", str(st.session_state.ah_line),
                   "--ah_odds", str(st.session_state.ah_home), str(st.session_state.ah_away),
                   "--stake", str(stake),
                   "--choice", strategy_map[strategy],
                   "--custom-odds", custom_odds,
                   "--no-interactive"]

            if st.session_state.neutral:
                cmd.append("--neutral")
            if st.session_state.home_elo > 0:
                cmd.extend(["--home_elo", str(st.session_state.home_elo)])
            if st.session_state.away_elo > 0:
                cmd.extend(["--away_elo", str(st.session_state.away_elo)])
            if st.session_state.home_xg > 0:
                cmd.extend(["--home_xg", str(st.session_state.home_xg)])
            if st.session_state.away_xg > 0:
                cmd.extend(["--away_xg", str(st.session_state.away_xg)])

            with st.spinner("⏳ 正在计算投注凭证..."):
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30, env=my_env)
                    if result.returncode == 0:
                        st.success("✅ 投注凭证生成成功！")
                        st.markdown("---")
                        st.markdown(result.stdout)
                        record = {
                            "params": {
                                "home_team": st.session_state.home_team,
                                "away_team": st.session_state.away_team,
                                "odds_h": st.session_state.odds_h,
                                "odds_d": st.session_state.odds_d,
                                "odds_a": st.session_state.odds_a,
                                "ou_line": st.session_state.ou_line,
                                "ou_over": st.session_state.ou_over,
                                "ou_under": st.session_state.ou_under,
                                "ah_line": st.session_state.ah_line,
                                "ah_home": st.session_state.ah_home,
                                "ah_away": st.session_state.ah_away,
                                "neutral": st.session_state.neutral,
                                "stake": stake,
                                "strategy": strategy_map[strategy]
                            },
                            "report": result.stdout
                        }
                        with st.spinner("正在保存投注凭证到云端..."):
                            if save_history_to_github(record):
                                st.success("✅ 投注凭证已同步至 GitHub 仓库")
                    else:
                        st.error("❌ 生成失败！")
                        st.code(result.stderr)
                except Exception as e:
                    st.error(f"发生错误: {e}")

# ================= 侧边栏历史记录 =================

with st.sidebar:
    st.header("📜 历史分析记录")
    history, _ = load_history_from_github()

    if not history:
        st.info("暂无历史记录")
    else:
        for i, rec in enumerate(history):
            params = rec.get("params", {})
            label = f"{rec.get('timestamp', '')[:16]} | {params.get('home_team', '')} vs {params.get('away_team', '')}"

            st.button(
                label,
                key=f"hist_{i}",
                use_container_width=True,
                on_click=fill_form_callback,
                args=(params,)
            )

# streamlit run app.py
# streamlit run app.py
