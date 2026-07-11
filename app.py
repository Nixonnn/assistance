import streamlit as st
import subprocess
import sys
import os
import json, base64, requests
from datetime import datetime
import time  # 【修改点 1】引入 time 用于重试等待

# ================= GitHub 读写函数 =================

@st.cache_data(ttl=60)  # 【修改点 2】缓存时间从 300 缩短到 60 秒，减少冲突
def load_history_from_github():
    """从GitHub仓库读取历史记录"""
    try:
        url = f"https://api.github.com/repos/{st.secrets['github']['repo']}/contents/{st.secrets['github']['file_path']}"
        headers = {"Authorization": f"token {st.secrets['github']['token']}"}
        params = {"ref": st.secrets['github'].get('branch', 'main')}
        
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 404:
            return [], None
        resp.raise_for_status()
        
        content = base64.b64decode(resp.json()["content"]).decode("utf-8")
        sha = resp.json()["sha"]
        return json.loads(content), sha
    except Exception as e:
        st.error(f"读取历史记录失败: {e}")
        return [], None

def save_history_to_github(record: dict, max_retries: int = 3):
    """通过GitHub API追加保存分析记录（含冲突自动重试）"""
    for attempt in range(max_retries):
        load_history_from_github.clear()  # 【修改点 3】每次重试前强制清缓存
        history, old_sha = load_history_from_github()
        
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
            load_history_from_github.clear()
            return True
        elif resp.status_code == 409 and attempt < max_retries - 1:
            wait_time = (attempt + 1) * 1.5
            st.warning(f"⚠️ 检测到并发冲突，{wait_time}s 后自动重试 ({attempt+1}/{max_retries})...")
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

st.set_page_config(page_title="⚽ 足球概率预测系统", layout="centered", page_icon="⚽")
st.title("⚽ Multi-Model Bayesian Football Engine")
st.caption("v1.2 — 基于 Elo / xG / Dixon-Coles / Monte Carlo 的贝叶斯融合预测")

# 【修改点 4】在渲染任何 UI 之前，统一初始化所有表单字段的默认值到 session_state
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

tab1, tab2 = st.tabs(["📊 第一阶段：生成预测报告", "💰 第二阶段：生成投注凭证"])

# ================= Tab 1: 预测报告 =================

with tab1:
    st.header("🎯 输入比赛数据")

    col1, col2 = st.columns(2)
    # 【修改点 5】所有 widget 放弃 value=硬编码，全部使用 key= 绑定 session_state
    with col1:
        st.text_input("主队名称", key="home_team")
        st.number_input("主队 Elo (可选, 留空自动反推)", step=1.0, format="%.0f", key="home_elo")
        st.number_input("主队 xG (可选, 留空自动反推)", step=0.01, format="%.2f", key="home_xg")
    with col2:
        st.text_input("客队名称", key="away_team")
        st.number_input("客队 Elo (可选)", step=1.0, format="%.0f", key="away_elo")
        st.number_input("客队 xG (可选)", step=0.01, format="%.2f", key="away_xg")

    st.subheader("赔率与盘口")
    col3, col4 = st.columns(2)
    with col3:
        st.number_input("主胜赔率", step=0.01, format="%.2f", key="odds_h")
        st.number_input("平局赔率", step=0.01, format="%.2f", key="odds_d")
        st.number_input("客胜赔率", step=0.01, format="%.2f", key="odds_a")
    with col4:
        st.text_input("大小球盘口 (如 2.5)", key="ou_line")
        st.number_input("大球赔率", step=0.01, format="%.2f", key="ou_over")
        st.number_input("小球赔率", step=0.01, format="%.2f", key="ou_under")
        st.text_input("让球盘口 (如 -1)", key="ah_line")
        st.number_input("让球方赔率", step=0.01, format="%.2f", key="ah_home")
        st.number_input("受让方赔率", step=0.01, format="%.2f", key="ah_away")

    st.checkbox("中立场", key="neutral")

    if st.button("🚀 生成预测报告", type="primary"):
        # 【修改点 6】构建命令时，直接从 st.session_state 读取数据
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
                        
                        # 【修改点 7】保存记录时，把所有参数都存进 params，方便以后回填
                        record = {
                            "params": {
                                "home_team": st.session_state.home_team, "away_team": st.session_state.away_team,
                                "home_elo": st.session_state.home_elo, "away_elo": st.session_state.away_elo,
                                "home_xg": st.session_state.home_xg, "away_xg": st.session_state.away_xg,
                                "odds_h": st.session_state.odds_h, "odds_d": st.session_state.odds_d, "odds_a": st.session_state.odds_a,
                                "ou_line": st.session_state.ou_line, "ou_over": st.session_state.ou_over, "ou_under": st.session_state.ou_under,
                                "ah_line": st.session_state.ah_line, "ah_home": st.session_state.ah_home, "ah_away": st.session_state.ah_away,
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
    st.header("💰 投注策略与凭证生成")
    st.info("请先在第一阶段生成预测报告，再基于实际看到的波胆赔率输入下方。")

    stake = st.number_input("投注本金 (元)", value=100, step=10, min_value=1)
    strategy = st.radio("选择策略", ["A (等收益分投)", "B (保本主攻分投)"], index=0)
    strategy_map = {"A (等收益分投)": "A", "B (保本主攻分投)": "B"}

    st.caption("📝 在下方表格中逐行录入波胆赔率（支持增删改）")
    
    if "cs_odds_data" not in st.session_state:
        st.session_state.cs_odds_data = [
            {"score": "1-0", "odds": 6.40}, {"score": "2-0", "odds": 7.10},
            {"score": "1-1", "odds": 8.00}, {"score": "2-1", "odds": 8.90},
            {"score": "0-0", "odds": 11.50},
        ]
    
    edited_df = st.data_editor(
        st.session_state.cs_odds_data,
        column_config={
            "score": st.column_config.TextColumn("比分", width="small"),
            "odds": st.column_config.NumberColumn("赔率", format="%.2f", step=0.01, min_value=1.0),
        },
        num_rows="dynamic", hide_index=True, use_container_width=True, key="cs_odds_editor"
    )
    st.session_state.cs_odds_data = edited_df
    
    valid_rows = [r for r in edited_df if r["score"] and r["odds"] > 0]
    custom_odds = ",".join(f"{r['score']}:{r['odds']:.2f}" for r in valid_rows)
    
    if st.button("📜 生成投注凭证", type="primary"):
        # 【修改点 8】Tab2 同样直接从 session_state 获取 Tab1 填写的球队和赔率数据
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
               
        if st.session_state.neutral: cmd.append("--neutral")
        if st.session_state.home_elo > 0: cmd.extend(["--home_elo", str(st.session_state.home_elo)])
        if st.session_state.away_elo > 0: cmd.extend(["--away_elo", str(st.session_state.away_elo)])
        if st.session_state.home_xg > 0: cmd.extend(["--home_xg", str(st.session_state.home_xg)])
        if st.session_state.away_xg > 0: cmd.extend(["--away_xg", str(st.session_state.away_xg)])

        with st.spinner("⏳ 正在计算投注凭证..."):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30, env=my_env)
                if result.returncode == 0:
                    st.success("✅ 投注凭证生成成功！")
                    st.markdown("---")
                    st.markdown(result.stdout)
                    record = {
                        "params": {
                            "home_team": st.session_state.home_team, "away_team": st.session_state.away_team,
                            "odds_h": st.session_state.odds_h, "odds_a": st.session_state.odds_a,
                            "stake": stake, "strategy": strategy_map[strategy]
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
            label = f"{rec.get('timestamp', '')[:16]} | {params.get('home_team','')} vs {params.get('away_team','')}"
            
            # 【修改点 9】点击历史记录时，直接把参数写回 session_state 对应的 key，然后 rerun
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                for field in DEFAULTS.keys():
                    if field in params:
                        st.session_state[field] = params[field]
                st.rerun()

# streamlit run app.py
