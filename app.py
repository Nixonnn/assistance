import streamlit as st
import subprocess
import sys
import os
import json, base64, requests, streamlit as st
from datetime import datetime

@st.cache_data(ttl=300)  # 缓存5分钟，避免频繁调用API触发限流
def load_history_from_github():
    """从GitHub仓库读取历史记录"""
    try:
        url = f"https://api.github.com/repos/{st.secrets['github']['repo']}/contents/{st.secrets['github']['file_path']}"
        headers = {"Authorization": f"token {st.secrets['github']['token']}"}
        params = {"ref": st.secrets['github'].get('branch', 'main')}
        
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 404:
            return [], None  # 文件不存在
        resp.raise_for_status()
        
        content = base64.b64decode(resp.json()["content"]).decode("utf-8")
        sha = resp.json()["sha"]  # 用于后续更新时防止冲突
        return json.loads(content), sha
    except Exception as e:
        st.error(f"读取历史记录失败: {e}")
        return [], None

def save_history_to_github(record: dict):
    """通过GitHub API追加保存分析记录"""
    history, old_sha = load_history_from_github()
    
    record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.insert(0, record)
    history = history[:100]  # 限制最多100条
    
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
        payload["sha"] = old_sha  # 防止并发覆盖
        
    resp = requests.put(url, json=payload, headers=headers, timeout=15)
    if resp.status_code in (200, 201):
        load_history_from_github.clear()  # 清除缓存，下次读取最新数据
        return True
    else:
        st.error(f"保存失败 ({resp.status_code}): {resp.text}")
        return False

my_env = os.environ.copy()
my_env['PYTHONIOENCODING'] = 'utf-8'

# 设置页面标题
st.set_page_config(page_title="⚽ 足球概率预测系统", layout="centered", page_icon="⚽")
st.title("⚽ Multi-Model Bayesian Football Engine")
st.caption("v1.2 — 基于 Elo / xG / Dixon-Coles / Monte Carlo 的贝叶斯融合预测")

# 分离两个阶段：预测报告 vs 投注凭证
tab1, tab2 = st.tabs(["📊 第一阶段：生成预测报告", "💰 第二阶段：生成投注凭证"])

# ---- 第一阶段：预测报告 ----
with tab1:
    st.header("🎯 输入比赛数据")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.text_input("主队名称", value="法国")
        home_elo = st.number_input("主队 Elo (可选, 留空自动反推)", value=0.0, step=1.0, format="%.0f")
        home_xg = st.number_input("主队 xG (可选, 留空自动反推)", value=0.0, step=0.01, format="%.2f")
    with col2:
        away_team = st.text_input("客队名称", value="摩洛哥")
        away_elo = st.number_input("客队 Elo (可选)", value=0.0, step=1.0, format="%.0f")
        away_xg = st.number_input("客队 xG (可选)", value=0.0, step=0.01, format="%.2f")

    st.subheader("赔率与盘口")
    col3, col4 = st.columns(2)
    with col3:
        odds_h = st.number_input("主胜赔率", value=1.55, step=0.01, format="%.2f")
        odds_d = st.number_input("平局赔率", value=4.10, step=0.01, format="%.2f")
        odds_a = st.number_input("客胜赔率", value=5.80, step=0.01, format="%.2f")
    with col4:
        ou_line = st.text_input("大小球盘口 (如 2.5)", value="2.5")
        ou_over = st.number_input("大球赔率", value=2.04, step=0.01, format="%.2f")
        ou_under = st.number_input("小球赔率", value=1.84, step=0.01, format="%.2f")
        ah_line = st.text_input("让球盘口 (如 -1)", value="-1")
        ah_home = st.number_input("让球方赔率", value=2.01, step=0.01, format="%.2f")
        ah_away = st.number_input("受让方赔率", value=1.89, step=0.01, format="%.2f")

    neutral = st.checkbox("中立场", value=True)

    # 生成预测按钮
    if st.button("🚀 生成预测报告", type="primary"):
        # 构建命令行参数
        cmd = [sys.executable, "predict2.py",
               "--home", home_team,
               "--away", away_team,
               "--odds", str(odds_h), str(odds_d), str(odds_a),
               "--ou_line", str(ou_line),
               "--ou_odds", str(ou_over), str(ou_under),
               "--ah_line", str(ah_line),
               "--ah_odds", str(ah_home), str(ah_away),
               "--no-interactive"]

        if neutral:
            cmd.append("--neutral")
        if home_elo > 0:
            cmd.extend(["--home_elo", str(home_elo)])
        if away_elo > 0:
            cmd.extend(["--away_elo", str(away_elo)])
        if home_xg > 0:
            cmd.extend(["--home_xg", str(home_xg)])
        if away_xg > 0:
            cmd.extend(["--away_xg", str(away_xg)])

        with st.spinner("⏳ 正在调用物理计算引擎 `predict2.py`..."):
            try:
                result = subprocess.run(
                  cmd,
                  capture_output=True,
                  text=True,
                  encoding='utf-8',          # 明确指定编码
                  errors='replace',          # 遇到无法解码的字符用 � 代替
                  timeout=30,
                  env=my_env
                )
                if result.returncode == 0:
                    st.success("✅ 预测报告生成成功！")
                    st.markdown("---")
                    # 调试：先显示原始输出长度和开头，确认有没有内容
                    if result.stdout and result.stdout.strip():
                       st.code(f"调试：捕获到 {len(result.stdout)} 个字符", language="text")
                       # 正式渲染 Markdown 报告
                       st.markdown(result.stdout)
                       record = {
                           "params": {
                               "home_team": home_team,
                               "away_team": away_team,
                               "home_odds": odds_h,
                               "away_odds": odds_a
                           },
                           "report": result.stdout
                       }
                       with st.spinner("正在保存历史记录到云端..."):
                           if save_history_to_github(record):
                               st.success("✅ 分析记录已同步至 GitHub 仓库")
                    else:
                       st.error("⚠️ 虽然脚本成功运行，但捕获到的标准输出为空！")
                       st.code("（空输出）")
                else:
                    st.error("❌ 引擎运行出错！")
                    st.code(result.stderr)
            except FileNotFoundError:
                st.error("❌ 未找到 `predict2.py` 文件，请确保该文件与 `app.py` 在同一目录下。")
            except subprocess.TimeoutExpired:
                st.error("⏰ 计算超时，请检查脚本是否卡死。")

# ---- 第二阶段：投注凭证 ----
with tab2:
    st.header("💰 投注策略与凭证生成")
    st.info("请先在第一阶段生成预测报告，再基于实际看到的波胆赔率输入下方。")

    stake = st.number_input("投注本金 (元)", value=100, step=10, min_value=1)
    strategy = st.radio("选择策略", ["A (等收益分投)", "B (保本主攻分投)"], index=0)
    strategy_map = {"A (等收益分投)": "A", "B (保本主攻分投)": "B"}

    st.caption("输入您实际看到的 TOP 5 波胆赔率 (格式: 比分:赔率，用逗号分隔)")
    custom_odds = st.text_input("波胆赔率", value="1-0:6.40,2-0:7.10,1-1:8.00,2-1:8.90,0-0:11.5")

    if st.button("📜 生成投注凭证", type="primary"):
        # 这里需要重复第一阶段的部分参数构建，但我们直接调用脚本
        # 简便起见，由于界面已经填了参数，复用逻辑，但为了严谨，我们在后台重新构建完整命令
        # 为了简化用户操作，这里需要重新读取上面的输入框。因为Streamlit的tab是独立的，需要重新获取值（或者用session_state，但为简化，我们用相同的变量名，注意作用域）
        # 实际上，上面的变量在tab1中定义，在tab2中也能访问到（只要不重名）。
        # 但为了保险，直接从tab1拿数据会有作用域问题，这里重新定义一遍（或者用全局变量）。最简单的方式：复制上面的输入框？但那样太冗余。
        # 更优方案：使用st.session_state存储，但为了代码简洁，我直接在此处重新用同名的text_input获取？但会有重复输入的问题。
        # 针对这个演示，我建议用户直接在第二阶段重新输入一次球队和赔率，或者我们通过修改代码使用session_state。
        # 为了不让代码过于复杂，我写一个简易版本：提示用户手动输入完整命令行，或者我们直接将完整的参数塞进一个会话中。
        # 鉴于用户是技术向，我推荐在第二阶段提供"快速生成"按钮，自动复用第一阶段填写的参数（但需要跨tab共享数据）。
        
        st.warning("⚠️ 为简化逻辑，第二阶段将默认复用第一阶段输入的球队和赔率数据（需确保第一阶段已填写）。")
        # 构造完整命令
        cmd = [sys.executable, "predict2.py",
               "--home", home_team,
               "--away", away_team,
               "--odds", str(odds_h), str(odds_d), str(odds_a),
               "--ou_line", str(ou_line),
               "--ou_odds", str(ou_over), str(ou_under),
               "--ah_line", str(ah_line),
               "--ah_odds", str(ah_home), str(ah_away),
               "--stake", str(stake),
               "--choice", strategy_map[strategy],
               "--custom-odds", custom_odds,
               "--no-interactive"]
        if neutral:
            cmd.append("--neutral")
        if home_elo > 0:
            cmd.extend(["--home_elo", str(home_elo)])
        if away_elo > 0:
            cmd.extend(["--away_elo", str(away_elo)])
        if home_xg > 0:
            cmd.extend(["--home_xg", str(home_xg)])
        if away_xg > 0:
            cmd.extend(["--away_xg", str(away_xg)])

        with st.spinner("⏳ 正在计算投注凭证..."):
            try:
                result = subprocess.run(
                  cmd,
                  capture_output=True,
                  text=True,
                  encoding='utf-8',          # 明确指定编码
                  errors='replace',          # 遇到无法解码的字符用 � 代替
                  timeout=30,
                  env=my_env
                )
                if result.returncode == 0:
                    st.success("✅ 投注凭证生成成功！")
                    st.markdown("---")
                    st.markdown(result.stdout)
                    record = {
                        "params": {
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_odds": odds_h,
                            "away_odds": odds_a,
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

# ---- 侧边栏历史记录 ----
with st.sidebar:
    st.header("📜 历史分析记录")
    
    history, _ = load_history_from_github()
    
    if not history:
        st.info("暂无历史记录")
    else:
        for i, rec in enumerate(history):
            params = rec.get("params", {})
            label = f"{rec['timestamp'][:16]} | {params.get('home_team','')} vs {params.get('away_team','')}"
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                # 回填表单参数到 session_state
                st.session_state["home_team"] = params.get("home_team", "")
                st.session_state["away_team"] = params.get("away_team", "")
                st.session_state["odds_h"] = params.get("home_odds", 1.55)
                st.session_state["odds_a"] = params.get("away_odds", 5.80)
                st.rerun()
# streamlit run app.py
