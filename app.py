import streamlit as st
import requests
import numpy as np
import time
import json
import os
import sys
import tempfile
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

API_BASE_URL = "http://127.0.0.1:8000"


def safe_api_get(endpoint, params=None):
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API 请求失败 (HTTP {response.status_code}): {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ 无法连接到后端服务，请确认后端已启动 (python server/app.py)")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ API 请求超时")
        return None
    except json.JSONDecodeError:
        st.error("❌ API 返回数据格式错误")
        return None
    except Exception as e:
        st.error(f"❌ API 请求异常: {str(e)}")
        return None


def safe_api_post(endpoint, files=None, params=None, json_data=None):
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.post(url, files=files, params=params, json=json_data, timeout=30)
        if response.status_code in (200, 201):
            return response.json()
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text
            st.error(f"API 请求失败 (HTTP {response.status_code}): {error_detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ 无法连接到后端服务，请确认后端已启动 (python server/app.py)")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ API 请求超时")
        return None
    except json.JSONDecodeError:
        st.error("❌ API 返回数据格式错误")
        return None
    except Exception as e:
        st.error(f"❌ API 请求异常: {str(e)}")
        return None


def safe_api_delete(endpoint):
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.delete(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text
            st.error(f"API 请求失败 (HTTP {response.status_code}): {error_detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ 无法连接到后端服务，请确认后端已启动 (python server/app.py)")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ API 请求超时")
        return None
    except json.JSONDecodeError:
        st.error("❌ API 返回数据格式错误")
        return None
    except Exception as e:
        st.error(f"❌ API 请求异常: {str(e)}")
        return None

st.set_page_config(
    page_title="语音唤醒词检测与命令识别系统",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 语音唤醒词检测与命令识别系统")

st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择功能",
    ["实时检测", "文件测试", "唤醒词管理", "命令模型管理", "数字模型管理", "说话人管理", "验证统计", "检测历史", "系统状态"]
)


def plot_waveform(samples, sample_rate=16000, title="音频波形"):
    fig, ax = plt.subplots(figsize=(10, 3))
    duration = len(samples) / sample_rate
    time_axis = np.linspace(0, duration, len(samples))
    ax.plot(time_axis, samples)
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("幅度")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_mfcc(mfcc_data, title="MFCC 热力图"):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mfcc_data, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel("帧")
    ax.set_ylabel("MFCC 系数")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_confidence_curve(confidence_history, threshold=0.75):
    fig, ax = plt.subplots(figsize=(10, 3))
    times = [item["time"] for item in confidence_history]
    confidences = [item["confidence"] for item in confidence_history]
    ax.plot(times, confidences, label="置信度")
    ax.axhline(y=threshold, color='r', linestyle='--', label=f"阈值 ({threshold})")
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("置信度")
    ax.set_title("唤醒词检测置信度曲线")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_vad_result(samples, vad_segments, sample_rate=16000):
    fig, ax = plt.subplots(figsize=(10, 3))
    duration = len(samples) / sample_rate
    time_axis = np.linspace(0, duration, len(samples))
    ax.plot(time_axis, samples, label="音频波形", alpha=0.7)

    for seg in vad_segments:
        ax.axvspan(seg["start_time"], seg["end_time"],
                   alpha=0.3, color='green', label='语音段' if seg == vad_segments[0] else "")

    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("幅度")
    ax.set_title("VAD 语音活动检测结果")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


if page == "实时检测":
    st.header("🎙️ 实时录音检测")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("实时检测")

        if "is_recording" not in st.session_state:
            st.session_state.is_recording = False
        if "audio_buffer" not in st.session_state:
            st.session_state.audio_buffer = np.array([], dtype=np.float32)
        if "confidence_history" not in st.session_state:
            st.session_state.confidence_history = []
        if "detection_results" not in st.session_state:
            st.session_state.detection_results = []
        if "last_wake_word" not in st.session_state:
            st.session_state.last_wake_word = None
        if "last_command" not in st.session_state:
            st.session_state.last_command = None

        start_col, stop_col, clear_col = st.columns(3)
        with start_col:
            if st.button("开始录音", type="primary", disabled=st.session_state.is_recording):
                st.session_state.is_recording = True
                st.session_state.audio_buffer = np.array([], dtype=np.float32)
                st.session_state.confidence_history = []
                st.session_state.detection_results = []
                st.session_state.last_wake_word = None
                st.session_state.last_command = None
                st.success("开始录音...")

        with stop_col:
            if st.button("停止录音", disabled=not st.session_state.is_recording):
                st.session_state.is_recording = False
                st.info("已停止录音")

        with clear_col:
            if st.button("清空结果"):
                st.session_state.confidence_history = []
                st.session_state.detection_results = []
                st.session_state.last_wake_word = None
                st.session_state.last_command = None
                st.rerun()

        status_placeholder = st.empty()
        if st.session_state.is_recording:
            status_placeholder.info("🔴 正在录音 - 请说出唤醒词...")
        else:
            status_placeholder.info("⏸️ 已停止录音")

    with col2:
        st.subheader("检测结果")

        if st.session_state.last_wake_word:
            st.success(f"✅ 唤醒词: **{st.session_state.last_wake_word}**")
        else:
            st.info("等待唤醒词检测...")

        if st.session_state.last_command:
            st.success(f"📢 命令: **{st.session_state.last_command.get('command', 'N/A')}**")
            if st.session_state.last_command.get('params'):
                st.write(f"参数: {st.session_state.last_command['params']}")
            st.write(f"置信度: {st.session_state.last_command.get('confidence', 0):.2%}")
        else:
            st.info("等待命令识别...")

        if st.session_state.detection_results:
            with st.expander("历史检测记录"):
                for i, result in enumerate(reversed(st.session_state.detection_results[-10:])):
                    st.write(f"{len(st.session_state.detection_results) - i}. "
                             f"{result.get('wake_word', '未检测')} - "
                             f"置信度: {result.get('confidence', 0):.2f}")

    st.divider()
    st.subheader("可视化展示")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        if len(st.session_state.audio_buffer) > 0:
            samples = st.session_state.audio_buffer[::10]
            fig = plot_waveform(samples, config.SAMPLE_RATE)
            st.pyplot(fig)
            plt.close(fig)

    with viz_col2:
        if st.session_state.confidence_history:
            fig = plot_confidence_curve(
                st.session_state.confidence_history,
                config.WAKE_WORD_THRESHOLD
            )
            st.pyplot(fig)
            plt.close(fig)

    st.info("💡 提示：由于浏览器安全限制，完整的实时录音功能需要通过浏览器API实现。"
            "您可以使用文件上传功能或后端WebSocket API进行测试。")

elif page == "文件测试":
    st.header("📁 音频文件测试")

    uploaded_file = st.file_uploader("上传音频文件 (WAV/MP3)", type=["wav", "mp3"])

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")

        col1, col2 = st.columns(2)

        with col1:
            test_type = st.radio(
                "测试类型",
                ["完整识别 (唤醒词+命令)", "仅唤醒词检测", "仅命令识别"]
            )

        with col2:
            st.write("")
            st.write("")
            analyze_button = st.button("开始分析", type="primary")

        if analyze_button:
            with st.spinner("正在分析音频..."):
                files = {"file": uploaded_file.getvalue()}

                if test_type == "完整识别 (唤醒词+命令)":
                    result = safe_api_post("/api/recognize/full", files=files)
                    if result is None:
                        st.error("❌ 识别请求失败，请检查后端服务")
                    else:
                        st.subheader("识别结果")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            if result.get("wake_word_detected"):
                                st.success(f"✅ 唤醒词: {result.get('wake_word', 'N/A')}")
                            else:
                                st.error("❌ 未检测到唤醒词")
                        with col2:
                            st.metric("唤醒词置信度", f"{result.get('wake_word_confidence', 0):.2%}")
                        with col3:
                            st.metric("音频时长", f"{result.get('duration', 0):.2f} 秒")
                        with col4:
                            noise_map = {"quiet": "🟢 安静", "moderate": "🟡 一般", "noisy": "🔴 嘈杂"}
                            nl = result.get("noise_level", "moderate")
                            st.metric("环境噪声", noise_map.get(nl, nl))
                            st.caption(f"动态阈值: {result.get('adaptive_threshold', config.WAKE_WORD_THRESHOLD):.3f}")

                        if result.get("speaker_verified") is not None:
                            spk_col1, spk_col2, spk_col3, spk_col4 = st.columns(4)
                            with spk_col1:
                                if result.get("speaker_verified"):
                                    level = result.get("confidence_level", "high")
                                    level_map = {"high": "🟢 高置信度", "medium": "🟡 中置信度"}
                                    st.success(f"✅ 说话人验证通过 ({level_map.get(level, level)})")
                                else:
                                    blocked_by = result.get("blocked_by")
                                    if blocked_by:
                                        blocked_name = result.get("blocked_by_name", "")
                                        st.error(f"🚫 黑名单拦截: {blocked_name}")
                                    elif result.get("speaker_confidence", 0) > 0:
                                        st.error("❌ 说话人验证未通过 (低置信度)")
                                    else:
                                        st.info("⏭️ 说话人验证跳过")
                            with spk_col2:
                                spk_id = result.get("speaker_id")
                                if spk_id:
                                    st.metric("匹配说话人ID", spk_id[:8] + "...")
                                else:
                                    st.metric("匹配说话人", "无")
                            with spk_col3:
                                st.metric("说话人相似度", f"{result.get('speaker_confidence', 0):.2%}")
                            with spk_col4:
                                level = result.get("confidence_level", "rejected")
                                level_label = {"high": "🟢 高", "medium": "🟡 中", "rejected": "🔴 拒绝"}
                                st.metric("置信度级别", level_label.get(level, level))
                            st.caption(f"高阈值: {config.SPEAKER_HIGH_CONFIDENCE_THRESHOLD} | 中阈值: {config.SPEAKER_MEDIUM_CONFIDENCE_THRESHOLD} | 黑名单阈值: {config.SPEAKER_BLACKLIST_THRESHOLD}")

                        if result.get("command"):
                            st.divider()
                            st.subheader("命令识别结果")
                            cmd = result.get("command")
                            if cmd == "unclear_command":
                                st.warning("⚠️ 命令识别不确定，需要用户重说")
                                reason = result.get("rejection_reason", "")
                                if reason == "low_confidence":
                                    st.caption("原因: 最高置信度过低")
                                elif reason == "low_margin":
                                    st.caption("原因: 最高与次高置信度差距太小")
                            elif cmd == "unknown_command":
                                st.error("❌ 无法识别命令")
                            else:
                                st.success(f"📢 命令类型: **{cmd}**")
                            st.metric("命令置信度", f"{result.get('command_confidence', 0):.2%}")
                            if result.get("params"):
                                st.write(f"提取参数: {result['params']}")
                            top_candidates = result.get("top_candidates", [])
                            if top_candidates:
                                with st.expander("🔍 候选命令 (Top3)"):
                                    for i, cand in enumerate(top_candidates, 1):
                                        st.write(f"{i}. **{cand['command']}** - 置信度: {cand['confidence']:.2%}")

                elif test_type == "仅唤醒词检测":
                    result = safe_api_post("/api/detect/wake-word", files=files)
                    if result is None:
                        st.error("❌ 检测请求失败，请检查后端服务")
                    else:
                        st.subheader("唤醒词检测结果")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if result.get("detected"):
                                st.success(f"✅ 检测到唤醒词: **{result.get('wake_word')}**")
                            else:
                                st.error("❌ 未检测到唤醒词")
                        with col2:
                            st.metric("置信度", f"{result.get('confidence', 0):.2%}")
                        with col3:
                            noise_map = {"quiet": "🟢 安静", "moderate": "🟡 一般", "noisy": "🔴 嘈杂"}
                            nl = result.get("noise_level", "moderate")
                            st.metric("环境噪声", noise_map.get(nl, nl))
                            st.caption(f"动态阈值: {result.get('adaptive_threshold', config.WAKE_WORD_THRESHOLD):.3f}")

                        if not result.get("detected"):
                            st.metric("最高置信度", f"{result.get('confidence', 0):.2%}")

                        with st.expander("详细置信度"):
                            st.write("DTW 置信度:", result.get("dtw_confidence", {}))
                            st.write("CNN 置信度:", result.get("cnn_confidence", {}))
                            st.write("融合置信度:", result.get("combined_confidence", {}))

                else:
                    result = safe_api_post("/api/recognize/command", files=files)
                    if result is None:
                        st.error("❌ 识别请求失败，请检查后端服务")
                    else:
                        st.subheader("命令识别结果")

                        cmd = result.get("command")
                        if cmd == "unclear_command":
                            st.warning("⚠️ 命令识别不确定，需要用户重说")
                            reason = result.get("rejection_reason", "")
                            if reason == "low_confidence":
                                st.caption("原因: 最高置信度过低 (< {:.2f})".format(config.COMMAND_MIN_CONFIDENCE))
                            elif reason == "low_margin":
                                st.caption("原因: 最高与次高置信度差距太小 (< {:.2f})".format(config.COMMAND_REJECT_MARGIN))
                        elif cmd == "unknown_command":
                            st.error("❌ 无法识别命令")
                        else:
                            st.success(f"📢 识别到命令: **{cmd}**")
                        st.metric("置信度", f"{result.get('confidence', 0):.2%}")
                        if result.get("params"):
                            st.write(f"提取参数: {result['params']}")

                        top_candidates = result.get("top_candidates", [])
                        if top_candidates:
                            with st.expander("🔍 候选命令 (Top3)"):
                                for i, cand in enumerate(top_candidates, 1):
                                    st.write(f"{i}. **{cand['command']}** - 置信度: {cand['confidence']:.2%}")

                st.divider()
                st.subheader("音频分析")

                temp_path = os.path.join(config.TEMP_DIR, "test_audio.wav")
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                from core.audio_preprocessing import load_audio, compute_mfcc, VAD
                signal, sr = load_audio(temp_path)

                viz1, viz2 = st.columns(2)

                with viz1:
                    fig = plot_waveform(signal[::10], sr, "音频波形")
                    st.pyplot(fig)
                    plt.close(fig)

                vad = VAD()
                vad_result = vad.detect(signal)
                fig = plot_vad_result(signal[::10], vad_result["speech_segments"], sr)
                st.pyplot(fig)
                plt.close(fig)

                mfcc = compute_mfcc(signal)
                fig = plot_mfcc(mfcc.T, "MFCC 热力图 (39维)")
                st.pyplot(fig)
                plt.close(fig)

                if os.path.exists(temp_path):
                    os.remove(temp_path)

elif page == "唤醒词管理":
    st.header("🔑 唤醒词管理")

    st.subheader("已注册的唤醒词")

    ww_data = safe_api_get("/api/wake-words")
    if ww_data is None:
        st.warning("⚠️ 无法获取唤醒词列表，请检查后端服务")
        wake_words = []
    else:
        wake_words = ww_data.get("wake_words", [])

    if wake_words:
        for ww in wake_words:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.write(f"**{ww['name']}**")
            with col2:
                st.write(f"样本数: {ww['samples']}")
            with col3:
                st.write(f"准确率: {ww['accuracy']:.2%}")
            with col4:
                if st.button(f"删除", key=f"del_{ww['name']}"):
                    delete_result = safe_api_delete(f"/api/wake-words/{ww['name']}")
                    if delete_result is not None:
                        st.success(f"已删除唤醒词: {ww['name']}")
                        st.rerun()
    else:
        st.info("暂无已注册的唤醒词")

    st.divider()
    st.subheader("注册新唤醒词")

    new_name = st.text_input("唤醒词名称", placeholder="例如: 小助手")

    sample_files = st.file_uploader(
        "上传3-5个正样本音频 (WAV格式)",
        type=["wav"],
        accept_multiple_files=True
    )

    if sample_files:
        st.write(f"已选择 {len(sample_files)} 个样本文件")

    if st.button("注册唤醒词", type="primary", disabled=not (new_name and sample_files and len(sample_files) >= 3)):
        if len(sample_files) < 3:
            st.error("至少需要3个正样本音频")
        else:
            with st.spinner("正在注册唤醒词并训练模型..."):
                files = [("files", (f.name, f.getvalue(), "audio/wav")) for f in sample_files]
                result = safe_api_post(f"/api/wake-words/{new_name}/register", files=files)
                if result is not None:
                    st.success(result["message"])
                    st.rerun()

    st.divider()
    st.subheader("添加样本")

    existing_names = [ww["name"] for ww in wake_words]
    if existing_names:
        selected_ww = st.selectbox("选择唤醒词", existing_names, key="add_sample_select")
        add_sample_file = st.file_uploader("上传样本音频", type=["wav"], key="add_sample_upload")

        if st.button("添加样本", disabled=not (selected_ww and add_sample_file)):
            files = {"file": (add_sample_file.name, add_sample_file.getvalue(), "audio/wav")}
            result = safe_api_post(f"/api/wake-words/{selected_ww}/add-sample", files=files)
            if result is not None:
                st.success(result["message"])
                st.rerun()

    st.divider()
    if st.button("🔄 重新加载所有模型"):
        with st.spinner("重新加载模型..."):
            result = safe_api_post("/api/models/reload")
            if result is not None:
                st.success("模型重新加载完成")
                st.rerun()

elif page == "命令模型管理":
    st.header("📋 命令模型管理")

    st.subheader("已训练的命令模型")

    cmd_data = safe_api_get("/api/commands")
    if cmd_data is None:
        st.warning("⚠️ 无法获取命令模型列表，请检查后端服务")
        commands = []
    else:
        commands = cmd_data.get("commands", [])

    if commands:
        for cmd in commands:
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
            with col1:
                st.write(f"**{cmd['name']}**")
            with col2:
                st.write(f"样本: {cmd['samples']}")
            with col3:
                st.write(f"状态: {cmd['states']}")
            with col4:
                st.write(f"混合: {cmd['mixtures']}")
            with col5:
                if st.button(f"删除", key=f"del_cmd_{cmd['name']}"):
                    delete_result = safe_api_delete(f"/api/commands/{cmd['name']}")
                    if delete_result is not None:
                        st.success(f"已删除命令: {cmd['name']}")
                        st.rerun()
    else:
        st.info("暂无已训练的命令模型")

    st.divider()
    st.subheader("预定义命令列表")

    predefined_data = safe_api_get("/api/commands/predefined")
    if predefined_data is None:
        st.warning("⚠️ 无法获取预定义命令列表")
        predefined = []
    else:
        predefined = predefined_data.get("commands", [])

    st.write("系统支持的预定义命令类型:")
    for cmd_name in predefined:
        cmd_config = config.COMMANDS.get(cmd_name, {})
        has_number = cmd_config.get("has_number", False)
        st.write(f"- **{cmd_name}** {'(含数字参数)' if has_number else ''}")

    st.divider()
    st.subheader("训练新命令模型")

    new_cmd = st.text_input("命令名称", placeholder="例如: turn_on_light")

    col1, col2 = st.columns(2)
    with col1:
        num_states = st.slider("HMM 状态数", min_value=2, max_value=10, value=5)
    with col2:
        num_mixtures = st.slider("高斯混合数", min_value=1, max_value=8, value=3)

    cmd_samples = st.file_uploader(
        "上传2-10个训练样本 (WAV格式)",
        type=["wav"],
        accept_multiple_files=True,
        key="cmd_samples"
    )

    if st.button("训练命令", type="primary", disabled=not (new_cmd and cmd_samples and len(cmd_samples) >= 2)):
        if len(cmd_samples) < 2:
            st.error("至少需要2个训练样本")
        else:
            with st.spinner("正在训练命令模型..."):
                files = [("files", (f.name, f.getvalue(), "audio/wav")) for f in cmd_samples]
                params = {"num_states": num_states, "num_mixtures": num_mixtures}
                result = safe_api_post(f"/api/commands/{new_cmd}/train", files=files, params=params)
                if result is not None:
                    st.success(result["message"])
                    st.rerun()

elif page == "数字模型管理":
    st.header("🔢 数字识别模型管理")

    st.subheader("已训练的数字模型")

    digit_data = safe_api_get("/api/digits")
    if digit_data is None:
        st.warning("⚠️ 无法获取数字模型列表，请检查后端服务")
        digits = []
    else:
        digits = digit_data.get("digits", [])

    if digits:
        for d in digits:
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                st.write(f"**数字 {d['digit']}**")
            with col2:
                st.write(f"样本: {d['samples']}")
            with col3:
                st.write(f"状态: {d['states']}")
            with col4:
                st.write(f"混合: {d['mixtures']}")
    else:
        st.info("暂无已训练的数字模型")

    st.divider()
    st.subheader("训练数字模型")

    digit_to_train = st.selectbox("选择数字", [str(i) for i in range(10)])

    col1, col2 = st.columns(2)
    with col1:
        digit_states = st.slider("HMM 状态数", min_value=2, max_value=6, value=3, key="digit_states")
    with col2:
        digit_mixtures = st.slider("高斯混合数", min_value=1, max_value=5, value=2, key="digit_mixtures")

    digit_samples = st.file_uploader(
        "上传2-5个训练样本 (WAV格式)",
        type=["wav"],
        accept_multiple_files=True,
        key="digit_samples"
    )

    if st.button("训练数字模型", type="primary", disabled=not digit_samples or len(digit_samples) < 2):
        if len(digit_samples) < 2:
            st.error("至少需要2个训练样本")
        else:
            with st.spinner("正在训练数字模型..."):
                files = [("files", (f.name, f.getvalue(), "audio/wav")) for f in digit_samples]
                params = {"num_states": digit_states, "num_mixtures": digit_mixtures}
                result = safe_api_post(f"/api/digits/{digit_to_train}/train", files=files, params=params)
                if result is not None:
                    st.success(result["message"])
                    st.rerun()

elif page == "说话人管理":
    st.header("👤 说话人管理")

    spk_data = safe_api_get("/api/speakers")
    if spk_data is None:
        st.warning("⚠️ 无法获取说话人列表，请检查后端服务")
        speakers = []
    else:
        speakers = spk_data.get("speakers", [])

    whitelist = [s for s in speakers if s.get('speaker_type', 'whitelist') == 'whitelist']
    blacklist = [s for s in speakers if s.get('speaker_type') == 'blacklist']

    spk_tab1, spk_tab2 = st.tabs(["✅ 白名单说话人", "🚫 黑名单说话人"])

    with spk_tab1:
        st.subheader("白名单说话人")
        if whitelist:
            for spk in whitelist:
                col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
                with col1:
                    st.write(f"**{spk['name']}**")
                    st.caption(f"ID: {spk['id']}")
                with col2:
                    st.write(f"样本数: {spk['num_samples']}")
                with col3:
                    st.write(f"注册时间:")
                    st.caption(spk['created_at'])
                with col4:
                    st.write(f"验证次数:")
                    st.caption(str(spk.get('verify_count', 0)))
                with col5:
                    last_time = spk.get('last_verify_time')
                    st.write(f"最近验证:")
                    st.caption(last_time if last_time else "从未")
                with col6:
                    if st.button(f"删除", key=f"del_spk_{spk['id']}"):
                        delete_result = safe_api_delete(f"/api/speakers/{spk['id']}")
                        if delete_result is not None:
                            st.success(f"已删除说话人: {spk['name']}")
                            st.rerun()
        else:
            st.info("暂无白名单说话人")

    with spk_tab2:
        st.subheader("黑名单说话人")
        if blacklist:
            for spk in blacklist:
                col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
                with col1:
                    st.write(f"🚫 **{spk['name']}**")
                    st.caption(f"ID: {spk['id']}")
                with col2:
                    st.write(f"样本数: {spk['num_samples']}")
                with col3:
                    st.write(f"注册时间:")
                    st.caption(spk['created_at'])
                with col4:
                    if st.button(f"删除", key=f"del_blk_{spk['id']}"):
                        delete_result = safe_api_delete(f"/api/speakers/{spk['id']}")
                        if delete_result is not None:
                            st.success(f"已删除黑名单说话人: {spk['name']}")
                            st.rerun()
        else:
            st.info("暂无黑名单说话人")

    st.divider()
    st.subheader("注册新说话人")

    new_spk_name = st.text_input("说话人名称", placeholder="例如: 张三", key="spk_name_input")

    new_spk_type = st.radio(
        "注册类型",
        ["whitelist", "blacklist"],
        format_func=lambda x: "✅ 白名单 (允许通过)" if x == "whitelist" else "🚫 黑名单 (强制拒绝)",
        horizontal=True
    )

    spk_sample_files = st.file_uploader(
        f"上传 {config.SPEAKER_MIN_SAMPLES} 段以上音频样本 (WAV格式，任意内容)",
        type=["wav"],
        accept_multiple_files=True,
        key="spk_samples"
    )

    if spk_sample_files:
        st.write(f"已选择 {len(spk_sample_files)} 个样本文件")

    if st.button("注册说话人", type="primary", disabled=not (new_spk_name and spk_sample_files and len(spk_sample_files) >= config.SPEAKER_MIN_SAMPLES)):
        if len(spk_sample_files) < config.SPEAKER_MIN_SAMPLES:
            st.error(f"至少需要 {config.SPEAKER_MIN_SAMPLES} 段音频样本")
        else:
            with st.spinner("正在注册说话人并提取声纹特征..."):
                files = [("files", (f.name, f.getvalue(), "audio/wav")) for f in spk_sample_files]
                params = {"name": new_spk_name, "speaker_type": new_spk_type}
                result = safe_api_post("/api/speakers/register", files=files, params=params)
                if result is not None:
                    st.success(result["message"])
                    st.rerun()

    st.divider()
    st.subheader("验证说明")
    st.info(
        f"💡 **说话人验证流程 (多阈值分级)**\n\n"
        f"- 使用 **余弦相似度** 算法比对声纹特征\n"
        f"- 🟢 **高置信度** (≥ {config.SPEAKER_HIGH_CONFIDENCE_THRESHOLD}): 直接通过 + 自动更新声纹模板\n"
        f"- 🟡 **中置信度** (≥ {config.SPEAKER_MEDIUM_CONFIDENCE_THRESHOLD} 且 < {config.SPEAKER_HIGH_CONFIDENCE_THRESHOLD}): 通过但不更新声纹\n"
        f"- 🔴 **低置信度** (< {config.SPEAKER_MEDIUM_CONFIDENCE_THRESHOLD}): 拒绝\n"
        f"- 🚫 **黑名单拦截**: 与黑名单说话人相似度 ≥ {config.SPEAKER_BLACKLIST_THRESHOLD} 时强制拒绝\n"
        f"- 📈 **声纹更新**: 高置信度通过时使用指数移动平均(EMA)融合新特征，衰减系数: {config.SPEAKER_EMA_DECAY}\n"
        f"- 如果没有白名单说话人，验证步骤自动跳过"
    )

elif page == "验证统计":
    st.header("📊 说话人验证统计")

    stats = safe_api_get("/api/speakers/verification-stats")
    if stats is None:
        st.warning("⚠️ 无法获取验证统计数据，请检查后端服务")
        stats = {}

    st.subheader("统计摘要")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("总验证次数", stats.get("total_verifications", 0))
    with col2:
        st.metric("通过率", f"{stats.get('pass_rate', 0):.1%}")
    with col3:
        st.metric("高置信度", stats.get("high_count", 0))
    with col4:
        st.metric("中置信度", stats.get("medium_count", 0))
    with col5:
        st.metric("拒绝次数", stats.get("rejected_count", 0))

    st.divider()

    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.subheader("置信度级别分布")
        total = stats.get("total_verifications", 0)
        if total > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            labels = ["高置信度", "中置信度", "拒绝"]
            sizes = [stats.get("high_count", 0), stats.get("medium_count", 0), stats.get("rejected_count", 0)]
            colors = ["#4CAF50", "#FFC107", "#F44336"]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                startangle=90, colors=colors, pctdistance=0.85
            )
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig.gca().add_artist(centre_circle)
            ax.axis('equal')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("暂无验证数据")

    with col_bar:
        st.subheader("关键指标")
        st.metric("🚫 黑名单拦截次数", stats.get("blacklist_blocks", 0))
        st.metric("🟢 高置信度占比", f"{stats.get('high_ratio', 0):.1%}")
        st.metric("🟡 中置信度占比", f"{stats.get('medium_ratio', 0):.1%}")
        st.metric("🔴 拒绝占比", f"{stats.get('rejected_ratio', 0):.1%}")

    st.divider()
    st.subheader("各说话人验证详情")

    speaker_stats = stats.get("speaker_stats", {})
    if speaker_stats:
        spk_detail_rows = []
        for spk_id, spk_stat in speaker_stats.items():
            spk_detail_rows.append({
                "说话人": spk_stat.get("name", "未知"),
                "ID": spk_id[:8] + "...",
                "验证通过次数": spk_stat.get("verify_count", 0),
                "最近验证时间": spk_stat.get("last_verify_time", "从未")
            })
        spk_detail_rows.sort(key=lambda x: x["验证通过次数"], reverse=True)
        st.dataframe(spk_detail_rows, use_container_width=True, hide_index=True)
    else:
        st.info("暂无说话人验证记录")

    st.divider()
    col_reset1, col_reset2 = st.columns([3, 1])
    with col_reset1:
        st.caption("重置统计将清除所有验证历史数据，此操作不可恢复")
    with col_reset2:
        if st.button("🗑️ 重置统计"):
            result = safe_api_post("/api/speakers/verification-stats/reset")
            if result is not None:
                st.success("验证统计已重置")
                st.rerun()

elif page == "检测历史":
    st.header("📋 检测历史统计")

    stats = safe_api_get("/api/detection/stats")
    if stats is None:
        st.warning("⚠️ 无法获取统计数据，请检查后端服务")
        stats = {"total_detections": 0, "hit_rate": 0.0, "avg_confidence": 0.0, "wake_word_distribution": {}}

    st.subheader("统计摘要")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总检测次数", stats.get("total_detections", 0))
    with col2:
        st.metric("命中率", f"{stats.get('hit_rate', 0):.2%}")
    with col3:
        st.metric("平均置信度", f"{stats.get('avg_confidence', 0):.2%}")

    distribution = stats.get("wake_word_distribution", {})
    if distribution and len(distribution) > 0:
        st.divider()
        col_pie, col_table = st.columns([2, 1])

        with col_pie:
            st.subheader("各唤醒词命中次数分布")
            fig, ax = plt.subplots(figsize=(8, 6))
            labels = list(distribution.keys())
            sizes = list(distribution.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                startangle=90, colors=colors, pctdistance=0.85
            )
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig.gca().add_artist(centre_circle)
            ax.axis('equal')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_table:
            st.subheader("分布详情")
            dist_data = [{"唤醒词": k, "次数": v} for k, v in distribution.items()]
            dist_data.sort(key=lambda x: x["次数"], reverse=True)
            st.dataframe(dist_data, use_container_width=True, hide_index=True)
    else:
        st.info("📊 暂无检测分布数据，饼图将在产生检测记录后显示")

    st.divider()
    st.subheader("检测记录 (最近100条)")

    ww_data = safe_api_get("/api/wake-words")
    if ww_data is None:
        wake_words_data = []
    else:
        wake_words_data = ww_data.get("wake_words", [])
    ww_options = ["全部"] + [ww["name"] for ww in wake_words_data] + ["未命中"]

    filter_col, refresh_col = st.columns([3, 1])
    with filter_col:
        selected_ww = st.selectbox("按唤醒词筛选", ww_options)
    with refresh_col:
        st.write("")
        st.write("")
        if st.button("🔄 刷新"):
            st.rerun()

    history_params = {"limit": 100}
    if selected_ww != "全部":
        history_params["wake_word"] = selected_ww

    history_resp = safe_api_get("/api/detection/history", params=history_params)
    if history_resp is None:
        history_data = []
    else:
        history_data = history_resp.get("history", [])

    if history_data:
        display_records = []
        noise_level_map = {"quiet": "🟢 安静", "moderate": "🟡 一般", "noisy": "🔴 嘈杂"}
        for r in history_data:
            nl = r.get("noise_level", "moderate")
            display_records.append({
                "时间": r["timestamp"],
                "唤醒词": r["wake_word"],
                "是否命中": "✅ 是" if r["detected"] else "❌ 否",
                "置信度": f"{r['confidence']:.2%}",
                "音频时长(秒)": f"{r['duration']:.2f}",
                "环境噪声": noise_level_map.get(nl, f"未知({nl})"),
                "动态阈值": f"{r.get('adaptive_threshold', config.WAKE_WORD_THRESHOLD):.2f}"
            })

        st.dataframe(display_records, use_container_width=True, hide_index=True)
        st.info(f"共显示 {len(display_records)} 条记录（按时间倒序）")
    else:
        st.info("暂无检测记录，请先进行一些检测测试。")

elif page == "系统状态":
    st.header("📊 系统状态")

    status = safe_api_get("/api/status")
    if status is None:
        st.warning("⚠️ 无法获取系统状态，请检查后端服务是否启动")
        status = {
            "is_initialized": False,
            "num_wake_words": 0,
            "num_commands": 0,
            "num_digits": 0,
            "num_speakers": 0,
            "speaker_verification_enabled": config.SPEAKER_VERIFICATION_ENABLED,
            "speaker_verification_threshold": config.SPEAKER_VERIFICATION_THRESHOLD,
            "wake_word_threshold": config.WAKE_WORD_THRESHOLD,
            "command_confidence_threshold": config.COMMAND_CONFIDENCE_THRESHOLD,
            "sample_rate": config.SAMPLE_RATE,
            "environment": {"noise_level": "moderate", "noise_rms": 0.0, "adaptive_threshold": config.WAKE_WORD_THRESHOLD, "base_threshold": config.WAKE_WORD_THRESHOLD}
        }
    environment = status.get("environment", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("唤醒词数量", status.get("num_wake_words", 0))
    with col2:
        st.metric("命令模型数量", status.get("num_commands", 0))
    with col3:
        st.metric("数字模型数量", status.get("num_digits", 0))
    with col4:
        st.metric("说话人数量", status.get("num_speakers", 0))

    spk_enabled = status.get("speaker_verification_enabled", False)
    if spk_enabled:
        st.success(f"✅ 说话人验证功能已启用")
    else:
        st.info("⏸️ 说话人验证功能未启用")

    st.divider()
    st.subheader("🌿 当前环境状态 (实时)")

    noise_level = environment.get("noise_level", "moderate")
    noise_rms = environment.get("noise_rms", 0.0)
    adaptive_thresh = environment.get("adaptive_threshold", config.WAKE_WORD_THRESHOLD)
    base_thresh = environment.get("base_threshold", config.WAKE_WORD_THRESHOLD)

    noise_colors = {
        "quiet": ("🟢", "安静", "#4CAF50"),
        "moderate": ("🟡", "一般", "#FFC107"),
        "noisy": ("🔴", "嘈杂", "#F44336")
    }
    emoji, label, color = noise_colors.get(noise_level, ("⚪", "未知", "#9E9E9E"))

    env_col1, env_col2, env_col3 = st.columns(3)
    with env_col1:
        st.markdown(f"### {emoji} 环境噪声等级")
        st.markdown(f"<h2 style='color:{color}; text-align:center;'>{label}</h2>", unsafe_allow_html=True)
        st.caption(f"RMS值: {noise_rms:.6f}")
    with env_col2:
        st.markdown("### 📊 当前动态阈值")
        st.markdown(f"<h2 style='text-align:center;'>{adaptive_thresh:.3f}</h2>", unsafe_allow_html=True)
        st.caption(f"基础阈值: {base_thresh:.3f}")
        delta = adaptive_thresh - base_thresh
        if abs(delta) > 0.001:
            direction = "↑ 提高" if delta > 0 else "↓ 降低"
            st.info(f"{direction} {abs(delta):.3f}（自适应调整）")
    with env_col3:
        st.markdown("### 📈 阈值范围")
        progress = (adaptive_thresh - config.ADAPTIVE_THRESHOLD_MIN) / (config.ADAPTIVE_THRESHOLD_MAX - config.ADAPTIVE_THRESHOLD_MIN)
        st.progress(max(0.0, min(1.0, progress)))
        st.caption(f"范围: [{config.ADAPTIVE_THRESHOLD_MIN:.2f}, {config.ADAPTIVE_THRESHOLD_MAX:.2f}]")

    st.divider()

    st.subheader("系统配置")

    st.write(f"- **采样率**: {status.get('sample_rate', 0)} Hz")
    st.write(f"- **唤醒词基础阈值**: {status.get('wake_word_threshold', 0)}")
    st.write(f"- **命令置信度阈值**: {status.get('command_confidence_threshold', 0)}")
    st.write(f"- **说话人验证**: {'✅ 已启用' if spk_enabled else '⏸️ 未启用'}")
    st.write(f"- **说话人高置信度阈值**: {status.get('speaker_high_confidence_threshold', config.SPEAKER_HIGH_CONFIDENCE_THRESHOLD)}")
    st.write(f"- **说话人中置信度阈值**: {status.get('speaker_medium_confidence_threshold', config.SPEAKER_MEDIUM_CONFIDENCE_THRESHOLD)}")
    st.write(f"- **说话人黑名单阈值**: {status.get('speaker_blacklist_threshold', config.SPEAKER_BLACKLIST_THRESHOLD)}")
    st.write(f"- **声纹EMA衰减系数**: {status.get('speaker_ema_decay', config.SPEAKER_EMA_DECAY)}")
    st.write(f"- **说话人最少样本数**: {config.SPEAKER_MIN_SAMPLES}")
    st.write(f"- **自适应阈值范围**: [{config.ADAPTIVE_THRESHOLD_MIN:.2f}, {config.ADAPTIVE_THRESHOLD_MAX:.2f}]")
    st.write(f"- **命令拒识最低置信度**: < {config.COMMAND_MIN_CONFIDENCE:.2f}")
    st.write(f"- **命令拒识最小间隔**: < {config.COMMAND_REJECT_MARGIN:.2f}")
    st.write(f"- **系统初始化状态**: {'✅ 已初始化' if status.get('is_initialized') else '⚠️ 未初始化'}")

    st.divider()

    st.subheader("性能参数")

    st.write(f"- **单帧处理目标**: < 50ms")
    st.write(f"- **最大并发流数**: {config.MAX_CONCURRENT_STREAMS}")
    st.write(f"- **最大唤醒词数量**: {config.MAX_WAKE_WORDS}")
    st.write(f"- **命令窗口时长**: {config.COMMAND_WINDOW_DURATION} 秒")

    st.divider()

    st.subheader("模型目录")

    st.write(f"- **模型根目录**: {config.MODELS_DIR}")
    st.write(f"- **唤醒词模型**: {config.WAKE_WORDS_DIR}")
    st.write(f"- **命令模型**: {config.COMMANDS_DIR}")
    st.write(f"- **数字模型**: {config.DIGITS_DIR}")
    st.write(f"- **说话人模型**: {config.SPEAKERS_DIR}")

    st.divider()

    if st.button("🔄 刷新系统状态"):
        st.rerun()

st.sidebar.divider()
st.sidebar.info("📌 提示：请确保后端服务已启动 (python server/app.py)")
