"""
登录注册页面组件
Streamlit Web 应用的认证模块
"""
import streamlit as st
import hashlib
import time
from typing import Optional, Any


def hash_password(password: str) -> str:
    """SHA256 密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()


def show_auth_page(db):
    """
    登录/注册页面
    db: DatabaseManager 实例
    """
    st.markdown("""
    <div style="text-align:center;padding:40px 20px 20px;">
        <div style="font-size:48px;margin-bottom:8px;">📚</div>
        <h1 style="color:#1a1a2e;margin:0;">RAG 文档问答系统</h1>
        <p style="color:#666;margin-top:8px;">基于招股说明书的智能问答</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["登录", "注册"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="输入用户名")
            password = st.text_input("密码", type="password", placeholder="输入密码")
            submitted = st.form_submit_button("登录", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("用户名和密码不能为空")
                elif db is None:
                    st.error("数据库未连接，无法登录")
                else:
                    result = db.login_user(username, hash_password(password))
                    if result["success"]:
                        st.session_state.logged_in = True
                        st.session_state.user_id = result["user_id"]
                        st.session_state.username = result["username"]
                        st.session_state.session_id = f"ses_{int(time.time())}"
                        st.session_state.chat_messages = []
                        st.session_state.mode = "rag"
                        st.success(f"欢迎回来，{username}！")
                        st.rerun()
                    else:
                        st.error("用户名或密码错误")

    with tab2:
        with st.form("register_form"):
            reg_user = st.text_input("用户名", placeholder="设置用户名", key="reg_user")
            reg_pass = st.text_input("密码", type="password", placeholder="设置密码", key="reg_pass")
            reg_confirm = st.text_input("确认密码", type="password", placeholder="再次输入密码", key="reg_confirm")
            submitted2 = st.form_submit_button("注册", use_container_width=True)

            if submitted2:
                if not reg_user or not reg_pass:
                    st.error("用户名和密码不能为空")
                elif reg_pass != reg_confirm:
                    st.error("两次密码不一致")
                elif len(reg_user) < 2:
                    st.error("用户名至少 2 个字符")
                elif len(reg_pass) < 4:
                    st.error("密码至少 4 个字符")
                elif db is None:
                    st.error("数据库未连接，无法注册")
                else:
                    reg_result = db.register_user(reg_user, hash_password(reg_pass))
                    if reg_result["success"]:
                        st.success("注册成功！请切换到登录页面登录。")
                    else:
                        st.error(reg_result.get("message", "注册失败：用户名可能已存在"))

    # 页脚信息
    st.markdown("""
    <div style="text-align:center;padding:40px 20px 0;color:#999;font-size:12px;">
        <p>向量模型: BGE-M3 | 向量数据库: Milvus | 数据库: MariaDB</p>
    </div>
    """, unsafe_allow_html=True)
