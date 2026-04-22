"""
Simple app login using environment variables.

Set APP_AUTH_USER and APP_AUTH_PASSWORD (e.g. in a .env file). When both are
non-empty, the app shows a login form until credentials match. If either
variable is missing or empty, authentication is disabled.
"""
import hmac
import os

import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]


def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv()
        


def _auth_configured() -> bool:
    _load_env()
    user = (os.environ.get("APP_AUTH_USER") or "").strip()
    password = (os.environ.get("APP_AUTH_PASSWORD") or "").strip()
    return bool(user and password)


def _safe_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(
        a.encode("utf-8", errors="strict"),
        b.encode("utf-8", errors="strict"),
    )


def require_login() -> None:
    """If APP_AUTH_USER and APP_AUTH_PASSWORD are set, block the app until login succeeds."""
    if not _auth_configured():
        return
    if st.session_state.get("auth_ok"):
        return

    st.title("Sign in")
    st.caption("This application is protected. Enter the credentials from your environment configuration.")

    with st.form("app_login", clear_on_submit=False):
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)

    if submitted:
        _load_env()
        expected_user = (os.environ.get("APP_AUTH_USER") or "").strip()
        expected_password = (os.environ.get("APP_AUTH_PASSWORD") or "").strip()
        if _safe_equal(username.strip(), expected_user) and _safe_equal(password, expected_password):
            st.session_state.auth_ok = True
            st.rerun()
        st.error("Invalid username or password.")

    st.stop()


def render_logout_in_sidebar() -> None:
    """Show a log out control when auth is enabled and the user is signed in."""
    if not _auth_configured() or not st.session_state.get("auth_ok"):
        return
    with st.sidebar:
        st.markdown("**Session**")
        if st.button("Log out", use_container_width=True):
            st.session_state.auth_ok = False
            st.rerun()
