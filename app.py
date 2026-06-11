import json
import html
import os, sys
import re
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path
import uuid
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
from PIL import Image
import yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from wordcloud import WordCloud
from streamlit_option_menu import option_menu
from supabase import create_client, Client
from auth_supabase import (
    signup,
    login,
    logout,
    reset_password,
    get_profile,
    supabase
)
from datetime import timezone as _tz


# ── load .env for local development ───────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
    load_dotenv(override=False)
except ImportError:
    pass

try:
    from groq import Groq
except Exception:
    Groq = None

favicon = Image.open("assets/favicon2.png")
st.set_page_config(page_title="InstaScribe", page_icon=favicon,
                   layout="wide", initial_sidebar_state="expanded")

import base64, io
def _get_favicon_b64():
    try:
        img = Image.open("assets/favicon3.png")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""
FAVICON_B64 = _get_favicon_b64()
FAVICON_IMG = f'<img src="data:image/png;base64,{FAVICON_B64}" style="width:44px;height:44px;object-fit:cover;border-radius:12px;display:block;">'

AUTH_STORE_PATH = Path(__file__).with_name("auth_store.yaml")


# ══════════════════════════════════════════════════════════════════
# SUPABASE CONNECTION
# ══════════════════════════════════════════════════════════════════

@st.cache_resource
def _supabase_client() -> Client:
    """
    Reads SUPABASE_URL and SUPABASE_KEY from Streamlit secrets.
    Add these to your .streamlit/secrets.toml:

        [supabase]
        url = "https://xxxxxxxxxxxxxxxxxxxx.supabase.co"
        key = "your-anon-or-service-role-key"
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def _sb() -> Client:
    return _supabase_client()


# ── helpers ──────────────────────────────────────────────────────

def _details_from_text(details_text):
    if not details_text:
        return {}
    if isinstance(details_text, dict):
        return details_text
    try:
        loaded = json.loads(details_text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _details_to_text(details):
    return json.dumps(_details_from_text(details), ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════
# SUPABASE SCHEMA BOOTSTRAP
# Run this SQL once in the Supabase SQL editor to create tables:
#
#   CREATE TABLE IF NOT EXISTS users (
#       username TEXT PRIMARY KEY,
#       name TEXT NOT NULL,
#       email TEXT NOT NULL UNIQUE,
#       password TEXT NOT NULL,
#       role TEXT NOT NULL DEFAULT 'member',
#       created_at TIMESTAMPTZ,
#       details JSONB NOT NULL DEFAULT '{}'
#   );
#
#   CREATE TABLE IF NOT EXISTS active_sessions (
#       session_id TEXT PRIMARY KEY,
#       username TEXT NOT NULL,
#       name TEXT,
#       role TEXT,
#       started_at TIMESTAMPTZ NOT NULL,
#       last_seen TIMESTAMPTZ NOT NULL
#   );
# ══════════════════════════════════════════════════════════════════


def _default_auth_store():
    return {
        "credentials": {
            "usernames": {
                "admin": {
                    "name": "Admin User",
                    "email": "admin@example.com",
                    "password": stauth.Hasher.hash("admin123"),
                    "role": "admin",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "details": {
                        "company": "InstaScribe",
                        "notes": "Bootstrap admin account",
                    },
                }
            }
        },
        "cookie": {
            "name": "instascribe_auth",
            "key": "9f2c7a8d4b1e6c3f0a5d8e7b1c9f4a2d",
            "expiry_days": 1,
        },
        "preauthorized": {"emails": []},
    }


def _secret_value(*keys, default=None):
    current = st.secrets
    try:
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = current[key]
            if current is None:
                return default
        return current if current is not None else default
    except Exception:
        return default


def _cookie_config(defaults):
    cookie = dict(defaults)
    secret_cookie_key = _secret_value("auth", "cookie_key", default=None)
    if secret_cookie_key:
        cookie["key"] = secret_cookie_key
    return cookie


def _load_yaml_auth_store():
    if AUTH_STORE_PATH.exists():
        with AUTH_STORE_PATH.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


# ══════════════════════════════════════════════════════════════════
# USER CRUD — Supabase
# ══════════════════════════════════════════════════════════════════

def _read_users_from_db():
    res = _sb().table("users").select("*").order("username").execute()
    users = {}
    for row in (res.data or []):
        users[row["username"]] = {
            "name": row["name"],
            "email": row["email"],
            "password": row["password"],
            "role": row["role"],
            "created_at": row.get("created_at") or "",
            "details": _details_from_text(row.get("details") or {}),
        }
    return users


def _user_count_in_db():
    res = _sb().table("users").select("username", count="exact").execute()
    return res.count or 0


def _seed_auth_db(yaml_data):
    seed_users = (yaml_data.get("credentials") or {}).get("usernames") or {}
    defaults = _default_auth_store()["credentials"]["usernames"]
    if not seed_users:
        seed_users = defaults
    for username_value, record in seed_users.items():
        _sb().table("users").upsert({
            "username":   username_value,
            "name":       record.get("name", username_value),
            "email":      record.get("email", ""),
            "password":   record.get("password", ""),
            "role":       record.get("role", "member"),
            "created_at": record.get("created_at", ""),
            "details":    _details_from_text(record.get("details", {})),
        }, on_conflict="username", ignore_duplicates=True).execute()


def _save_auth_store(data):
    with AUTH_STORE_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def _load_auth_store():
    yaml_data = _load_yaml_auth_store()
    defaults = _default_auth_store()
    yaml_data.setdefault("cookie", _cookie_config(defaults["cookie"]))
    yaml_data.setdefault("preauthorized", defaults["preauthorized"])
    if _user_count_in_db() == 0:
        _seed_auth_db(yaml_data)
    data = {
        "credentials": {"usernames": _read_users_from_db()},
        "cookie": yaml_data["cookie"],
        "preauthorized": yaml_data["preauthorized"],
    }
    _save_auth_store(data)
    return data


def _refresh_auth_store_cache():
    global auth_store
    auth_store = _load_auth_store()
    return auth_store


def _save_user_record(username_value, record):
    _sb().table("users").upsert({
        "username":   username_value,
        "name":       record.get("name", username_value),
        "email":      record.get("email", ""),
        "password":   record.get("password", ""),
        "role":       record.get("role", "member"),
        "created_at": record.get("created_at", ""),
        "details":    _details_from_text(record.get("details", {})),
    }, on_conflict="username").execute()
    _refresh_auth_store_cache()


def _set_password(username_value, new_password):
    _sb().table("users").update(
        {"password": _hash_password(new_password)}
    ).eq("username", username_value).execute()
    _refresh_auth_store_cache()


def _delete_user_record(username_value):
    try:
        result = (
            _sb()
            .table("users")
            .delete()
            .eq("username", username_value)
            .execute()
        )

        print("DELETE USER:", username_value)
        print("DELETE RESULT:", result)

        _refresh_auth_store_cache()
        return True

    except Exception as e:
        print("DELETE ERROR:", e)
        st.error(f"Delete failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
# SUBSCRIPTION CONSTANTS
# ══════════════════════════════════════════════════════════════════

PLANS = {
    "free": {
        "label":       "Free",
        "monthly_inr": 0.00,
        "yearly_inr":  0.00,
        "color":       "#818cf8",
        "color2":      "#6366f1",
        "icon":        "🌱",
        "features": [
            "Core dashboards & lead scoring",
            "Community support via email"
        ],
        "limits": "50 records · no AI",
    },
    "pro": {
        "label":       "Pro",
        "monthly_inr": 200.00,
        "yearly_inr":  2000.00,
        "color":       "#a855f7",
        "color2":      "#ec4899",
        "icon":        "⚡",
        "features": [
            "All Free features",
            "Full AI Insights — all 6 tabs unlocked",
        ],
        "limits": "Unlimited · AI included",
    },
    "business": {
        "label":       "Business",
        "monthly_inr": 500.00,
        "yearly_inr":  5000.00,
        "color":       "#f59e0b",
        "color2":      "#ef4444",
        "icon":        "🏢",
        "features": [
            "Everything in Pro",
            "Dedicated priority support & SLA",
        ],
        "limits": "Team · SLA · Dedicated",
    },
}

# ══════════════════════════════════════════════════════════════════
# SUBSCRIPTION DB HELPERS
# ══════════════════════════════════════════════════════════════════

def _get_subscription(username_val):
    """Return the latest active subscription row for a user, or None."""
    try:
        res = (
            _sb().table("subscriptions")
            .select("*")
            .eq("username", username_val)
            .eq("status", "active")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return rows[0] if rows else None
    except Exception:
        return None


def _get_all_subscriptions():
    """Return all subscription rows (all users, all statuses)."""
    try:
        res = (
            _sb().table("subscriptions")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return res.data or []
    except Exception:
        return []


def _create_subscription(username_val, plan, billing_cycle):
    """Insert a new active subscription, cancel any existing one first."""
    # Cancel existing active subs
    try:
        _sb().table("subscriptions").update({
            "status": "cancelled",
            "cancelled_at": datetime.now(timezone.utc).isoformat(),
        }).eq("username", username_val).eq("status", "active").execute()
    except Exception:
        pass

    plan_info = PLANS.get(plan, PLANS["free"])
    amount    = plan_info["yearly_inr"] if billing_cycle == "yearly" else plan_info["monthly_inr"]
    now       = datetime.now(timezone.utc)
    renews_at = (now + timedelta(days=365) if billing_cycle == "yearly"
                 else now + timedelta(days=30))
    import uuid as _uuid
    payment_ref = "PAY-" + _uuid.uuid4().hex[:12].upper()

    try:
        _sb().table("subscriptions").insert({
            "username":      username_val,
            "plan":          plan,
            "status":        "active",
            "billing_cycle": billing_cycle,
            "amount_usd":    float(amount),
            "started_at":    now.isoformat(),
            "renews_at":     renews_at.isoformat(),
            "payment_ref":   payment_ref,
            "created_at":    now.isoformat(),
        }).execute()
        return True, payment_ref
    except Exception as e:
        return False, str(e)


def _cancel_subscription(username_val):
    """Cancel the active subscription for a user."""
    try:
        _sb().table("subscriptions").update({
            "status":       "cancelled",
            "cancelled_at": datetime.now(timezone.utc).isoformat(),
        }).eq("username", username_val).eq("status", "active").execute()
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
# SUBSCRIPTION PAGE (shown to non-admin users)
# ══════════════════════════════════════════════════════════════════

def render_subscription_page():
    current_sub = _get_subscription(username)
    current_plan = current_sub["plan"] if current_sub else "free"

    st.markdown(
        '<div class="ai-page-heading">Subscription Plans</div>'
        '<div class="ai-page-subheading">Choose the plan that fits your team. '
        'Upgrade or downgrade at any time.</div>',
        unsafe_allow_html=True,
    )

    # ── Flash messages ──────────────────────────────────────────
    sub_flash = st.session_state.pop("sub_flash", None)
    if sub_flash:
        if sub_flash["kind"] == "success":
            st.success(sub_flash["msg"])
        else:
            st.error(sub_flash["msg"])

    # ── Current plan banner ─────────────────────────────────────
    plan_info = PLANS.get(current_plan, PLANS["free"])
    renews_str = ""
    if current_sub and current_sub.get("renews_at"):
        try:
            rdt = datetime.fromisoformat(current_sub["renews_at"].replace("Z", "+00:00"))
            renews_str = f" · renews {rdt.strftime('%d %b %Y')}"
        except Exception:
            pass

    st.markdown(
        f'<div style="background:linear-gradient(135deg,{plan_info["color"]}18,{plan_info["color2"]}0f);'
        f'border:1px solid {plan_info["color"]}44;border-radius:16px;padding:16px 22px;'
        f'margin-bottom:24px;display:flex;align-items:center;gap:12px;">'
        f'<span style="font-size:24px">{plan_info["icon"]}</span>'
        f'<div>'
        f'<div style="font-size:14px;font-weight:700;color:#1e293b">Current Plan: '
        f'<span style="color:{plan_info["color"]}">{plan_info["label"]}</span></div>'
        f'<div style="font-size:11px;color:#64748b">{plan_info["limits"]}{renews_str}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── Billing toggle ──────────────────────────────────────────
    billing = st.radio(
        "Billing cycle",
        ["Monthly", "Yearly (save ~17%)"],
        horizontal=True,
        key="sub_billing_toggle",
    )
    billing_cycle = "yearly" if "Yearly" in billing else "monthly"

    # ── Plan cards ──────────────────────────────────────────────
    cols = st.columns(3, gap="large")
    for i, (plan_key, plan_data) in enumerate(PLANS.items()):
        with cols[i]:
            price_inr = plan_data["yearly_inr"] if billing_cycle == "yearly" else plan_data["monthly_inr"]
            price_str = "Free" if price_inr == 0 else f"₹ {price_inr:.0f}/{billing_cycle[:1]}"
            is_current = plan_key == current_plan
            border_style = f"3px solid {plan_data['color']}" if is_current else f"1px solid {plan_data['color']}44"

            features_html = "".join(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                f'<span style="color:{plan_data["color"]};font-size:13px">✓</span>'
                # f'<span style="font-size:12px;color:#475569">{feat}</span></div>'
                f'<span style="font-size:12px;color:#0f172a;font-weight:500">{feat}</span></div>'
                for feat in plan_data["features"]
            )
            badge = (
                f'<span style="background:{plan_data["color"]}22;color:{plan_data["color"]};'
                f'font-size:10px;font-weight:700;padding:2px 8px;border-radius:20px;'
                f'border:1px solid {plan_data["color"]}44;margin-left:8px;">CURRENT</span>'
                if is_current else ""
            )

            st.markdown(
                f'<div style="background:#ffffff;'
                f'border:{border_style};border-radius:18px;padding:24px 22px;'
                f'position:relative;overflow:hidden;min-height:360px;">'
                f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
                f'background:linear-gradient(90deg,{plan_data["color"]},{plan_data["color2"]})" ></div>'
                f'<div style="font-size:22px;margin-bottom:6px">{plan_data["icon"]}</div>'
                f'<div style="font-size:16px;font-weight:700;color:#1e293b">'
                f'{plan_data["label"]}{badge}</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:26px;font-weight:700;'
                f'background:linear-gradient(135deg,{plan_data["color"]},{plan_data["color2"]});'
                f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
                f'margin:10px 0 16px;">{price_str}</div>'
                f'{features_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

            if is_current:
                if plan_key != "free":
                    if st.button("Cancel Plan", key=f"cancel_{plan_key}", use_container_width=True):
                        if _cancel_subscription(username):
                            st.session_state["sub_flash"] = {
                                "kind": "success",
                                "msg": f"Your {plan_data['label']} plan has been cancelled.",
                            }
                        else:
                            st.session_state["sub_flash"] = {
                                "kind": "error",
                                "msg": "Cancellation failed. Please try again.",
                            }
                        st.rerun()
                else:
                    st.button("Current Plan", key=f"cur_{plan_key}", disabled=True, use_container_width=True)
            else:
                label = f"Upgrade to {plan_data['label']}" if plan_key != "free" else "Downgrade to Free"
                if st.button(label, key=f"buy_{plan_key}", use_container_width=True):
                    ok, ref = _create_subscription(username, plan_key, billing_cycle)
                    if ok:
                        st.session_state["sub_flash"] = {
                            "kind": "success",
                            "msg": (
                                f"🎉 Successfully subscribed to **{plan_data['label']}** "
                                f"({billing_cycle}) — Ref: `{ref}`"
                                if plan_key != "free"
                                else "Downgraded to Free plan."
                            ),
                        }
                    else:
                        st.session_state["sub_flash"] = {"kind": "error", "msg": f"Error: {ref}"}
                    st.rerun()

    # ── FAQ ─────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)
    sec("💬 Frequently Asked Questions")
    faq_cols = st.columns(2)
    faqs = [
        ("Can I switch plans?", "Yes — upgrade or downgrade any time. Changes take effect immediately."),
        ("Is there a free trial?", "The Free plan is permanently free with no credit card required."),
        ("How are yearly savings calculated?", "Yearly billing gives ~2 months free compared to paying monthly."),
        ("What payment methods are accepted?", "This is a demo — payments are simulated with a mock reference."),
    ]
    for i, (q, a) in enumerate(faqs):
        with faq_cols[i % 2]:
            st.markdown(
                f'<div class="insight" style="--ac:#818cf8;margin-bottom:12px;">'
                f'<b>{q}</b><br><span style="font-size:12px">{a}</span></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════
# ADMIN REVENUE DASHBOARD (tab inside admin view)
# ══════════════════════════════════════════════════════════════════

def render_admin_revenue_tab():
    all_subs = _get_all_subscriptions()

    if not all_subs:
        st.info("No subscription records yet. Users need to purchase a plan first.")
        return

    subs_df = pd.DataFrame(all_subs)
    subs_df["created_at"] = pd.to_datetime(subs_df["created_at"], utc=True, errors="coerce")
    subs_df["started_at"] = pd.to_datetime(subs_df["started_at"], utc=True, errors="coerce")
    subs_df["Month"]      = subs_df["created_at"].dt.to_period("M").dt.to_timestamp()
    # DB column is still amount_usd — rename it after loading
    if "amount_usd" in subs_df.columns and "amount_inr" not in subs_df.columns:
        subs_df = subs_df.rename(columns={"amount_usd": "amount_inr"})
    subs_df["amount_inr"] = pd.to_numeric(subs_df["amount_inr"], errors="coerce").fillna(0)

    active_df = subs_df[subs_df["status"] == "active"]

    # ── KPIs ────────────────────────────────────────────────────
    total_rev   = subs_df[subs_df["status"] != "free"]["amount_inr"].sum()
    mrr         = active_df[active_df["billing_cycle"] == "monthly"]["amount_inr"].sum()
    arr         = active_df[active_df["billing_cycle"] == "yearly"]["amount_inr"].sum() / 12
    mrr_total   = mrr + arr
    paying_subs = len(active_df[active_df["plan"] != "free"])
    churn_count = len(subs_df[subs_df["status"] == "cancelled"])

    k1, k2, k3, k4 = st.columns(4, gap="large")
    k1.markdown(kpi("Total Revenue",    f"₹{total_rev:,.0f}",   "all-time · all plans",      "#4ade80","#22c55e",min(int(total_rev/100),100)), unsafe_allow_html=True)
    k2.markdown(kpi("MRR",              f"₹{mrr_total:,.0f}",   "monthly recurring revenue", "#818cf8","#6366f1",70), unsafe_allow_html=True)
    k3.markdown(kpi("Paying Subs",      str(paying_subs),       "active paid plans",         "#a855f7","#ec4899",min(paying_subs*20,100)), unsafe_allow_html=True)
    k4.markdown(kpi("Churned",          str(churn_count),       "cancelled subscriptions",   "#f87171","#ef4444",min(churn_count*15,100)), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    # ── Chart row 1 ─────────────────────────────────────────────
    rc1, rc2 = st.columns([3, 2], gap="large")

    with rc1:
        sec("📈 Monthly Revenue (MRR trend)")
        monthly_rev = (
            subs_df[subs_df["amount_inr"] > 0]
            .groupby("Month")["amount_inr"]
            .sum()
            .reset_index()
            .sort_values("Month")
        )
        if len(monthly_rev) > 0:
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Bar(
                x=monthly_rev["Month"], y=monthly_rev["amount_inr"],
                marker_color="#a855f7", marker_line_width=0, opacity=0.85,
                name="Revenue",
                text=monthly_rev["amount_inr"].apply(lambda v: f"${v:,.0f}"),
                textposition="outside", textfont=dict(color="#d8b4fe", size=11),
            ))
            fig_rev.add_trace(go.Scatter(
                x=monthly_rev["Month"], y=monthly_rev["amount_inr"],
                mode="lines", line=dict(color="#ec4899", width=2), showlegend=False,
            ))
            dark(fig_rev, 280)
            fig_rev.update_layout(
                title=dict(text="Revenue by Month", font=dict(size=12, color="#6b4fa0")),
                bargap=0.30, yaxis_title="USD",
            )
            st.plotly_chart(fig_rev, use_container_width=True)
        else:
            st.info("No revenue data yet (only Free subscriptions recorded).")

    with rc2:
        sec("🥧 Plan Distribution")
        plan_counts = active_df["plan"].value_counts().reset_index()
        plan_counts.columns = ["Plan", "Count"]
        plan_colors_map = {"free": "#818cf8", "pro": "#a855f7", "business": "#f59e0b"}
        plan_counts["Color"] = plan_counts["Plan"].map(plan_colors_map)

        fig_pie = go.Figure(go.Pie(
            labels=plan_counts["Plan"].str.title(),
            values=plan_counts["Count"],
            hole=0.58,
            marker=dict(
                colors=plan_counts["Color"].tolist(),
                line=dict(color="#0e0814", width=2),
            ),
            textinfo="percent+label",
            textfont=dict(size=11, color="#f0eaf6"),
            hovertemplate="<b>%{label}</b>: %{value} users<extra></extra>",
        ))
        dark(fig_pie, 280)
        fig_pie.update_layout(
            title=dict(text="Active Plan Distribution", font=dict(size=12, color="#6b4fa0")),
            legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Chart row 2 ─────────────────────────────────────────────
    rc3, rc4 = st.columns(2, gap="large")

    with rc3:
        sec("💳 Billing Cycle Split")
        cycle_counts = active_df[active_df["plan"] != "free"]["billing_cycle"].value_counts().reset_index()
        cycle_counts.columns = ["Cycle", "Count"]
        if len(cycle_counts) > 0:
            fig_cycle = go.Figure(go.Bar(
                x=cycle_counts["Cycle"].str.title(),
                y=cycle_counts["Count"],
                marker_color=["#4ade80", "#fbbf24"],
                marker_line_width=0, opacity=0.88,
                text=cycle_counts["Count"], textposition="outside",
                textfont=dict(color="#ffffff", size=12),
            ))
            dark(fig_cycle, 240)
            fig_cycle.update_layout(
                title=dict(text="Monthly vs Yearly Subscriptions", font=dict(size=12, color="#6b4fa0")),
                bargap=0.4, showlegend=False,
            )
            st.plotly_chart(fig_cycle, use_container_width=True)
        else:
            st.info("No paid billing cycle data yet.")

    with rc4:
        sec("📊 Revenue by Plan")
        plan_rev = (
            subs_df[subs_df["amount_inr"] > 0]
            .groupby("plan")["amount_inr"]
            .sum()
            .reset_index()
            .sort_values("amount_inr", ascending=False)
        )
        if len(plan_rev) > 0:
            plan_rev["Color"] = plan_rev["plan"].map(plan_colors_map)
            fig_planrev = go.Figure()
            for _, row_pr in plan_rev.iterrows():
                fig_planrev.add_trace(go.Bar(
                    x=[row_pr["plan"].title()],
                    y=[row_pr["amount_inr"]],
                    marker_color=row_pr["Color"],
                    marker_line_width=0, opacity=0.88,
                    text=[f"${row_pr['amount_inr']:,.0f}"],
                    textposition="outside",
                    textfont=dict(color="#ffffff", size=11),
                    name=row_pr["plan"].title(),
                ))
            dark(fig_planrev, 240)
            fig_planrev.update_layout(
                title=dict(text="All-Time Revenue by Plan", font=dict(size=12, color="#6b4fa0")),
                showlegend=False, bargap=0.35,
            )
            st.plotly_chart(fig_planrev, use_container_width=True)
        else:
            st.info("No revenue data yet.")

    # ── Subscriber table ─────────────────────────────────────────
    sec("📋 All Subscriptions")
    display_subs = subs_df.copy()
    display_subs["created_at"] = display_subs["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    display_subs["started_at"] = display_subs["started_at"].dt.strftime("%Y-%m-%d")
    show_sub_cols = [c for c in ["username", "plan", "status", "billing_cycle",
                                  "amount_inr", "started_at", "created_at", "payment_ref"]
                     if c in display_subs.columns]
    display_subs = display_subs[show_sub_cols].rename(columns={
        "username": "User", "plan": "Plan", "status": "Status",
        "billing_cycle": "Cycle", "amount_inr": "Amount (INR)",
        "started_at": "Started", "created_at": "Created At",
        "payment_ref": "Payment Ref",
    })
    st.dataframe(display_subs.reset_index(drop=True), use_container_width=True, height=320)
    st.download_button(
        "⬇️ Export Revenue CSV",
        display_subs.to_csv(index=False).encode(),
        "instascribe_revenue.csv",
        "text/csv",
        key="rev_export_btn",
    )


def _find_user_by_identifier(identifier):
    lookup = (identifier or "").strip().lower()
    if not lookup:
        return None, None
    res = (
        _sb().table("users")
        .select("*")
        .or_(f"username.ilike.{lookup},email.ilike.{lookup}")
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if rows:
        row = rows[0]
        return row["username"], {
            "name":       row["name"],
            "email":      row["email"],
            "password":   row["password"],
            "role":       row["role"],
            "created_at": row.get("created_at") or "",
            "details":    _details_from_text(row.get("details") or {}),
        }
    return None, None


# ══════════════════════════════════════════════════════════════════
# SESSION TRACKING — Supabase
# ══════════════════════════════════════════════════════════════════

def _ensure_session_id():
    if "session_uuid" not in st.session_state:
        st.session_state["session_uuid"] = uuid.uuid4().hex
    return st.session_state["session_uuid"]


def _touch_active_session(username_value, name_value=None, role_value=None):
    if not username_value:
        return
    session_id = _ensure_session_id()
    now = datetime.now(timezone.utc).isoformat()
    # Check if session already exists to preserve started_at
    res = _sb().table("active_sessions").select("started_at").eq("session_id", session_id).execute()
    existing = res.data or []
    started_at = existing[0]["started_at"] if existing else now
    _sb().table("active_sessions").upsert({
        "session_id": session_id,
        "username":   username_value,
        "name":       name_value or "",
        "role":       role_value or "",
        "started_at": started_at,
        "last_seen":  now,
    }, on_conflict="session_id").execute()


def _clear_active_session(_=None):
    session_id = st.session_state.get("session_uuid")
    if not session_id:
        return
    _sb().table("active_sessions").delete().eq("session_id", session_id).execute()


def _prune_active_sessions(max_age_minutes=10):
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
    _sb().table("active_sessions").delete().lt("last_seen", cutoff).execute()


def _get_active_sessions(max_age_minutes=10):
    _prune_active_sessions(max_age_minutes=max_age_minutes)
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
    res = (
        _sb().table("active_sessions")
        .select("*")
        .gte("last_seen", cutoff)
        .order("last_seen", desc=True)
        .execute()
    )
    return res.data or []


# ══════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════════════════════════

def _refresh_admin_view(tab_name=None):
    if tab_name:
        st.session_state["admin_page"] = tab_name
    st.rerun()


def _hash_password(password):
    return stauth.Hasher.hash(password)


def _mask_hash(password_hash):
    if not password_hash:
        return "—"
    if len(password_hash) <= 12:
        return password_hash[:2] + "…" + password_hash[-2:]
    return password_hash[:6] + "…" + password_hash[-4:]


def _is_strong_password(password):
    return (
        len(password) >= 8
        and any(ch.islower() for ch in password)
        and any(ch.isupper() for ch in password)
        and any(ch.isdigit() for ch in password)
    )


def _set_page_mode(page_mode):
    st.session_state["page_mode"] = page_mode
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()


def _get_page_mode_from_query():
    try:
        value = st.query_params.get("page_mode")
        if isinstance(value, list):
            value = value[0] if value else None
    except Exception:
        params = st.experimental_get_query_params()
        value = params.get("page_mode", [None])[0]
    if value in {"Login", "Sign Up", "Forgot Password"}:
        return value
    return None


def _set_admin_flash(message, kind="success"):
    st.session_state["admin_flash"] = {"kind": kind, "message": message}


# ── Bootstrap ─────────────────────────────────────────────────────
try:
    auth_store = _load_auth_store()
except Exception as auth_init_error:
    st.error(f"Authentication storage could not be initialized: {auth_init_error}")
    st.stop()

authenticator = stauth.Authenticate(
    auth_store['credentials'],
    auth_store['cookie']['name'],
    auth_store['cookie']['key'],
    auth_store['cookie']['expiry_days'],
    auto_hash=False,
)

# Restore the session from the auth cookie before the login gate runs.
authenticator.login(location="unrendered")

name = st.session_state.get('name')
auth_status = st.session_state.get('authentication_status')
username = st.session_state.get('username')

# ══════════════════════════════════════════════════════════════════
# GLOBAL THEME — applied to login page and dashboard
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
# import url('https://fonts.googleapis.com/css2?family=Urbanist:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Dancing+Script:wght@400;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Urbanist', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top left, rgba(244, 63, 94, 0.12), transparent 28%),
                radial-gradient(circle at bottom right, rgba(249, 115, 22, 0.08), transparent 24%),
                linear-gradient(180deg, #0f051d 0%, #18072b 100%);
    min-height: 100vh;
}
.stApp::before {
    content: '';
    position: fixed; top: -120px; right: -120px;
    width: 500px; height: 500px; border-radius: 50%; pointer-events: none;
    background: radial-gradient(circle, rgba(244, 63, 94, 0.12) 0%, transparent 65%);
    z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed; bottom: -100px; left: -100px;
    width: 420px; height: 420px; border-radius: 50%; pointer-events: none;
    background: radial-gradient(circle, rgba(249, 115, 22, 0.08) 0%, transparent 65%);
    z-index: 0;
}

.block-container { padding: 3.1rem 2rem 3rem; max-width: 1320px; position: relative; z-index: 1; }

/* ── LOGIN PAGE ──────────────────────────────────────────────── */
.login-outer {
    min-height: 48vh;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 0.6rem 0 1.5rem;
    margin-top: -1.0rem;
}
.login-left {
    padding: 52px 44px 52px 36px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 100%;
    position: relative;
}
.login-left-border {
    border-right: 1px solid #1a2740;
}
.login-logo {
    width: 52px; height: 52px; border-radius: 15px;
    background: linear-gradient(135deg,#4f7cff,#7c3aed);
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; margin-bottom: 24px;
}
.login-brand {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(135deg, #f4f7ff, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin-bottom: 10px;
}
.login-brand { font-family: 'Dancing Script', cursive; }
.login-tagline {
    font-size: 1.2rem; font-weight: 500;
    color: #9fb0d2; line-height: 1.5; margin-bottom: 32px;
}
.login-pill {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(79,124,255,.1); color: #9fb0d2;
    font-size: 11px; font-weight: 500;
    padding: 5px 13px; border-radius: 20px;
    border: 1px solid rgba(79,124,255,.2);
    margin-bottom: 8px; width: fit-content;
}
.login-right {
    padding: 52px 36px 52px 44px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.login-form-title {
    font-size: 1.3rem; font-weight: 700; color: #f4f7ff;
    margin-bottom: 4px;
}
.login-form-title { font-family: 'Dancing Script', cursive; }
.login-form-sub {
    font-size: 12px; color: #6b4fa0; margin-bottom: 24px;
}
.login-card-wrapper {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    overflow: visible !important;
    position: relative;
    max-width: 820px;
    margin: 0 auto;
    box-shadow: none !important;
}
.login-card-wrapper::before {
    display: none !important;
}

[data-testid="stTextInput"] input {
    background: rgba(15,22,40,0.95) !important;
    border: 1px solid #1a2740 !important;
    border-radius: 10px !important;
    color: #f4f7ff !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #4f7cff !important;
    box-shadow: 0 0 0 2px rgba(79,124,255,0.15) !important;
}
[data-testid="stTextInput"] label {
    color: #8ea0c7 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: .07em;
}

.stButton > button, .btn-secondary > button {
    height: 50px !important;
}

/* Force full width on the button container */
[data-testid="stButton"] {
    width: 100% !important;
    max-width: 480px !important;
}
[data-testid="stButton"] > button {
    width: 100% !important;
    min-width: 0 !important;
    display: block !important;
}
[data-testid="stFormSubmitButton"] {
    width: 100% !important;
}
[data-testid="stFormSubmitButton"] > button {
    width: 100% !important;
    min-width: 0 !important;
    display: block !important;
}
div[data-testid="stForm"] [data-testid="stButton"],
div[data-testid="stForm"] [data-testid="stButton"] > button,
div[data-testid="stForm"] [data-testid="stFormSubmitButton"],
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] > button {
    width: 100% !important;
    max-width: 100% !important;
    display: block !important;
}
.stButton > button:hover { opacity: .88 !important; }

[data-testid="stCheckbox"] label { color: #8ea0c7 !important; font-size: 13px !important; }
[data-testid="stRadio"] label { color: #9fb0d2 !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050814 0%, #090d1a 100%);
    border-right: 1px solid #18233a;
}
[data-testid="stSidebar"] * { color: #9fb0d2 !important; }
section[data-testid="stSidebar"] label {
  font-size: 0.7rem !important;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: #111111 !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div { background: #18233a !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div { background: linear-gradient(90deg,#4f7cff,#7c3aed) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #eef3ff !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] * { color: #eef3ff !important; }
[role="listbox"] [role="option"] * { color: #eef3ff !important; }

.app-header { margin-top: 0.95rem; margin-bottom: 1rem; }
.app-header-card {
    background: linear-gradient(180deg, rgba(10,16,30,0.98), rgba(6,10,19,0.98));
    backdrop-filter: blur(12px);
    border: 1px solid #1a2740;
    border-radius: 18px;
    padding: 14px 20px 13px;
    position: relative;
    overflow: hidden;
}
.app-header-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(79,124,255,0.7), rgba(124,58,237,0.7), transparent);
}
# AFTER — line 1111
.app-logo-icon {
  width: 44px; height: 44px; border-radius: 12px;
  background: transparent;                               ← no gradient, favicon has its own
  display: inline-flex; align-items: center; justify-content: center;
  overflow: hidden;                                      ← clips favicon to rounded corners
  margin-right: 10px; vertical-align: middle;
}
.app-title { font-size: 1.2rem; font-weight: 700; color: #f4f7ff; vertical-align: middle; }
.app-subtitle { font-size: 10px; color: #8ea0c7; text-transform: uppercase; letter-spacing: .12em; margin-top: 2px; }

.kpi {
    background: linear-gradient(180deg, rgba(10,16,30,0.98), rgba(6,10,19,0.98));
    border: 1px solid #1a2740;
    border-radius: 18px;
    padding: 18px 20px;
    position: relative; overflow: hidden;
}
.kpi::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--ac,#4f7cff), var(--ac2,#7c3aed));
}
.kpi::after {
  content: ''; position: absolute; top: -20px; right: -20px;
  width: 70px; height: 70px; border-radius: 50%;
  background: radial-gradient(circle, var(--ac,#4f7cff) 0%, transparent 70%);
  opacity: 0.18;
}
.kpi-label { font-size: 10px; color: #8ea0c7; font-weight: 700; letter-spacing: .7px; text-transform: uppercase; margin-bottom: 10px; }
.kpi-value {
    font-family: 'DM Mono', monospace; font-size: 26px; font-weight: 700; line-height: 1;
    background: linear-gradient(135deg, var(--ac,#9fb8ff), var(--ac2,#7c3aed));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.kpi-sub { font-size: 11px; color: #8ea0c7; margin-top: 7px; }
.kpi-bar { height: 3px; border-radius: 2px; margin-top: 12px; background: linear-gradient(90deg, var(--ac,#4f7cff), var(--ac2,#7c3aed)); width: var(--bar, 70%); }

.sec {
    font-size: 10px; font-weight: 700; color: #8ea0c7;
    text-transform: uppercase; letter-spacing: .8px;
    padding-bottom: 8px; margin-bottom: 16px; margin-top: 28px;
    border-bottom: 1px solid #1a2740;
    position: relative;
}
.sec::after {
  content: '';
  position: absolute; bottom: -1px; left: 0;
  width: 48px; height: 1px;
  background: linear-gradient(90deg,#4f7cff,#7c3aed);
}

.insight {
    background: linear-gradient(180deg, rgba(10,16,30,0.94), rgba(6,10,19,0.98));
    backdrop-filter: blur(8px);
    border: 1px solid #1a2740;
    border-left: 3px solid var(--ac,#4f7cff);
    border-radius: 14px;
    padding: 14px 16px; font-size: 13px; color: #9fb0d2; line-height: 1.6;
}
.insight b { color: #f4f7ff; }

.chip { display: inline-block; background: rgba(79,124,255,.14); color: #dbe7ff; font-size: 10px; font-weight: 500; padding: 2px 9px; border-radius: 20px; border: 1px solid rgba(79,124,255,.24); margin: 2px 3px; }

.pb-wrap { margin-bottom: 12px; }
.pb-row { display: flex; justify-content: space-between; font-size: 12px; color: #8ea0c7; margin-bottom: 5px; }
.pb-val { font-family: 'DM Mono', monospace; font-size: 12px; }
.pb-track { height: 4px; background: #14203a; border-radius: 3px; overflow: hidden; }
.pb-fill  { height: 100%; border-radius: 3px; }

.post-banner {
    background: linear-gradient(180deg, rgba(10,16,30,0.98), rgba(6,10,19,0.98));
    border: 1px solid #1a2740; border-radius: 18px;
    padding: 22px 26px; margin-bottom: 18px;
    position: relative; overflow: hidden;
}
.post-banner::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg,#4f7cff,#7c3aed); }
.post-banner-id { font-family: 'DM Mono', monospace; font-size: 11px; color: #9fb0d2; letter-spacing: .05em; margin-bottom: 4px; }
.post-banner-handle { font-size: 1.1rem; font-weight: 700; color: #f4f7ff; margin-bottom: 2px; }
.post-banner-meta { font-size: 12px; color: #8ea0c7; }

.stat-pill { display: inline-flex; align-items: center; gap: 6px; background: rgba(10,16,30,0.96); border: 1px solid #1a2740; border-radius: 10px; padding: 6px 12px; font-size: 12px; color: #a9b8d7; margin: 3px 4px; }
.stat-pill b { color: var(--pc,#dbe7ff); font-family: 'DM Mono', monospace; }

.desc-box { background: linear-gradient(180deg, rgba(10,16,30,0.94), rgba(6,10,19,0.98)); backdrop-filter: blur(6px); border: 1px solid #1a2740; border-left: 3px solid #4f7cff; border-radius: 12px; padding: 12px 16px; font-size: 12px; color: #9fb0d2; line-height: 1.65; margin-bottom: 14px; }
.desc-box b { color: #f4f7ff; }

.about-hero { background: linear-gradient(180deg, rgba(10,16,30,0.98), rgba(6,10,19,0.98)); border: 1px solid #1a2740; border-radius: 18px; padding: 32px 36px; margin-bottom: 20px; position: relative; overflow: hidden; }
.about-hero::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg,transparent,#4f7cff,#7c3aed,transparent); }
.about-card { background: linear-gradient(180deg, rgba(10,16,30,0.98), rgba(6,10,19,0.98)); border: 1px solid #1a2740; border-radius: 16px; padding: 22px 24px; height: 100%; }
.about-card-green  { border-left: 4px solid #4ade80; }
.about-card-blue   { border-left: 4px solid #818cf8; }
.about-card-amber  { border-left: 4px solid #fbbf24; }
.about-card-pink   { border-left: 4px solid #ec4899; }
.about-card-purple { border-left: 4px solid #a855f7; }
.about-title { font-size: 13px; font-weight: 600; color: #f4f7ff; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
.about-tag { display: inline-block; font-family: 'DM Mono',monospace; font-size: 11px; background: rgba(79,124,255,.12); color: #dbe7ff; padding: 1px 7px; border-radius: 4px; border: 1px solid #1a2740; margin: 1px 2px; }
.about-li { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; font-size: 13px; color: #9fb0d2; line-height: 1.6; }
.about-li-dot { width: 6px; height: 6px; border-radius: 50%; margin-top: 6px; flex-shrink: 0; background: var(--dot,#a855f7); }
.about-divider { border: none; border-top: 1px solid #1a2740; margin: 8px 0 16px; }

.grad-divider { height: 1px; margin: 28px 0 8px; background: linear-gradient(90deg, #4f7cff, #7c3aed, transparent); border-radius: 2px; }

[data-testid="stDataFrame"] * { font-family: 'DM Mono', monospace !important; font-size: 12px !important; }

.stDownloadButton > button { background: linear-gradient(135deg,#2563eb,#7c3aed) !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
<style>
.stApp {
    background:
        /* Soft ellipses for texture (mimicking the SVG shapes) */
        radial-gradient(ellipse at 15% 20%, rgba(165, 180, 252, 0.13) 0%, transparent 40%),
        radial-gradient(ellipse at 85% 80%, rgba(249, 168, 212, 0.15) 0%, transparent 40%),
        /* Primary Aurora radial glows */
        radial-gradient(circle at 20% 30%, rgba(199, 210, 254, 0.7) 0%, transparent 55%),
        radial-gradient(circle at 80% 70%, rgba(251, 207, 232, 0.75) 0%, transparent 50%),
        radial-gradient(circle at 60% 20%, rgba(191, 219, 254, 0.55) 0%, transparent 40%),
        /* Base linear gradient (indigo to blush) */
        linear-gradient(135deg, #f8f4ff 0%, #dbeafe 40%, #fce7f3 100%) !important;
    background-attachment: fixed !important;
}

.block-container { color: #162945 !important; }

[data-testid="stSidebar"] {
    /* Soft, semi-transparent gradient blending indigo to blush */
    background: linear-gradient(180deg, rgba(248, 244, 255, 0.75) 0%, rgba(252, 231, 243, 0.75) 100%) !important;
    
    /* Frosted glass blur effect */
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    
    /* Subtle white border to separate it cleanly from the main app */
    border-right: 1px solid rgba(255, 255, 255, 0.6) !important;
}
[data-testid="stSidebar"] * { color: #000000 !important; }

.app-header-card,
.kpi,
.insight,
.post-banner,
.desc-box,
.about-hero,
.about-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.99), rgba(244,248,255,0.99)) !important;
    border-color: #dbe6f5 !important;
    box-shadow: 0 18px 40px rgba(20, 42, 84, 0.08) !important;
}

.app-header-card::before,
.kpi::before,
.sec::after,
.post-banner::before,
.about-hero::before {
    background: linear-gradient(90deg, transparent, #4f7cff, #ff4dad, #ffce5a, transparent) !important;
}

.app-title,
.about-title,
.post-banner-handle,
.login-form-title,
.card-title,
.sec,
.hero-brand {
    color: #14213d !important;
}

    .app-subtitle,
    .kpi-label,
    .kpi-sub,
    .post-banner-id,
    .post-banner-meta,
    .about-li,
    .desc-box,
    .insight,
    .pb-row,
    .auth-note,
    .login-tagline,
    .card-subtitle,
    .hero-copy,
    .login-form-sub {
    color: #231a3b !important;
    font-weight: 700 !important;
}

.kpi {
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(244,248,255,0.98)) !important;
    border-color: #dbe6f5 !important;
    color: #231a3b !important;
    font-weight: 700 !important;
}

.app-header-card, .insight, .insight *, .post-banner, .post-banner *, .chip, .about-li, .about-li * {
    color: #231a3b !important;
    font-weight: 700 !important;
}
.insight b, .about-title, .kpi-label { color: #231a3b !important; }

.insight .category { color: #4f7cff !important; font-weight: 800 !important; }

.post-banner-meta, .post-banner-id, .pb-row, .card-subtitle, .login-tagline, .kpi-sub {
    color: #6b4fa0 !important;
    font-weight: 600 !important;
}
.desc-box, .about-li, .insight { color: #4f7cff !important; font-weight: 600 !important; }
.insight .category, .chip { color: #ec4899 !important; }
.kpi-sub, .post-banner-meta small, .hero-copy { color: #fbbf24 !important; font-weight: 600 !important; }

*[style*="color:#9fb0d2"] { color: #4f7cff !important; font-weight: 700 !important; }
*[style*="color:#c4d0e8"] { color: #6b4fa0 !important; font-weight: 700 !important; }
*[style*="color:#f4f7ff"] { color: #231a3b !important; font-weight: 700 !important; }
*[style*="color:#e8d5ff"] { color: #231a3b !important; font-weight: 700 !important; }
*[style*="color:#9b7ec8"] { color: #4f7cff !important; font-weight: 700 !important; }
*[style*="color:#c4d0e8"] { color: #6b4fa0 !important; font-weight: 700 !important; }

.block-container .insight, .block-container .insight *,
.block-container .post-banner, .block-container .post-banner *,
.block-container .about-hero, .block-container .about-hero *,
.block-container .about-card, .block-container .about-card *,
.block-container .desc-box, .block-container .desc-box *,
.block-container .kpi, .block-container .kpi * {
    color: #231a3b !important;
    font-weight: 600 !important;
}

.insight .category { color: #4f7cff !important; font-weight: 800 !important; }
.chip { color: #ec4899 !important; }
.kpi-sub { color: #6b4fa0 !important; }

.ai-generated-panel {
    background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(255,245,252,0.98)) !important;
    border: 1px solid #d9e4f7 !important;
    border-left: 4px solid #ec4899 !important;
    border-radius: 16px !important;
    box-shadow: 0 16px 38px rgba(20, 42, 84, 0.08) !important;
    padding: 20px 22px !important;
}
.ai-generated-panel .ai-generated-title {
    font-size: 12px !important;
    font-weight: 800 !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    color: #14213d !important;
    margin-bottom: 12px !important;
}
.ai-generated-panel .ai-generated-body {
    font-size: 13px !important;
    line-height: 1.8 !important;
    color: #41506d !important;
    font-weight: 400 !important;
}
.app-title, .ai-page-heading, .ai-generated-title, .login-brand, .post-banner-handle {
    background: linear-gradient(135deg,#000000 0%, #14213d 50%, #ec4899 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}
[data-testid="stSidebar"] .stButton > button { color: #000000 !important; }
.block-container .stButton > button { color: #000000 !important; }
.ai-page-heading {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #000000 !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 6px !important;
}
.ai-page-subheading {
    font-size: 0.98rem !important;
    font-weight: 400 !important;
    color: #5f6f8d !important;
    line-height: 1.6 !important;
    margin-bottom: 16px !important;
}
.ai-page-subheading b {
    font-weight: 800 !important;
    color: #14213d !important;
}
.ai-status-badge {
    display: inline-block !important;
    padding: 2px 8px !important;
    border-radius: 999px !important;
    background: rgba(79,124,255,0.08) !important;
    border: 1px solid rgba(79,124,255,0.16) !important;
    color: #4f7cff !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    margin-left: 6px !important;
}
.ai-response-heading {
    font-size: 1rem !important;
    font-weight: 800 !important;
    color: #14213d !important;
    margin: 14px 0 6px !important;
}
.ai-response-text {
    margin: 0 0 8px !important;
    color: #41506d !important;
    font-size: 13px !important;
    line-height: 1.8 !important;
    font-weight: 400 !important;
}
.ai-response-list {
    margin: 0 0 8px 1.2rem !important;
    padding: 0 !important;
    color: #41506d !important;
    font-size: 13px !important;
    line-height: 1.8 !important;
    font-weight: 400 !important;
}
.ai-response-list li { margin-bottom: 4px !important; }
.ai-response-step {
    margin: 0 0 8px !important;
    color: #41506d !important;
    font-size: 13px !important;
    line-height: 1.8 !important;
    font-weight: 400 !important;
}
.ai-response-step-no { font-weight: 800 !important; color: #14213d !important; }
.ai-response-step-body { font-weight: 400 !important; color: #41506d !important; }

.kpi-value {
    background: linear-gradient(135deg, var(--ac, #4f7cff), var(--ac2, #ff4dad)) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.kpi-bar,
.login-logo,
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button,
.st-key-FormSubmitter-login_form-LOGIN > div > button,
.btn-secondary > button,
[data-testid="stRadio"] input:checked + label {
    background: linear-gradient(135deg, #4f7cff, #ff4dad) !important;
    color: #ffffff !important;
}

.chip,
.stat-pill,
.about-tag,
.login-pill,
.hero-badges span {
    background: rgba(79,124,255,0.08) !important;
    border-color: rgba(79,124,255,0.18) !important;
    color: #3f5478 !important;
}

.btn-secondary > button {
    background: rgba(255,255,255,0.96) !important;
    color: #4b5e80 !important;
}

[data-testid="stTextInput"] input,
div[data-testid="stForm"] input,
.stTextInput input {
    background: rgba(255,255,255,0.98) !important;
    border-color: #d7e0ee !important;
    color: #15223b !important;
}

[data-testid="stTextInput"] label,
[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label {
    color: #6a7f9f !important;
}

.stCheckbox svg { fill: #ff4dad !important; }

[data-testid="stSidebar"] .stSlider > div > div > div { background: #dbe6f5 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div { background: linear-gradient(90deg, #4f7cff, #ff4dad) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="tag"] *,
[role="listbox"] [role="option"] * { color: #183153 !important; }

[data-testid="stDataFrame"] * { color: #162945 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── COLOR PALETTES ─────────────────────────────────────────────
CAT_CLR = {"tech":"#818cf8","fashion":"#ec4899","fitness":"#4ade80",
           "travel":"#c084fc","food":"#fbbf24"}
Q_CLR   = {"high":"#4ade80","medium":"#fbbf24","low":"#f87171"}

# DL = dict(
#     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#     font=dict(family="DM Sans", color="#5f6f8d", size=12),
#     xaxis=dict(gridcolor="#dde6f4", linecolor="#d559f1", zeroline=False),
#     yaxis=dict(gridcolor="#dde6f4", linecolor="#d559f1", zeroline=False),
#     legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#0a0c0f", font=dict(size=11)),
#     margin=dict(l=24, r=24, t=40, b=24))
DL = dict(
    paper_bgcolor="rgba(0,0,0,0)", 
    plot_bgcolor="rgba(0,0,0,0)",
    # 1. Change the main global font color to solid black
    font=dict(family="Cantarell", color="#000000", size=12),
    
    # 2. Force the X and Y axis numbers to inherit that black color explicitly
    xaxis=dict(gridcolor="#dde6f4", linecolor="#1e1c1e", zeroline=False, tickfont=dict(color="#000000")),
    yaxis=dict(gridcolor="#dde6f4", linecolor="#1e1e1f", zeroline=False, tickfont=dict(color="#000000")),
    
    # 3. Ensure the legend text is black
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#0a0c0f", font=dict(size=11, color="#000000")),
    
    margin=dict(l=24, r=24, t=40, b=24)
)


def dark(fig, h=300):
    fig.update_layout(**DL, height=h)
    return fig

# ── DATA LOADING ───────────────────────────────────────────────
def _data_dir():
    try:    sd = os.path.dirname(os.path.abspath(__file__))
    except: sd = os.path.dirname(os.path.abspath(sys.argv[0]))
    for p in [os.path.join(sd,"data"), os.path.join(os.getcwd(),"data"), sd, os.getcwd()]:
        if os.path.isdir(p) and any(os.path.exists(os.path.join(p,f))
           for f in ["influencer_master.csv","post_metrics.csv"]):
            return p
    return sd

def _csv(name):
    p = os.path.join(_data_dir(), name)
    if not os.path.exists(p):
        st.error(f"❌ `{name}` not found in `{_data_dir()}`."); st.stop()
    d = pd.read_csv(p)
    d.columns = d.columns.str.strip().str.replace(" ","_")
    return d

@st.cache_data(show_spinner="Loading data…")
def load_data():
    inf  = _csv("influencer_master.csv")
    post = _csv("post_metrics.csv")
    cat  = _csv("category_dim.csv")

    inf  = inf.rename(columns={"Followers":"Follower_Count","Following":"Following_Count",
                                "Total_Engagement":"Engagement","engagement":"Engagement",
                                "engagement_rate":"Engagement_Rate","handle":"Handle",
                                "category_id":"Category_ID"})
    post = post.rename(columns={"Total_Engagement":"Engagement","engagement":"Engagement",
                                 "post_date":"Post_Date","likes":"Likes","comments":"Comments",
                                 "handle":"Handle","sentiment_score":"Sentiment_Score",
                                 "hashtags":"Hashtags","post_id":"Post_ID"})
    if "Post_ID" not in post.columns:
        post["Post_ID"] = ["POST_" + str(i+1).zfill(6) for i in range(len(post))]
    cat  = cat.rename(columns={"saas_relevance":"SaaS_Relevance_Weight",
                                "SaaS_Relevance":"SaaS_Relevance_Weight",
                                "category_id":"Category_ID","category_name":"Category_Name"})

    if "Category_ID" in inf.columns and "Category_ID" in cat.columns:
        inf = inf.merge(cat, on="Category_ID", how="left")
    inf["SaaS_Relevance"] = inf["SaaS_Relevance_Weight"].fillna(0.5) \
                            if "SaaS_Relevance_Weight" in inf.columns else 0.5

    for c in ["Handle","Follower_Count","Engagement_Rate","Engagement"]:
        if c not in inf.columns:
            st.error(f"Missing column `{c}`"); st.stop()

    post["Post_Date"] = pd.to_datetime(post["Post_Date"], dayfirst=True, errors="coerce")
    post["Month"]     = post["Post_Date"].dt.to_period("M").dt.to_timestamp()
    post["MonthName"] = post["Post_Date"].dt.strftime("%b %Y")
    post["Year"]      = post["Post_Date"].dt.year
    if "Sentiment_Score" in post.columns:
        post["Sentiment_Bin"] = pd.cut(post["Sentiment_Score"],
            bins=[-1.1,-0.1,0.1,1.1], labels=["Negative","Neutral","Positive"])

    if "Sentiment_Score" in post.columns:
        avg_s = post.groupby("Handle")["Sentiment_Score"].mean().rename("Avg_Sentiment")
        inf   = inf.merge(avg_s, on="Handle", how="left")
    inf["Avg_Sentiment"] = inf.get("Avg_Sentiment", pd.Series(0,index=inf.index)).fillna(0)

    er_score   = inf["Engagement_Rate"] * 0.4
    fol_score  = (inf["Follower_Count"].clip(upper=2_000_000) / 2_000_000) * 30
    sent_score = ((inf["Avg_Sentiment"] + 1) / 2) * 20
    saas_score = inf["SaaS_Relevance"] * 10
    raw = (er_score + fol_score + sent_score + saas_score).round(4)
    r_min, r_max = raw.min(), raw.max()
    inf["Lead_Score"] = ((raw - r_min) / (r_max - r_min) * 100).round(2)

    inf["Lead_Quality"] = pd.cut(
        inf["Lead_Score"], bins=[-np.inf, 30, 60, np.inf],
        labels=["low","medium","high"]).astype(str)

    def _category_lead_quality(group):
        if len(group) < 3:
            return pd.Series(["medium"] * len(group), index=group.index)
        ranked = group["Lead_Score"].rank(method="first")
        return pd.qcut(ranked, q=3, labels=["low", "medium", "high"]).astype(str)

    if "Category_Name" in inf.columns:
        inf["Category_Lead_Quality"] = (
            inf.groupby("Category_Name", group_keys=False).apply(_category_lead_quality)
        )
    else:
        inf["Category_Lead_Quality"] = inf["Lead_Quality"]

    def _tier(f):
        if f < 10_000:    return "Nano (<10K)"
        if f < 100_000:   return "Micro (10K-100K)"
        if f < 500_000:   return "Mid (100K-500K)"
        if f < 1_000_000: return "Macro (500K-1M)"
        return "Mega (1M+)"

    inf["Follower_Tier"] = inf["Follower_Count"].apply(_tier)
    fc = inf.get("Following_Count", pd.Series(0,index=inf.index))
    inf["FF_Ratio"] = (fc / inf["Follower_Count"].replace(0,np.nan)).round(4).fillna(0)
    inf["Action"]   = inf["Lead_Quality"].map(
        {"high":"🔥 Contact Now","medium":"⏳ Nurture","low":"🗑 Ignore"})

    mc   = [c for c in ["Handle","Category_Name","Lead_Quality","Lead_Score"] if c in inf.columns]
    post = post.merge(inf[mc], on="Handle", how="left")
    return inf, post


def _groq_api_key():
    api_key = _secret_value("groq", "api_key", default=None) or os.getenv("GROQ_API_KEY")
    return api_key.strip() if isinstance(api_key, str) else api_key


def _groq_model_name():
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _groq_client():
    api_key = _groq_api_key()
    if not api_key or Groq is None:
        return None
    return Groq(api_key=api_key)


def _call_groq(system_prompt, user_prompt):
    """Single Groq call. Returns (answer_text, None) or (None, error_string)."""
    client = _groq_client()
    if client is None:
        return None, ("⚠️ Groq API not configured. "
                      "Add GROQ_API_KEY to your .env file or set groq.api_key in secrets.toml.")
    try:
        completion = client.chat.completions.create(
            model=_groq_model_name(),
            temperature=0.35,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content, None
    except Exception as exc:
        return None, f"Groq error: {exc}"


def _clean_token(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _find_entity_mentions(question, posts_df, leads_df):
    question_text = str(question or "").lower()
    compact_question = _clean_token(question_text)

    post_matches = []
    if "Post_ID" in posts_df.columns:
        for pid in posts_df["Post_ID"].dropna().astype(str).unique().tolist():
            pid_low = pid.lower()
            if pid_low in question_text or _clean_token(pid_low) in compact_question:
                post_matches.append(pid)

    handle_matches = []
    if "Handle" in posts_df.columns:
        for handle in posts_df["Handle"].dropna().astype(str).unique().tolist():
            handle_low = handle.lower().strip()
            if handle_low in question_text or f"@{handle_low}" in question_text or _clean_token(handle_low) in compact_question:
                handle_matches.append(handle)

    return list(dict.fromkeys(post_matches))[:5], list(dict.fromkeys(handle_matches))[:5]


def _safe_metric_text(value, digits=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def _profile_context_for_handle(handle, leads_df, posts_df):
    handle_key = str(handle).strip().lower()
    lead_row = None
    if "Handle" in leads_df.columns:
        matches = leads_df[leads_df["Handle"].astype(str).str.strip().str.lower() == handle_key]
        if len(matches) > 0:
            lead_row = matches.iloc[0]

    handle_posts = pd.DataFrame()
    if "Handle" in posts_df.columns:
        handle_posts = posts_df[posts_df["Handle"].astype(str).str.strip().str.lower() == handle_key].copy()
        if len(handle_posts) > 0 and "Post_Date" in handle_posts.columns:
            handle_posts = handle_posts.sort_values("Post_Date", ascending=False)

    lines = [f"Handle: {handle}"]
    if lead_row is not None:
        lines.extend([
            f"Name: {lead_row.get('Name', '—')}",
            f"Category: {lead_row.get('Category_Name', '—')}",
            f"Followers: {_safe_metric_text(lead_row.get('Follower_Count'), 0)}",
            f"Following: {_safe_metric_text(lead_row.get('Following_Count'), 0)}",
            f"Engagement Rate: {_safe_metric_text(lead_row.get('Engagement_Rate'))}%",
            f"Lead Score: {_safe_metric_text(lead_row.get('Lead_Score'))}/100",
            f"Lead Quality: {lead_row.get('Lead_Quality', '—')}",
            f"Avg Sentiment: {_safe_metric_text(lead_row.get('Avg_Sentiment'), 3)}",
        ])
    else:
        lines.append("No matching influencer master row was found.")

    if len(handle_posts) > 0:
        lines.append(f"Total posts: {len(handle_posts)}")
        preview_cols = [c for c in ["Post_ID", "Post_Date", "Likes", "Comments", "Engagement", "Hashtags"] if c in handle_posts.columns]
        preview = handle_posts.head(3)[preview_cols].copy()
        if "Post_Date" in preview.columns:
            preview["Post_Date"] = preview["Post_Date"].dt.strftime("%Y-%m-%d")
        lines.append("Recent posts:\n" + preview.to_string(index=False))
    else:
        lines.append("No post history found for this handle.")

    return "\n".join(lines)


def _profile_context_for_post(post_id, posts_df):
    post_key = str(post_id).strip().lower()
    if "Post_ID" not in posts_df.columns:
        return f"Post ID: {post_id}\nNo post_id column is available in the current dataset."

    matches = posts_df[posts_df["Post_ID"].astype(str).str.strip().str.lower() == post_key]
    if len(matches) == 0:
        return f"Post ID: {post_id}\nNo matching post record was found."

    row = matches.iloc[0]
    lines = [f"Post ID: {row.get('Post_ID', '—')}"]
    for label, key, digits in [
        ("Handle", "Handle", None),
        ("Date", "Post_Date", None),
        ("Likes", "Likes", 0),
        ("Comments", "Comments", 0),
        ("Engagement", "Engagement", 0),
        ("Sentiment Score", "Sentiment_Score", 3),
        ("Category", "Category_Name", None),
        ("Lead Score", "Lead_Score", 1),
        ("Lead Quality", "Lead_Quality", None),
    ]:
        value = row.get(key, "—")
        if key == "Post_Date" and pd.notna(value):
            value = pd.to_datetime(value).strftime("%Y-%m-%d")
        elif digits is not None:
            value = _safe_metric_text(value, digits)
        lines.append(f"{label}: {value}")

    if "Hashtags" in row.index and pd.notna(row.get("Hashtags")):
        lines.append(f"Hashtags: {row.get('Hashtags')}")
    return "\n".join(lines)


def _system_summary_context(leads_df, posts_df):
    lines = [
        f"Influencers in scope: {len(leads_df):,}",
        f"Posts in scope: {len(posts_df):,}",
    ]
    if "Lead_Quality" in leads_df.columns and len(leads_df) > 0:
        quality_counts = leads_df["Lead_Quality"].value_counts(dropna=False).to_dict()
        lines.append("Lead quality split: " + ", ".join(f"{k}: {v}" for k, v in quality_counts.items()))
    if "Category_Name" in leads_df.columns and len(leads_df) > 0:
        category_counts = leads_df["Category_Name"].value_counts().head(5)
        lines.append("Top categories: " + ", ".join(f"{idx} ({val})" for idx, val in category_counts.items()))
    if "Lead_Score" in leads_df.columns and len(leads_df) > 0:
        lines.append(f"Average lead score: {leads_df['Lead_Score'].mean():.1f}/100")
    if "Engagement_Rate" in leads_df.columns and len(leads_df) > 0:
        lines.append(f"Average engagement rate: {leads_df['Engagement_Rate'].mean():.2f}%")
    if "Handle" in posts_df.columns and len(posts_df) > 0 and "Engagement" in posts_df.columns:
        top_handle = posts_df.groupby("Handle")["Engagement"].sum().idxmax()
        lines.append(f"Top handle by total engagement: {top_handle}")
    return "\n".join(lines)


def _build_ai_context(question, leads_df, posts_df):
    post_matches, handle_matches = _find_entity_mentions(question, posts_df, leads_df)
    blocks = [_system_summary_context(leads_df, posts_df)]
    if post_matches:
        blocks.append("Matched post details:\n" + "\n\n".join(_profile_context_for_post(pid, posts_df) for pid in post_matches))
    if handle_matches:
        blocks.append("Matched handle details:\n" + "\n\n".join(_profile_context_for_handle(handle, leads_df, posts_df) for handle in handle_matches))
    if not post_matches and not handle_matches:
        blocks.append("No direct Post ID or Handle match was detected in the question.")
    return "\n\n---\n\n".join(blocks), post_matches, handle_matches


def _generate_ai_insight(question, leads_df, posts_df):
    context_text, post_matches, handle_matches = _build_ai_context(question, leads_df, posts_df)
    client = _groq_client()
    if client is None:
        summary = [
            "Groq API is not configured. Set `groq.api_key` in Streamlit secrets or `GROQ_API_KEY` in the environment.",
            "",
            "Local dataset summary:",
            context_text,
        ]
        return "\n".join(summary), post_matches, handle_matches

    system_prompt = (
        "You are an AI insights assistant for InstaScribe. "
        "Use only the provided dataset context. "
        "Return concise, useful answers in plain English with short bullet points when helpful. "
        "If the user asks about a Post ID or Handle, include the matching details from the context. "
        "If the answer is not available in the dataset context, say so clearly."
    )
    user_prompt = f"User question:\n{question}\n\nDataset context:\n{context_text}"

    completion = client.chat.completions.create(
        model=_groq_model_name(),
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content if completion and completion.choices else "No response returned by Groq."
    return content, post_matches, handle_matches


def _ai_section_prompt(section, leads_df, posts_df, post_id="", handle="", custom_question=""):
    context_text, post_matches, handle_matches = _build_ai_context(
        f"{section} {custom_question} {post_id} {handle}", leads_df, posts_df
    )
    section_prompts = {
        "Executive Summary": (
            "Give a concise executive summary of the overall system. "
            "Cover volume, lead quality, engagement, strongest categories/handles, and the key story from the data."
        ),
        "Carrier Performance": (
            "Summarize creator performance and content efficiency. "
            "Highlight who is performing best, average engagement, and any standout handles in the context."
        ),
        "Category Analysis": (
            "Analyze category-level performance. "
            "Compare categories by lead quality, engagement rate, lead score, and which categories deserve priority."
        ),
        "Risk Analysis": (
            "Identify risk signals in the dataset. "
            "Mention weak engagement, low lead quality, suspicious follower behavior if visible, or handles/posts needing caution."
        ),
        "Recommendations": (
            "Provide action-oriented recommendations for outreach and prioritization. "
            "Suggest who to contact first, who to nurture, and what operational focus would improve results."
        ),
        "Custom Question": (
            "Answer the user's custom question using the dataset context. "
            "If the user asks about a specific Post ID or Handle, include the matching details clearly."
        ),
    }
    section_instruction = section_prompts.get(section, section_prompts["Executive Summary"])
    if custom_question.strip():
        user_question = custom_question.strip()
    elif section == "Custom Question":
        user_question = "No custom question was entered."
    elif post_id.strip() or handle.strip():
        user_question = f"Please analyze {post_id or handle}"
    else:
        user_question = f"Provide {section.lower()} for the current data."

    return section_instruction, user_question, context_text, post_matches, handle_matches

# ══════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════
if not auth_status:
    query_page_mode = _get_page_mode_from_query()
    if query_page_mode:
        st.session_state["page_mode"] = query_page_mode
    if st.session_state.get("page_mode") not in {"Login", "Sign Up", "Forgot Password"}:
        st.session_state["page_mode"] = "Login"

    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:radial-gradient(circle at 15% 18%, rgba(255,59,212,0.26) 0%, rgba(255,59,212,0.08) 18%, rgba(255,59,212,0) 34%),
           radial-gradient(circle at 82% 24%, rgba(24,200,255,0.24) 0%, rgba(24,200,255,0.08) 18%, rgba(24,200,255,0) 34%),
           radial-gradient(circle at 48% 78%, rgba(93,86,255,0.18) 0%, rgba(93,86,255,0.05) 18%, rgba(93,86,255,0) 30%),
           linear-gradient(180deg, #050816 0%, #090014 100%) !important;min-height:100vh;}
header, footer {visibility:hidden;}
[data-testid="stSidebar"]{display:none !important;}
[data-testid="stAppViewContainer"] > section.main{padding:0 !important;}
[data-testid="stAppViewContainer"] > section.main > div.block-container{
    max-width:none !important;
    width:calc(100vw - 40px) !important;
    margin:0px 20px !important;
    padding:18px !important;
    border-radius:56px !important;
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
    min-height:calc(100vh - 40px) !important;
    overflow:hidden !important;
}
#login-architecture{display:none;}
#login-architecture + div[data-testid="stHorizontalBlock"]{
    gap:28px !important;
    align-items:center !important;
    min-height:calc(100vh - 76px) !important;
    display:flex !important;
}
#login-architecture + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child{
    position:relative !important;
    overflow:hidden !important;
    border-radius:40px !important;
    padding:16px 42px !important;
    background:
        radial-gradient(circle at 18% 20%, rgba(255,59,212,0.30) 0%, rgba(255,59,212,0.10) 16%, rgba(255,59,212,0) 34%),
        radial-gradient(circle at 78% 28%, rgba(24,200,255,0.22) 0%, rgba(24,200,255,0.08) 18%, rgba(24,200,255,0) 34%),
        radial-gradient(circle at 50% 82%, rgba(93,86,255,0.16) 0%, rgba(93,86,255,0.04) 16%, rgba(93,86,255,0) 28%),
        linear-gradient(135deg, rgba(7,10,18,0.98) 0%, rgba(16,7,24,0.95) 44%, rgba(7,12,20,0.99) 100%) !important;
    border:none !important;
    box-shadow:0 24px 70px rgba(0,0,0,0.42) !important;
}
#login-architecture + div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child{
    position:relative !important;
    overflow:hidden !important;
    border-radius:40px !important;
    padding:30px 34px 28px !important;
    background:
        radial-gradient(circle at 15% 12%, rgba(255,59,212,0.08) 0%, rgba(255,59,212,0) 24%),
        radial-gradient(circle at 84% 20%, rgba(24,200,255,0.08) 0%, rgba(24,200,255,0) 22%),
        linear-gradient(180deg, rgba(9,13,24,0.98) 0%, rgba(6,9,18,0.98) 100%) !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    box-shadow:0 24px 70px rgba(0,0,0,0.34) !important;
}
div[data-testid="stForm"]{
    width:min(100%, 480px) !important;
    max-width:480px !important;
    margin-left:auto !important;
    margin-right:0 !important;
    margin-top:-380px !important;
    padding-top:0 !important;
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
}
div[data-testid="stForm"],
div[data-testid="stForm"] * {
    box-sizing: border-box !important;
}
div[data-testid="stForm"] .stFormSubmitButton,
div[data-testid="stForm"] .stFormSubmitButton > button,
div[data-testid="stForm"] .stButton > button,
div[data-testid="stButton"] > button {
    width:100% !important;
    max-width:100% !important;
    margin-left:0 !important;
}
div[data-testid="stButton"]{
    width:min(100%, 480px) !important;
    max-width:480px !important;
    margin-left:auto !important;
    margin-right:0 !important;
}
.login-hero-orb{position:absolute;border-radius:50%;pointer-events:none;filter:blur(3px);}
.login-hero-orb.one{width:330px;height:330px;left:-120px;bottom:-120px;background:radial-gradient(circle, rgba(255,59,212,0.24) 0%, rgba(255,59,212,0) 70%);}
.login-hero-orb.two{width:360px;height:360px;right:-140px;top:-140px;background:radial-gradient(circle, rgba(24,200,255,0.18) 0%, rgba(24,200,255,0) 70%);}
.card-head{position:relative;z-index:1;text-align:center;margin-bottom:6px !important;}
.card-title{font-size:1.08rem;font-weight:600;color:#f4f7ff;letter-spacing:.02em;}
.card-subtitle{margin-top:5px;color:#8ea0c7;font-size:0.88rem;}
.hero-brand{color:#ffffff;font-size:clamp(3.3rem,5vw,5.8rem);line-height:1.01;font-weight:600;letter-spacing:-0.06em;}
[data-testid="stRadio"]{margin-bottom:14px;}
[data-testid="stRadio"] > div{justify-content:center !important;gap:8px !important;}
[data-testid="stRadio"] label{
    background:rgba(11,18,34,0.42) !important;
    border:1px solid rgba(255,255,255,0.18) !important;
    border-radius:999px !important;
    padding:8px 20px !important;
    color:#ffffff !important;
    font-size:12px !important;
    font-weight:600 !important;
    cursor:pointer !important;
    box-shadow:0 10px 20px rgba(0,0,0,0.20) !important;
}
[data-testid="stRadio"] input:checked + label,
[data-testid="stRadio"] [aria-checked="true"] ~ label {
    background:linear-gradient(135deg,#ff4ad7 0%,#6adfff 100%) !important;
    border-color:rgba(255,255,255,0.45) !important;
    color:#000000 !important;
}
[data-testid="stTextInput"]{margin-bottom:10px;width:100% !important;}
[data-testid="stTextInput"] label{color:#8ea0c7 !important;font-size:11px !important;letter-spacing:.09em !important;font-weight:700 !important;padding-left:4px !important;margin-bottom:4px !important;text-transform:uppercase !important;}
[data-testid="stTextInput"] input{
    background:rgba(15,22,40,0.96) !important;
    border:1px solid #22304a !important;
    border-radius:11px !important;
    color:#f4f7ff !important;
    padding:0.58rem 0.85rem 0.56rem !important;
    box-shadow:none !important;
}
.stTextInput input::placeholder{color:#64748b !important;}
[data-testid="stTextInput"] input:focus{box-shadow:0 0 0 2px rgba(79,124,255,0.14) !important;border-color:#4f7cff !important;}
[data-testid="stCheckbox"] label{color:#dbe7ff !important;font-size:11px !important;}
.stCheckbox svg{fill:#ff4ad7 !important;}
.auth-link-row{display:flex;justify-content:space-between;align-items:center;gap:14px;margin:-2px 0 10px;}
.auth-link-row a{color:#a78bfa !important;font-size:11px;text-decoration:none;font-weight:500;}
.auth-link-row a:hover{text-decoration:underline;}
/* ── SAMPLE 1 HERO (login left column) ───────────────────────── */
.login-hero-content {
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: none;
    border-radius: 28px;
    padding: 32px 36px;
    max-width: 760px;
    box-shadow: 0 20px 50px rgba(15,23,42,0.05);
}
.login-hero-content .s1-badge{display:inline-flex;align-items:center;gap:10px;background:rgba(219,39,119,0.08);border:1px solid rgba(219,39,119,0.18);color:#db2777;padding:8px 16px;border-radius:30px;font-weight:700;margin-bottom:18px;}
.login-hero-content .s1-title{font-family:'Space Grotesk',sans-serif;font-size:clamp(2.6rem,6.5vw,4.6rem);font-weight:800;line-height:1.02;margin-bottom:12px;color:#0f172a;letter-spacing:-1px}
.login-hero-content .s1-title .grad-1{background:linear-gradient(90deg,#0f172a 0%, #2563eb 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.login-hero-content .s1-title .grad-2{background:linear-gradient(90deg,#db2777 0%, #eab308 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.login-hero-content .s1-copy{color:#475569;font-size:1rem;line-height:1.6;margin-bottom:20px}
.login-hero-content .s1-action-row{display:flex;gap:14px}
.login-hero-content .btn-premium{background:linear-gradient(135deg,#2563eb 0%, #db2777 100%);color:#fff;border:none;padding:12px 20px;border-radius:14px;font-weight:800;cursor:default}
.login-hero-content .btn-secondary{background:rgba(15,23,42,0.04);border:1px solid rgba(15,23,42,0.08);color:#0f172a;padding:12px 20px;border-radius:14px;font-weight:700;cursor:default}
[data-testid="stHorizontalBlock"]{align-items:flex-start !important;}
[data-testid="stRadio"]{margin-bottom:14px;}
[data-testid="stRadio"] > div{justify-content:center !important;gap:8px !important;}
[data-testid="stRadio"] label{
    background:rgba(11,18,34,0.42) !important;
    border:1px solid rgba(255,255,255,0.18) !important;
    border-radius:999px !important;
    padding:8px 20px !important;
    color:#ffffff !important;
    font-size:12px !important;
    font-weight:600 !important;
    cursor:pointer !important;
    box-shadow:0 10px 20px rgba(0,0,0,0.20) !important;
}
[data-testid="stRadio"] [aria-checked="true"] ~ label,
[data-testid="stRadio"] input:checked + label{
    background:linear-gradient(135deg,#ff4ad7 0%,#6adfff 100%) !important;
    border-color:rgba(255,255,255,0.45) !important;
    color:#000 !important;
}
[data-testid="stTextInput"]{margin-bottom:10px;width:100% !important;}
[data-testid="stTextInput"]{max-width:none !important;margin-left:0 !important;margin-right:0 !important;}
[data-testid="stTextInput"] label{color:#8ea0c7 !important;font-size:11px !important;letter-spacing:.09em !important;font-weight:700 !important;padding-left:4px !important;margin-bottom:4px !important;text-transform:uppercase !important;}
[data-testid="stTextInput"] input{
    background:rgba(15,22,40,0.96) !important;
    border:1px solid #22304a !important;
    border-radius:11px !important;
    color:#f4f7ff !important;
    padding:0.58rem 0.85rem 0.56rem !important;
    box-shadow:none !important;
}
.stTextInput input::placeholder{color:#64748b !important;}
[data-testid="stTextInput"] input:focus{box-shadow:0 0 0 2px rgba(79,124,255,0.14) !important;border-color:#4f7cff !important;}
[data-testid="stCheckbox"] label{color:#dbe7ff !important;font-size:11px !important;}
.stCheckbox svg{fill:#ff4ad7 !important;}
.auth-link-row{display:flex;justify-content:space-between;align-items:center;gap:14px;margin:-2px 0 10px;}
.auth-link-row a{color:#a78bfa !important;font-size:11px;text-decoration:none;font-weight:500;}
.auth-link-row a:hover{text-decoration:underline;}
.st-key-FormSubmitter-login_form-LOGIN{display:block !important;width:100% !important;max-width:none !important;flex:1 1 100% !important;align-self:stretch !important;min-width:0 !important;}
.st-key-FormSubmitter-login_form-LOGIN > div{display:block !important;width:100% !important;max-width:none !important;}
.stFormSubmitButton{display:block !important;width:100% !important;max-width:none !important;flex:1 1 100% !important;align-self:stretch !important;margin-top:2px !important;}
div[data-testid="stFormSubmitButton"],
div[data-testid="stButton"] {
    width: 100% !important;
    max-width: 480px !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    display: block !important;
}
div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stButton"] > button,
.stFormSubmitButton > button,
.stButton > button,
.btn-secondary > button {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 50px !important;
    border: none !important;
    border-radius: 14px !important;
    background: linear-gradient(90deg, #4f7cff 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    box-shadow: 0 8px 24px rgba(79,124,255,0.28) !important;
    cursor: pointer !important;
    box-sizing: border-box !important;
}
div[data-testid="stFormSubmitButton"] > button:hover,
div[data-testid="stButton"] > button:hover,
.stFormSubmitButton > button:hover,
.stButton > button:hover,
.btn-secondary > button:hover {
    filter: brightness(0.95) !important;
    transform: none !important;
}
.auth-note{font-size:11px;color:#8ea0c7;text-align:center;margin-top:8px;}
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<style>
.stApp{background:
    radial-gradient(circle at 15% 18%, rgba(255,77,171,0.16) 0%, rgba(255,77,171,0.06) 18%, rgba(255,77,171,0) 34%),
    radial-gradient(circle at 82% 24%, rgba(79,124,255,0.16) 0%, rgba(79,124,255,0.06) 18%, rgba(79,124,255,0) 34%),
    radial-gradient(circle at 48% 78%, rgba(255,206,90,0.14) 0%, rgba(255,206,90,0.05) 18%, rgba(255,206,90,0) 30%),
    linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%) !important;}
.card-title, .login-form-title{color:#14213d !important;}
.card-subtitle, .login-form-sub{color:#6a7f9f !important;}
.auth-link-row a{color:#4f7cff !important;}
.auth-note{color:#6a7f9f !important;}
.stFormSubmitButton > button,
.stButton > button,
.btn-secondary > button {
    background: linear-gradient(90deg, #4f7cff 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 8px 24px rgba(79,124,255,0.28) !important;
    width: 100% !important;
    height: 50px !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stRadio"] label {
    background: rgba(255,255,255,0.86) !important;
    border: 1px solid #dbe6f5 !important;
    color: #3f5478 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    border-radius: 999px !important;
}
[data-testid="stRadio"] input:checked + label,
[data-testid="stRadio"] [aria-checked="true"] ~ label {
    background: linear-gradient(135deg,#4f7cff,#ff4dad) !important;
    color: #ffffff !important;
    border-color: transparent !important;
}

/* Dark theme override — forces white text on unselected pills */
@media (prefers-color-scheme: dark) {
    [data-testid="stRadio"] label {
        background: rgba(11,18,34,0.42) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        color: #ffffff !important;
    }
    [data-testid="stRadio"] input:checked + label,
    [data-testid="stRadio"] [aria-checked="true"] ~ label {
        background: linear-gradient(135deg,#ff4ad7 0%,#6adfff 100%) !important;
        color: #000000 !important;
        border-color: rgba(255,255,255,0.45) !important;
    }
}
.stCheckbox svg{fill:#ff4dad !important;}
</style>
""",
        unsafe_allow_html=True,
    )

    login_mode = st.radio(" ", options=["User", "Admin"], horizontal=True, key="login_mode", label_visibility="collapsed")
    page_mode = st.session_state.get("page_mode", "Login")
    st.markdown('<div id="login-architecture"></div>', unsafe_allow_html=True)
    hero_col, form_col = st.columns([1.28, 0.72], gap="large")

    with hero_col:
        st.markdown(
        """
<div class="login-hero-orb one"></div>
<div class="login-hero-orb two"></div>
<div style="padding: 10px 8px;">
    <div style="font-size: 12px; font-weight: 800; letter-spacing: .34em;
                text-transform: uppercase; margin-bottom: 16px;
                background: linear-gradient(90deg, #f09433, #e6683c, #dc2743, #cc2366, #bc1888);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text; width: fit-content;">
        WELCOME TO
    </div>
    <h1 style="font-size: clamp(2.5rem, 5.5vw, 4.5rem); font-weight: 600;
               line-height: 1.08; letter-spacing: -0.03em; margin: 0;
               font-family: Optima, sans-serif;
               background: linear-gradient(135deg,
                   #f09433 0%,
                   #e6683c 15%,
                   #dc2743 30%,
                   #7C48F0 80%,
                   #bc1888 70%,
                   #833ab4 85%,
                   #5851db 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               background-clip: text;
               display: block;">
        InstaScribe<br/>
        AI Creator<br/>
        Intelligence<br/>
        Platform
    </h1>
</div>
""",
        unsafe_allow_html=True,
    )

    with form_col:
        subtitle_text = "Use your username or email to sign in" if login_mode == "User" else "Admin credentials required"

    if page_mode == "Login":
        with st.form("login_form", clear_on_submit=False):
            st.markdown(
                f"""
<div class="card-head">
    <div class="card-title">Login</div>
    <div class="card-subtitle">{subtitle_text}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            login_id = st.text_input("Email ID / Username", placeholder="Email ID / Username")
            password = st.text_input("Password", type="password", placeholder="Password")
            st.markdown(
                '''
<div class="auth-link-row">
  <a href="?page_mode=Forgot%20Password" target="_self">Forgot Password?</a>
  <a href="?page_mode=Sign%20Up" target="_self">Create new account</a>
</div>
''',
                unsafe_allow_html=True,
            )
            submitted = st.form_submit_button("LOGIN")

        if submitted:
            username_found, user_record = _find_user_by_identifier(login_id)
            if not username_found or not user_record:
                st.error("Username or email not found.")
            elif login_mode == "Admin" and user_record.get("role", "member") != "admin":
                st.error("Use the admin button for admin login.")
            elif login_mode == "User" and user_record.get("role", "member") == "admin":
                st.error("Please use admin login for this account.")
            elif not stauth.Hasher.check_pw(password, user_record.get("password", "")):
                st.error("Password is incorrect.")
            else:
                st.session_state["authentication_status"] = True
                st.session_state["name"] = user_record.get("name", username_found)
                st.session_state["username"] = username_found
                st.session_state["role"] = user_record.get("role", "member")
                st.session_state["email"] = user_record.get("email", "")
                st.rerun()

    elif page_mode == "Sign Up":
        st.markdown('<div class="auth-note"></div>', unsafe_allow_html=True)
        with st.form("signup_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            full_name = c1.text_input("Full Name")
            company = c2.text_input("Company / Organization", value="")
            email = st.text_input("Email")
            username_input = st.text_input("Username")
            title = st.text_input("Title / Role", value="")
            d1, d2 = st.columns(2)
            password = d1.text_input("Password", type="password", help="Min 8 chars · upper · lower · digit")
            confirm = d2.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("CREATE ACCOUNT")

        if submitted:
            username_value = username_input.strip().lower()
            email_value = email.strip().lower()
            if not all([full_name.strip(), email_value, username_value, password, confirm]):
                st.error("Name, email, username, and password are required.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif not _is_strong_password(password):
                st.error("Password needs 8+ chars, upper-case, lower-case, and a digit.")
            elif username_value in auth_store["credentials"]["usernames"]:
                st.error("Username already taken.")
            elif any(u.get("email", "").strip().lower() == email_value for u in auth_store["credentials"]["usernames"].values()):
                st.error("Email already registered.")
            else:
                _save_user_record(username_value, {
                    "name": full_name.strip(),
                    "email": email_value,
                    "password": _hash_password(password),
                    "role": "member",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "details": {"company": company.strip(), "title": title.strip()},
                })
                st.success("Account created. Switch to Login to sign in.")

        back_left, back_right = st.columns([1, 1])
        if back_left.button("Back to Login", use_container_width=True):
            _set_page_mode("Login")
            st.rerun()

    else:
        st.markdown('<div class="auth-note"></div>', unsafe_allow_html=True)
        with st.form("reset_form", clear_on_submit=True):
            reset_id = st.text_input("Username or Email")
            new_password = st.text_input("New Password", type="password")
            confirm_reset = st.text_input("Confirm New Password", type="password")
            submitted = st.form_submit_button("RESET PASSWORD")

        if submitted:
            username_value, user_record = _find_user_by_identifier(reset_id)
            if not username_value or not user_record:
                st.error("Username or email not found.")
            elif new_password != confirm_reset:
                st.error("Passwords do not match.")
            elif not _is_strong_password(new_password):
                st.error("Password needs 8+ chars, upper-case, lower-case, and a digit.")
            else:
                _set_password(username_value, new_password)
                st.success("Password updated. Go back to Login and sign in.")

        back_left, back_right = st.columns([1, 1])
        if back_left.button("Back to Login", use_container_width=True):
            _set_page_mode("Login")
            st.rerun()

    st.stop()

if auth_status is False:
    st.error("Username/password is incorrect")
    st.stop()
elif auth_status is None:
    st.warning("Please enter your username and password")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# AUTHENTICATED — check role
# ══════════════════════════════════════════════════════════════════
current_user = auth_store["credentials"]["usernames"].get((username or "").lower(), {})
is_admin = current_user.get("role") == "admin"

_touch_active_session(username, name, current_user.get("role", "member"))

# ── Shared helper functions ────────────────────────────────────
def fmt(n):
    try: n = float(n)
    except: return str(n)
    if np.isnan(n): return "—"
    if n>=1e9: return f"{n/1e9:.1f}B"
    if n>=1e6: return f"{n/1e6:.1f}M"
    if n>=1e3: return f"{n/1e3:.1f}K"
    return f"{n:.1f}"

def safe_int(v):
    try:
        f = float(v)
        return 0 if np.isnan(f) else int(f)
    except (TypeError, ValueError):
        return 0

def safe_float(v, default=0.0):
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default

def sentiment_code(score):
    score = safe_float(score, 0.0)
    if score > 0.1:  return "Positive", "#4ade80"
    if score < -0.1: return "Negative", "#f87171"
    return "Neutral", "#fbbf24"

def kpi(label, value, sub="", ac="#a855f7", ac2="#ec4899", bar=70):
    return (f'<div class="kpi" style="--ac:{ac};--ac2:{ac2};--bar:{bar}%">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'<div class="kpi-bar"></div>'
            f'</div>')

def sec(t):
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)

def pb(label, val_s, pct, color):
    return (f'<div class="pb-wrap"><div class="pb-row">'
            f'<span>{label}</span>'
            f'<span class="pb-val" style="color:{color}">{val_s}</span></div>'
            f'<div class="pb-track">'
            f'<div class="pb-fill" style="width:{min(pct,100):.1f}%;background:{color}"></div>'
            f'</div></div>')

def desc(text):
    st.markdown(f'<div class="desc-box">{text}</div>', unsafe_allow_html=True)

def grad_divider():
    st.markdown('<div class="grad-divider"></div>', unsafe_allow_html=True)

def _format_ai_response(text):
    raw_lines = str(text or "").splitlines()
    parts = []
    in_list = False
 
    FS_HEADING  = "16px"
    FS_BODY     = "15px"
    FS_BULLET   = "15px"
    FS_INLINE_H = "16px"
 
    def escape_and_inline_bold(raw_text):
        converted = re.sub(
            r'\*\*(.+?)\*\*',
            lambda m: f'\x00STRONG\x00{m.group(1)}\x00/STRONG\x00',
            raw_text
        )
        escaped = html.escape(converted)
        escaped = escaped.replace('\x00STRONG\x00', '<strong>').replace('\x00/STRONG\x00', '</strong>')
        return escaped
 
    def close_list():
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False
 
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            close_list()
            continue
 
        # Pattern 1: "1. LABEL: body" or "1) LABEL — body" or "1. LABEL - body"
        inline_numbered = re.match(
    r'^(\d+)[.)]\s+(.+?)(?::|\ —|\ -)\s+(.+)$', line
)
        # Pattern 2: "1. LABEL:" with no body — standalone heading
        heading_only_numbered = re.match(
    r'^(\d+)[.)]\s+(.+?):?\s*$', line
)
        # Pattern 3: ## Markdown heading
        md_heading = re.match(r'^#{1,6}\s+(.+)$', line)
        # Pattern 4: **Bold only line**
        bold_only = re.match(r'^\*\*([^*]+)\*\*:?\s*$', line)
        # Pattern 5: bullet
        bullet = re.match(r'^[-•*]\s+(.+)$', line)
 
        if inline_numbered:
            close_list()
            num   = inline_numbered.group(1)
            label = html.escape(inline_numbered.group(2))
            body  = escape_and_inline_bold(inline_numbered.group(3))
            parts.append(
                f'<p style="margin:18px 0 8px;font-size:{FS_BODY};line-height:1.8;">'
                f'<strong style="color:#000000;font-size:{FS_INLINE_H};">{num}. {label}:</strong>'
                f'<span style="color:#000000;font-weight:400 !important;"> {body}</span>'
                f'</p>'
            )
 
        elif heading_only_numbered:
            close_list()
            num   = heading_only_numbered.group(1)
            label = html.escape(heading_only_numbered.group(2))
            parts.append(
                f'<h4 style="font-size:{FS_HEADING};font-weight:700;color:#000000;'
                f'margin:22px 0 6px 0;padding:0;">{num}. {label}</h4>'
            )
 
        elif md_heading:
            close_list()
            parts.append(
                f'<h4 style="font-size:{FS_HEADING};font-weight:700;color:#000000;'
                f'margin:22px 0 6px 0;padding:0;">'
                f'{escape_and_inline_bold(md_heading.group(1))}</h4>'
            )
 
        elif bold_only:
            close_list()
            parts.append(
                f'<h4 style="font-size:{FS_HEADING};font-weight:700;color:#000000;'
                f'margin:22px 0 6px 0;padding:0;">'
                f'{html.escape(bold_only.group(1))}</h4>'
            )
 
        elif bullet:
            if not in_list:
                parts.append(
                    f'<ul style="margin:6px 0 12px 1.3rem;padding:0;list-style-type:disc;">'
                )
                in_list = True
            parts.append(
                f'<li style="color:#000000;font-size:{FS_BULLET};line-height:1.8;margin-bottom:5px;">'
                f'{escape_and_inline_bold(bullet.group(1))}</li>'
            )
 
        else:
            fallback_num = re.match(r'^(\d+)[.)]\s+(.+)$', line)
            if fallback_num:
                close_list()
                parts.append(
                    f'<h4 style="font-size:{FS_HEADING};font-weight:700;color:#000000;'
                    f'margin:22px 0 6px 0;padding:0;">'
                    f'{fallback_num.group(1)}. {escape_and_inline_bold(fallback_num.group(2))}</h4>'
                )
            else:
                close_list()
                parts.append(
                    f'<p style="margin:0 0 10px;color:#000000;font-size:{FS_BODY};line-height:1.8;">'
                    f'{escape_and_inline_bold(line)}</p>'
                )
 
    close_list()
    return "\n".join(parts)



def _render_ai_panel(title, body, accent="#4f7cff", margin_top=False):
    formatted = _format_ai_response(body)
    mt = "margin-top:12px;" if margin_top else ""
    st.markdown(
        f'<div style="'
        f'background:linear-gradient(135deg,rgba(255,255,255,0.99),rgba(248,252,255,0.99));'
        f'border:1px solid #dbe6f5;'
        f'border-left:4px solid {accent};'
        f'border-radius:16px;'
        f'padding:24px 28px 20px;'
        f'box-shadow:0 8px 32px rgba(20,42,84,0.07);'
        f'{mt}">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:18px;'
        f'padding-bottom:12px;border-bottom:1px solid #eef2fa;">'
        f'<span style="font-size:12px;font-weight:800;letter-spacing:.1em;'
        f'text-transform:uppercase;color:{accent}">✦ {html.escape(title)}</span>'
        f'</div>'
        f'<div>{formatted}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# ADMIN-ONLY VIEW
# ══════════════════════════════════════════════════════════════════
if is_admin:
    active_sessions = _get_active_sessions()
    active_session_count = len(active_sessions)

    with st.sidebar:
        st.markdown(
            '<div style="padding:.8rem 0 1.4rem">'
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
            '<div style="width:32px;height:32px;border-radius:9px;'
            'background:linear-gradient(135deg,#a855f7,#ec4899);'
            'display:flex;align-items:center;justify-content:center;font-size:16px">🛡️</div>'
            '<div>'
            '<div style="font-size:1.05rem;font-weight:700;color:#f0e6ff">InstaScribe</div>'
            '<div style="font-size:10px;color:#6b4fa0;text-transform:uppercase;'
            'letter-spacing:.1em">Admin Console</div>'
            '</div></div></div>',
            unsafe_allow_html=True)
        st.write(f'Welcome **{name}**')
        authenticator.logout('Logout', 'sidebar', callback=_clear_active_session)

    st.markdown(
        f'<div class="app-header">'
        f'<div class="app-header-card">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<div class="app-logo-icon">🛡️</div>'
        f'<div>'
        f'<div class="app-title">Admin Console</div>'
        f'<div class="app-subtitle">User Store · Account Management · Session Monitor</div>'
        f'</div></div>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:11px;color:#6b4fa0;'
        f'border:1px solid #2d1555;padding:2px 10px;border-radius:20px;">🔑 Admin Access</span>'
        f'</div>'
        f'</div></div>',
        unsafe_allow_html=True)

    if "admin_page" not in st.session_state:
        st.session_state["admin_page"] = "users"
    if "edit_user" not in st.session_state:
        st.session_state["edit_user"] = None
    if "confirm_delete" not in st.session_state:
        st.session_state["confirm_delete"] = None
    admin_flash = st.session_state.pop("admin_flash", None)
    if admin_flash:
        flash_kind = admin_flash.get("kind", "success")
        flash_message = admin_flash.get("message", "")
        if flash_kind == "success":
            st.success(flash_message)
        elif flash_kind == "error":
            st.error(flash_message)
        else:
            st.info(flash_message)

    all_users  = auth_store["credentials"]["usernames"]
    total_u    = len(all_users)
    admin_u    = sum(1 for u in all_users.values() if u.get("role") == "admin")
    member_u   = total_u - admin_u

    k1, k2, k3, k4 = st.columns(4, gap="large")
    k1.markdown(kpi("Registered Users",  f"{total_u:,}",           "stored securely in Supabase",      "#818cf8","#6366f1", 75), unsafe_allow_html=True)
    k2.markdown(kpi("Active Sessions",   f"{active_session_count}","browser sessions right now",        "#4ade80","#22c55e", min(active_session_count*25,100)), unsafe_allow_html=True)
    k3.markdown(kpi("Admins",            f"{admin_u}",             "with elevated privileges",          "#f87171","#ef4444", int(admin_u/max(total_u,1)*100)), unsafe_allow_html=True)
    k4.markdown(kpi("Members",           f"{member_u}",            "standard user accounts",            "#c084fc","#a855f7", int(member_u/max(total_u,1)*100)), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

    tab_users, tab_sessions, tab_revenue = st.tabs([
        "👥  User Store", "🖥️  Active Sessions", "💰  Revenue"
    ])

    with tab_users:
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        search_col, refresh_col = st.columns([0.88, 0.12], gap="small")
        with search_col:
            search_q = st.text_input(
                "🔍 Search by username or email",
                placeholder="Type username or email…",
                key="admin_search"
            )
        with refresh_col:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            if st.button("🔄", use_container_width=True, key="admin_refresh_btn"):
                _refresh_auth_store_cache()
                st.rerun()

        user_rows = []
        for username_k, record in auth_store["credentials"]["usernames"].items():
            user_rows.append({
                "Username": username_k,
                "Name": record.get("name", ""),
                "Email": record.get("email", ""),
                "Role": record.get("role", "member"),
                "Company": record.get("details", {}).get("company", "") if isinstance(record.get("details"), dict) else "",
                "Created At": record.get("created_at", ""),
                "Password Hash": _mask_hash(record.get("password", "")),
            })

        users_df = pd.DataFrame(user_rows).sort_values("Username").reset_index(drop=True)

        if search_q.strip():
            q_low = search_q.strip().lower()
            mask = (
                users_df["Username"].str.lower().str.contains(q_low, na=False) |
                users_df["Email"].str.lower().str.contains(q_low, na=False)
            )
            filtered_df = users_df[mask].reset_index(drop=True)
        else:
            filtered_df = users_df

        st.markdown(
            f'<div style="font-size:12px;color:#6b4fa0;margin-bottom:10px;">'
            f'{len(filtered_df)} of {len(users_df)} user(s) shown</div>',
            unsafe_allow_html=True
        )

        display_cols = ["Username", "Name", "Email", "Role", "Company", "Created At", "Password Hash"]
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)
        st.caption("Passwords are never stored in plain text. Only salted hashes are persisted.")

        st.markdown('<div class="grad-divider"></div>', unsafe_allow_html=True)
        sec("⚙️ Manage User")

        if len(filtered_df) == 0:
            st.info("No users match your search.")
        else:
            manage_options = filtered_df["Username"].tolist()
            sel_manage = st.selectbox("Select user to manage", manage_options, key="admin_manage_sel")

            if sel_manage:
                record_m = auth_store["credentials"]["usernames"].get(sel_manage, {})
                details_m = record_m.get("details", {}) if isinstance(record_m.get("details", {}), dict) else {}

                action_col1, action_col2 = st.columns(2, gap="small")

                with action_col1:
                    st.markdown(
                        '<div style="background:linear-gradient(135deg,#1a0d2e,#150a24);'
                        'border:1px solid #2d1555;border-left:3px solid #818cf8;'
                        'border-radius:14px;padding:18px 20px;">'
                        '<div style="font-size:12px;font-weight:700;color:#818cf8;'
                        'text-transform:uppercase;letter-spacing:.07em;margin-bottom:14px">✏️ Edit User</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    with st.form(f"edit_form_{sel_manage}", clear_on_submit=False):
                        new_name    = st.text_input("Full Name",  value=record_m.get("name", ""),       key="ef_name")
                        new_email   = st.text_input("Email",      value=record_m.get("email", ""),      key="ef_email")
                        new_company = st.text_input("Company",    value=details_m.get("company", ""),   key="ef_company")
                        role_opts   = ["member", "admin"]
                        cur_role    = record_m.get("role", "member")
                        new_role    = st.selectbox("Role", role_opts,
                                                   index=role_opts.index(cur_role) if cur_role in role_opts else 0,
                                                   key="ef_role")
                        save_edit   = st.form_submit_button("💾 Save Changes")

                    if save_edit:
                        new_email_val = new_email.strip().lower()
                        conflict = any(
                            u.get("email", "").strip().lower() == new_email_val and uname != sel_manage
                            for uname, u in auth_store["credentials"]["usernames"].items()
                        )
                        if conflict:
                            st.error("Email already used by another account.")
                        else:
                            updated_details = dict(details_m)
                            updated_details["company"] = new_company.strip()
                            updated_record = dict(record_m)
                            updated_record["name"]    = new_name.strip()
                            updated_record["email"]   = new_email_val
                            updated_record["role"]    = new_role
                            updated_record["details"] = updated_details
                            _save_user_record(sel_manage, updated_record)
                            _set_admin_flash(f"User **{sel_manage}** details changed.")
                            st.rerun()

                with action_col2:
                    st.markdown(
                        '<div style="background:linear-gradient(135deg,#2e0d0d,#240a0a);'
                        'border:1px solid #551515;border-left:3px solid #f87171;'
                        'border-radius:14px;padding:18px 20px;">'
                        '<div style="font-size:12px;font-weight:700;color:#f87171;'
                        'text-transform:uppercase;letter-spacing:.07em;margin-bottom:14px">🗑️ Delete User</div>'
                        f'<div style="font-size:12px;color:#c4a0a0;margin-bottom:12px;">'
                        f'Permanently remove <b style="color:#ffdddd">{sel_manage}</b>.'
                        f' This action cannot be undone.</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    if sel_manage == username:
                        st.warning("⚠️ You cannot delete your own account.")
                    else:
                        if st.session_state.get("confirm_delete") != sel_manage:
                            if st.button("🗑️ Delete User", key="del_btn", use_container_width=True):
                                st.session_state["confirm_delete"] = sel_manage
                                st.rerun()
                        else:
                            st.warning(f"Confirm deletion of **{sel_manage}**?")
                            yes_col, no_col = st.columns(2)
                            if yes_col.button("✅ Yes, Delete", key="del_yes", use_container_width=True):

                                st.write("Trying to delete:", sel_manage)

                                success = _delete_user_record(sel_manage)

                                if success:
                                    st.success(f"{sel_manage} deleted")

                                    auth_store["credentials"]["usernames"].pop(sel_manage, None)

                                    st.session_state["confirm_delete"] = None
                                    st.session_state.pop("admin_manage_sel", None)

                                    _set_admin_flash(f"User **{sel_manage}** deleted.")
                                st.rerun()
                                auth_store["credentials"]["usernames"].pop(sel_manage, None)
                                st.session_state["confirm_delete"] = None
                                st.session_state.pop("admin_manage_sel", None)
                                _set_admin_flash(f"User **{sel_manage}** deleted.")
                                st.rerun()
                            if no_col.button("❌ Cancel", key="del_no", use_container_width=True):
                                st.session_state["confirm_delete"] = None
                                st.rerun()
    with tab_sessions:
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        sessions = _get_active_sessions()
        if not sessions:
            st.info("No active sessions recorded.")
        else:
            sec(f"🖥️ Active Sessions — {len(sessions)} online")
            def _to_ist(utc_str):
                try:
                    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
                    ist = dt + timedelta(hours=5, minutes=30)
                    return ist.strftime("%Y-%m-%d %H:%M:%S IST")
                except Exception:
                    return utc_str

            sess_df = pd.DataFrame([
            {
                "Username": row["username"],
                "Name": row["name"],
                "Role": row["role"],
                "Started At": _to_ist(row["started_at"]),
                "Last Seen": _to_ist(row["last_seen"]),
            }
            for row in sessions
        ])
            st.dataframe(sess_df, use_container_width=True, hide_index=True)

            st.markdown(
                '<div class="insight" style="--ac:#4ade80;margin-top:12px">'
                '💡 <b>Session tracking</b> is persisted in Supabase. '
                'Active sessions are refreshed whenever an authenticated user loads a page. '
                'Old sessions are pruned automatically after inactivity.</div>',
                unsafe_allow_html=True
            )

        if st.button("🔄 Refresh Sessions", use_container_width=False):
            _refresh_admin_view("sessions")

    with tab_revenue:
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        render_admin_revenue_tab()

    st.stop()

# ══════════════════════════════════════════════════════════════════
# NON-ADMIN — full dashboard
# ══════════════════════════════════════════════════════════════════
leads_full, posts_full = load_data()

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="padding:.8rem 0 1.4rem">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">'
        f'<img src="data:image/png;base64,{FAVICON_B64}" '
        f'style="width:44px;height:44px;border-radius:12px;object-fit:cover;display:block;flex-shrink:0;">'
        f'<div style="display:flex;flex-direction:column;gap:2px;">'
        f'<div style="font-size:16px;font-weight:800;color:#1a1a2e;letter-spacing:.01em;line-height:1.2;">InstaScribe</div>'
        f'<div style="font-size:12px;font-weight:600;color:#a855f7;text-transform:uppercase;letter-spacing:.12em;line-height:1.2;">Creator Intelligence</div>'
        f'</div>'
        f'</div></div>',
        unsafe_allow_html=True)

    st.markdown(
    f"""
    <div style="
        color:#0F0F0F;
        font-size:15px;
        font-weight:700;
        margin-bottom:12px;
    ">
        Welcome {name}
    </div>
    """,
    unsafe_allow_html=True
)
    if auth_status:
        authenticator.logout('Logout', 'sidebar', callback=_clear_active_session)

    st.markdown('<div style="font-size:15px;font-weight:600;color:#a855f7;'
                'text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">🎛 Filters</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    all_cats = sorted(leads_full["Category_Name"].dropna().unique()) \
               if "Category_Name" in leads_full.columns else []
    sel_cats = st.multiselect("📂 Category", all_cats, default=[], placeholder="All categories")
    sel_qual = st.multiselect("🎯 Lead Quality", ["high","medium","low"], default=[], placeholder="All quality tiers")

    er_lo = float(leads_full["Engagement_Rate"].min())
    er_hi = float(leads_full["Engagement_Rate"].max())
    sel_er = st.slider("📈 Engagement Rate (%)", er_lo, er_hi, (er_lo, er_hi), step=0.1)

    fo_lo = int(leads_full["Follower_Count"].min())
    fo_hi = int(leads_full["Follower_Count"].max())
    sel_fo = st.slider("👥 Followers", fo_lo, fo_hi, (fo_lo, fo_hi), step=10_000)

    sel_score = st.slider("🤖 Min Lead Score", 0, 100, 0)
    st.markdown("---")

    all_tiers = sorted(leads_full["Follower_Tier"].unique())
    sel_tiers = st.multiselect("🏷 Follower Tier", all_tiers, default=[], placeholder="All tiers")

    d_lo = posts_full["Post_Date"].dropna().min().date()
    d_hi = posts_full["Post_Date"].dropna().max().date()
    sel_dates = st.date_input("📅 Date Range", (d_lo, d_hi))

    all_yrs = sorted(posts_full["Year"].dropna().astype(int).unique())
    sel_yrs = st.multiselect("📅 Year", all_yrs, default=[], placeholder="All years")

    st.markdown("---")
    if st.button("↺ Reset Filters", use_container_width=True):
        st.rerun()

# ── FILTER FUNCTIONS ───────────────────────────────────────────
def fl(d):
    if sel_cats and "Category_Name" in d.columns:
        d = d[d["Category_Name"].isin(sel_cats)]
    if sel_cats and "Category_Lead_Quality" in d.columns:
        d = d.copy()
        d["Lead_Quality"] = d["Category_Lead_Quality"]
    if sel_qual:
        d = d[d["Lead_Quality"].isin(sel_qual)]
    d = d[d["Engagement_Rate"].between(*sel_er)]
    d = d[d["Follower_Count"].between(*sel_fo)]
    if sel_tiers: d = d[d["Follower_Tier"].isin(sel_tiers)]
    return d[d["Lead_Score"] >= sel_score]

def fp(d):
    if len(sel_dates)==2:
        d = d[(d["Post_Date"].dt.date >= sel_dates[0]) &
              (d["Post_Date"].dt.date <= sel_dates[1])]
    if sel_yrs:  d = d[d["Year"].isin(sel_yrs)]
    if sel_cats and "Category_Name" in d.columns:
        d = d[d["Category_Name"].isin(sel_cats)]
    if sel_qual and "Lead_Quality" in d.columns:
        d = d[d["Lead_Quality"].isin(sel_qual)]
    return d

leads = fl(leads_full.copy())
posts = fp(posts_full.copy())

is_filtered = (bool(sel_cats) or bool(sel_qual) or bool(sel_tiers) or bool(sel_yrs)
               or sel_er!=(er_lo,er_hi) or sel_fo!=(fo_lo,fo_hi) or sel_score>0)

# ── HEADER ─────────────────────────────────────────────────────
chips = "".join(
    f'<span class="chip">{v}</span>'
    for v in list(sel_cats)+list(sel_qual)+list(sel_tiers)+list(sel_yrs))

st.markdown(
    f'<div class="app-header">'
    f'<div class="app-header-card">'
    f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
    f'<div style="display:flex;align-items:center;gap:10px">'
    f'<div class="app-logo-icon">{FAVICON_IMG}</div>'
    f'<div>'
    f'<div class="app-title">InstaScribe</div>'
    f'<div class="app-subtitle">Creator Intelligence Platform</div>'
    f'</div></div>'
    f'<span style="font-family:\'DM Mono\',monospace;font-size:12px;color:#6b4fa0;'
    f'border:1px solid #2d1555;padding:2px 10px;border-radius:20px;">'
    f'{"🔍 " if is_filtered else "📊 "}{len(leads):,} / {len(leads_full):,} records</span>'
    f'</div>'
    f'{"<div style=margin-top:6px>"+chips+"</div>" if chips else ""}'
    f'</div></div>',
    unsafe_allow_html=True)

# ── NAV ────────────────────────────────────────────────────────
nav_pages = ["Executive Overview","Lead Intelligence","Post Analytics","Lead Scoring","AI Insights","Subscription","About"]
nav_icons = ["bar-chart-fill","people-fill","chat-dots-fill","robot","stars","credit-card-fill","info-circle-fill"]

page = option_menu(None,
    nav_pages,
    icons=nav_icons,
    orientation="horizontal",
    styles={
        "container": {
            "background": "linear-gradient(135deg,#ffffff,#f5f8ff)",
            "border": "1px solid #dbe6f5",
            "border-radius": "12px", "padding": "5px 10px"
        },
        "nav-link": {
            "font-size": "13px", "color": "#5f6f8d",
            "border-radius": "8px", "padding": "7px 14px"
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg,#4f7cff,#ff4dad)",
            "color": "#fff", "font-weight": "600"
        },
        "icon": {"font-size": "13px"},
    })

# ==========================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ==========================================================
if page == "Executive Overview":

    total  = len(leads)
    t_eng  = leads["Engagement"].sum()
    avg_er = leads["Engagement_Rate"].mean() if total else 0
    hq     = (leads["Lead_Quality"]=="high").sum()
    n_cats = leads["Category_Name"].nunique() if "Category_Name" in leads.columns else 0

    eqr_raw = (leads["Engagement"] / leads["Follower_Count"].replace(0, np.nan)).mean()
    eqr_raw = 0.0 if (eqr_raw is None or (isinstance(eqr_raw, float) and np.isnan(eqr_raw))) else eqr_raw
    if eqr_raw > 0.10:
        eqr_label = "Positive"; eqr_color = "#4ade80"; eqr_c2 = "#22c55e"
        eqr_sub   = f"ratio {eqr_raw:.4f} · strong audience quality"
    elif eqr_raw >= 0.04:
        eqr_label = "Neutral"; eqr_color = "#fbbf24"; eqr_c2 = "#f59e0b"
        eqr_sub   = f"ratio {eqr_raw:.4f} · average audience quality"
    else:
        eqr_label = "Negative"; eqr_color = "#f87171"; eqr_c2 = "#ef4444"
        eqr_sub   = f"ratio {eqr_raw:.4f} · weak audience quality"

    k1, k2, k3, k4 = st.columns(4, gap="large")
    k1.markdown(kpi("Total Influencers",   fmt(total), f"{n_cats} {'category' if n_cats==1 else 'categories'}", "#a855f7", "#ec4899", 80), unsafe_allow_html=True)
    k2.markdown(kpi("Total Engagement",    fmt(t_eng), "across selected handles", "#818cf8", "#6366f1", 95), unsafe_allow_html=True)
    k3.markdown(kpi("Avg Engagement Rate", f"{avg_er:.2f}%", "average across selection", "#c084fc", "#a855f7", 60), unsafe_allow_html=True)
    k4.markdown(kpi("Engagement Quality",  eqr_label, eqr_sub, eqr_color, eqr_c2, 72), unsafe_allow_html=True)

    sec("🧠 Smart Insights")
    if total > 0 and "Category_Name" in leads.columns:
        top_er   = leads.groupby("Category_Name")["Engagement_Rate"].mean().idxmax()
        top_fol  = leads.groupby("Category_Name")["Follower_Count"].mean().idxmax()
        med      = (leads["Lead_Quality"]=="medium").sum()
        top_hq   = (leads[leads["Lead_Quality"]=="high"].groupby("Category_Name").size().idxmax() if hq else "—")
        avg_sc   = leads["Lead_Score"].mean()
        tier_top = leads["Follower_Tier"].value_counts().idxmax()
        top_eqr  = (leads.assign(EQR=leads["Engagement"]/leads["Follower_Count"].replace(0,np.nan)).groupby("Category_Name")["EQR"].mean().idxmax())

        i1, i2, i3 = st.columns(3)
        i1.markdown(f'<div class="insight" style="--ac:#818cf8">📌 <b class="category">{top_er.title()}</b> leads with the highest avg engagement rate across {total:,} influencers in view.</div>', unsafe_allow_html=True)
        i2.markdown(f'<div class="insight" style="--ac:#4ade80">📡 <b class="category">{top_fol.title()}</b> dominates follower reach — best for brand awareness campaigns.</div>', unsafe_allow_html=True)
        i3.markdown(f'<div class="insight" style="--ac:#fbbf24">🎯 <b>{hq:,}</b> leads outreach-ready ({hq/max(total,1)*100:.1f}%). Top vertical: <b class="category">{str(top_hq).title()}</b>.</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
        i4, i5, i6 = st.columns(3)
        i4.markdown(f'<div class="insight" style="--ac:#c084fc">🏷 Dominant tier: <b>{tier_top}</b>. Avg lead score: <b>{avg_sc:.1f} / 100</b>.</div>', unsafe_allow_html=True)
        i5.markdown(f'<div class="insight" style="--ac:#ec4899">💡 <b class="category">{top_eqr.title()}</b> has the best engagement quality ratio — most impactful audience per follower.</div>', unsafe_allow_html=True)
        i6.markdown(f'<div class="insight" style="--ac:#4ade80">⏳ <b>{med:,}</b> medium leads ({med/max(total,1)*100:.1f}%) in the nurture pipeline.</div>', unsafe_allow_html=True)
    else:
        st.info("No data matches the current filters — adjust the sidebar.")

    sec("Engagement Trend & Lead Pipeline")
    ch1, ch2 = st.columns([3, 2])

    with ch1:
        monthly = (posts.groupby("Month")["Engagement"].sum().reset_index().sort_values("Month"))
        fig = go.Figure(go.Scatter(
            x=monthly["Month"], y=monthly["Engagement"],
            mode="lines+markers",
            line=dict(color="#423dec", width=2.5),
            marker=dict(size=5, color="#210915"),
            fill="tozeroy", fillcolor="rgba(168,85,247,0.07)"))
        dark(fig, 300)
        fig.update_layout(title=dict(text="Monthly Engagement Trend", font=dict(size=12,color="#6b4fa0")))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        qc = leads["Lead_Quality"].value_counts().reset_index()
        qc.columns = ["Quality", "Count"]
        order = ["high","medium","low"]
        qc["_ord"] = qc["Quality"].map({v:i for i,v in enumerate(order)})
        qc = qc.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
        total_q = qc["Count"].sum()
        qc["Pct"] = (qc["Count"] / max(total_q,1) * 100).round(1)
        fig2 = go.Figure(go.Pie(
            labels=qc["Quality"].str.title(), values=qc["Count"], hole=0.62,
            marker=dict(colors=[Q_CLR.get(q,"#888") for q in qc["Quality"]], line=dict(color="#0e0814",width=2)),
            text=qc["Pct"].map(lambda p: f"{p:.1f}%"), textinfo="text", textposition="inside",
            textfont=dict(size=10, color="#141414"),
            hovertemplate="<b>%{label}</b><extra></extra>",
            direction="clockwise", sort=False))
        dark(fig2, 300)
        fig2.update_layout(title=dict(text="Lead Quality Distribution", font=dict(size=12,color="#6b4fa0")), legend=dict(orientation="h",y=-0.18,font=dict(size=10)))
        st.plotly_chart(fig2, use_container_width=True)

    sec("Category Comparison Charts")
    if "Category_Name" in leads.columns:
        avail_cats = sorted(leads["Category_Name"].dropna().unique().tolist())
        compare_cats = sel_cats if sel_cats else avail_cats[:2]
        if not compare_cats:
            st.info("No categories available to compare.")
        else:
            compare_cats = compare_cats[:5]
            cmp_df = leads[leads["Category_Name"].isin(compare_cats)]
            cc1, cc2 = st.columns(2)

            with cc1:
                eng_bar = (cmp_df.groupby("Category_Name")["Engagement"].sum().reset_index().sort_values("Engagement", ascending=False))
                eng_bar["Color"] = eng_bar["Category_Name"].map(lambda x: CAT_CLR.get(str(x).lower(), "#a855f7"))
                fig_a = go.Figure()
                for _, r in eng_bar.iterrows():
                    fig_a.add_trace(go.Bar(
                        x=[r["Category_Name"]],
                        y=[r["Engagement"]],
                        marker_color=r["Color"],
                        marker_line_width=0,
                        opacity=0.92,
                        text=[fmt(int(r["Engagement"]))],
                        textposition="outside",
                        textfont=dict(color="#0d0c0c", size=11),
                        name=r["Category_Name"],
                    ))
                dark(fig_a, 280)
                fig_a.update_layout(title=dict(text="Total Engagement", font=dict(size=12,color="#6b4fa0")), showlegend=False, bargap=0.4)
                st.plotly_chart(fig_a, use_container_width=True)

            with cc2:
                fol_bar = (cmp_df.groupby("Category_Name")["Follower_Count"].mean().reset_index().sort_values("Follower_Count", ascending=False))
                fol_bar["Color"] = fol_bar["Category_Name"].map(lambda x: CAT_CLR.get(str(x).lower(), "#a855f7"))
                fig_b = go.Figure()
                for _, r in fol_bar.iterrows():
                    fig_b.add_trace(go.Bar(
                        x=[r["Category_Name"]],
                        y=[r["Follower_Count"]],
                        marker_color=r["Color"],
                        marker_line_width=0,
                        opacity=0.92,
                        text=[fmt(int(r["Follower_Count"]))],
                        textposition="outside",
                        textfont=dict(color="#080707", size=11),
                        name=r["Category_Name"],
                    ))
                dark(fig_b, 280)
                fig_b.update_layout(title=dict(text="Avg Follower Count", font=dict(size=12,color="#6b4fa0")), showlegend=False, bargap=0.4)
                st.plotly_chart(fig_b, use_container_width=True)

# ==========================================================
# PAGE 2 — LEAD INTELLIGENCE
# ==========================================================
elif page == "Lead Intelligence":
    total  = len(leads)
    hq     = (leads["Lead_Quality"]=="high").sum()
    hi_er  = (leads["Engagement_Rate"] > 15).sum()

    st.markdown(f'<p style="color:#7036E0;font-size:15px;margin-bottom:1rem">Influencer quality signals, authenticity analysis, and profile ranking · <b style="color:#A127E8">{total:,} records in view</b></p>', unsafe_allow_html=True)

    avg_following = leads["Following_Count"].mean() if "Following_Count" in leads.columns else 0
    avg_following = 0 if (avg_following is None or (isinstance(avg_following, float) and np.isnan(avg_following))) else avg_following

    k1, k2, k3, k4 = st.columns(4, gap="large")
    k1.markdown(kpi("Total Influencers",      fmt(total),                             "matching current filter",          "#a855f7","#ec4899",80), unsafe_allow_html=True)
    k2.markdown(kpi("High-Quality Leads",     fmt(hq),                               f"{hq/max(total,1)*100:.1f}% ready for outreach", "#4ade80","#22c55e", int(hq/max(total,1)*100)), unsafe_allow_html=True)
    k3.markdown(kpi("Avg Accounts Following", f"{round(avg_following):,}",            "Avg accounts each influencer follows", "#c084fc","#a855f7",55), unsafe_allow_html=True)
    k4.markdown(kpi("High Engagement",        fmt(hi_er),                            "ER above 15% — contact first",     "#ec4899","#be185d", int(hi_er/max(total,1)*100)), unsafe_allow_html=True)

    if sel_cats:
        st.markdown('<div class="insight" style="--ac:#818cf8">📂 <b>Category-aware quality is active</b> — <b>high</b> means top-tier within the selected category.</div>', unsafe_allow_html=True)

    desc("<b>Avg Accounts Following</b> = average number of other accounts each influencer follows. A <b>lower number</b> signals authentic organic growth. A <b>high number</b> (5,000+) suggests mass-following — a common fake-follower tactic.")

    sec("Follower Reach vs Engagement Rate")
    ch1, ch2 = st.columns([3, 2])

    with ch1:
        samp = leads.sample(min(2000,len(leads)), random_state=42)
        kws  = dict(opacity=0.55)
        if "Category_Name" in samp.columns: kws["color"] = "Category_Name"; kws["color_discrete_map"] = CAT_CLR
        if "Engagement" in samp.columns: kws["size"] = "Engagement"
        if "Handle"     in samp.columns: kws["hover_data"] = ["Handle","Lead_Quality"]
        fig = px.scatter(samp, x="Follower_Count", y="Engagement_Rate", **kws)
        fig.update_traces(marker_line_width=0)
        dark(fig, 320)
        fig.update_layout(title=dict(text="Followers vs Engagement Rate", font=dict(size=12,color="#6b4fa0")))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        sec("Avg Accounts Following per Category")
        if "Category_Name" in leads.columns and "Following_Count" in leads.columns:
            fc_cat = leads.groupby("Category_Name")["Following_Count"].mean().sort_values()
            max_fc = fc_cat.max() if fc_cat.max() > 0 else 1
            html_str = '<div style="background:rgba(255,255,255,0.6);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid rgba(255,255,255, 0.5);border-radius:14px;padding:18px 20px;color:#000000">'
            for cat, val in fc_cat.items():
                color = CAT_CLR.get(str(cat).lower(), "#a855f7")
                html_str += pb(str(cat).title(), f"{round(val):,}", val / max_fc * 100, color)
            html_str += '<div style="font-size:10.5px;color:#7036E0;margin-top:10px">Lower number = fewer accounts followed = organic creator</div></div>'
            st.markdown(html_str, unsafe_allow_html=True)

    sec("Follower Tier Distribution & Engagement Rate by Category")
    ch3, ch4 = st.columns(2)

    with ch3:
        if "Category_Name" in leads.columns:
            tier_order = ["Nano (<10K)","Micro (10K-100K)","Mid (100K-500K)","Macro (500K-1M)","Mega (1M+)"]
            td = leads.groupby(["Category_Name","Follower_Tier"]).size().reset_index(name="Count")
            fig3 = px.bar(
                td,
                x="Category_Name",
                y="Count",
                color="Follower_Tier",
                barmode="stack",
                category_orders={"Follower_Tier":tier_order},
                color_discrete_sequence=["#2563eb", "#db2777", "#eab308", "#3b82f6", "#be185d"],
            )
            fig3.update_traces(marker_line_width=1, opacity=1, textfont=dict(color="#050404"), hovertemplate=None)
            dark(fig3, 290)
            fig3.update_layout(title=dict(text="Follower Tier Stack by Category",font=dict(size=12,color="#6b4fa0")), bargap=0.25)
            st.plotly_chart(fig3, use_container_width=True)

    with ch4:
        if "Category_Name" in leads.columns:
            er_cat = (leads.groupby("Category_Name")["Engagement_Rate"].agg(["mean","min","max"]).reset_index())
            er_cat["Color"] = er_cat["Category_Name"].map(lambda x: CAT_CLR.get(str(x).lower(), "#a855f7"))
            fig4 = go.Figure()
            for _, r in er_cat.sort_values("mean", ascending=False).iterrows():
                fig4.add_trace(go.Bar(
                        x=[r["Category_Name"]], y=[r["mean"]],
                        marker_color=r["Color"], marker_line_width=0,
                        opacity=0.92,
                        error_y=dict(type="data", array=[max(r["max"]-r["mean"],0)], arrayminus=[max(r["mean"]-r["min"],0)], thickness=1, width=4, color="#9b7ec8"),
                        text=[f"{r['mean']:.2f}%"], textposition="outside",
                        textfont=dict(color="#080808", size=11),
                        hovertemplate=f"<b>{r['Category_Name']}</b><br>Avg ER: {r['mean']:.2f}%<br>Min: {r['min']:.2f}%<br>Max: {r['max']:.2f}%<extra></extra>",
                        name=r["Category_Name"],
                    ))
            dark(fig4, 290)
            fig4.update_layout(title=dict(text="Average Engagement Rate by Category",font=dict(size=12,color="#6b4fa0")), showlegend=False, bargap=0.35)
            st.plotly_chart(fig4, use_container_width=True)

    sec("Top Influencer Profiles")
    s1, s2 = st.columns([2,1])
    sort_by = s1.selectbox("Sort by", ["Lead_Score","Follower_Count","Engagement_Rate","Engagement"], key="li_s")
    n_show  = s2.selectbox("Show", [20,50,100], key="li_n")

    sc  = [c for c in ["Handle","Category_Name","Follower_Count","Engagement_Rate","Lead_Score","Lead_Quality","Action"] if c in leads.columns]
    tbl = leads.nlargest(n_show, sort_by)[sc].copy()
    if "Follower_Count"  in tbl.columns: tbl["Follower_Count"]  = tbl["Follower_Count"].apply(fmt)
    if "Engagement_Rate" in tbl.columns: tbl["Engagement_Rate"] = tbl["Engagement_Rate"].round(2)
    if "Lead_Score"      in tbl.columns: tbl["Lead_Score"]      = tbl["Lead_Score"].round(1)
    st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=320)


# ==========================================================
# PAGE 3 — POST ANALYTICS
# ==========================================================
elif page == "Post Analytics":
    posts_pa = posts_full.copy()
    avg_l  = posts_pa["Likes"].mean()    if "Likes"    in posts_pa.columns else 0
    avg_c  = posts_pa["Comments"].mean() if "Comments" in posts_pa.columns else 0
    avg_l  = 0 if (avg_l is None or (isinstance(avg_l, float) and np.isnan(avg_l))) else avg_l
    avg_c  = 0 if (avg_c is None or (isinstance(avg_c, float) and np.isnan(avg_c))) else avg_c

    st.markdown(f'<p style="color:#6b4fa0;font-size:15px;margin-bottom:1rem">Content performance, engagement patterns, and post inspector · <b style="color:#d8b4fe">{len(posts_pa):,} posts in view</b></p>', unsafe_allow_html=True)

    if len(posts_pa) == 0:
        st.info("No posts match the current filters."); st.stop()

    grad_divider()
    sec("🔍 Post Inspector — Search & Drill Down")
    desc("Search any <b>Post ID</b> or <b>Handle</b> to view the post's metrics, engagement history, and the influencer's full profile below.")

    pt_nonce = st.session_state.get("pt_refresh_nonce", 0)
    src1, src2, src3 = st.columns([3,2,2])
    search_text = src1.text_input("🔍 Search Post ID or Handle", placeholder="e.g. POST000001 or username", key=f"pt_search_{pt_nonce}")

    if search_text.strip():
        q = search_text.strip().lower()
        search_hits = posts_full.copy()
        mask = pd.Series([False] * len(search_hits), index=search_hits.index)
        if "Post_ID" in search_hits.columns: mask |= search_hits["Post_ID"].astype(str).str.lower().str.contains(q, na=False)
        if "Handle"  in search_hits.columns: mask |= search_hits["Handle"].astype(str).str.lower().str.contains(q, na=False)
        search_hits = search_hits[mask]
        if len(search_hits) > 0:
            first_hit = search_hits.iloc[0]
            hit_month = first_hit.get("MonthName", None)
            hit_cat   = first_hit.get("Category_Name", None)
            if pd.notna(hit_month) and hit_month in posts_full["MonthName"].dropna().unique().tolist():
                st.session_state["pt_month"] = hit_month
            if pd.notna(hit_cat) and str(hit_cat) in posts_full["Category_Name"].dropna().unique().tolist():
                st.session_state["pt_cat"] = str(hit_cat)

    month_opts  = ["All Months"] + sorted(posts_full["MonthName"].dropna().unique().tolist(), key=lambda x: pd.to_datetime(x, format="%b %Y"))
    sel_month   = src2.selectbox("📅 Month", month_opts, key="pt_month")
    cat_opts_pt = ["All Categories"] + sorted(posts_full["Category_Name"].dropna().unique().tolist() if "Category_Name" in posts_full.columns else [])
    sel_cat_pt  = src3.selectbox("📂 Category", cat_opts_pt, key="pt_cat")

    pool = posts_full.copy()
    if sel_month != "All Months": pool = pool[pool["MonthName"] == sel_month]
    if search_text.strip():
        q = search_text.strip().lower()
        mask = pd.Series([False]*len(pool), index=pool.index)
        if "Post_ID" in pool.columns: mask |= pool["Post_ID"].astype(str).str.lower().str.contains(q, na=False)
        if "Handle"  in pool.columns: mask |= pool["Handle"].astype(str).str.lower().str.contains(q, na=False)
        pool = pool[mask]
    elif sel_cat_pt != "All Categories" and "Category_Name" in pool.columns:
        pool = pool[pool["Category_Name"] == sel_cat_pt]

    st.markdown(f'<div style="font-size:12px;color:#6b4fa0;margin-bottom:12px;">{len(pool):,} post(s) match · select a Post ID below to inspect</div>', unsafe_allow_html=True)

    def _reset_post_inspector():
        st.session_state["pt_refresh_nonce"] = st.session_state.get("pt_refresh_nonce", 0) + 1
        st.session_state.pop("pt_search", None)
        st.session_state.pop("pt_pid", None)

    inspector_found = len(pool) > 0
    if not inspector_found:
        st.info("No posts match — try a different search term, month, or category.")
        st.stop()
    else:
        post_id_col = "Post_ID" if "Post_ID" in pool.columns else pool.columns[0]
        post_ids    = pool[post_id_col].astype(str).tolist()
        pid_nonce   = st.session_state.get("pt_refresh_nonce", 0)
        pid_col, refresh_col = st.columns([6, 1], gap="small")
        sel_pid = pid_col.selectbox("📋 Select Post ID", post_ids, key=f"pt_pid_{pid_nonce}")
        refresh_col.button("↻ Refresh", use_container_width=True, on_click=_reset_post_inspector)
        row = pool[pool[post_id_col].astype(str) == sel_pid].iloc[0]

        handle     = str(row.get("Handle","—"))
        handle_key = handle.strip().lower()
        row_category_key      = str(row.get('Category_Name','')).strip().lower()
        selected_category_key = sel_cat_pt.strip().lower()
        category_matches      = (sel_cat_pt == "All Categories") or (row_category_key == selected_category_key)
        lead_mask  = leads_full["Handle"].astype(str).str.strip().str.lower() == handle_key
        lead_found = bool(lead_mask.any())

        post_date = str(row.get("Post_Date",""))[:10]
        category  = str(row.get("Category_Name","—")).title()
        quality   = str(row.get("Lead_Quality","—"))
        q_color   = Q_CLR.get(quality,"#9451f9")
        q_badge   = f'<span style="background:{q_color}22;color:{q_color};font-size:11px;font-weight:500;padding:2px 9px;border-radius:20px;border:1px solid {q_color}44">{quality.title()}</span>'
        cat_color = CAT_CLR.get(str(row.get("Category_Name","")).lower(),"#a855f7")

        if not category_matches:
            st.info("No posts match — try a different category.")
            st.stop()

        st.markdown(f"""
        <div class="post-banner">
          <div class="post-banner-id">{sel_pid}</div>
          <div class="post-banner-handle">{handle}</div>
          <div class="post-banner-meta">{post_date} &nbsp;·&nbsp; <span style="color:{cat_color}">{category}</span> &nbsp;·&nbsp; Lead Quality: {q_badge}</div>
        </div>""", unsafe_allow_html=True)

        pills_html = '<div style="margin-bottom:18px;display:flex;flex-wrap:wrap;">'
        stats = [
            ("Likes",      fmt(safe_int(row.get("Likes",0))),           "#2a10ef"),
            ("Comments",   fmt(safe_int(row.get("Comments",0))),         "#821af2"),
            ("Engagement", fmt(safe_int(row.get("Engagement",0))),       "#f31882"),
            ("Lead Score", f"{safe_float(row.get('Lead_Score',0)):.1f}", "#ff000d"),
        ]
        if lead_found:
            inf_row = leads_full[lead_mask].iloc[0]
            stats += [
                ("Followers", fmt(safe_int(inf_row["Follower_Count"])),              "#ec4899"),
                ("ER %",      f"{safe_float(inf_row['Engagement_Rate']):.2f}%",       "#f87171"),
            ]
        for lbl, val, color in stats:
            pills_html += f'<span class="stat-pill" style="--pc:{color}"><b>{val}</b> {lbl}</span>'
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)

        handle_posts = posts_full[posts_full["Handle"].astype(str).str.strip().str.lower() == handle_key].copy().sort_values("Post_Date")

        sec("Influencer Profile")
        if lead_found:
            inf_r = leads_full[lead_mask].iloc[0]
            sent_text, sent_color = sentiment_code(inf_r.get("Avg_Sentiment",0))
            p1, p2, p3, p4 = st.columns(4)
            p1.markdown(kpi("Total Posts",    f"{len(handle_posts):,}",                                  "by this user",   "#a855f7","#ec4899",60), unsafe_allow_html=True)
            p2.markdown(kpi("Followers",      fmt(safe_int(inf_r.get("Follower_Count",0))),        "master profile", "#4ade80","#22c55e",65), unsafe_allow_html=True)
            p3.markdown(kpi("Following",f"{round(safe_float(inf_r.get('Following_Count',0))):,}", "master profile","#c084fc","#a855f7",45), unsafe_allow_html=True)
            p4.markdown(kpi("Sentiment",      sent_text,                                            "master profile", sent_color,sent_color,70), unsafe_allow_html=True)
        else:
            st.info(f"No influencer master record found for {handle}.")

        if len(handle_posts) > 1:
            sec(f"{handle} — Engagement History")
            hc1, hc2 = st.columns(2)
            with hc1:
                fig_h1 = go.Figure()
                fig_h1.add_trace(go.Scatter(x=handle_posts["Post_Date"], y=handle_posts["Engagement"], mode="lines+markers", line=dict(color="#a855f7", width=2), marker=dict(size=5, color="#ec4899"), fill="tozeroy", fillcolor="rgba(168,85,247,0.07)", name="Engagement"))
                sel_date = row.get("Post_Date"); sel_eng = row.get("Engagement",0)
                if pd.notna(sel_date):
                    fig_h1.add_trace(go.Scatter(x=[sel_date], y=[sel_eng], mode="markers", marker=dict(size=13, color="#f87171", symbol="star"), name="Selected post"))
                dark(fig_h1, 280)
                fig_h1.update_layout(title=dict(text="Engagement Over Time  (★ = selected post)", font=dict(size=12,color="#6b4fa0")))
                st.plotly_chart(fig_h1, use_container_width=True)

            with hc2:
                if "Likes" in handle_posts.columns and "Comments" in handle_posts.columns:
                    hp = handle_posts.copy()
                    hp["Label"] = hp["Post_Date"].dt.strftime("%b %d")
                    hp = hp.tail(15).reset_index(drop=True)
                    fig_h2 = go.Figure()
                    fig_h2.add_trace(go.Bar(
                        y=hp["Label"], x=hp["Likes"], name="Likes", orientation="h",
                        marker_color="#2563eb", marker_line_width=0, opacity=0.92,
                    ))
                    fig_h2.add_trace(go.Bar(
                        y=hp["Label"], x=hp["Comments"], name="Comments", orientation="h",
                        marker_color="#db2777", marker_line_width=0, opacity=0.92,
                    ))
                    sel_label = pd.to_datetime(sel_date).strftime("%b %d") if pd.notna(sel_date) else None
                    if sel_label and sel_label in hp["Label"].values:
                        idx = hp[hp["Label"]==sel_label].index[0]
                        fig_h2.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=idx-0.5, y1=idx+0.5, fillcolor="rgba(248,113,113,0.08)", line=dict(color="#f87171", width=1, dash="dot"))
                    dark(fig_h2, 280)
                    fig_h2.update_layout(title=dict(text="Likes vs Comments — last 15 posts", font=dict(size=12,color="#6b4fa0")), barmode="group", bargap=0.22, xaxis_title="Count", legend=dict(orientation="h",y=-0.2))
                    st.plotly_chart(fig_h2, use_container_width=True)

            sec(f"Monthly Engagement — @{handle}")
            monthly_h = handle_posts.groupby("Month")["Engagement"].sum().reset_index().sort_values("Month")
            fig_h3 = px.bar(
                monthly_h,
                x="Month",
                y="Engagement",
                color_discrete_sequence=["#2563eb"],
            )
            fig_h3.update_traces(marker_line_width=0, opacity=0.92, textposition="outside", textfont=dict(color="#ffffff"))
            dark(fig_h3, 240)
            fig_h3.update_layout(title=dict(text="Monthly Engagement for this Influencer", font=dict(size=12,color="#6b4fa0")), bargap=0.25)
            st.plotly_chart(fig_h3, use_container_width=True)
        else:
            st.info(f"Only one post found for {handle} in the dataset.")

    grad_divider()
    sec("Post Volume & Content Distribution")
    ch1, ch2 = st.columns([3,2])

    with ch1:
        me = posts_pa.groupby("Month")["Engagement"].sum().reset_index().sort_values("Month")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=me["Month"], y=me["Engagement"],
            marker_color="#2563eb", marker_line_width=0, opacity=0.92, name="Engagement",
        ))
        fig.add_trace(go.Scatter(x=me["Month"], y=me["Engagement"], mode="lines", line=dict(color="#06010b",width=1.5), showlegend=False))
        dark(fig, 300)
        fig.update_layout(title=dict(text="Monthly Post Engagement", font=dict(size=12,color="#6b4fa0")), bargap=0.22)
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        if "Likes" in posts_pa.columns and "Comments" in posts_pa.columns:
            sp = posts_pa.sample(min(1200,len(posts_pa)), random_state=7).copy()
            med_likes = sp["Likes"].median(); med_comments = sp["Comments"].median()
            def _quadrant(r):
                hi_l = r["Likes"] >= med_likes; hi_c = r["Comments"] >= med_comments
                if hi_l and hi_c:  return "High Likes & Comments"
                if hi_l:           return "High Likes, Low Comments"
                if hi_c:           return "Low Likes, High Comments"
                return "Low Likes & Comments"
            sp["Quadrant"] = sp.apply(_quadrant, axis=1)
            quad_colors = {"High Likes & Comments":"#4ade80","High Likes, Low Comments":"#818cf8","Low Likes, High Comments":"#fbbf24","Low Likes & Comments":"#f87171"}
            fig2 = px.scatter(sp, x="Likes", y="Comments", color="Quadrant", size="Engagement", size_max=18, color_discrete_map=quad_colors, opacity=0.65, hover_data=(["Handle","Post_Date","Engagement"] if "Handle" in sp.columns else ["Post_Date","Engagement"]))
            fig2.add_vline(x=med_likes, line_dash="dot", line_color="#2d1555", line_width=1.2)
            fig2.add_hline(y=med_comments, line_dash="dot", line_color="#2d1555", line_width=1.2)
            x_max = sp["Likes"].quantile(0.97); y_max = sp["Comments"].quantile(0.97)
            for txt, x, y, col in [("🔥 Top",x_max*0.82,y_max*0.90,"#4ade80"),("👍 Likes",x_max*0.82,y_max*0.12,"#818cf8"),("💬 Comments",x_max*0.08,y_max*0.90,"#fbbf24"),("📉 Low",x_max*0.08,y_max*0.12,"#f87171")]:
                fig2.add_annotation(x=x, y=y, text=txt, showarrow=False, font=dict(color=col, size=10), bgcolor="rgba(14,8,20,0.7)", borderpad=3)
            fig2.update_traces(marker_line_width=0)
            dark(fig2, 300)
            fig2.update_layout(title=dict(text="Likes vs Comments — Quadrant View", font=dict(size=12,color="#6b4fa0")), legend=dict(orientation="v", x=1.01, y=1, font=dict(size=9), itemsizing="constant"))
            st.plotly_chart(fig2, use_container_width=True)

    sec("Engagement Heatmap — Month × Category")
    if "Category_Name" in posts_pa.columns:
        pivot = (posts_pa.groupby(["MonthName","Category_Name"])["Engagement"].sum().unstack("Category_Name").fillna(0))
        fig3 = px.imshow(pivot, color_continuous_scale=["#0e0814","#2d1555","#7e22ce","#d8b4fe"], aspect="auto")
        dark(fig3, 280)
        fig3.update_layout(title=dict(text="Total Engagement by Month & Category", font=dict(size=12,color="#6b4fa0")), coloraxis_colorbar=dict(tickfont=dict(color="#121112")))
        st.plotly_chart(fig3, use_container_width=True)

    sec("☁️ Trending Hashtags")
    if "Hashtags" in posts_pa.columns:
        import re
        from collections import Counter
        source_df = posts_pa
        try:
            if 'handle_posts' in locals() and isinstance(handle_posts, pd.DataFrame) and len(handle_posts) > 0:
                source_df = handle_posts
        except Exception:
            source_df = posts_pa
        token_re = re.compile(r"#[-\w]+"); word_re = re.compile(r"[A-Za-z0-9_\-]{2,}")
        tags = []
        for val in source_df["Hashtags"].dropna().astype(str):
            s = val.strip(); found = token_re.findall(s); tags.extend(found)
            for part in re.split(r"[\s,;|/\\]+", s):
                if part and not part.startswith("#"):
                    for w in word_re.findall(part):
                        if len(w) > 1 and not w.isdigit(): tags.append("#" + w)
        tags = [t.lower() for t in tags if isinstance(t, str)]
        freqs = Counter(tags)
        if freqs:
            from collections import Counter as _C
            freqs_display = _C()
            for tag, cnt in freqs.items():
                freqs_display[tag[1:] if tag.startswith('#') else tag] += cnt
            freqs_top = dict(freqs_display.most_common(150))
            wc = WordCloud(width=1400, height=420, background_color="#0e0814", colormap="PuRd", prefer_horizontal=0.7, max_words=150)
            wc.generate_from_frequencies(freqs_top)
            fig_wc, ax = plt.subplots(figsize=(14,4))
            fig_wc.patch.set_facecolor("#0e0814"); ax.set_facecolor("#0e0814")
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            st.pyplot(fig_wc, use_container_width=True); plt.close(fig_wc)
        else:
            st.info("No hashtag data in current filter.")

    try:
        handle_exists = 'handle_posts' in locals() and isinstance(handle_posts, pd.DataFrame)
    except Exception:
        handle_exists = False
    if handle_exists and len(handle_posts) > 0:
        sec(f"All Posts by {handle}  ({len(handle_posts)} total)")
        show_cols = [c for c in ["Post_ID","Post_Date","Likes","Comments","Engagement","Hashtags"] if c in handle_posts.columns]
        disp = handle_posts.sort_values("Post_Date", ascending=False)[show_cols].copy()
        if "Post_Date" in disp.columns: disp["Post_Date"] = disp["Post_Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(disp.reset_index(drop=True), use_container_width=True, height=300)


# ==========================================================
# PAGE 4 — AI LEAD SCORING
# ==========================================================
elif page == "Lead Scoring":
    hq  = (leads["Lead_Quality"]=="high").sum()
    med = (leads["Lead_Quality"]=="medium").sum()
    low = (leads["Lead_Quality"]=="low").sum()
    tot = max(len(leads),1)

    st.markdown(f'<p style="color:#6b4fa0;font-size:13px;margin-bottom:1rem">AI-driven outreach prioritisation — score, segment, and act · <b style="color:#d8b4fe">{len(leads):,} records in view</b></p>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("🔥 High Priority",  fmt(hq),  f"{hq/tot*100:.1f}% · Contact now",   "#4ade80","#22c55e", int(hq/tot*100)), unsafe_allow_html=True)
    k2.markdown(kpi("⏳ Medium Priority", fmt(med), f"{med/tot*100:.1f}% · Nurture",       "#fbbf24","#f59e0b", int(med/tot*100)), unsafe_allow_html=True)
    k3.markdown(kpi("🗑 Low Priority",    fmt(low), f"{low/tot*100:.1f}% · Ignore",        "#f87171","#ef4444", int(low/tot*100)), unsafe_allow_html=True)
    k4.markdown(kpi("Avg Lead Score",    f"{leads['Lead_Score'].mean():.1f}", "out of 100 · current view", "#a855f7","#ec4899", int(leads['Lead_Score'].mean())), unsafe_allow_html=True)

    sec("Pipeline Funnel & Score Distribution")
    ch1, ch2 = st.columns([2, 3])

    with ch1:
        desc("<b>Pipeline Funnel</b> — the narrowing of your lead pool from all influencers → medium only → high only.")
        funnel = pd.DataFrame({"Stage":["All Leads","Medium","High Only"],"Count":[tot, med, hq]})
        fig = go.Figure(go.Funnel(y=funnel["Stage"], x=funnel["Count"], marker_color=["#2d1555","#fbbf24","#4ade80"], textinfo="value+percent initial", connector=dict(fillcolor="#0e0814")))
        dark(fig, 300)
        fig.update_layout(title=dict(text="Lead Pipeline Funnel", font=dict(size=12,color="#6b4fa0")))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        desc("<b>Score Distribution Histogram</b> — each bar shows influencer count per score range. KDE curve shows density. Dashed lines at 30 and 60 mark quality thresholds.")
        fig_score = go.Figure()
        palette = {"high":"#4ade80","medium":"#fbbf24","low":"#f87171"}
        nbins = 30; bin_edges = np.linspace(0, 100, nbins + 1); bin_width = bin_edges[1] - bin_edges[0]
        max_bin_count = 0; hist_counts = {}
        for quality in palette.keys():
            subset = leads[leads["Lead_Quality"]==quality]["Lead_Score"].dropna().to_numpy()
            if len(subset) > 0:
                counts, _ = np.histogram(subset, bins=bin_edges)
                hist_counts[quality] = (counts, subset)
                max_bin_count = max(max_bin_count, counts.max())
        for quality, color in palette.items():
            data = hist_counts.get(quality)
            if not data: continue
            counts, subset = data
            fig_score.add_trace(go.Histogram(x=subset, xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=bin_width), name=quality.title(), marker=dict(color=color, line=dict(color="#0e0814", width=1)), opacity=0.65, hovertemplate="<b>%{fullData.name}</b><br>Score: %{x:.1f}<br>Count: %{y}<extra></extra>"))
            if len(subset) >= 2 and np.unique(subset).size > 1:
                x_kde = np.linspace(0, 100, 300); kde = gaussian_kde(subset)
                y_scaled = kde(x_kde) * len(subset) * bin_width
                fig_score.add_trace(go.Scatter(x=x_kde, y=y_scaled, mode="lines", line=dict(color=color, width=2.6), name=f"{quality.title()} KDE", showlegend=False))
        fig_score.add_vline(x=30, line_dash="dash", line_color="#3d1f70", line_width=1)
        fig_score.add_vline(x=60, line_dash="dash", line_color="#3d1f70", line_width=1)
        dark(fig_score, 360)
        fig_score.update_layout(title=dict(text="Lead Score Distribution by Quality Tier", font=dict(size=12,color="#6b4fa0")), barmode="overlay", hovermode="closest", xaxis_title="Lead Score", yaxis_title="Count", yaxis=dict(range=[0, max(10, int(max_bin_count*1.15))]), legend=dict(orientation="h", y=1.05, x=1, xanchor="right"))
        for lbl, xpos, col in [("Low",15,"#f87171"),("Medium",45,"#fbbf24")]:
            fig_score.add_annotation(x=xpos, y=1.0, yref="paper", text=lbl, showarrow=False, font=dict(color=col, size=10, family="DM Mono"))
        st.plotly_chart(fig_score, use_container_width=True)

    sec("Score Distribution by Category")
    desc("<b>Violin Chart</b> — each violin shows the full lead score distribution per category. Wider = more influencers at that score. White box = IQR. White line = median.")
    if "Category_Name" in leads.columns:
        fig_v = go.Figure()
        for cat, color in CAT_CLR.items():
            subset = leads[leads["Category_Name"]==cat]["Lead_Score"].dropna()
            if len(subset) > 0:
                fig_v.add_trace(go.Violin(x=[cat]*len(subset), y=subset, name=cat, line_color=color, fillcolor=color, opacity=0.30, box_visible=True, meanline_visible=True, points=False))
        dark(fig_v, 300)
        fig_v.update_layout(title=dict(text="Lead Score Violin — spread and median per category", font=dict(size=12,color="#6b4fa0")), showlegend=False, violingap=0.25)
        st.plotly_chart(fig_v, use_container_width=True)

    sec("🎯 Priority Outreach List")
    desc("<b>Priority Outreach Table</b> — top influencers ranked by Lead Score. 🔥 Contact Now · ⏳ Nurture · 🗑 Ignore. Export as CSV for your sales team.")

    f1, f2, f3 = st.columns(3)
    cat_opts = ["All"] + (sorted(leads["Category_Name"].dropna().unique().tolist()) if "Category_Name" in leads.columns else [])
    fcat = f1.selectbox("Category", cat_opts, key="sc_cat")
    fact = f2.selectbox("Action", ["All","🔥 Contact Now","⏳ Nurture","🗑 Ignore"], key="sc_act")
    fn   = f3.selectbox("Show", [20,50,100], key="sc_n")

    pt = leads.copy()
    if fcat!="All" and "Category_Name" in pt.columns: pt = pt[pt["Category_Name"]==fcat]
    if fact!="All" and "Action"        in pt.columns: pt = pt[pt["Action"]==fact]

    show_c = [c for c in ["Handle","Category_Name","Lead_Score","Lead_Quality","Action","Follower_Count","Engagement_Rate","Avg_Sentiment"] if c in pt.columns]
    pt = pt.nlargest(fn,"Lead_Score")[show_c].rename(columns={"Category_Name":"Category","Lead_Score":"Score","Lead_Quality":"Quality","Follower_Count":"Followers","Engagement_Rate":"ER %","Avg_Sentiment":"Avg Sentiment"})
    if "Followers"     in pt.columns: pt["Followers"]     = pt["Followers"].apply(fmt)
    if "Score"         in pt.columns: pt["Score"]         = pt["Score"].round(1)
    if "ER %"          in pt.columns: pt["ER %"]          = pt["ER %"].round(2)
    if "Avg Sentiment" in pt.columns: pt["Avg Sentiment"] = pt["Avg Sentiment"].round(3)

    st.dataframe(pt.reset_index(drop=True), use_container_width=True, height=380)
    st.download_button("⬇️ Export as CSV", pt.to_csv(index=False).encode(), "priority_leads.csv","text/csv")


# ==========================================================
# PAGE 5 — AI INSIGHTS
# ==========================================================
elif page == "AI Insights":
    for key in ["ai_chat_history","ai_exec_summary","ai_post_result",
                "ai_content_result","ai_rec_result","ai_custom_result"]:
        if key not in st.session_state:
            st.session_state[key] = [] if key == "ai_chat_history" else ""

    groq_ok = _groq_client() is not None

    st.markdown(
    f'<div class="ai-page-heading">AI Insights</div>'
    f'<div class="ai-page-subheading"><b>Groq-powered AI Intelligence</b> — chat, analyse, and generate insights from your data '
    f'<span class="ai-status-badge">{"✅ Groq connected" if groq_ok else "⚠️ Groq not configured — add GROQ_API_KEY to .env or secrets.toml"}</span></div>',
    unsafe_allow_html=True)

    scope_choice = st.radio("Data scope",["All data","Current dashboard filters"],
                            index=0, horizontal=True)
    ai_leads = leads_full if scope_choice == "All data" else leads
    ai_posts = posts_full if scope_choice == "All data" else posts

    (tab_chat, tab_exec, tab_post,
     tab_content, tab_rec, tab_custom) = st.tabs([
        "💬 AI Chat", "📊 Executive Summary", "🔍 Post ID Analyser",
        "📝 Content Analyser", "🎯 Recommendations", "❓ Custom Question",
    ])

    with tab_chat:
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        desc("<b>AI Dashboard Chat</b> — ask anything about your influencer data. "
             "Mention a <b>Handle</b> or <b>Post ID</b> for drill-down detail. "
             "Conversation history is kept for this session.")

        if not st.session_state["ai_chat_history"]:
            st.markdown(
                '<div class="insight" style="--ac:#818cf8;text-align:center;padding:28px 20px;">'
                '<div style="font-size:22px;margin-bottom:8px">💬</div>'
                '<div style="font-size:13px;color:#9fb0d2">Start by asking a question below.<br>'
                'Try: <i>"Which category has the best engagement rate?"</i></div></div>',
                unsafe_allow_html=True)
        for msg in st.session_state["ai_chat_history"]:
            role    = msg["role"]
            content = str(msg["content"])
            if role == "user":
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end;margin-bottom:10px;">'
                    f'<div style="background:linear-gradient(135deg,#4f7cff22,#7c3aed22);'
                    f'border:1px solid #4f7cff44;border-radius:14px 14px 4px 14px;'
                    f'padding:10px 16px;max-width:75%;font-size:13px;color:#dbe7ff;">'
                    f'{html.escape(content)}</div></div>', unsafe_allow_html=True)
            else:
                escaped = html.escape(content).replace("\n","<br>")
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-start;margin-bottom:10px;">'
                    f'<div style="background:linear-gradient(135deg,#ffffff,#f5f8ff);'
                    f'border:1px solid #dbe6f5;border-left:3px solid #a855f7;'
                    f'border-radius:14px 14px 14px 4px;padding:12px 16px;max-width:82%;'
                    f'font-size:13px;color:#1e293b;line-height:1.75;'
                    f'box-shadow:0 4px 18px rgba(79,124,255,0.09);">'
                    f'<span style="font-size:10px;color:#a855f7;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:.06em;display:block;margin-bottom:6px">✨ InstaScribe AI</span>'
                    f'{escaped}</div></div>', unsafe_allow_html=True)

        cc1,cc2,cc3 = st.columns([5,1,1], gap="small")
        chat_input   = cc1.text_input("msg", label_visibility="collapsed",
                                       placeholder="Ask about your data, a handle, or a post ID…",
                                       key="ai_chat_input_field")
        send_clicked  = cc2.button("Send ➤", use_container_width=True, key="ai_chat_send")
        clear_clicked = cc3.button("🗑 Clear", use_container_width=True, key="ai_chat_clear")

        if clear_clicked:
            st.session_state["ai_chat_history"] = []; st.rerun()

        if send_clicked and chat_input.strip():
            user_msg = chat_input.strip()
            st.session_state["ai_chat_history"].append({"role":"user","content":user_msg})
            context_text, _, _ = _build_ai_context(user_msg, ai_leads, ai_posts)
            history_for_groq   = st.session_state["ai_chat_history"][-8:]
            sys_p = ("You are InstaScribe AI, an expert influencer intelligence assistant. "
                     "Answer using ONLY the dataset context. Be concise and actionable. "
                     "Use bullet points where helpful.")
            messages = [{"role":"system","content":sys_p}]
            for h in history_for_groq[:-1]:
                messages.append({"role":h["role"],"content":h["content"]})
            messages.append({"role":"user","content":f"{user_msg}\n\nDataset context:\n{context_text}"})
            client = _groq_client()
            if client is None:
                answer = f"⚠️ Groq not configured.\n\nDataset context:\n{context_text}"
            else:
                try:
                    completion = client.chat.completions.create(
                        model=_groq_model_name(), temperature=0.35, max_tokens=1024, messages=messages)
                    answer = completion.choices[0].message.content
                except Exception as exc:
                    answer = f"Error calling Groq: {exc}"
            st.session_state["ai_chat_history"].append({"role":"assistant","content":answer})
            st.rerun()

        st.markdown('<div style="margin-top:10px;font-size:10px;color:#6b4fa0;text-transform:uppercase;letter-spacing:.07em">Quick prompts</div>', unsafe_allow_html=True)
        qp_cols = st.columns(4, gap="small")
        quick_prompts = [
            "Which category has the best engagement rate?",
            "Who are the top 5 high-priority leads?",
            "What is the overall sentiment trend?",
            "Which follower tier has the most leads?",
        ]
        for i, qp in enumerate(quick_prompts):
            if qp_cols[i].button(qp, key=f"qp_{i}", use_container_width=True):
                st.session_state["ai_chat_history"].append({"role":"user","content":qp})
                ctx, _, _ = _build_ai_context(qp, ai_leads, ai_posts)
                sys_p = "You are InstaScribe AI. Answer using only the dataset context. Be concise."
                answer, err = _call_groq(sys_p, f"{qp}\n\nDataset context:\n{ctx}")
                st.session_state["ai_chat_history"].append({"role":"assistant","content":err if err else answer})
                st.rerun()

    with tab_exec:
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        desc("<b>Executive Summary</b> — one-click AI briefing: lead quality, top categories, engagement health, and key actions.")
        if st.button("📊 Generate Executive Summary", key="exec_gen_btn"):
            ctx, _, _ = _build_ai_context("executive summary", ai_leads, ai_posts)
            sys_p = ("You are InstaScribe AI. Write a structured executive summary. "
                     "Include: 1) Overall health, 2) Top categories, 3) Lead pipeline, "
                     "4) Sentiment overview, 5) Top 3 recommended actions. Use clear section headers.")
            answer, err = _call_groq(sys_p, f"Generate executive summary.\n\nDataset:\n{ctx}")
            st.session_state["ai_exec_summary"] = err if err else answer

        if st.session_state["ai_exec_summary"]:
            _render_ai_panel("✨ AI Executive Summary", st.session_state["ai_exec_summary"], accent="#a855f7")
            

    # ── REPLACE the entire "with tab_post:" block with this ──────────────────

    with tab_post:
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        desc(
            "<b>Post ID Analyser</b> — breakdown of a <i>single post</i>. "
            "How did it perform vs this creator's average? Was the sentiment strong? "
            "Did the hashtags pull their weight?"
        )

        all_post_ids = (
            sorted(ai_posts["Post_ID"].dropna().astype(str).unique().tolist())
            if "Post_ID" in ai_posts.columns else []
        )
        filtered_pids = all_post_ids

        # ── Post ID dropdown ──────────────────────────────────────────────
        with st.form("post_analyser_form", clear_on_submit=False):
            if filtered_pids:
                sel_pid_tab = st.selectbox(
                    f"📋 Post ID ({len(filtered_pids):,} available — type to search)",
                    filtered_pids,
                    key="ai_tab_post_sel",
                )
            else:
                sel_pid_tab = None
                st.warning("No Post IDs found for the selected handle.")
            analyse_clicked = st.form_submit_button("🔬 Analyse This Post")

        post_id_val = sel_pid_tab or ""

        # ── Post snapshot metrics ─────────────────────────────────────────
        if post_id_val and "Post_ID" in ai_posts.columns:
            raw_row = ai_posts[
                ai_posts["Post_ID"].astype(str).str.strip().str.lower()
                == post_id_val.strip().lower()
            ]
            if len(raw_row) > 0:
                pr = raw_row.iloc[0]
                handle_of_post = str(pr.get("Handle", ""))

                

                h_posts_all = pd.DataFrame()
                if handle_of_post and "Handle" in ai_posts.columns:
                    h_posts_all = ai_posts[
                        ai_posts["Handle"].astype(str).str.strip().str.lower()
                        == handle_of_post.strip().lower()
                    ]
                    avg_eng     = h_posts_all["Engagement"].mean() if len(h_posts_all) > 0 else 0
                    this_eng    = safe_float(pr.get("Engagement", 0))
                    delta_pct   = ((this_eng - avg_eng) / max(avg_eng, 1)) * 100
                    delta_color = "#4ade80" if delta_pct >= 0 else "#f87171"
                    delta_sign  = "+" if delta_pct >= 0 else ""
                    
                st.markdown('<div style="margin-bottom:8px"></div>', unsafe_allow_html=True)

                if handle_of_post and len(h_posts_all) > 1 and "Engagement" in h_posts_all.columns:
                    this_eng   = safe_float(pr.get("Engagement", 0))
                    rank       = int((h_posts_all["Engagement"] > this_eng).sum() + 1)
                    total_hp   = len(h_posts_all)
                    pct_rank   = (1 - rank / total_hp) * 100
                    rank_color = "#4ade80" if pct_rank >= 66 else ("#fbbf24" if pct_rank >= 33 else "#f87171")
                    st.markdown(
                        f'<div class="insight" style="--ac:{rank_color};margin-bottom:12px">'
                        f'📊 This post ranks <b style="color:{rank_color}">#{rank} of {total_hp}</b> '
                        f'posts by @{handle_of_post} — top <b>{100 - int(pct_rank):.0f}%</b>. '
                        f'{"🔥 Standout post." if pct_rank >= 66 else ("📈 Above average." if pct_rank >= 33 else "📉 Below average.")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        if analyse_clicked:
            if not post_id_val:
                st.warning("Please select a Post ID first.")
            else:
                post_ctx  = _profile_context_for_post(post_id_val.strip(), ai_posts)
                extra_ctx = ""
                pr2 = ai_posts[
                    ai_posts["Post_ID"].astype(str).str.strip().str.lower()
                    == post_id_val.strip().lower()
                ]
                if len(pr2) > 0:
                    h2 = str(pr2.iloc[0].get("Handle", ""))
                    if h2 and "Handle" in ai_posts.columns:
                        h2_posts = ai_posts[
                            ai_posts["Handle"].astype(str).str.strip().str.lower()
                            == h2.strip().lower()
                        ]
                        if len(h2_posts) > 0:
                            extra_ctx = (
                                f"\n\nHandle averages for @{h2} across {len(h2_posts)} posts:\n"
                                f"Avg Likes: {h2_posts['Likes'].mean():.0f}\n"
                                f"Avg Comments: {h2_posts['Comments'].mean():.0f}\n"
                                f"Avg Engagement: {h2_posts['Engagement'].mean():.0f}\n"
                            )
                            if "Sentiment_Score" in h2_posts.columns:
                                extra_ctx += f"Avg Sentiment: {h2_posts['Sentiment_Score'].mean():.3f}\n"

                sys_p = (
                    "You are InstaScribe AI performing SINGLE-POST FORENSICS. "
                    "Your job is to diagnose why this specific post performed the way it did. "
                    "Structure your response with these exact sections:\n"
                    "1) POST VERDICT — one sentence: was this post a hit, average, or miss?\n"
                    "2) PERFORMANCE VS AVERAGE — compare this post's numbers to the handle's averages. Use exact % differences.\n"
                    "3) HASHTAG ANALYSIS — which hashtags likely helped or hurt reach?\n"
                    "4) SENTIMENT SIGNAL — what does the sentiment score tell us about audience reaction?\n"
                    "5) ONE ACTIONABLE FINDING — what should the creator do differently or repeat?\n"
                    "Be specific. Use exact numbers. Keep each section to 2-3 sentences."
                )
                answer, err = _call_groq(
                    sys_p,
                    f"Analysis of post {post_id_val}:\n{post_ctx}{extra_ctx}",
                )
                st.session_state["ai_post_result"] = err if err else answer

        if st.session_state["ai_post_result"]:
            _handle_label = ""
            _pr_check = ai_posts[ai_posts["Post_ID"].astype(str).str.strip().str.lower() == post_id_val.strip().lower()]
            if len(_pr_check) > 0:
                 _handle_label = str(_pr_check.iloc[0].get("Handle", ""))


            _render_ai_panel(
                f"🔬 Post Analysis of — {post_id_val}  ·  {_handle_label}" if _handle_label else f"🔬 Post Analysis — {post_id_val}",
                st.session_state["ai_post_result"],
                accent="#364af8",
                margin_top=True,
            )
            if post_id_val and "Post_ID" in ai_posts.columns:
                raw_row2 = ai_posts[
                    ai_posts["Post_ID"].astype(str).str.strip().str.lower()
                    == post_id_val.strip().lower()
                ]
                if len(raw_row2) > 0:
                    with st.expander("Raw post record", expanded=False):
                        show_cols = [c for c in [
                            "Post_ID", "Handle", "Post_Date", "Likes",
                            "Comments", "Engagement", "Sentiment_Score",
                            "Hashtags", "Category_Name", "Lead_Score",
                        ] if c in raw_row2.columns]
                        st.dataframe(
                            raw_row2[show_cols].reset_index(drop=True),
                            use_container_width=True, hide_index=True,
                        )
        else:
            st.markdown(
                '<div class="insight" style="--ac:#818cf8;text-align:center;padding:32px 20px;">'
                '<div style="font-size:28px;margin-bottom:10px">🔬</div>'
                '<div style="font-size:13px;color:#9fb0d2">'
                'Type a handle name above to filter, or go straight to picking a '
                '<b style="color:#f4f7ff">Post ID</b> and click '
                '<b style="color:#f4f7ff">Analyse This Post</b>.'
                '</div></div>',
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════════
    # TAB: CONTENT STRATEGY ANALYSER  — full creator profile
    # ════════════════════════════════════════════════════════════════
    with tab_content:
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        desc(
            "<b>Content Strategy Analyser</b> — macro-level creator intelligence across "
            "<i>all their posts</i>. Posting patterns, hashtag DNA, engagement trajectory, "
            "and whether this creator is worth a partnership."
        )

        # ── Step 1: Category pre-filter ───────────────────────────────────
        cat_opts_content = [""] + (
            sorted(ai_leads["Category_Name"].dropna().astype(str).unique().tolist())
            if "Category_Name" in ai_leads.columns else []
        )
        sel_cat_content = st.selectbox(
            "📂 Filter by Category (optional — type to search)",
            cat_opts_content,
            key="ai_content_cat_filter",
            format_func=lambda x: "— all categories —" if x == "" else x,
        )

        # ── Step 2: Handle list filtered by category ──────────────────────
        if sel_cat_content and "Category_Name" in ai_leads.columns:
            cat_mask = (
                ai_leads["Category_Name"].astype(str).str.strip().str.lower()
                == sel_cat_content.strip().lower()
            )
            handles_in_cat = sorted(
                ai_leads.loc[cat_mask, "Handle"].dropna().astype(str).unique().tolist()
            )
        else:
            handles_in_cat = (
                sorted(ai_leads["Handle"].dropna().astype(str).unique().tolist())
                if "Handle" in ai_leads.columns else []
            )

        if "Handle" in ai_posts.columns:
            post_handles = set(ai_posts["Handle"].dropna().astype(str).unique().tolist())
            extra_handles = sorted(post_handles - set(handles_in_cat))
            all_content_handles = [""] + handles_in_cat + extra_handles
        else:
            all_content_handles = [""] + handles_in_cat

        if sel_cat_content:
            st.markdown(
                f'<div style="font-size:12px;color:#6b4fa0;margin-bottom:4px;">'
                f'{len(handles_in_cat):,} handle(s) in <b>{sel_cat_content}</b></div>',
                unsafe_allow_html=True,
            )

        # ── Step 3: Handle dropdown ───────────────────────────────────────
        content_handle = st.selectbox(
            f"👤 Handle ( type to search)",
            all_content_handles,
            key="ai_content_handle_select",
            format_func=lambda x: "— choose a handle —" if x == "" else x,
        )

        # ── Creator summary stats across ALL posts ────────────────────────
        if content_handle and "Handle" in ai_posts.columns:
            h_all = ai_posts[
                ai_posts["Handle"].astype(str).str.strip().str.lower()
                == content_handle.strip().lower()
            ].copy()

            if len(h_all) > 0:
                total_posts_h  = len(h_all)
                avg_likes_h    = h_all["Likes"].mean()    if "Likes"    in h_all.columns else 0
                avg_comments_h = h_all["Comments"].mean() if "Comments" in h_all.columns else 0
                Likes     = h_all["Likes"].sum()    if "Likes"    in h_all.columns else 0
                Followers= h_all["Follower_Count"].max() if "Follower_Count" in h_all.columns else 0
                best_post_eng  = h_all["Engagement"].max()
                

                
                st.markdown('<div style="margin-bottom:4px"></div>', unsafe_allow_html=True)

                # ── Engagement trajectory sparkline ───────────────────────
                if "Post_Date" in h_all.columns and len(h_all) > 2:
                    h_sorted = h_all.sort_values("Post_Date")
                    monthly_h = h_sorted.groupby("Month")["Engagement"].sum().reset_index()
                    if len(monthly_h) > 1:
                        sec(f"📈 Engagement Trajectory — @{content_handle}")
                        # Trend direction
                        first_half  = monthly_h["Engagement"].iloc[:len(monthly_h)//2].mean()
                        second_half = monthly_h["Engagement"].iloc[len(monthly_h)//2:].mean()
                        trend_pct   = ((second_half - first_half) / max(first_half, 1)) * 100
                        trend_color = "#4ade80" if trend_pct >= 0 else "#f87171"
                        trend_label = f"{'📈' if trend_pct >= 0 else '📉'} {'+' if trend_pct >= 0 else ''}{trend_pct:.1f}% vs first half of history"

                        fig_traj = go.Figure()
                        fig_traj.add_trace(go.Scatter(
                            x=monthly_h["Month"], y=monthly_h["Engagement"],
                            mode="lines+markers",
                            line=dict(color="#4ade80", width=2.5),
                            marker=dict(size=6, color="#22c55e"),
                            fill="tozeroy", fillcolor="rgba(74,222,128,0.07)",
                            name="Monthly Engagement",
                        ))
                        # Rolling average
                        if len(monthly_h) >= 3:
                            monthly_h["Rolling"] = monthly_h["Engagement"].rolling(3, min_periods=1).mean()
                            fig_traj.add_trace(go.Scatter(
                                x=monthly_h["Month"], y=monthly_h["Rolling"],
                                mode="lines",
                                line=dict(color="#fbbf24", width=1.5, dash="dot"),
                                name="3-month avg",
                            ))
                        dark(fig_traj, 240)
                        fig_traj.update_layout(
                            title=dict(text=f"Monthly Engagement  ·  {trend_label}", font=dict(size=12, color=trend_color)),
                            legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
                        )
                        st.plotly_chart(fig_traj, use_container_width=True)

                
        # ── Analyse button ────────────────────────────────────────────────
        if st.button("📝 Analyse Content Strategy", key="ai_content_analyse_btn"):
            if not content_handle:
                st.warning("Please select a handle first.")
            else:
                hctx = _profile_context_for_handle(content_handle, ai_leads, ai_posts)

                # Build richer context: top/bottom posts, posting cadence
                extra_content_ctx = ""
                if "Handle" in ai_posts.columns:
                    h_ctx_posts = ai_posts[
                        ai_posts["Handle"].astype(str).str.strip().str.lower()
                        == content_handle.strip().lower()
                    ].copy()
                    if len(h_ctx_posts) > 0:
                        top3 = h_ctx_posts.nlargest(3, "Engagement")[["Post_ID","Engagement","Likes","Comments"]].to_string(index=False)
                        bot3 = h_ctx_posts.nsmallest(3, "Engagement")[["Post_ID","Engagement","Likes","Comments"]].to_string(index=False)
                        if "Post_Date" in h_ctx_posts.columns:
                            h_ctx_posts["Post_Date"] = pd.to_datetime(h_ctx_posts["Post_Date"], errors="coerce")
                            h_ctx_posts["DayOfWeek"] = h_ctx_posts["Post_Date"].dt.day_name()
                            best_day = h_ctx_posts.groupby("DayOfWeek")["Engagement"].mean().idxmax()
                            extra_content_ctx += f"\nBest posting day (avg engagement): {best_day}\n"
                        extra_content_ctx += f"\nTop 3 posts by engagement:\n{top3}\n"
                        extra_content_ctx += f"\nBottom 3 posts by engagement:\n{bot3}\n"

                sys_p = (
                    "You are InstaScribe AI performing a CREATOR CONTENT STRATEGY AUDIT. "
                    "This is NOT about a single post — it's about the creator's overall content DNA. "
                    "Structure your response with these exact sections:\n"
                    "1) CREATOR VERDICT — is this creator worth partnering with? One sentence.\n"
                    "2) ENGAGEMENT PATTERN — how consistent is their engagement? Growing or declining? Use numbers.\n"
                    "3) CONTENT DNA — what types of posts (based on hashtags) drive the most engagement for them?\n"
                    "4) AUDIENCE BEHAVIOUR — what does the likes-to-comments ratio tell us about their audience?\n"
                    "5) BEST POSTING STRATEGY — when should they post and what content format works best?\n"
                    "6) PARTNERSHIP RECOMMENDATION — concrete yes/no with budget tier suggestion and campaign angle.\n"
                    "Be strategic and specific. Use exact numbers. Think like a CMO."
                )
                answer, err = _call_groq(
                    sys_p,
                    f"Full content strategy audit for @{content_handle}:\n{hctx}{extra_content_ctx}",
                )
                st.session_state["ai_content_result"] = err if err else answer

        if st.session_state["ai_content_result"]:
            _render_ai_panel(
                f"📝 Content Strategy Audit — {content_handle or ''}",
                st.session_state["ai_content_result"],
                accent="#fd7e00",
            )
        else:
            st.markdown(
                '<div class="insight" style="--ac:#4ade80;text-align:center;padding:32px 20px;">'
                '<div style="font-size:28px;margin-bottom:10px">📝</div>'
                '<div style="font-size:13px;color:#9fb0d2">'
                'Filter by Category, pick a Handle, then click '
                '<b style="color:#f4f7ff">Analyse Content Strategy</b> '
                'for a full creator audit.'
                '</div></div>',
                unsafe_allow_html=True,
            )

    with tab_rec:
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        desc("<b>AI Recommendations</b> — action-oriented outreach and campaign strategy based on your live lead pipeline.")
        rec_focus = st.selectbox("Recommendation focus",
            ["Full Outreach Strategy","Top Handles to Contact Now",
             "Nurture Pipeline","Category Priority","Risk & Watch List"],
            key="ai_rec_focus")
        if st.button("🎯 Generate Recommendations", key="ai_rec_gen_btn"):
            ctx, _, _ = _build_ai_context(rec_focus, ai_leads, ai_posts)
            focus_prompts = {
                "Full Outreach Strategy":      "Provide complete outreach strategy: top 5 to contact now, top 5 to nurture, category priorities, timing recommendations.",
                "Top Handles to Contact Now":  "List top 10 handles to contact immediately. For each: reason, outreach angle, risks.",
                "Nurture Pipeline":            "Analyse medium-quality leads. Which are closest to high-priority? What signals should trigger outreach?",
                "Category Priority":           "Rank all categories by outreach priority with ER, lead quality split, score, and campaign recommendation.",
                "Risk & Watch List":           "Identify influencers with risk signals: declining engagement, negative sentiment, mass-following, score vs follower mismatch.",
            }
            sys_p = ("You are InstaScribe AI, an expert influencer marketing strategist. "
                     "Provide specific, actionable recommendations using exact numbers from context.")
            answer, err = _call_groq(sys_p, f"{focus_prompts.get(rec_focus,'')}\n\nDataset:\n{ctx}")
            st.session_state["ai_rec_result"] = err if err else answer

        if st.session_state["ai_rec_result"]:
            _render_ai_panel(f"🎯 {rec_focus}", st.session_state["ai_rec_result"], accent="#fbbf24")
            st.download_button("⬇️ Export as TXT",
                data=str(st.session_state["ai_rec_result"]).encode(),
                file_name=f"recommendation_{rec_focus.lower().replace(' ','_')}.txt",
                mime="text/plain", key="ai_rec_export")
        else:
            st.markdown('<div class="insight" style="--ac:#fbbf24;text-align:center;padding:32px 20px;"><div style="font-size:28px;margin-bottom:10px">🎯</div><div style="font-size:13px;color:#9fb0d2">Choose a focus and click <b style="color:#f4f7ff">Generate Recommendations</b>.</div></div>', unsafe_allow_html=True)

    with tab_custom:
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        desc("<b>Custom Question</b> — ask anything specific. Press <b>Enter</b> or click "
             "<b>Ask AI</b>. Supports ranked lists, category comparisons, outreach emails, and more.")

        with st.form("custom_question_form", clear_on_submit=False):
            custom_q = st.text_area(
                "Your question",
                placeholder=(
                    "Examples:\n"
                    "• Generate top 20 influencers with high leads for fashion category\n"
                    "• What is the engagement rate of @handlename?\n"
                    "• Compare Tech vs Fashion lead quality\n"
                    "• Generate an outreach email for the top lead\n"
                    "• Which handles have the best engagement-to-follower ratio?"
                ),
                height=130,
                key="ai_custom_q_input",
            )
            ask_clicked = st.form_submit_button("✨ Ask AI")

        if ask_clicked:
            if not custom_q.strip():
                st.warning("Please enter a question first.")
            else:
                question = custom_q.strip()
                ctx, p_matches, h_matches = _build_ai_context(question, ai_leads, ai_posts)
                q_lower = question.lower()
                extra_blocks = []

                # Detect top-N request
                n_match = re.search(r'\btop[- ]?(\d+)\b', q_lower)
                n_count = int(n_match.group(1)) if n_match else 20

                # Detect category filter
                cat_filter = None
                if "Category_Name" in ai_leads.columns:
                    for _cat in ai_leads["Category_Name"].dropna().unique():
                        if str(_cat).lower() in q_lower:
                            cat_filter = _cat
                            break

                # Detect quality filter
                quality_filter = None
                if any(kw in q_lower for kw in ["high lead","high quality","high priority"]):
                    quality_filter = "high"
                elif any(kw in q_lower for kw in ["medium lead","medium quality"]):
                    quality_filter = "medium"
                elif any(kw in q_lower for kw in ["low lead","low quality"]):
                    quality_filter = "low"

                # Build the requested subset for list questions
                subset_df = ai_leads.copy()
                if cat_filter and "Category_Name" in subset_df.columns:
                    subset_df = subset_df[
                        subset_df["Category_Name"].astype(str).str.lower() == str(cat_filter).lower()
                    ]
                if quality_filter and "Lead_Quality" in subset_df.columns:
                    subset_df = subset_df[subset_df["Lead_Quality"] == quality_filter]
                if "Lead_Score" in subset_df.columns:
                    subset_df = subset_df.nlargest(n_count, "Lead_Score")

                show_cols = [c for c in [
                    "Handle","Category_Name","Lead_Score","Lead_Quality",
                    "Follower_Count","Engagement_Rate","Avg_Sentiment","Action",
                ] if c in subset_df.columns]
                if len(subset_df) > 0:
                    tbl_str = subset_df[show_cols].to_string(index=False)
                    filter_desc = []
                    if cat_filter:     filter_desc.append(f"category={cat_filter}")
                    if quality_filter: filter_desc.append(f"quality={quality_filter}")
                    label = f"Top {n_count} influencers" + (
                        f" ({', '.join(filter_desc)})" if filter_desc else ""
                    )
                    extra_blocks.append(f"{label}:\n{tbl_str}")

                # Inject category summary when relevant
                if "category" in q_lower and "Category_Name" in ai_leads.columns:
                    cat_summary = (
                        ai_leads.groupby("Category_Name").agg(
                            Count=("Handle","count"),
                            Avg_Lead_Score=("Lead_Score","mean"),
                            Avg_ER=("Engagement_Rate","mean"),
                            High_Leads=("Lead_Quality", lambda x: (x=="high").sum()),
                        ).round(2).to_string()
                    )
                    extra_blocks.append(f"Category summary:\n{cat_summary}")

                full_context = ctx
                if extra_blocks:
                    full_context += "\n\n---\n\n" + "\n\n---\n\n".join(extra_blocks)

                sys_p = (
                    "You are InstaScribe AI, an expert influencer marketing analyst. "
                    "Answer the user's question using ONLY the dataset context and tables provided. "
                    "When asked for a list or top-N, output a clear numbered list "
                    "with the exact handles, scores, and metrics from the supplied table. "
                    "When asked to compare categories, use the category summary table. "
                    "When asked to generate an email or copy, write it in full. "
                    "Be specific, use exact numbers, and be actionable. "
                    "If information is not in the dataset, say so clearly."
                )
                answer, err = _call_groq(
                    sys_p,
                    f"Question: {question}\n\nDataset context:\n{full_context}",
                )
                st.session_state["ai_custom_result"] = err if err else answer
                st.session_state["ai_custom_matches"] = {"posts": p_matches, "handles": h_matches}

        if st.session_state["ai_custom_result"]:
            _render_ai_panel(
                "✨ AI Answer",
                st.session_state["ai_custom_result"],
                accent="#ec4899",
                margin_top=True,
            )
            matches = st.session_state.get("ai_custom_matches", {})
            if matches.get("posts") or matches.get("handles"):
                with st.expander("Dataset records used", expanded=False):
                    if matches.get("posts"):
                        st.markdown(f"**Post IDs:** {', '.join(matches['posts'])}")
                    if matches.get("handles"):
                        st.markdown(f"**Handles:** {', '.join(matches['handles'])}")
        else:
            st.markdown(
                '<div class="insight" style="--ac:#ec4899;text-align:center;padding:32px 20px;">'
                '<div style="font-size:28px;margin-bottom:10px">❓</div>'
                '<div style="font-size:13px;color:#9fb0d2">Type your question and press '
                '<b style="color:#f4f7ff">Enter</b> or click '
                '<b style="color:#f4f7ff">✨ Ask AI</b>.</div></div>',
                unsafe_allow_html=True,
            )

        sec("💡 Suggested Questions")
        sg1, sg2, sg3 = st.columns(3)
        suggestions = [
            ("Top 20 high-lead influencers in fashion category", "#818cf8"),
            ("Compare Tech vs Fashion lead quality",             "#4ade80"),
            ("Generate an outreach email for the top lead",      "#fbbf24"),
            ("Which handles have the best engagement rate?",     "#f87171"),
            ("Which handles have declining engagement?",         "#c084fc"),
            ("Best hashtag strategy for Fitness category?",      "#ec4899"),
        ]
        sg_cols = [sg1, sg2, sg3, sg1, sg2, sg3]
        for i, (sug, col) in enumerate(suggestions):
            if sg_cols[i].button(sug, key=f"sg_btn_{i}", use_container_width=True):
                st.session_state["ai_custom_q_input"] = sug
                _ctx, _pm, _hm = _build_ai_context(sug, ai_leads, ai_posts)
                _ans, _err = _call_groq(
                    "You are InstaScribe AI. Answer using only the dataset context. Be specific.",
                    f"Question: {sug}\n\nDataset context:\n{_ctx}",
                )
                st.session_state["ai_custom_result"] = _err if _err else _ans
                st.session_state["ai_custom_matches"] = {"posts": _pm, "handles": _hm}
                st.rerun()


elif page == "Subscription":
    render_subscription_page()

# ==========================================================
# PAGE 6 — ABOUT
# ==========================================================
elif page == "About":
    total_inf   = len(leads_full)
    total_posts = len(posts_full)
    cats_count  = leads_full["Category_Name"].nunique() if "Category_Name" in leads_full.columns else 5

    def _safe_csv_rows(name):
        p = os.path.join(_data_dir(), name)
        if not os.path.exists(p): return 0
        try: return len(pd.read_csv(p))
        except Exception: return 0

    date_dim_rows     = _safe_csv_rows("date_dim.csv")
    lead_scoring_rows = _safe_csv_rows("lead_scoring.csv")

    st.markdown(f"""
    <div class="about-hero">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px;">
        <div>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <div style="width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#a855f7,#ec4899);display:flex;align-items:center;justify-content:center;">{FAVICON_IMG}</div>'            <div>
              <div style="font-size:1.5rem;font-weight:700;color:#f0e6ff">InstaScribe</div>
              <div style="font-size:10px;color:#6b4fa0;text-transform:uppercase;letter-spacing:.12em">Creator Intelligence Platform</div>
            </div>
          </div>
          <div style="font-size:13px;color:#9b7ec8;max-width:640px;line-height:1.7;">
            InstaScribe is an <b style="color:#e8d5ff">AI-powered influencer intelligence platform</b>
            built with Streamlit and Python. It helps founders, agencies, and marketing teams detect
            high-quality influencer leads, analyse engagement authenticity, track individual posts, and
            prioritise outreach — all powered by real CSV data.
          </div>
        </div>
                <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start;">
                    <div style="background:linear-gradient(135deg,#ffffff,#f4f7ff);border:1px solid #dbe6f5;border-left:3px solid #4f7cff;border-radius:12px;padding:14px 20px;text-align:center;box-shadow:0 10px 24px rgba(15,23,42,0.05);">
                        <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:700;color:#2563eb">{total_inf:,}</div>
                        <div style="font-size:10px;color:#4b5e80;text-transform:uppercase;letter-spacing:.5px;margin-top:3px;font-weight:700">Influencers</div>
                    </div>
                    <div style="background:linear-gradient(135deg,#ffffff,#f8fafc);border:1px solid #dbe6f5;border-left:3px solid #ec4899;border-radius:12px;padding:14px 20px;text-align:center;box-shadow:0 10px 24px rgba(15,23,42,0.05);">
                        <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:700;color:#db2777">{total_posts:,}</div>
                        <div style="font-size:10px;color:#4b5e80;text-transform:uppercase;letter-spacing:.5px;margin-top:3px;font-weight:700">Posts</div>
                    </div>
                    <div style="background:linear-gradient(135deg,#ffffff,#f8fafc);border:1px solid #dbe6f5;border-left:3px solid #22c55e;border-radius:12px;padding:14px 20px;text-align:center;box-shadow:0 10px 24px rgba(15,23,42,0.05);">
                        <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:700;color:#16a34a">{cats_count}</div>
                        <div style="font-size:10px;color:#4b5e80;text-transform:uppercase;letter-spacing:.5px;margin-top:3px;font-weight:700">Categories</div>
                    </div>
                </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="about-card about-card-blue">
          <div class="about-title">📊 Dashboard Sections</div>
          <hr class="about-divider">
          <div class="about-li"><div class="about-li-dot" style="--dot:#818cf8"></div><span><b style="color:#e8d5ff">Executive Overview</b> — KPIs, smart insights, engagement trend, quality donut, category comparison</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#4ade80"></div><span><b style="color:#e8d5ff">Lead Intelligence</b> — Followers vs ER scatter, Avg Accounts Following bars, tier stack, ER chart, ranked table</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#c084fc"></div><span><b style="color:#e8d5ff">Post Analytics</b> — Post Inspector (search by ID or Handle) + monthly engagement, quadrant scatter, heatmap, hashtag cloud</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#f87171"></div><span><b style="color:#e8d5ff">Lead Scoring</b> — Pipeline funnel, score histogram with KDE, violin chart, priority outreach export</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#fbbf24"></div><span><b style="color:#e8d5ff">AI Insights</b> — Groq-powered Q&A for whole-system insights plus Post ID and Handle lookups</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card about-card-green">
          <div class="about-title">🔑 Key Features</div>
          <hr class="about-divider">
          <div class="about-li"><div class="about-li-dot" style="--dot:#4ade80"></div><span><b style="color:#e8d5ff">Multi Filter Sidebar</b> category, quality, ER, followers, tier, date, year</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#4ade80"></div><span><b style="color:#e8d5ff">Engagement Quality KPI</b> — shown as Positive / Neutral / Negative</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#4ade80"></div><span><b style="color:#e8d5ff">Post Inspector</b> — search any Post ID or Handle with ★ highlight</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#4ade80"></div><span>Priority outreach list with one-click <b style="color:#e8d5ff">CSV export</b></span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#4ade80"></div><span>Auth & sessions backed by <b style="color:#e8d5ff">Supabase</b> — persistent across reboots</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div class="about-card about-card-amber">
          <div class="about-title">🗂 Data Model</div>
          <hr class="about-divider">
          <div class="about-li"><div class="about-li-dot" style="--dot:#fbbf24"></div><span><span class="about-tag">influencer_master.csv</span> — {total_inf:,} records · core profiles</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#fbbf24"></div><span><span class="about-tag">post_metrics.csv</span> — {total_posts:,} records · likes, comments, hashtags</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#fbbf24"></div><span><span class="about-tag">category_dim.csv</span> — {cats_count} categories · SaaS relevance weights</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#fbbf24"></div><span><span class="about-tag">date_dim.csv</span> — {date_dim_rows:,} records · calendar attributes</span></div>
          <div class="about-li"><div class="about-li-dot" style="--dot:#fbbf24"></div><span><span class="about-tag">lead_scoring.csv</span> — {lead_scoring_rows:,} records · scored leads</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="about-card about-card-pink">
          <div class="about-title">🤖 Lead Score Formula</div>
          <hr class="about-divider">
          <div style="font-family:'DM Mono',monospace;font-size:11.5px;color:#9b7ec8;background:#FFFFFF;border:0px solid #2d1555;border-radius:8px;padding:14px 16px;line-height:2;">
            <span style="color:#3d1f70"># Weighted raw score</span><br>
            ER_score   = Engagement_Rate × <span style="color:#818cf8">0.40</span><br>
            Fol_score  = clip(Followers, 2M) / 2M × <span style="color:#4ade80">30</span><br>
            Sent_score = norm(Avg_Sentiment) × <span style="color:#c084fc">20</span><br>
            SaaS_score = SaaS_Relevance × <span style="color:#fbbf24">10</span><br>
            raw = (er_score + fol_score + sent_score + saas_score)<br><br>
            <span style="color:#3d1f70"># Normalise 0 → 100</span><br>
            Score = (raw − min) / (max − min) × <span style="color:#ec4899">100</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:11px;font-family:\'DM Mono\',monospace;'
    'font-weight:bold;'
    'background:linear-gradient(90deg,#a855f7,#ec4899);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent">'
    'InstaScribe · Creator Intelligence · Streamlit + Supabase + Plotly</div>',
    unsafe_allow_html=True)