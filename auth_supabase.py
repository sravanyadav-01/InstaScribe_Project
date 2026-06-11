from supabase import create_client
import streamlit as st

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


def signup(email, password, username, full_name):
    response = supabase.auth.sign_up(
        {
            "email": email,
            "password": password
        }
    )

    if response.user:

        supabase.table("profiles").insert(
            {
                "id": response.user.id,
                "username": username,
                "full_name": full_name,
                "role": "member"
            }
        ).execute()

    return response


def login(email, password):

    return supabase.auth.sign_in_with_password(
        {
            "email": email,
            "password": password
        }
    )


def logout():

    return supabase.auth.sign_out()


def get_profile(user_id):

    return (
        supabase.table("profiles")
        .select("*")
        .eq("id", user_id)
        .single()
        .execute()
    )


def reset_password(email):

    return supabase.auth.reset_password_email(email)