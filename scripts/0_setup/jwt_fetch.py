



import os

import requests
from dotenv import load_dotenv

load_dotenv()

TB_URL = os.getenv("TB_URL")
USERNAME = os.getenv("TB_USERNAME")
PASSWORD = os.getenv("TB_PASSWORD")

if not all([TB_URL, USERNAME, PASSWORD]):
    raise ValueError(
        "Missing one or more required variables: TB_URL, TB_USERNAME, TB_PASSWORD."
    )

login_url = f"{TB_URL}/api/auth/login"
credentials = {"username": USERNAME, "password": PASSWORD}

try:
    response = requests.post(login_url, json=credentials, timeout=30)

    if response.status_code == 200:
        jwt_token = response.json().get("token")
        if jwt_token:
            print("Successfully obtained JWT.")
            print(jwt_token)
        else:
            print("Login succeeded, but token was not found in response.")
    else:
        print(f"Login failed: {response.status_code}")
        print(response.text)
except requests.RequestException as exc:
    print(f"Connection error: {exc}")
