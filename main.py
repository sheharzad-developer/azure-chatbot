"""Streamlit chatbot backed by an Azure AI Foundry deployment.

    python -m streamlit run main.py --server.port 8000 --server.address 0.0.0.0

Configuration is read from environment variables, or from a .env file next to
this script:
    AZURE_API_KEY      (required)
    AZURE_ENDPOINT     default: https://testp1.services.ai.azure.com/
    AZURE_DEPLOYMENT   default: gpt-4.1-mini

The API key stays on the Streamlit server; the browser never sees it.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import streamlit as st

SYSTEM_PROMPT = "You are a helpful, concise assistant."


def load_dotenv(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def stream_chat(endpoint, api_key, deployment, messages):
    """Send the conversation and yield the reply text as it streams in."""
    url = endpoint.rstrip("/") + "/openai/v1/chat/completions"
    body = json.dumps({"model": deployment, "messages": messages, "stream": True}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"api-key": api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if not chunk.get("choices"):
                continue
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content:
                yield content


def main():
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.environ.get("AZURE_API_KEY")
    endpoint = os.environ.get("AZURE_ENDPOINT", "https://testp1.services.ai.azure.com/")
    deployment = os.environ.get("AZURE_DEPLOYMENT", "gpt-4.1-mini")

    st.set_page_config(page_title="Chatbot", page_icon="💬")
    st.title("Chatbot")
    st.caption(f"Model: {deployment}")

    if not api_key:
        st.error("AZURE_API_KEY is not set. Add it to .env or your environment.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Type a message")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_chat(endpoint, api_key, deployment, messages))
        except urllib.error.HTTPError as e:
            st.error(f"Azure error {e.code}: {e.read().decode('utf-8', 'replace')}")
            st.session_state.messages.pop()
            return
        except urllib.error.URLError as e:
            st.error(f"Connection error: {e.reason}")
            st.session_state.messages.pop()
            return

    st.session_state.messages.append({"role": "assistant", "content": reply})


main()
