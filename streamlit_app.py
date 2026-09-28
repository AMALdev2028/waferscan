"""WaferScan on Streamlit Community Cloud — the real website (web/), not a Streamlit look-alike.

The page in web/ runs inside a Streamlit component. Its calls to /api/v1/... are forwarded here by
web/st_bridge.js and answered by the same FastAPI app (via TestClient), so predictions, verification,
routing and error messages are identical to the self-hosted site.
"""
import base64
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="WaferScan", page_icon="◉", layout="wide", initial_sidebar_state="collapsed")

# Let the page use the whole window: hide Streamlit's header, padding and footer.
st.markdown(
    """<style>
    header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
    .block-container {padding: 0 !important; max-width: 100% !important;}
    [data-testid="stAppViewContainer"], .stApp {background: #000;}
    iframe {display: block; border: 0;}
    </style>""",
    unsafe_allow_html=True,
)

WEB = Path(__file__).parent / "web"
waferscan_page = components.declare_component("waferscan", path=str(WEB))


@st.cache_resource(show_spinner="Loading the WaferScan model…")
def api_client():
    from fastapi.testclient import TestClient
    import waferscan.api.main as api
    api.predictor()                      # load the ONNX bundle once per server
    return TestClient(api.app)


def answer(req: dict) -> dict:
    """Run one request from the page through the FastAPI app."""
    client, path, method = api_client(), req["path"], req.get("method", "GET")
    try:
        if req.get("file"):
            f = req["file"]
            files = {f["field"]: (f["name"], base64.b64decode(f["b64"]), f["type"])}
            r = client.request(method, path, data=req.get("form") or {}, files=files)
        elif req.get("json") is not None:
            r = client.request(method, path, content=req["json"], headers={"Content-Type": "application/json"})
        else:
            r = client.request(method, path)
        return {"id": req["id"], "status": r.status_code, "body": r.text}
    except Exception as exc:             # never leave the page waiting
        return {"id": req["id"], "status": 500, "body": json.dumps({"detail": f"server error: {exc}"})}


api_client()                             # warm up before the page asks for anything
st.session_state.setdefault("response", None)
st.session_state.setdefault("last_id", None)

request = waferscan_page(response=st.session_state.response, key="waferscan", default=None)
if request and request.get("id") != st.session_state.last_id:
    st.session_state.last_id = request["id"]
    st.session_state.response = answer(request)
    st.rerun()
