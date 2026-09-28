# Deploy WaferScan on Streamlit Community Cloud (about 10 minutes)

This folder is complete: the website (`web/`), the model (`model_bundle/`), the code (`waferscan/`)
and `streamlit_app.py`, which shows the real website and answers its API calls.

## 1. Put the folder on GitHub
1. Sign in at https://github.com and click **New repository**.
2. Name it `waferscan`, choose **Public**, and click **Create repository**. Don't add a README.
3. On the new repository page, click **uploading an existing file**.
4. Open this folder on your computer, select **everything inside it** (Ctrl+A), and drag it onto the page.
   All files are under GitHub's 25 MB limit. Check that `.streamlit/config.toml` was included;
   if not, add it later with **Add file → Create new file**, named `.streamlit/config.toml`.
5. Click **Commit changes** and wait for the upload to finish.

## 2. Create the app
1. Go to https://share.streamlit.io and **Continue with GitHub**.
2. Click **Create app → Deploy a public app from GitHub**.
3. Repository: `<your-name>/waferscan` · Branch: `main` · Main file path: `streamlit_app.py`.
4. **Advanced settings → Python version: 3.11**.
5. Click **Deploy**. The first build takes 3–6 minutes while it installs the packages.

## 3. Check it
- The status in the top-right should say **MODEL READY**.
- Pick **Scratch → Run scan**, click a red die, and try **Upload** with a wafer image.
- Copy the app's URL (it ends in `.streamlit.app`). That is your public link.

## If something goes wrong
- **"Error installing requirements"**: open **Manage app → Logs**. It is almost always a Python version
  mismatch. Set 3.11 in **Settings → General** and **Reboot**.
- **Status says OFFLINE DEMO**: the model failed to load. Check that the `model_bundle/` folder is in the
  repository, then **Reboot**.
- **The app is asleep**: free apps sleep after 12 hours without visitors. Click **Yes, get this app back up**
  and wait about a minute. Open the link 5 minutes before you present.

## Run it on your laptop instead (backup for the presentation)
```
pip install -r requirements.txt
python -m uvicorn waferscan.api.main:app --port 8000
```
Open http://localhost:8000/site/ — no internet needed except for the fonts.
