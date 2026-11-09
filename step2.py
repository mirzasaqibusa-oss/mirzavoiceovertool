# MirZa Tools — launcher (hosted file)
import builtins, os, subprocess, urllib.request
if not getattr(builtins, '_ok', False):
    print('❌ Please run Step 1 (Verify Your Access) above first.')
else:
    from IPython.display import clear_output, display, HTML
    _gpu = False
    try:
        _gpu = (subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0)
    except Exception:
        _gpu = False
    if not _gpu:
        clear_output()
        display(HTML("""
          <div style="font-family:Inter,system-ui,sans-serif;background:linear-gradient(135deg,#1a1222,#0f0d18);border:1.5px solid #f59e0b;border-radius:16px;padding:22px 26px;max-width:760px;box-shadow:0 0 26px rgba(245,158,11,.2)">
            <div style="font-size:1.3em;font-weight:800;color:#fbbf24">&#9888;&#65039; GPU is not turned on</div>
            <div style="color:#d8d4ec;margin-top:8px;line-height:1.6">On Kaggle you turn the GPU on yourself &mdash; it's free (about 30 hours/week). Do this:</div>
            <ol style="color:#cfd3e6;margin:12px 0;line-height:1.9;padding-left:20px">
              <li>Right-side panel &rarr; <b>&#9881;&#65039; Settings</b> &rarr; <b>Accelerator</b> &rarr; choose <b>GPU T4 x2</b>.</li>
              <li><b>Settings &rarr; Internet &rarr; On</b> (needed to download the model &amp; open the link).</li>
              <li>Then run <b>Step 2</b> again.</li>
            </ol>
            <div style="color:#8b8f9e;font-size:.82rem;margin-top:8px">If GPU is greyed out, verify your phone number in Kaggle settings first (one-time, free). Weekly ~30h resets every week.</div>
            <div style="color:#8b8f9e;font-size:.82rem;margin-top:10px">&#128172; Help: WhatsApp 0321-2018655</div>
          </div>"""))
    else:
        display(HTML("<h3 style='font-family:Inter,system-ui,sans-serif;color:#b78bff;margin:6px 0;'>⏳ Setting up MirZa Tools…</h3><p style='font-family:system-ui,sans-serif;color:#9a94b4;margin:0;'>First run takes about 2–3 minutes (it downloads the AI model). Please keep this tab open.</p>"))
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        os.environ['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
        os.environ['PIP_NO_INPUT'] = '1'
        print('\U0001f4e6  Step 1/3 — Installing the engine…  (about 2 minutes, please wait)')
        get_ipython().system('pip install -q --prefer-binary omnivoice transformers==5.3 pydub soundfile librosa accelerate hf_transfer "gradio==5.50.0"')
        print('\U0001f3a8  Step 2/3 — Preparing your studio…')
        _BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"
        _app = os.path.join(_BASE, "mirza_app.py")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/mirzasaqibusa-oss/mirzavoiceovertool/main/mirza_app.py", _app)
        clear_output()
        display(HTML("""
          <div style="font-family:Inter,system-ui,sans-serif; background:linear-gradient(135deg,#0b0a14,#171029); border:1.5px solid #7c5cff; border-radius:16px; padding:18px 24px; max-width:760px; box-shadow:0 0 26px rgba(124,92,255,.28);">
            <div style="font-size:1.4em;font-weight:800;background:linear-gradient(135deg,#b78bff,#a855f7);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;">\U0001f680 Launching MirZa Tools…</div>
            <div style="color:#d8d4ec;margin-top:6px;">A public link (ending in <b>.gradio.live</b>) will appear just below in a minute. Click it to open your studio. Keep this tab open while you work.</div>
          </div>"""))
        print('\U0001f680  Step 3/3 — Loading model & launching…')
        get_ipython().system('python ' + _app)
