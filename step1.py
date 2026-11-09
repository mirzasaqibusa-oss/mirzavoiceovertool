# MirZa Tools — access gate (hosted file)
import urllib.request, json, urllib.parse, builtins
builtins._ok = False

api_url = 'https://script.google.com/macros/s/AKfycbyTufQiM1kZ_UrV_EqrxxzpJA9d75YKtmsY6DNts8v2oiqRC5G4-ufaVBy2ohRzyLGt/exec'
secret_key = 'mirza_secure_key_2026'

try:
    em = input('Enter your registered Gmail, then press Enter: ').strip().lower()
except Exception:
    em = ''

if not em:
    print('⚠️  No email entered. Run this cell again and type your registered Gmail.')
else:
    try:
        req = urllib.request.Request(
            api_url + '?key=' + secret_key + '&email=' + urllib.parse.quote(em),
            headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        status = data.get('status', '')
        if status == 'ok':
            builtins._ok = True
            print('✅ Access Granted!  Now run Step 2 below.  \U0001f447')
        elif status == 'off':
            print('\U0001f534 The service is currently unavailable. Please try again later.')
        elif status == 'expired':
            from IPython.display import display, HTML
            exp = str(data.get('expires', ''))
            exp_line = ("<div style='color:#f2c9c9;margin-top:6px;font-size:.9em;'>Your access ended on <b>" + exp + "</b>.</div>") if exp else ""
            display(HTML(
                "<div style='font-family:Inter,Arial,sans-serif;max-width:580px;background:linear-gradient(135deg,#1c0f14,#2a1420);border:1.5px solid #ff6b6b;border-radius:14px;padding:24px 28px;'>"
                "<div style='font-size:1.3em;font-weight:800;color:#ff9a9a;'>&#9203; Your Subscription Has Expired</div>"
                + exp_line +
                "<div style='color:#e8d4d4;margin-top:10px;line-height:1.65;'>Your access to <b style='color:#fff;'>MirZa Tools</b> has ended. To renew and reactivate your account, please contact us:</div>"
                "<div style='margin-top:16px;color:#fff;line-height:2;font-size:1em;'>&#128231; &nbsp;<b>Email:</b> &nbsp;mirzasaqibali837@gmail.com<br>&#128172; &nbsp;<b>WhatsApp:</b> &nbsp;0321-2018655</div>"
                "<a href='https://wa.me/923212018655' target='_blank' style='display:inline-block;margin-top:18px;background:linear-gradient(135deg,#25d366,#128c7e);color:#fff;text-decoration:none;padding:11px 24px;border-radius:9px;font-weight:700;font-size:.95em;'>&#128172; Renew on WhatsApp</a>"
                "<div style='color:#b89090;margin-top:16px;font-size:.85em;'>Thank you for choosing MirZa Tools. We will reactivate your access right away.</div>"
                "</div>"))
        else:
            from IPython.display import display, HTML
            display(HTML(
                "<div style='font-family:Inter,Arial,sans-serif;max-width:580px;background:linear-gradient(135deg,#141018,#1e1420);border:1.5px solid #e0685a;border-radius:14px;padding:24px 28px;'>"
                "<div style='font-size:1.3em;font-weight:800;color:#ff9a8f;'>&#128274; Access Denied</div>"
                "<div style='color:#e0d2d2;margin-top:10px;line-height:1.65;'>This account is not approved yet. To get access to <b style='color:#fff;'>MirZa Tools</b>, please contact us:</div>"
                "<div style='margin-top:16px;color:#fff;line-height:2;font-size:1em;'>&#128231; &nbsp;<b>Email:</b> &nbsp;mirzasaqibali837@gmail.com<br>&#128172; &nbsp;<b>WhatsApp:</b> &nbsp;0321-2018655</div>"
                "<a href='https://wa.me/923212018655' target='_blank' style='display:inline-block;margin-top:18px;background:linear-gradient(135deg,#25d366,#128c7e);color:#fff;text-decoration:none;padding:11px 24px;border-radius:9px;font-weight:700;font-size:.95em;'>&#128172; Get Access on WhatsApp</a>"
                "</div>"))
    except Exception:
        print('⚠️  Connection error. Make sure "Internet" is ON in the right-side Settings panel, then run again.')
