#!/usr/bin/env python3
# mtmd + dflash repro (v2): /v1/chat/completions + image_url (the upstream
# refactor dropped legacy /completion image_data). Tests issue ggml-org#27408:
# draft-cache position holes / M-RoPE rows -> llama_decode(ctx_dft) rc=-1
# -> HTTP 500. Usage: mtmd_repro2.py <image> [n_requests]
import base64, json, sys, time, urllib.request, urllib.error

img_path = sys.argv[1]
n_req = int(sys.argv[2]) if len(sys.argv) > 2 else 1
img_b64 = base64.b64encode(open(img_path, "rb").read()).decode()
body = json.dumps({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in one sentence."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ],
    }],
    "max_tokens": 64, "temperature": 0, "top_p": 1,
}).encode()
fails = 0
for i in range(n_req):
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=1800)
        d = json.loads(r.read())
        txt = d["choices"][0]["message"]["content"]
        u = d.get("usage", {})
        print(f"[{i+1}/{n_req}] HTTP {r.status} ({time.time()-t0:.1f}s) "
              f"pt={u.get('prompt_tokens')} ct={u.get('completion_tokens')} "
              f"| {txt[:100]!r}")
    except urllib.error.HTTPError as e:
        fails += 1
        print(f"[{i+1}/{n_req}] HTTP {e.code} ({time.time()-t0:.1f}s) | {e.read()[:200]!r}")
    except Exception as e:
        fails += 1
        print(f"[{i+1}/{n_req}] EXC {type(e).__name__}: {e}")
print(f"RESULT: {n_req - fails}/{n_req} ok")
sys.exit(1 if fails else 0)
