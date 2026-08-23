#!/usr/bin/env python3
# Verify the model actually sees the image (not text-only hallucination):
# two DIFFERENT images must produce DIFFERENT descriptions; print reasoning.
import base64, json, sys, urllib.request

def ask(img_path, tag):
    img_b64 = base64.b64encode(open(img_path, "rb").read()).decode()
    body = json.dumps({
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image in one sentence."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]}],
        "max_tokens": 128, "temperature": 0,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    msg = d["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    print(f"--- {tag} ---")
    print(f"content: {content[:200]!r}")
    if not content and reasoning:
        print(f"reasoning(tail): {reasoning[-300:]!r}")
    return content or reasoning

a = ask(sys.argv[1], "image-A")
b = ask(sys.argv[2], "image-B")
print(f"\nVERDICT: {'DIFFERENT' if a != b else 'IDENTICAL(!?)'} descriptions for different images")
