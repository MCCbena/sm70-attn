#!/usr/bin/env python3
import json, urllib.request
body = json.dumps({
    "messages": [{"role": "user",
                  "content": "Write a haiku about GPUs. Reply with the haiku only."}],
    "max_tokens": 48, "temperature": 0,
}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                             data=body, headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=300).read())
u = d.get("usage", {})
print((d["choices"][0]["message"]["content"] or "(in reasoning)").strip()[:200])
print(f"pt={u.get('prompt_tokens')} ct={u.get('completion_tokens')}")
