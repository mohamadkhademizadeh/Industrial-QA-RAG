import os, requests, json
from typing import List, Dict, Optional
from openai import OpenAI

def openai_chat(messages: List[Dict], base_url: Optional[str], model: str) -> str:
    client = OpenAI(base_url=base_url) if base_url else OpenAI()
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return resp.choices[0].message.content

def ollama_chat(messages: List[Dict], host: str, model: str) -> str:
    # messages -> string prompt
    sys = ""
    user = ""
    for m in messages:
        if m['role'] == 'system':
            sys += m['content'] + "\n"
        elif m['role'] == 'user':
            user += m['content'] + "\n"
    prompt = (sys + "\n" + user).strip()
    r = requests.post(f"{host}/api/chat", json={"model": model, "messages": messages, "stream": False}, timeout=120)
    r.raise_for_status()
    data = r.json()
    if 'message' in data and 'content' in data['message']:
        return data['message']['content']
    if 'choices' in data and data['choices']:
        return data['choices'][0]['message']['content']
    return data.get('response', '')

def chat(provider: str, messages: List[Dict], cfg: Dict) -> str:
    if provider == 'ollama':
        return ollama_chat(messages, cfg['ollama']['host'], cfg['ollama']['model'])
    else:
        base = cfg['openai'].get('base_url', None)
        model = cfg['openai'].get('model', 'gpt-4o-mini')
        return openai_chat(messages, base, model)
