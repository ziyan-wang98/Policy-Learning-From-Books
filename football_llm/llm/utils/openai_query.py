from openai import OpenAI
import os


def _openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set before calling OpenAI.")
    kwargs = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def query(global_prompt, user_prompt, model_name, req_json=False):
    
    client = _openai_client()
    
    if req_json:
        other_kwargs = {"response_format": {"type": "json_object"}}
    else:
        other_kwargs = {}
    response = client.chat.completions.create(
        messages=[
        {"role": "system", "content": "You are a concise domain reasoning assistant. " + global_prompt},
        {"role": "user", "content": user_prompt},
        ],
        model=model_name,
        temperature=0,
        **other_kwargs,
    )
    
    return response