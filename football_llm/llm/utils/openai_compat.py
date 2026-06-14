'''

  @python version : 3.6.4
  @author : pangjc
  @time : 2023/10/30
'''

import requests
import json
import signal
import time
import os

class TimeoutError(Exception):
    pass


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _default_chat_model():
    return os.environ.get("PLFB_OPENAI_CHAT_MODEL", "gpt-4o-mini")


def _default_embedding_model():
    return os.environ.get("PLFB_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def handler(signum, frame):
    raise TimeoutError("Timeout occurred")


def execute_with_timeout(function, timeout, *args):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)  # Set timeout.

    try:
        result = function(*args)  # Run the wrapped function.
        return result
    except TimeoutError as e:
        # Handle timeout errors from the wrapped call.
        print("Function timed out")
        raise e
    finally:
        signal.alarm(0)  # Clear timeout.


class OpenAIChatAPI():
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set before calling an OpenAI-compatible endpoint.")
        self.request_timeout = _env_int("OPENAI_REQUEST_TIMEOUT", 600)
        self.max_retries = _env_int("OPENAI_MAX_RETRIES", 3)
        self.retry_sleep = _env_float("OPENAI_RETRY_SLEEP", 2.0)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def post_request(self,
                    #  question,
                     global_prompt,
                     user_prompt,
                     n=1,
                     model=None,
                     req_json=True, temp=0.0, max_tokens=2500):
        """API for calling an OpenAI-compatible chat-completions endpoint."""
        model = model or _default_chat_model()

        if req_json:
            other_kwargs = {"response_format": {"type": "json_object"}}
        else:
            other_kwargs = {}
        
        
        data = {
            "max_tokens": max_tokens,
            "model": model,
             **other_kwargs,
            "temperature": temp,
            "top_p": 1,
            "presence_penalty": 1,
            "messages": [
                {"role": "system", "content": "You are a concise domain reasoning assistant. " + global_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "n": n
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            data=json.dumps(data).encode('utf-8'),
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        
        return response.content.decode("utf-8")
        
        # result = json.loads(response.content.decode("utf-8"))
        # # print(result)

        # if n == 1:
        #     res = result["choices"][0]["message"]["content"]
        # else:
        #     res = []
        #     for i in range(n):
        #         res.append(result["choices"][i]["message"]["content"])

        # return res



    def ask(self,
            global_prompt,
            user_prompt,
            model,
            req_json,
            n, temp, max_tokens
    ):
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return execute_with_timeout(
                    self.post_request,
                    self.request_timeout,
                    global_prompt,
                    user_prompt,
                    n,
                    model,
                    req_json,
                    temp,
                    max_tokens,
                )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    print(f"OpenAI request failed on attempt {attempt}/{self.max_retries}: {e}")
                    time.sleep(self.retry_sleep)

        raise RuntimeError(f"OpenAI request failed after {self.max_retries} attempts") from last_error


    def get_embedding(self, input):

        data = {
            "model": _default_embedding_model(),
            "input": input
        }
        url = f"{self.base_url}/embeddings"
        response = requests.post(
            url,
            headers=self.headers,
            data=json.dumps(data).encode('utf-8'),
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        result = response.content.decode("utf-8")
        result = json.loads(result)["data"][0]["embedding"]

        return result

def openai_chat_query(global_prompt, user_prompt, model_name=None, req_json=True, n=1, temp=0.0, max_tokens=2500):
    bot = OpenAIChatAPI()
    response = bot.ask(global_prompt, user_prompt, model_name or _default_chat_model(), req_json, n, temp=temp, max_tokens=max_tokens)
    
    return response

def openai_chat_query_json(
    global_prompt,
    user_prompt,
    model_name=None,
    req_json=True,
    print=True,
    n=1,
    temp=0.0,
    max_tokens=2500,
):
    bot = OpenAIChatAPI()
    import pprint
    if print:
        pprint.pprint(user_prompt, width=400)
    response = bot.ask(global_prompt, user_prompt, model_name or _default_chat_model(), req_json, n, temp, max_tokens)
    try:
        res = json.loads(json.loads(response)["choices"][0]["message"]["content"].replace("```json\n", "").replace("\n```", ""))
        # if print:
        #     pprint.pprint(res, width=400)
        return res
    except:
        return None
    
    
    
