from openai import OpenAI
import os
import json
from llm.utils.openai_compat import openai_chat_query


def _env(*names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _require_key(name, value):
    if not value:
        raise RuntimeError(f"{name} is not set. Export it before using this model backend.")
    return value

class OpenAIServer:

    def __init__(self, model: str = "Meta-Llama-3-8B", top_p=0.9, temp=0.75, max_token=800, seed=42, *args, **kwargs):
        super().__init__()
        self.top_p = top_p
        self.temp = temp
        self.max_token = max_token
        self.seed = seed
        print("model", model)
        self.model = model
        if 'Llama' in model:
            endpoint = _require_key("PLFB_LLAMA_ENDPOINT or LLAMA_ENDPOINT", _env("PLFB_LLAMA_ENDPOINT", "LLAMA_ENDPOINT"))
            self.client = OpenAI(base_url=endpoint, api_key=_env("PLFB_LLAMA_API_KEY", "LLAMA_API_KEY", default="EMPTY"))
        elif 'deepseek' in model:
            self.client = OpenAI(
                base_url=_env("DEEPSEEK_BASE_URL", default="https://api.deepseek.com"),
                api_key=_require_key("DEEPSEEK_API_KEY", _env("DEEPSEEK_API_KEY")),
            )
        elif 'llama' in model:
            self.client = OpenAI(
                api_key=_require_key("OPENROUTER_API_KEY", _env("OPENROUTER_API_KEY")),
                base_url=_env("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1"),
            )
        elif 'gpt' in model:

            pass
        else:
            self.client = OpenAI(
                api_key=_require_key("DASHSCOPE_API_KEY", _env("DASHSCOPE_API_KEY")),
                base_url=_env("DASHSCOPE_BASE_URL", default="https://dashscope.aliyuncs.com/compatible-mode/v1"),
            )

        self.model = model

    def chat(self, message: str, system_msg: str = '', json_mode: bool = False, n=1,
             raw_ret=False):
        if 'gpt' in self.model:
            response = openai_chat_query(system_msg, message, model_name=self.model, req_json=True)
            response = json.loads(response)
            # response = response["choices"][0]["message"]["content"]
            # rj = json.loads(repair_json(json_str_clean(response, single_line=True)))
            if raw_ret:
                return response
            else:
                response = response["choices"][0]["message"]["content"]
                return response
        else:
            if system_msg == '':
                system_msg = "Return only valid JSON. Do not include additional prose such as `Here is the response in JSON format`."
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": message},
            ]
            # if 'llama' in self.model:
            #     assert n==1, "Dashscope only supports n=1"
            #     import dashscope
            #     response = dashscope.Generation.call(
            #         model=self.model,
            #         messages=messages,
            #         seed=self.seed,
            #         top_p=self.top_p,
            #         temperature=self.temp,
            #         max_tokens=self.max_token,
            #         n=n,
            #     )
            #     if raw_ret:
            #         return response
            #     else:
            #         response = response.output.text
            #         return response
            # else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temp,
                max_tokens=self.max_token,
                top_p=self.top_p,
                n=n,
                seed=self.seed,
                stop=["<|im_end|>"],
            )
            if raw_ret:
                return response
            else:
                response = response.choices[0].message.content
                return response
