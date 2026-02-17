from abc import ABC, abstractmethod
import os
from typing import Iterable, Iterator, List, Literal, Tuple, Union

from fastapi.encoders import jsonable_encoder
from utils.log import quick_log
from utils.rwkv import ModelConfigBody, get_model_path, AbstractRWKV


class AbstractLlama(ABC):
    def __init__(self, model):
        self.name = "llama"
        self.model_path = ""
        self.version = 100
        self.model = model

        self.max_tokens_per_generation = 500
        self.temperature = 1.0
        self.top_p = 0.3
        self.top_k = 40
        self.penalty_alpha_presence = 0.0
        self.penalty_alpha_frequency = 0.0

    @abstractmethod
    def delta_postprocess(self, delta: str) -> str:
        pass

    def generate(
        self,
        body: ModelConfigBody,
        prompt: str,
        stop: Union[str, List[str], None] = None,
        stop_token_ids: Union[List[int], None] = None,
    ) -> Iterable[Tuple[Literal["text", "tool"], str, str, int, int]]:
        quick_log(None, None, "Generation Prompt:\n" + prompt)
        completion_token_len = 0
        response = ""


        from routes.completion import ChatCompletionBody

        if not is_rwkv_model(self) and isinstance(body, ChatCompletionBody):
            from llama_cpp import CreateChatCompletionStreamResponse

            stream_chat: Iterator[CreateChatCompletionStreamResponse] = (
                self.model.create_chat_completion(
                    messages=body.messages,
                    tools=jsonable_encoder(body.tools) if body.tools else None,
                    tool_choice=body.tool_choice,
                    max_tokens=self.max_tokens_per_generation,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stream=True,
                    stop=stop,
                    frequency_penalty=self.penalty_alpha_frequency,
                    presence_penalty=self.penalty_alpha_presence,
                )
            )

            for chunk in stream_chat:
                if not chunk.get("choices"):
                    continue
                completion_token_len = completion_token_len + 1
                delta = chunk["choices"][0].get("delta", {})
                content = self.delta_postprocess(delta.get("content", ""))

                if content:
                    response += content
                    yield "text", response, content, 0, completion_token_len
                # for tool in delta.get("tool_calls", []) or []:
                #     yield "tool", response, json.dumps(
                #         tool["function"]
                #     ), 0, completion_token_len
        else:
            from llama_cpp import CreateCompletionStreamResponse

            stream: Iterator[CreateCompletionStreamResponse] = (
                self.model.create_completion(
                    prompt=prompt,
                    max_tokens=self.max_tokens_per_generation,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stream=True,
                    stop=stop,
                    frequency_penalty=self.penalty_alpha_frequency,
                    presence_penalty=self.penalty_alpha_presence,
                )
            )

            for chunk in stream:
                if not chunk.get("choices"):
                    continue
                completion_token_len = completion_token_len + 1
                delta = self.delta_postprocess(chunk["choices"][0].get("text", ""))
                response += delta

                yield "text", response, delta, 0, completion_token_len


class TextLlama(AbstractLlama):
    def __init__(self, model) -> None:
        super().__init__(model)

        self.max_tokens_per_generation = 500
        self.temperature = 1.0
        self.top_p = 0.3
        self.top_k = 40
        self.penalty_alpha_presence = 0.0
        self.penalty_alpha_frequency = 1.0

        self.interface = ":"
        self.user = "User"
        self.bot = "Assistant"

        self.__preload()

    def delta_postprocess(self, delta: str) -> str:
        return delta

    def __preload(self):
        pass
    
    def clear_rwkv_state(self):
        """Properly clear RWKV recurrent state"""
        from llama_cpp import llama_get_memory, llama_memory_clear
        memory = llama_get_memory(self.model._ctx.ctx)
        llama_memory_clear(memory, True)
        self.model.n_tokens = 0


def Llama(model_path: str, strategy: str) -> AbstractLlama:
    model_path = get_model_path(model_path)

    from llama_cpp import Llama

    filename, _ = os.path.splitext(os.path.basename(model_path))
    
    # Parse strategy: "cuda fp16 [n_ctx] [n_gpu_layers]"
    # Example: "cuda fp16 524288 55" = 512K context, 55 GPU layers
    n_ctx = 0  # 0 = use model's native context length
    n_gpu = -1 if "cpu" not in strategy else 0

    try:
        parts = strategy.split()
        # Check parts[2] and parts[3] (after "cuda fp16")
        if len(parts) >= 3 and parts[2].isdigit():
            n_ctx = int(parts[2])  # Third param = context length
        if len(parts) >= 4 and parts[3].isdigit():
            n_gpu = int(parts[3])  # Fourth param = GPU layers
    except:
        pass

    model = Llama(
        model_path, 
        n_gpu_layers=n_gpu, 
        n_ctx=n_ctx
    )

    # Only patch generate function if it is an RWKV model
    if "rwkv" in filename.lower():
        original_generate = model.generate
        def rwkv_generate(tokens, **kwargs):
            kwargs['reset'] = False
            return original_generate(tokens, **kwargs)
        model.generate = rwkv_generate
    
    llama: AbstractLlama
    llama = TextLlama(model)
    llama.name = filename
    llama.model_path = model_path

    return llama


def set_llama_config(model: AbstractLlama, body: ModelConfigBody):
    if body.max_tokens is not None:
        model.max_tokens_per_generation = body.max_tokens
    if body.temperature is not None:
        if body.temperature < 0.1:
            model.temperature = 0.1
        else:
            model.temperature = body.temperature
    if body.top_p is not None:
        model.top_p = body.top_p
    if body.presence_penalty is not None:
        model.penalty_alpha_presence = body.presence_penalty
    if body.frequency_penalty is not None:
        model.penalty_alpha_frequency = body.frequency_penalty
    if body.top_k is not None:
        model.top_k = body.top_k


def get_llama_config(model: AbstractLlama) -> ModelConfigBody:
    return ModelConfigBody(
        max_tokens=model.max_tokens_per_generation,
        temperature=model.temperature,
        top_p=model.top_p,
        presence_penalty=model.penalty_alpha_presence,
        frequency_penalty=model.penalty_alpha_frequency,
        top_k=model.top_k,
    )


# you can rename gguf file to rwkv to use builtin rwkv prompt template
def is_rwkv_model(model: Union[AbstractRWKV, AbstractLlama]) -> bool:
    return isinstance(model, AbstractRWKV) or "rwkv" in model.name.lower()
