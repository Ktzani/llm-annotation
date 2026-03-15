from typing import Any, Mapping, Iterator, AsyncIterator, Sequence, cast
from typing_extensions import Self

from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.tracers.schemas import LangSmithParams
from pydantic import model_validator

# imports usados internamente pelo wrapper original
from ollama import Client, AsyncClient

from langchain_core.messages.ai import UsageMetadata

def _get_usage_metadata_from_generation_info(
    generation_info: Mapping[str, Any] | None,
) -> UsageMetadata | None:
    """Get usage metadata from Ollama generation info mapping."""
    if generation_info is None:
        return None
    input_tokens: int | None = generation_info.get("prompt_eval_count")
    output_tokens: int | None = generation_info.get("eval_count")
    if input_tokens is not None and output_tokens is not None:
        return UsageMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
    return None


class ChatOllamaLogProbs(ChatOllama):

    # novos atributos
    logprobs: bool | None = None
    top_logprobs: int | None = None

    def _chat_params(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:

        ollama_messages = self._convert_messages_to_ollama_messages(messages)

        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")

        if self.stop is not None:
            stop = self.stop

        options_dict = kwargs.pop("options", None)

        if options_dict is None:
            options_dict = {
                k: v
                for k, v in {
                    "mirostat": self.mirostat,
                    "mirostat_eta": self.mirostat_eta,
                    "mirostat_tau": self.mirostat_tau,
                    "num_ctx": self.num_ctx,
                    "num_gpu": self.num_gpu,
                    "num_thread": self.num_thread,
                    "num_predict": self.num_predict,
                    "repeat_last_n": self.repeat_last_n,
                    "repeat_penalty": self.repeat_penalty,
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "stop": self.stop if stop is None else stop,
                    "tfs_z": self.tfs_z,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                }.items()
                if v is not None
            }

        params = {
            "messages": ollama_messages,
            "stream": kwargs.pop("stream", True),
            "model": kwargs.pop("model", self.model),
            "think": kwargs.pop("reasoning", self.reasoning),
            "logprobs": kwargs.pop("logprobs", self.logprobs),
            "top_logprobs": kwargs.pop("top_logprobs", self.top_logprobs),
            "format": kwargs.pop("format", self.format),
            "options": options_dict,
            "keep_alive": kwargs.pop("keep_alive", self.keep_alive),
            **kwargs,
        }

        if tools := kwargs.get("tools"):
            params["tools"] = tools

        return params

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        client_kwargs = self.client_kwargs or {}

        cleaned_url, auth_headers = parse_url_with_auth(self.base_url)
        merge_auth_headers(client_kwargs, auth_headers)

        sync_client_kwargs = client_kwargs
        if self.sync_client_kwargs:
            sync_client_kwargs = {**sync_client_kwargs, **self.sync_client_kwargs}

        async_client_kwargs = client_kwargs
        if self.async_client_kwargs:
            async_client_kwargs = {**async_client_kwargs, **self.async_client_kwargs}

        self._client = Client(host=cleaned_url, **sync_client_kwargs)
        self._async_client = AsyncClient(host=cleaned_url, **async_client_kwargs)

        if self.validate_model_on_init:
            validate_model(self._client, self.model)

        return self

    def _iterate_over_stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:

        reasoning = kwargs.get("reasoning", self.reasoning)
        logprobs = kwargs.get("logprobs", self.logprobs)

        for stream_resp in self._create_chat_stream(messages, stop, **kwargs):

            if isinstance(stream_resp, str):
                continue

            content = (
                stream_resp["message"]["content"]
                if "message" in stream_resp and "content" in stream_resp["message"]
                else ""
            )

            generation_info = {}

            if logprobs and not stream_resp.get("done"):
                generation_info["logprobs"] = stream_resp.get("logprobs")

            if stream_resp.get("done") is True:
                generation_info = dict(stream_resp)
                if "model" in generation_info:
                    generation_info["model_name"] = generation_info["model"]
                generation_info["model_provider"] = "ollama"
                generation_info.pop("message", None)

            elif not logprobs:
                generation_info = None

            additional_kwargs = {}

            if (
                reasoning
                and "message" in stream_resp
                and (thinking_content := stream_resp["message"].get("thinking"))
            ):
                additional_kwargs["reasoning_content"] = thinking_content

            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=content,
                    additional_kwargs=additional_kwargs,
                    usage_metadata=_get_usage_metadata_from_generation_info(
                        stream_resp
                    ),
                    tool_calls=_get_tool_calls_from_response(stream_resp),
                ),
                generation_info=generation_info,
            )

            yield chunk

    async def _aiterate_over_stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:

        reasoning = kwargs.get("reasoning", self.reasoning)
        logprobs = kwargs.get("logprobs", self.logprobs)

        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):

            if isinstance(stream_resp, str):
                continue

            content = (
                stream_resp["message"]["content"]
                if "message" in stream_resp and "content" in stream_resp["message"]
                else ""
            )

            generation_info = {}

            if logprobs and not stream_resp.get("done"):
                generation_info["logprobs"] = stream_resp.get("logprobs")

            if stream_resp.get("done") is True:
                generation_info = dict(stream_resp)
                if "model" in generation_info:
                    generation_info["model_name"] = generation_info["model"]
                generation_info["model_provider"] = "ollama"
                generation_info.pop("message", None)

            elif not logprobs:
                generation_info = None

            additional_kwargs = {}

            if (
                reasoning
                and "message" in stream_resp
                and (thinking_content := stream_resp["message"].get("thinking"))
            ):
                additional_kwargs["reasoning_content"] = thinking_content

            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=content,
                    additional_kwargs=additional_kwargs,
                    usage_metadata=_get_usage_metadata_from_generation_info(
                        stream_resp
                    ),
                    tool_calls=_get_tool_calls_from_response(stream_resp),
                ),
                generation_info=generation_info,
            )

            yield chunk