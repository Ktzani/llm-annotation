"""
Transformers Chat Model - Executa modelos do HuggingFace localmente (sem Inference API)
"""

import asyncio
import gc
import json
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from pydantic import Field, PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


DEFAULT_LOAD_PARAMS: Dict[str, Any] = {"device_map": "auto", "dtype": "auto"}
DEFAULT_MAX_NEW_TOKENS = 512

_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}


class _LoadedModel:
    """
    Pesos + tokenizer de um modelo carregado
    O executor de 1 thread serializa as gerações do modelo sem bloquear o event loop
    """

    def __init__(self, model_name: str, model: Any, tokenizer: Any):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"hf-{model_name.split('/')[-1]}"
        )
        weakref.finalize(self, self.executor.shutdown, wait=False)


# Referência fraca: variações (_alt) reutilizam os pesos e a VRAM é liberada quando ninguém mais usa
_REGISTRY: "weakref.WeakValueDictionary[Tuple[str, str], _LoadedModel]" = weakref.WeakValueDictionary()
_REGISTRY_LOCK = threading.Lock()


def _hf_token() -> Optional[str]:
    """Token só é usado para baixar modelos gated"""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or None


def load_transformers_model(model_name: str, load_params: Optional[Dict[str, Any]] = None) -> _LoadedModel:
    """
    Carrega (ou reutiliza) um modelo do HF Hub ou de um caminho local

    Args:
        model_name: repo_id do HF Hub ou caminho do checkpoint
        load_params: kwargs do from_pretrained (sobrescrevem DEFAULT_LOAD_PARAMS)
    """
    params = {**DEFAULT_LOAD_PARAMS, **(load_params or {})}
    key = (model_name, json.dumps(params, sort_keys=True, default=str))

    with _REGISTRY_LOCK:
        loaded = _REGISTRY.get(key)
        if loaded is not None:
            logger.debug(f"Transformers: reutilizando {model_name} já carregado")
            return loaded

        # Libera modelos sem uso antes de ocupar mais VRAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Transformers: carregando {model_name} localmente ({params})...")
        token = _hf_token()
        trust_remote_code = params.get("trust_remote_code", False)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=token, trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, token=token, **params)
        model.eval()

        loaded = _LoadedModel(model_name, model, tokenizer)
        _REGISTRY[key] = loaded
        logger.success(f"Transformers: {model_name} carregado em {model.device}")
        return loaded


class _ChosenTokenLogprobs(LogitsProcessor):
    """
    Registra o logprob de cada token gerado sem guardar os logits de todos os passos
    O token escolhido só aparece no passo seguinte, por isso guarda o log-softmax anterior
    """

    def __init__(self):
        self.logprobs: List[float] = []
        self._prev: Optional[torch.Tensor] = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._prev is not None:
            self.logprobs.append(self._prev[input_ids[0, -1]].item())
        self._prev = torch.log_softmax(scores[0].float(), dim=-1)
        return scores

    def finalize(self, last_token_id: int) -> List[float]:
        """Resolve o logprob do último token gerado"""
        if self._prev is not None:
            self.logprobs.append(self._prev[last_token_id].item())
            self._prev = None
        return self.logprobs


class TransformersChatModel(BaseChatModel):
    """
    Chat model LangChain que executa um modelo do HF localmente
    Responsabilidades: aplicar chat template, gerar, separar thinking e calcular logprobs
    """

    model_name: str = Field(..., description="repo_id do HF Hub ou caminho local do checkpoint")
    generation_params: Dict[str, Any] = Field(default_factory=dict)
    load_params: Dict[str, Any] = Field(default_factory=dict)

    _loaded: _LoadedModel = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        # Carrega já na inicialização para falhar antes de começar a anotação
        super().model_post_init(__context)
        self._loaded = load_transformers_model(self.model_name, self.load_params)

    @property
    def _llm_type(self) -> str:
        return "transformers-local"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, **self.generation_params}

    # ------------------------------------------------------------------
    # Geração
    # ------------------------------------------------------------------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._loaded.executor.submit(self._generate_sync, messages, stop).result()

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._loaded.executor, self._generate_sync, messages, stop)

    def _generate_sync(self, messages: List[BaseMessage], stop: Optional[List[str]]) -> ChatResult:
        tokenizer = self._loaded.tokenizer
        model = self._loaded.model

        inputs = self._encode(messages)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        logprob_recorder = _ChosenTokenLogprobs()
        gen_kwargs = self._build_generation_kwargs()
        if stop:
            gen_kwargs["stop_strings"] = stop
            gen_kwargs["tokenizer"] = tokenizer

        with torch.inference_mode():
            sequences = model.generate(
                **inputs,
                **gen_kwargs,
                logits_processor=LogitsProcessorList([logprob_recorder]),
            )

        new_ids = sequences[0, prompt_len:].tolist()
        logprobs = logprob_recorder.finalize(new_ids[-1]) if new_ids else []

        thinking, content, content_ids, content_logprobs = self._split_thinking(new_ids, logprobs)
        if stop:
            content = self._truncate_at_stop(content, stop)

        token_logprobs = [
            {"token": tokenizer.decode([tid], skip_special_tokens=False), "logprob": lp}
            for tid, lp in zip(content_ids, content_logprobs)
        ]

        message = AIMessage(
            content=content,
            response_metadata={
                "model_name": self.model_name,
                "thinking": thinking,
                "logprobs": token_logprobs,
                "prompt_tokens": prompt_len,
                "completion_tokens": len(new_ids),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _encode(self, messages: List[BaseMessage]) -> Dict[str, torch.Tensor]:
        """Aplica o chat template do modelo (ou usa o texto cru se não houver)"""
        tokenizer = self._loaded.tokenizer

        if getattr(tokenizer, "chat_template", None):
            chat = [
                {"role": _ROLE_MAP.get(m.type, "user"), "content": m.content}
                for m in messages
            ]
            return tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )

        prompt = "\n\n".join(str(m.content) for m in messages)
        return tokenizer(prompt, return_tensors="pt")

    def _build_generation_kwargs(self) -> Dict[str, Any]:
        """Traduz os params do config para o generate (temperature <= 0 → guloso)"""
        params = dict(self.generation_params)
        params.setdefault("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)

        temperature = params.get("temperature")
        if temperature is not None and temperature <= 0:
            params["do_sample"] = False
        elif temperature is not None:
            params.setdefault("do_sample", True)

        # Não herda a amostragem do generation_config do modelo no modo guloso
        if params.get("do_sample") is False:
            params.update({"temperature": None, "top_p": None, "top_k": None})

        tokenizer = self._loaded.tokenizer
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            params["pad_token_id"] = tokenizer.eos_token_id

        return params

    def _split_thinking(
        self, new_ids: List[int], logprobs: List[float]
    ) -> Tuple[Optional[str], str, List[int], List[float]]:
        """Separa o raciocínio (<think>...</think>) da resposta final"""
        tokenizer = self._loaded.tokenizer
        end_think_id = tokenizer.convert_tokens_to_ids("</think>")
        has_end_token = (
            isinstance(end_think_id, int)
            and end_think_id != tokenizer.unk_token_id
            and end_think_id in new_ids
        )

        if has_end_token:
            cut = len(new_ids) - 1 - new_ids[::-1].index(end_think_id)
            thinking = tokenizer.decode(new_ids[:cut], skip_special_tokens=True)
            content_ids = new_ids[cut + 1:]
            content_logprobs = logprobs[cut + 1:]
            content = tokenizer.decode(content_ids, skip_special_tokens=True)
            return thinking.replace("<think>", "").strip(), content.strip(), content_ids, content_logprobs

        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if "</think>" in text:
            thinking, content = text.rsplit("</think>", 1)
            # Sem fronteira por token: descarta logprobs para não pegar número do raciocínio
            return thinking.replace("<think>", "").strip(), content.strip(), [], []

        return None, text.strip(), new_ids, logprobs

    @staticmethod
    def _truncate_at_stop(text: str, stop: List[str]) -> str:
        cut = min((text.find(s) for s in stop if s in text), default=-1)
        return text[:cut] if cut >= 0 else text
