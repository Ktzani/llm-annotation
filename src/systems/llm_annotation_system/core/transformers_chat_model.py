"""
Transformers Chat Model - Executa modelos do HuggingFace LOCALMENTE via `transformers`

Alternativa ao provider "huggingface" (Inference API), que consome créditos do
HF a cada chamada. Aqui os pesos são baixados uma única vez para o cache do HF
(`HF_HOME`) e a inferência roda na GPU/CPU da máquina.

A resposta segue o mesmo contrato usado pelo `AnnotationEngine` para as demais
chains LangChain: `content` (resposta final), e em `response_metadata` as chaves
`thinking` (raciocínio entre <think>...</think>, se houver) e `logprobs`
(lista de {"token", "logprob"} dos tokens da resposta final), consumidos pelo
`ResponseProcessor` para calcular a confiança do rótulo.
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
    Pesos + tokenizer de um modelo carregado, com um executor de 1 thread.

    O executor serializa as gerações do mesmo modelo (a GPU não ganha nada com
    `generate` concorrente sobre os mesmos pesos) sem bloquear o event loop:
    enquanto um modelo local gera, chamadas a outros providers seguem em paralelo.
    """

    def __init__(self, model_name: str, model: Any, tokenizer: Any):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"hf-{model_name.split('/')[-1]}"
        )
        weakref.finalize(self, self.executor.shutdown, wait=False)


# Referências fracas: variações do mesmo modelo (ex.: `_alt1`, `_alt2`) e folds
# da anotação em 2 fases reutilizam os mesmos pesos enquanto houver um annotator
# vivo usando-os; quando ninguém mais referencia, a VRAM é liberada — importante
# na API, onde o processo vive entre experimentos.
_REGISTRY: "weakref.WeakValueDictionary[Tuple[str, str], _LoadedModel]" = weakref.WeakValueDictionary()
_REGISTRY_LOCK = threading.Lock()


def _hf_token() -> Optional[str]:
    """Token só é necessário para BAIXAR modelos gated (ex.: meta-llama); a inferência é local."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or None


def load_transformers_model(model_name: str, load_params: Optional[Dict[str, Any]] = None) -> _LoadedModel:
    """
    Carrega (ou reutiliza) um modelo causal do HF Hub/disco.

    Args:
        model_name: repo_id no HF Hub (ex.: "meta-llama/Llama-3.1-8B-Instruct")
            ou caminho local para um checkpoint.
        load_params: kwargs repassados ao `from_pretrained` (sobrescrevem
            DEFAULT_LOAD_PARAMS). Ex.: {"dtype": "bfloat16", "device_map": "cuda:0"}.
    """
    params = {**DEFAULT_LOAD_PARAMS, **(load_params or {})}
    key = (model_name, json.dumps(params, sort_keys=True, default=str))

    with _REGISTRY_LOCK:
        loaded = _REGISTRY.get(key)
        if loaded is not None:
            logger.debug(f"Transformers: reutilizando {model_name} já carregado")
            return loaded

        # Libera modelos órfãos (sem annotator vivo) antes de ocupar mais VRAM.
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
    Registra o logprob de cada token gerado sem acumular os logits do vocabulário
    inteiro por passo (o que `output_logits=True` faria — GBs em respostas longas
    de modelos com raciocínio).

    No passo t o processor recebe os scores do passo t; o token escolhido só
    aparece em `input_ids` no passo t+1. Por isso guarda o log-softmax do passo
    anterior e, ao fim da geração, `finalize` resolve o último token.

    Roda depois dos processors padrão (ex.: repetition_penalty) e antes de
    temperature/top-k/top-p — o logprob reflete a distribuição do modelo, não a
    distribuição de amostragem.
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
        if self._prev is not None:
            self.logprobs.append(self._prev[last_token_id].item())
            self._prev = None
        return self.logprobs


class TransformersChatModel(BaseChatModel):
    """
    Chat model LangChain que executa um modelo do HF localmente com `transformers`.

    Compatível com `prompt | llm` (ver `LLMProvider.create_chain`).
    """

    model_name: str = Field(..., description="repo_id do HF Hub ou caminho local do checkpoint")
    generation_params: Dict[str, Any] = Field(default_factory=dict)
    load_params: Dict[str, Any] = Field(default_factory=dict)

    _loaded: _LoadedModel = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        # Carrega na inicialização (e não na 1ª chamada): erros de download,
        # modelo gated ou falta de memória aparecem antes de começar a anotação.
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
        # Mesmo caminho síncrono também passa pelo executor do modelo, para nunca
        # rodar dois `generate` simultâneos sobre os mesmos pesos.
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
        """Aplica o chat template do modelo; sem template (modelos base), usa o texto cru."""
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
        """
        Traduz os params do config para kwargs do `generate`.

        - temperature <= 0 → decodificação gulosa (determinística).
        - temperature > 0 sem `do_sample` explícito → amostragem.
        - Em modo guloso zera temperature/top_p/top_k para não herdar os valores
          de amostragem do `generation_config` do modelo (e evitar warnings).
        """
        params = dict(self.generation_params)
        params.setdefault("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)

        temperature = params.get("temperature")
        if temperature is not None and temperature <= 0:
            params["do_sample"] = False
        elif temperature is not None:
            params.setdefault("do_sample", True)

        if params.get("do_sample") is False:
            params.update({"temperature": None, "top_p": None, "top_k": None})

        tokenizer = self._loaded.tokenizer
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            params["pad_token_id"] = tokenizer.eos_token_id

        return params

    def _split_thinking(
        self, new_ids: List[int], logprobs: List[float]
    ) -> Tuple[Optional[str], str, List[int], List[float]]:
        """
        Separa o raciocínio (<think>...</think>) da resposta final.

        Alguns templates (ex.: DeepSeek-R1-Distill) já abrem o <think> no prompt,
        então a saída só contém o fechamento. Separar aqui evita que números do
        raciocínio sejam extraídos como rótulo pelo `ResponseProcessor`.
        """
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
            # Sem fronteira em nível de token: logprobs ficam de fora para não
            # casar um número do raciocínio como token do rótulo.
            return thinking.replace("<think>", "").strip(), content.strip(), [], []

        return None, text.strip(), new_ids, logprobs

    @staticmethod
    def _truncate_at_stop(text: str, stop: List[str]) -> str:
        cut = min((text.find(s) for s in stop if s in text), default=-1)
        return text[:cut] if cut >= 0 else text
