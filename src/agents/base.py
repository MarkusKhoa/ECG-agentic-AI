"""
LLM backend factory.

Provides a single ``get_llm()`` function that returns a LangChain-compatible
chat model based on ``CFG.llm_backend``.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import CFG


def get_llm() -> BaseChatModel:
    """Return the configured chat model."""
    if CFG.llm_backend == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=CFG.openai_model,
            temperature=CFG.temperature,
            max_tokens=CFG.max_tokens,
        )
    elif CFG.llm_backend == "huggingface":
        # Local HuggingFace pipeline wrapped for LangChain
        from langchain_community.chat_models import ChatHuggingFace
        from langchain_community.llms import HuggingFacePipeline
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(CFG.hf_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            CFG.hf_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=CFG.max_tokens,
            temperature=CFG.temperature,
            do_sample=True,
        )
        hf_llm = HuggingFacePipeline(pipeline=pipe)
        return ChatHuggingFace(llm=hf_llm)
    else:
        raise ValueError(f"Unknown llm_backend: {CFG.llm_backend}")
