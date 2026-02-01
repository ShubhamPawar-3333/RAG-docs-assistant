"""
LLM Module

Provides integration with Google Gemini and other LLM providers.
Handles model configuration, fallback logic, and response formatting.
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    GROQ = "groq"


# Model configurations
LLM_MODELS = {
    "gemini-2.5-flash": {
        "provider": LLMProvider.GEMINI,
        "model_name": "gemini-2.5-flash",
        "description": "Fast, efficient for most tasks",
        "context_window": 1_000_000,
        "free_tier": True,
    },
    "gemini-2.5-pro": {
        "provider": LLMProvider.GEMINI,
        "model_name": "gemini-2.5-pro",
        "description": "More capable, complex reasoning",
        "context_window": 1_000_000,
        "free_tier": True,
    },
    "gemini-2.0-flash": {
        "provider": LLMProvider.GEMINI,
        "model_name": "gemini-2.0-flash",
        "description": "Latest flash model",
        "context_window": 1_000_000,
        "free_tier": True,
    },
}


class LLMManager:
    """
    Manages LLM instances with configuration and fallback support.
    
    Provides a unified interface for interacting with Google Gemini
    and other LLM providers.
    
    Example:
        >>> llm_manager = LLMManager()
        >>> response = llm_manager.generate("What is RAG?")
        >>> # Or get the LangChain model directly
        >>> llm = llm_manager.get_llm()
    """
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the LLM manager.
        
        Args:
            model_name: Model to use (default from settings).
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum tokens in response.
            api_key: Google API key (default from settings).
        """
        self.model_name = model_name or settings.default_model
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens
        self.api_key = api_key or settings.google_api_key
        
        # Validate API key
        if not self.api_key:
            logger.warning(
                "No Google API key configured. "
                "Set GOOGLE_API_KEY environment variable."
            )
        
        # Lazy initialization
        self._llm: Optional[BaseChatModel] = None
        
        logger.info(
            f"Initialized LLMManager with model={self.model_name}, "
            f"temperature={self.temperature}"
        )
    
    def get_llm(self) -> BaseChatModel:
        """
        Get the LangChain LLM instance (lazy loaded).
        
        Returns:
            ChatGoogleGenerativeAI instance.
        """
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm
    
    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """Create the Gemini LLM instance."""
        logger.info(f"Creating Gemini LLM: {self.model_name}")
        
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            google_api_key=self.api_key,
            convert_system_message_to_human=True,  # Gemini compatibility
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt/question.
            system_prompt: Optional system instructions.
            
        Returns:
            Generated response text.
        """
        llm = self.get_llm()
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        logger.debug(f"Generating response for: {prompt[:50]}...")
        
        response = llm.invoke(messages)
        
        logger.debug(f"Generated {len(response.content)} characters")
        return response.content
    
    def generate_with_context(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response using retrieved context (for RAG).
        
        Args:
            question: User's question.
            context: Retrieved document context.
            system_prompt: Optional system instructions.
            
        Returns:
            Generated response text.
        """
        default_system = """You are a helpful documentation assistant. 
Answer questions based on the provided context. 
If the context doesn't contain the answer, say so honestly.
Be concise but comprehensive."""

        system = system_prompt or default_system
        
        prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""
        
        return self.generate(prompt, system_prompt=system)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model details.
        """
        if self.model_name in LLM_MODELS:
            info = LLM_MODELS[self.model_name].copy()
            info["provider"] = info["provider"].value
        else:
            info = {"model_name": self.model_name}
        
        info.update({
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key_configured": bool(self.api_key),
        })
        
        return info


# Singleton instance
_default_llm_manager: Optional[LLMManager] = None


def get_llm(
    model_name: str = None,
    temperature: float = None,
    force_new: bool = False,
) -> BaseChatModel:
    """
    Get LLM instance (convenience function).
    
    Uses singleton pattern - LLM is only created once.
    
    Args:
        model_name: Model to use.
        temperature: Sampling temperature.
        force_new: Force creation of new instance.
        
    Returns:
        LangChain LLM instance.
    """
    global _default_llm_manager
    
    if force_new or _default_llm_manager is None:
        _default_llm_manager = LLMManager(
            model_name=model_name,
            temperature=temperature,
        )
    
    return _default_llm_manager.get_llm()


def generate_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    model_name: str = None,
) -> str:
    """
    Generate a response (convenience function).
    
    Args:
        prompt: User prompt.
        system_prompt: Optional system instructions.
        model_name: Model to use.
        
    Returns:
        Generated response text.
    """
    global _default_llm_manager
    
    if _default_llm_manager is None:
        _default_llm_manager = LLMManager(model_name=model_name)
    
    return _default_llm_manager.generate(prompt, system_prompt)


def list_available_models() -> Dict[str, Dict]:
    """
    List all pre-configured LLM models.
    
    Returns:
        Dictionary of model configurations.
    """
    result = {}
    for name, config in LLM_MODELS.items():
        result[name] = {
            **config,
            "provider": config["provider"].value,
        }
    return result
