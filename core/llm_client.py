import base64
import io
import os
import logging
try:
    import requests
except ImportError:
    requests = None
    
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image
from dataclasses import dataclass

logger = logging.getLogger("LLMClient")

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
LLM_PRESETS = {
    'gemini': {'name': 'Google Gemini', 'api_format': 'gemini', 'base_url': 'https://generativelanguage.googleapis.com', 'models': ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash-exp'], 'default_model': 'gemini-1.5-flash', 'supports_vision': True},
    'openai': {'name': 'OpenAI', 'api_format': 'openai_chat', 'base_url': 'https://api.openai.com', 'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'], 'default_model': 'gpt-4o-mini', 'supports_vision': True},
    'openai_responses': {'name': 'OpenAI Responses', 'api_format': 'openai_responses', 'base_url': 'https://api.openai.com', 'models': ['gpt-4o', 'gpt-4o-mini'], 'default_model': 'gpt-4o-mini', 'supports_vision': False},
    'claude': {'name': 'Anthropic Claude', 'api_format': 'anthropic_messages', 'base_url': 'https://api.anthropic.com', 'models': ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307'], 'default_model': 'claude-3-5-sonnet-20241022', 'supports_vision': True},
    'qwen': {'name': '通义千问', 'api_format': 'openai_chat', 'base_url': 'https://dashscope.aliyuncs.com/compatible-mode', 'models': ['qwen-plus', 'qwen-turbo', 'qwen-max', 'qwen-vl-max', 'qwen-vl-plus'], 'default_model': 'qwen-plus', 'supports_vision': True},
    'ollama': {'name': 'Ollama (本地)', 'api_format': 'openai_chat', 'base_url': 'http://localhost:11434', 'models': ['llama3.1', 'qwen2.5', 'gemma2', 'llava'], 'default_model': 'llama3.1', 'supports_vision': True, 'note': '需要本地运行 Ollama'},
    'openrouter': {'name': 'OpenRouter', 'api_format': 'openai_chat', 'base_url': 'https://openrouter.ai/api', 'models': ['google/gemini-flash-1.5', 'anthropic/claude-3.5-sonnet', 'openai/gpt-4o'], 'default_model': 'google/gemini-flash-1.5', 'supports_vision': True, 'note': '聚合多个模型'},
    'custom': {'name': '自定义兼容接口', 'api_format': 'openai_chat', 'base_url': '', 'models': [], 'default_model': '', 'supports_vision': True},
}

class LLMClient:
    """
    Lightweight client for LLM services. 
    Now delegates to LLMEngine for central resource management.
    """

    def __init__(self, provider: str='gemini', api_key: str='', base_url: str='', model: str='', proxy: str='', safety_none: bool=True):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.proxy = proxy
        self.safety_none = safety_none
        self.last_usage: Optional[TokenUsage] = None
        self.total_usage = TokenUsage()

    def _image_to_base64(self, image: Image.Image, max_size: int=1024) -> tuple:
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size))
        buffer = io.BytesIO()
        img_format = 'PNG' if image.mode == 'RGBA' else 'JPEG'
        image.save(buffer, format=img_format)
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mime_type = 'image/png' if img_format == 'PNG' else 'image/jpeg'
        return (base64_data, mime_type)

    async def chat(self, messages: List[Dict[str, str]], config: Dict[str, Any] = None) -> str:
        """Text-only chat via LLMEngine."""
        from .llm_engine import llm_engine
        conf = {
            "provider": self.provider,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "proxy": self.proxy,
            **(config or {})
        }
        return await llm_engine.chat(messages, conf)

    def chat_with_image(self, image: Image.Image, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Vision-based chat. 
        Note: Currently still uses direct requests for online APIs. 
        TODO: Move this logic into LLMEngine as well.
        """
        # For now, maintain backward compatibility for existing Tagger/Scorer logic
        preset = LLM_PRESETS.get(self.provider, LLM_PRESETS['custom'])
        api_format = preset.get('api_format', 'openai')
        
        # Priority: explicit api_key or base_url
        effective_base = self.base_url or preset.get('base_url', '')
        
        # SECURITY CHECK (Audit Fix)
        # Ensure we have an API key before making external requests
        if not self.api_key:
            logger.error(f"[LLMClient] API Key missing for provider '{self.provider}'. Request blocked for security.")
            return None
        
        if api_format == 'gemini':
            return self._chat_gemini_legacy(image, system_prompt, user_prompt, effective_base)
        else:
            return self._chat_openai_legacy(image, system_prompt, user_prompt, effective_base)

    def _chat_gemini_legacy(self, image: Image.Image, system_prompt: str, user_prompt: str, base_url: str) -> Optional[str]:
        try:
            img_base64, mime_type = self._image_to_base64(image)
            final_base = base_url.rstrip('/')
            model_name = self.model or LLM_PRESETS['gemini']['default_model']
            api_url = f'{final_base}/v1beta/models/{model_name}:generateContent'
            headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
            payload = {'contents': [{'role': 'user', 'parts': [{'text': user_prompt}, {'inlineData': {'mimeType': mime_type, 'data': img_base64}}]}], 'systemInstruction': {'parts': [{'text': system_prompt}]}, 'generationConfig': {'maxOutputTokens': 1024}}
            if self.safety_none:
                payload['safetySettings'] = [{'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'}, {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}]
            proxies = {'http': self.proxy, 'https': self.proxy} if self.proxy else None
            # Handle potential missing API key for Gemini when using OpenRouter but preset is gemini (edge case)
            if not self.api_key:
                logger.warning("[LLMClient] Gemini API key missing.")
            
            response = requests.post(api_url, headers=headers, json=payload, proxies=proxies, timeout=120)
            response.raise_for_status()
            result = response.json()
            text = ''
            candidates = result.get('candidates', [])
            if candidates:
                parts = candidates[0].get('content', {}).get('parts', [])
                for part in parts:
                    if 'text' in part and (not part.get('thought', False)):
                        text += part['text']
            return text.strip() if text else None
        except Exception as e:
            logger.error(f'[LLMClient] Gemini error: {e}')
            return None

    def _chat_openai_legacy(self, image: Image.Image, system_prompt: str, user_prompt: str, base_url: str) -> Optional[str]:
        try:
            img_base64, mime_type = self._image_to_base64(image)
            final_base = base_url.rstrip('/')
            api_url = f'{final_base}/v1/chat/completions'
            headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
            model_name = self.model or LLM_PRESETS.get(self.provider, {}).get('default_model', 'gpt-4o-mini')
            payload = {'model': model_name, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': [{'type': 'text', 'text': user_prompt}, {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{img_base64}'}}]}], 'max_tokens': 1024}
            proxies = {'http': self.proxy, 'https': self.proxy} if self.proxy else None
            response = requests.post(api_url, headers=headers, json=payload, proxies=proxies, timeout=120)
            response.raise_for_status()
            result = response.json()
            text = ''
            choices = result.get('choices', [])
            if choices:
                text = choices[0].get('message', {}).get('content', '')
            return text.strip() if text else None
        except Exception as e:
            logger.error(f'[LLMClient] OpenAI error: {e}')
            return None

    def fetch_models(self) -> list:
        # Re-using the same logic for now, or could delegate to Engine
        from .llm_engine import llm_engine
        
        # Merge local GGUF models and online preset models
        local_models = [m['name'] for m in llm_engine.list_local_models()]
        
        # Add presets
        online_models = []
        preset = LLM_PRESETS.get(self.provider)
        if preset:
            online_models = preset.get('models', [])
            
        return list(set(local_models + online_models))
