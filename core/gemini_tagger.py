import re
import base64
import io
import requests
from pathlib import Path
from PIL import Image
from typing import Optional, Callable
import logging

logger = logging.getLogger("GeminiTagger")

class GeminiTagger:
    DEFAULT_PROMPT = 'You are an e621 image tagger. Output ONLY comma-separated tags. NO explanations, NO thinking, NO markdown, NO backticks. Do NOT include character names or art styles. Just output: tag1, tag2, tag3, ...'
    DEFAULT_EXAMPLES = 'Example output format:\nsolo, anthro, wolf, canine, mammal, male, blue_fur, looking_at_viewer, simple_background'

    def __init__(self, api_key: str, base_url: str=None, proxy: str=None, model: str='gemini-1.5-flash', safety_none: bool=True):
        self.api_key = api_key
        self.base_url = base_url or 'https://generativelanguage.googleapis.com'
        self.proxy = proxy
        self.model = model
        self.safety_none = safety_none
        self.system_prompt = self.DEFAULT_PROMPT
        self.examples = self.DEFAULT_EXAMPLES
        self.prefix_tags = ''

    def set_prompt(self, system_prompt: str, examples: str=None):
        self.system_prompt = system_prompt or self.DEFAULT_PROMPT
        if examples:
            self.examples = examples

    def set_prefix_tags(self, tags: str):
        self.prefix_tags = tags

    def _image_to_base64(self, image: Image.Image, max_size: int=1024) -> tuple:
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size))
        buffer = io.BytesIO()
        img_format = 'PNG' if image.mode == 'RGBA' else 'JPEG'
        image.save(buffer, format=img_format)
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mime_type = 'image/png' if img_format == 'PNG' else 'image/jpeg'
        return (base64_data, mime_type)

    def _extract_tags(self, raw_text: str) -> str:
        backtick_tags = re.findall('`([a-z0-9_]+(?:\\([^)]+\\))?)`', raw_text.lower())
        if len(backtick_tags) >= 5:
            tag_set = set(backtick_tags)
        else:
            cleaned = raw_text
            cleaned = re.sub('```[a-z]*\\n?', '', cleaned)
            cleaned = cleaned.replace('```', '')
            cleaned = cleaned.replace('`', '')
            prefixes = ['^tags?:\\s*', '^output:\\s*', '^result:\\s*', 'here are the tags:\\s*', 'the tags are:\\s*']
            for prefix in prefixes:
                cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
            parts = re.split('[,\\n]', cleaned)
            tag_set = set()
            for part in parts:
                tag = part.strip().lower()
                tag = tag.strip('.,;:!?\'"')
                tag = tag.replace(' ', '_')
                if re.match('^[a-z0-9_]+(?:\\([a-z0-9_]+\\))?$', tag) and 2 <= len(tag) <= 50:
                    tag_set.add(tag)
        invalid_tags = {'the', 'and', 'for', 'this', 'that', 'with', 'from', 'tags', 'tag', 'output', 'result', 'example', 'image'}
        tag_set = tag_set - invalid_tags
        return ', '.join(sorted(tag_set))

    def tag_image(self, image: Image.Image) -> Optional[str]:
        try:
            img_base64, mime_type = self._image_to_base64(image)
            final_base = self.base_url.rstrip('/')
            api_url = f'{final_base}/v1beta/models/{self.model}:generateContent'
            headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
            user_content = f'Here are some examples of tagging:\n{self.examples}\n\nPlease tag this image following the instructions and style shown in the examples.'
            payload = {'contents': [{'role': 'user', 'parts': [{'text': user_content}, {'inlineData': {'mimeType': mime_type, 'data': img_base64}}]}], 'systemInstruction': {'parts': [{'text': self.system_prompt}]}, 'generationConfig': {'maxOutputTokens': 1024}}
            if self.safety_none:
                payload['safetySettings'] = [{'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'}, {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}]
            proxies = {'http': self.proxy, 'https': self.proxy} if self.proxy else None
            response = requests.post(api_url, headers=headers, json=payload, proxies=proxies, timeout=120)
            response.raise_for_status()
            result = response.json()
            tags_raw = ''
            candidates = result.get('candidates', [])
            if candidates:
                parts = candidates[0].get('content', {}).get('parts', [])
                for part in parts:
                    if 'text' in part and (not part.get('thought', False)):
                        tags_raw += part['text']
            if not tags_raw.strip():
                return None
            final_tags = self._extract_tags(tags_raw)
            if self.prefix_tags:
                prefix_list = [t.strip().lower().replace(' ', '_') for t in self.prefix_tags.split(',') if t.strip()]
                if prefix_list:
                    prefix_str = ', '.join(prefix_list)
                    final_tags = f'{prefix_str}, {final_tags}' if final_tags else prefix_str
            return final_tags if final_tags else None
        except Exception as e:
            logger.error(f'[GeminiTagger] Error: {e}')
            return None

    def tag_folder(self, folder_path: Path, skip_existing: bool=True, progress_callback: Callable[[int, int, str], None]=None) -> dict:
        folder = Path(folder_path)
        extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
        image_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in extensions]
        stats = {'success': 0, 'skipped': 0, 'failed': 0}
        total = len(image_files)
        for i, img_file in enumerate(image_files):
            txt_file = img_file.with_suffix('.txt')
            if skip_existing and txt_file.exists() and (txt_file.stat().st_size > 0):
                stats['skipped'] += 1
                if progress_callback:
                    progress_callback(i + 1, total, f'跳过: {img_file.name}')
                continue
            if progress_callback:
                progress_callback(i + 1, total, f'处理: {img_file.name}')
            try:
                with Image.open(img_file) as opened:
                    image = opened.convert('RGB')
                tags = self.tag_image(image)
                if tags:
                    txt_file.write_text(tags, encoding='utf-8')
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
            except Exception as e:
                logger.error(f'[GeminiTagger] Failed to process {img_file.name}: {e}')
                stats['failed'] += 1
        return stats
