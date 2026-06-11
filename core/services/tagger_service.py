from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from PIL import Image
from core.job_manager import Job, JobType, JobStatus
from core.locator import Locator
from core.services.native_module_loader import native_with_entrypoints
import logging

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def native_tagger_image_listing_api() -> Any:
    return native_with_entrypoints('list_image_files')

class TaggerService:

    @staticmethod
    def submit_tagging_job(input_path: str, output_path: str='', config: Dict[str, Any]=None) -> str:
        if config is None:
            config = {}
        input_dir = Path(input_path)
        image_count = TaggerService._count_images(input_dir, config.get('recursive', False))
        job = Job(type=JobType.TAGGING, name=f'打标: {input_dir.name}', total_items=image_count, metadata={'input_path': input_path, 'output_path': output_path, 'config': config})
        job_id = Locator.jobs.submit(job, worker_func=TaggerService._run_tagging, args=(input_path, output_path, config))
        return job_id

    @staticmethod
    def _count_images(input_dir: Path, recursive: bool=False) -> int:
        return len(TaggerService._list_images(input_dir, recursive=recursive))

    @staticmethod
    def _list_images(input_dir: Path, recursive: bool=False) -> List[Path]:
        native = native_tagger_image_listing_api()
        if native is not None:
            try:
                image_paths = native.list_image_files(str(input_dir), bool(recursive))
                if isinstance(image_paths, list):
                    return sorted(
                        Path(str(path))
                        for path in image_paths
                        if Path(str(path)).suffix.lower() in IMAGE_EXTENSIONS
                    )
            except Exception:
                pass
        iterator = input_dir.rglob('*') if recursive else input_dir.iterdir()
        return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

    @staticmethod
    def _run_tagging(input_path: str, output_path: str, config: Dict[str, Any], progress_callback: Callable[[int, int], None]=None, cancel_check: Callable[[], bool]=None) -> Dict[str, Any]:
        input_dir = Path(input_path)
        output_dir = Path(output_path) if output_path else input_dir
        recursive = config.get('recursive', False)
        image_files = TaggerService._list_images(input_dir, recursive=recursive)
            
        if not image_files:
            raise ValueError('没有找到图片')
            
        total = len(image_files)
        method = config.get('method', 'wd14')
        results = {}
        total_tokens = 0
        saved_count = 0
        
        output_dir.mkdir(parents=True, exist_ok=True)

        if method in {'noop', 'stub'}:
            for idx, image_file in enumerate(image_files, start=1):
                if cancel_check and cancel_check():
                    raise InterruptedError('任务已取消')
                if progress_callback:
                    progress_callback(idx, total)
                tags = config.get('default_tags', '')
                results[image_file.name] = tags
                (output_dir / f'{image_file.stem}.txt').write_text(tags, encoding='utf-8')
                saved_count += 1
            return {
                'results': results,
                'saved_count': saved_count,
                'output_dir': str(output_dir),
                'total_tokens': 0,
                'truncated': False,
            }
        
        # Initialize Processor for WD14 once
        _truncation_warned = False
        wd14_processor = None
        if method != 'gemini':
            from core.tagger import TagProcessor
            model_name = config.get('model', 'wd-vit-v3')
            wd14_processor = TagProcessor(method='wd14', model_name=model_name)
            
        try:
            # Batch processing settings
            BATCH_SIZE = 32
            
            for i in range(0, total, BATCH_SIZE):
                if cancel_check and cancel_check():
                    raise InterruptedError('任务已取消')
                    
                batch_files = image_files[i:i+BATCH_SIZE]
                batch_images = []
                batch_filenames = []
                
                # Load Batch
                for f in batch_files:
                    try:
                        # Use context manager to prevent file handle leak
                        with Image.open(f) as img_file:
                            img = img_file.convert('RGB')
                            # Force load image data into memory before file closes
                            img.load()
                            batch_images.append(img)
                        batch_filenames.append(f.name)
                    except Exception as ex:
                        logger.error(f'[TaggerService] Failed to load {f}: {ex}')
                
                if not batch_images:
                    continue
                    
                # Process Batch
                batch_results = {}
                if method == 'gemini':
                    # Calculate global index for progress
                    def batch_progress(curr, tot):
                        if progress_callback:
                             progress_callback(i + curr, total)
                    
                    res, tokens = TaggerService._run_llm_tagging(
                        batch_images, batch_filenames, output_dir, config, 
                        progress_callback=batch_progress, cancel_check=cancel_check
                    )
                    batch_results = res
                    total_tokens += tokens
                elif wd14_processor:
                    # Provide a wrapped callback that maps batch progress to global progress
                    def wrapped_callback(current, batch_total):
                        if cancel_check and cancel_check():
                            raise InterruptedError('任务已取消')
                        if progress_callback:
                            # current is 1-based index in batch
                            global_current = i + current
                            progress_callback(global_current, total)
                            
                    batch_results = wd14_processor.process(batch_images, batch_filenames, config, wrapped_callback)
                
                # Only update results if we haven't exceeded a memory safety limit (e.g., 50k items)
                # For very large datasets, we rely on disk files, returning full results dict might be too heavy.
                if len(results) < 50000:
                    results.update(batch_results)
                elif not _truncation_warned:
                    logger.warning(f"[TaggerService] Result limit (50000) reached. Truncating in-memory results.")
                    _truncation_warned = True
                
                # Save immediately (reduce memory pressure and data loss risk)
                for filename, tags in batch_results.items():
                    try:
                        base_name = Path(filename).stem
                        txt_file = output_dir / f'{base_name}.txt'
                        with open(txt_file, 'w', encoding='utf-8') as f:
                            f.write(tags)
                        saved_count += 1
                    except Exception as ex:
                        logger.error(f'[TaggerService] Failed to save {filename}: {ex}')
                        
                # Explicitly clear batch images to free memory
                del batch_images
                del batch_filenames
                
        finally:
            if wd14_processor:
                wd14_processor.unload()
                
        return {
            'results': results, 
            'saved_count': saved_count, 
            'output_dir': str(output_dir), 
            'total_tokens': total_tokens,
            'truncated': _truncation_warned
        }

    # Removed _run_wd14_tagging as it is now inlined/replaced


    @staticmethod
    def _run_llm_tagging(images: List[Image.Image], filenames: List[str], output_dir: Path, config: Dict[str, Any], progress_callback: Callable=None, cancel_check: Callable=None) -> tuple:
        from core.llm_client import LLMClient
        from core.gemini_tagger import GeminiTagger
        llm_config = config.get('llm_settings', {})
        provider = llm_config.get('provider', 'gemini')
        api_key = llm_config.get('api_key', '')
        if provider != 'ollama' and (not api_key):
            raise ValueError('请先配置 LLM API Key')
        llm_client = LLMClient(provider=provider, api_key=api_key, base_url=llm_config.get('base_url', ''), model=config.get('gemini_model', llm_config.get('model', '')), proxy=llm_config.get('proxy', ''), safety_none=llm_config.get('safety_none', True))
        system_prompt = config.get('gemini_prompt', GeminiTagger.DEFAULT_PROMPT)
        user_prompt = f'Here are some examples of tagging:\n{GeminiTagger.DEFAULT_EXAMPLES}\n\nPlease tag this image.'
        results = {}
        skip_existing = config.get('gemini_skip_existing', True)
        # Helper for parsing tags from LLM response
        # We use a dummy instance because we only need _extract_tags method
        tagger = GeminiTagger(api_key="parsing_helper", base_url="", model="parsing", system_prompt="")
        for i, (img, fname) in enumerate(zip(images, filenames)):
            if cancel_check and cancel_check():
                raise InterruptedError('任务已取消')
            if progress_callback:
                progress_callback(i + 1, len(images))
            txt_file = output_dir / f'{Path(fname).stem}.txt'
            if skip_existing and txt_file.exists() and (txt_file.stat().st_size > 0):
                continue
            raw_text = llm_client.chat_with_image(img, system_prompt, user_prompt)
            if raw_text:
                tags = tagger._extract_tags(raw_text)
                trigger_word = config.get('trigger_word', '')
                if trigger_word and tags:
                    tags = f'{trigger_word}, {tags}'
                if tags:
                    results[fname] = tags
        total_tokens = llm_client.total_usage.total_tokens if llm_client.total_usage else 0
        return (results, total_tokens)

def submit_tagging(input_path: str, output_path: str='', config: Dict=None) -> str:
    return TaggerService.submit_tagging_job(input_path, output_path, config)
