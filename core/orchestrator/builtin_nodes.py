from typing import Dict, Any, Optional, Callable, List
from PIL import Image
from pathlib import Path
from .base_node import BaseNode, NodeResult, ConfigField
from .node_registry import register_node

@register_node('scorer')
class ScorerNode(BaseNode):
    _scorer = None
    _model_name = None

    @staticmethod
    def get_name() -> str:
        return '图片评分'

    @staticmethod
    def get_description() -> str:
        return '使用 CLIP 模型计算图片美学分数'

    @staticmethod
    def get_icon() -> str:
        return '⭐'

    @staticmethod
    def get_node_type() -> str:
        return BaseNode.TYPE_FILTER

    @staticmethod
    def get_config_schema() -> Dict[str, ConfigField]:
        return {
            'min_score': ConfigField('float', 0.5, '最低分数', description='低于此分数的图片将被过滤', range=(0.0, 1.0)),
            'model': ConfigField('str', 'openai/clip-vit-base-patch32', '模型', description='HuggingFace ID 或本地路径')
        }

    def on_load(self):
        model_name = self.config.get('model', 'openai/clip-vit-base-patch32')
        if ScorerNode._scorer is None or ScorerNode._model_name != model_name:
            try:
                from core.screener import CLIPScorer
                ScorerNode._scorer = CLIPScorer(model_path=model_name)
                ScorerNode._model_name = model_name
            except Exception as e:
                self.logger.error(f'Failed to load CLIP scorer: {e}')
        super().on_load()

    def on_unload(self):
        if ScorerNode._scorer:
            ScorerNode._scorer.unload()
            ScorerNode._scorer = None
        super().on_unload()

    def process(self, data: Any, progress_callback: Optional[Callable]=None) -> NodeResult:
        try:
            min_score = self.config.get('min_score', 0.5)
            if isinstance(data, (str, Path)):
                image = Image.open(data).convert('RGB')
            elif isinstance(data, Image.Image):
                image = data
            elif isinstance(data, dict) and 'image' in data:
                image = data['image']
            else:
                return NodeResult(success=False, error='无效的输入数据类型')
            if ScorerNode._scorer is None:
                self.on_load()
            score = ScorerNode._scorer.score(image)
            passed = score >= min_score
            return NodeResult(success=True, data=passed, metadata={'score': score})
        except Exception as e:
            return NodeResult(success=False, error=str(e))

@register_node('cropper')
class CropperNode(BaseNode):
    _cropper = None

    @staticmethod
    def get_name() -> str:
        return '智能裁剪'

    @staticmethod
    def get_description() -> str:
        return '根据分辨率桶裁剪图片到训练尺寸'

    @staticmethod
    def get_icon() -> str:
        return '✂️'

    @staticmethod
    def get_node_type() -> str:
        return BaseNode.TYPE_TRANSFORM

    @staticmethod
    def get_config_schema() -> Dict[str, ConfigField]:
        return {'mode': ConfigField('choice', 'bucket', '裁剪模式', choices=['bucket', 'ratio']), 'ratio': ConfigField('choice', '1:1', '固定比例', choices=['1:1', '4:3', '3:4', '16:9', '9:16']), 'exact_crop': ConfigField('bool', True, '精确裁剪', description='精确裁剪到目标尺寸')}

    def on_load(self):
        if CropperNode._cropper is None:
            try:
                from core.cropper import ImageCropper
                CropperNode._cropper = ImageCropper()
            except Exception as e:
                self.logger.error(f'Failed to load cropper: {e}')
        super().on_load()

    def process(self, data: Any, progress_callback: Optional[Callable]=None) -> NodeResult:
        try:
            filename = 'image'
            metadata = {}
            if isinstance(data, (str, Path)):
                image = Image.open(data).convert('RGB')
                filename = Path(data).name
            elif isinstance(data, Image.Image):
                image = data
            elif isinstance(data, dict) and 'image' in data:
                image = data['image']
                filename = data.get('filename', 'image')
                metadata = data.get('metadata', {}) or {}
            else:
                return NodeResult(success=False, error='无效的输入数据类型')
            if CropperNode._cropper is None:
                self.on_load()
            config = {'mode': self.config.get('mode', 'bucket'), 'ratio': self.config.get('ratio', '1:1'), 'exact_crop': self.config.get('exact_crop', True), 'target_size': 1024}
            cropped = CropperNode._cropper.process_single(image, config)
            payload = {
                'image': cropped,
                'filename': filename,
                'metadata': {**metadata, 'original_size': image.size, 'new_size': cropped.size}
            }
            return NodeResult(success=True, data=payload, metadata={'original_size': image.size, 'new_size': cropped.size})
        except Exception as e:
            return NodeResult(success=False, error=str(e))

@register_node('tagger')
class TaggerNode(BaseNode):
    _tagger = None
    _signature = None

    @staticmethod
    def get_name() -> str:
        return '自动打标'

    @staticmethod
    def get_description() -> str:
        return '使用 WD14/BLIP 生成图片标签'

    @staticmethod
    def get_icon() -> str:
        return '🏷️'

    @staticmethod
    def get_node_type() -> str:
        return BaseNode.TYPE_ANALYZE

    @staticmethod
    def get_config_schema() -> Dict[str, ConfigField]:
        return {
            'method': ConfigField('choice', 'wd14', '??', choices=['wd14', 'blip', 'llm', 'manual']),
            'model': ConfigField('choice', 'wd-vit-v3', '??', choices=['wd-vit-v3', 'wd-convnext-v3', 'wd-swinv2-v3']),
            'threshold': ConfigField('float', 0.35, '??', range=(0.0, 1.0)),
            'trigger_word': ConfigField('str', '', '???', description='???????'),
            'blacklist': ConfigField('str', '', '???', description='??????????'),
            'llm_provider': ConfigField('str', 'gemini', 'LLM Provider', description='gemini/openai/claude/qwen/ollama/openrouter/custom'),
            'llm_model': ConfigField('str', '', 'LLM Model', description='LLM model id'),
            'llm_api_key': ConfigField('str', '', 'LLM API Key', description='API key for LLM provider'),
            'llm_base_url': ConfigField('str', '', 'LLM Base URL', description='Optional custom base URL')
        }

    def on_load(self):
        method = self.config.get('method', 'wd14')
        model = self.config.get('model', 'wd-vit-v3')
        llm_provider = self.config.get('llm_provider', 'gemini')
        llm_model = self.config.get('llm_model', '')
        llm_api_key = self.config.get('llm_api_key', '')
        llm_base_url = self.config.get('llm_base_url', '')
        signature = (method, model, llm_provider, llm_model, llm_api_key, llm_base_url)
        if TaggerNode._tagger is None or TaggerNode._signature != signature:
            try:
                from core.tagger import TagProcessor
                TaggerNode._tagger = TagProcessor(
                    method=method,
                    model_name=model,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key,
                    llm_base_url=llm_base_url
                )
                TaggerNode._signature = signature
            except Exception as e:
                self.logger.error(f'Failed to load tagger: {e}')
        super().on_load()

    def on_unload(self):
        if TaggerNode._tagger:
            TaggerNode._tagger.unload()
            TaggerNode._tagger = None
            TaggerNode._signature = None
        super().on_unload()

    def process(self, data: Any, progress_callback: Optional[Callable]=None) -> NodeResult:
        try:
            if isinstance(data, (str, Path)):
                image = Image.open(data).convert('RGB')
                filename = Path(data).name
            elif isinstance(data, Image.Image):
                image = data
                filename = 'image.jpg'
            elif isinstance(data, dict):
                image = data.get('image')
                filename = data.get('filename', 'image.jpg')
            else:
                return NodeResult(success=False, error='无效的输入数据类型')
            if TaggerNode._tagger is None:
                self.on_load()
            config = {'threshold': self.config.get('threshold', 0.35), 'trigger_word': self.config.get('trigger_word', ''), 'blacklist': self.config.get('blacklist', '')}
            results = TaggerNode._tagger.process([image], [filename], config)
            tags = results.get(filename, '')
            metadata = {}
            if isinstance(data, dict):
                metadata = data.get('metadata', {}) or {}
            payload = {
                'image': image,
                'filename': filename,
                'metadata': {**metadata, 'tags': tags}
            }
            return NodeResult(success=True, data=payload, metadata={'tags': tags, 'filename': filename})
        except Exception as e:
            return NodeResult(success=False, error=str(e))

@register_node('saver')
class SaveNode(BaseNode):

    @staticmethod
    def get_name() -> str:
        return '保存文件'

    @staticmethod
    def get_description() -> str:
        return '将处理后的图片和标签保存到指定目录'

    @staticmethod
    def get_icon() -> str:
        return '💾'

    @staticmethod
    def get_node_type() -> str:
        return BaseNode.TYPE_OUTPUT

    @staticmethod
    def get_config_schema() -> Dict[str, ConfigField]:
        return {'output_dir': ConfigField('str', './output', '输出目录', description='保存文件的目标目录'), 'save_image': ConfigField('bool', True, '保存图片', description='是否保存处理后的图片'), 'save_tags': ConfigField('bool', True, '保存标签', description='是否将标签保存为 .txt 文件'), 'image_format': ConfigField('choice', 'png', '图片格式', choices=['png', 'jpg', 'webp']), 'overwrite': ConfigField('bool', False, '覆盖现有文件', description='如果文件已存在是否覆盖')}

    def process(self, data: Any, progress_callback: Optional[Callable]=None) -> NodeResult:
        try:
            output_dir = Path(self.config.get('output_dir', './output'))
            save_image = self.config.get('save_image', True)
            save_tags = self.config.get('save_tags', True)
            image_format = self.config.get('image_format', 'png')
            overwrite = self.config.get('overwrite', False)
            output_dir.mkdir(parents=True, exist_ok=True)
            image = None
            filename = 'image'
            tags = None
            metadata = {}
            if isinstance(data, dict):
                image = data.get('image')
                # [SECURITY] Sanitize filename to prevent traversal or injection
                raw_name = data.get('filename', 'image')
                if isinstance(raw_name, (str, Path)):
                    filename = Path(raw_name).stem
                    filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).strip()
                else:
                    filename = 'image'
                    
                metadata = data.get('metadata', {})
                tags = metadata.get('tags', data.get('tags'))
            elif isinstance(data, Image.Image):
                image = data
            elif isinstance(data, (str, Path)):
                image = Image.open(data).convert('RGB')
                # [SECURITY] Sanitize filename
                raw_name = Path(data).stem
                filename = "".join(c for c in raw_name if c.isalnum() or c in (' ', '-', '_')).strip()
            saved_files = []
            if save_image and image:
                ext_map = {'png': '.png', 'jpg': '.jpg', 'webp': '.webp'}
                ext = ext_map.get(image_format, '.png')
                image_path = output_dir / f'{filename}{ext}'
                if overwrite or not image_path.exists():
                    if image_format == 'jpg':
                        image.save(image_path, quality=95)
                    elif image_format == 'webp':
                        image.save(image_path, quality=95)
                    else:
                        image.save(image_path)
                    saved_files.append(str(image_path))
            if save_tags and tags:
                txt_path = output_dir / f'{filename}.txt'
                if overwrite or not txt_path.exists():
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(tags)
                    saved_files.append(str(txt_path))
            return NodeResult(success=True, data=data, metadata={'saved_files': saved_files, 'output_dir': str(output_dir)})
        except Exception as e:
            return NodeResult(success=False, error=str(e))

@register_node('loader')
class LoaderNode(BaseNode):

    @staticmethod
    def get_name() -> str:
        return '加载图片'

    @staticmethod
    def get_description() -> str:
        return '从指定目录加载图片文件'

    @staticmethod
    def get_icon() -> str:
        return '📁'

    @staticmethod
    def get_node_type() -> str:
        return BaseNode.TYPE_INPUT

    @staticmethod
    def get_config_schema() -> Dict[str, ConfigField]:
        return {'input_dir': ConfigField('str', '', '输入目录', description='包含图片的目录路径'), 'extensions': ConfigField('str', 'jpg,jpeg,png,webp,bmp', '文件扩展名', description='逗号分隔的扩展名列表'), 'recursive': ConfigField('bool', False, '递归搜索', description='是否搜索子目录')}

    def process(self, data: Any, progress_callback: Optional[Callable]=None) -> NodeResult:
        try:
            input_dir = Path(self.config.get('input_dir', '')).resolve()
            extensions = self.config.get('extensions', 'jpg,jpeg,png,webp,bmp')
            recursive = self.config.get('recursive', False)
            
            # [SECURITY] Traversal protection: Ensure it is a valid directory
            if not input_dir.exists():
                return NodeResult(success=False, error=f'目录不存在: {input_dir}')
            if not input_dir.is_dir():
                return NodeResult(success=False, error=f'路径不是目录: {input_dir}')
                
            ext_list = [f'.{e.strip().lower()}' for e in extensions.split(',')]
            files = []
            if recursive:
                for ext in ext_list:
                    files.extend(input_dir.rglob(f'*{ext}'))
            else:
                for ext in ext_list:
                    files.extend(input_dir.glob(f'*{ext}'))
            file_list = [str(f) for f in sorted(files)]
            return NodeResult(success=True, data=file_list, metadata={'count': len(file_list), 'input_dir': str(input_dir)})
        except Exception as e:
            return NodeResult(success=False, error=str(e))
