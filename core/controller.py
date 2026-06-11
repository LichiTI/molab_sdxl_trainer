import time
from core.loader import BatchLoader
from core.screener import ImageScreener
from core.tagger import TagProcessor
from core.exporter import BatchExporter
import logging

logger = logging.getLogger("WorkflowController")

class WorkflowController:

    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.is_paused = False
        self.is_stopped = False
        self.loader = BatchLoader()
        self.screener = ImageScreener()
        self.tagger = TagProcessor()
        self.exporter = BatchExporter()


    def run(self, input_dir, output_dir, steps):
        try:
            self._update_progress(0.1, '正在加载图片...', 0, 0)
            images, filenames = self.loader.load(input_dir, limit=0)
            total_images = len(images)
            if total_images == 0:
                self._update_progress(1.0, '未找到图片', 0, 0)
                return False
            self._update_progress(0.2, f'已加载 {total_images} 张图片', total_images, total_images)
            tags_dict = {}
            for i, step in enumerate(steps):
                if self.is_stopped:
                    self._update_progress(0, '已停止', 0, total_images)
                    return False
                while self.is_paused:
                    time.sleep(0.1)
                progress_base = 0.2 + (i + 1) * (0.6 / len(steps))
                if step['type'] == 'scorer':
                    self._update_progress(progress_base, '正在智能评分...', len(images), total_images)
                    config = step.get('config', {'method': 'Standard', 'top_k': 50})
                    
                    # Fix: Pass filenames and handle list[dict] return type
                    results = self.screener.score_and_filter(images, filenames, config)
                    
                    # Unpack results back to lists
                    images = [r['image'] for r in results]
                    filenames = [r['filename'] for r in results]
                elif step['type'] == 'tagger':
                    self._update_progress(progress_base, '正在处理标签...', len(images), total_images)
                    config = step.get('config', {'trigger_word': 'my_style', 'blacklist': 'watermark, text', 'ordering': 'Original'})
                    tags_dict = self.tagger.process(images, filenames, config)
            self._update_progress(0.9, '正在导出结果...', len(images), total_images)
            export_config = {'format': 'png', 'quality': 95, 'save_tags': True}
            success_count = self.exporter.export(images, filenames, tags_dict, output_dir, export_config)
            self._update_progress(1.0, f'处理完成！共导出 {success_count} 张图片', success_count, total_images)
            return True
        except Exception as e:
            logger.error(f'[WorkflowController] Error: {e}', exc_info=True)
            self._update_progress(0, f'错误: {str(e)}', 0, 0)
            return False

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.is_stopped = True

    def _update_progress(self, percent, message, current=0, total=0):
        if self.progress_callback:
            self.progress_callback(percent, message, current, total)