"""
SVD 智能模型融合器 (SVD Smart Merger)

在谱域中分析和操作 LoRA 特征，提供超越简单权重平均的高级融合功能。

核心概念:
1. 谱分析 (Spectral Analysis): 分解 LoRA 权重的奇异值谱
2. 结构/细节分离: 低秩=宏观结构, 高秩=精细纹理
3. 冲突检测: 识别两个模型试图控制相同特征方向的情况
4. 智能融合: 基于谱域的精细控制

算法选项:
- WEIGHTED_SUM: 传统加权平均 (基线)
- SVD_SMART: 谱域智能融合 (推荐)
- ORTHOGONALIZE: 正交化冲突特征
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    # We might want to warn or log here, but let class handle it gracefully


import logging
logger = logging.getLogger("SVDMerger")


class MergeMode(Enum):
    """融合模式"""
    WEIGHTED_SUM = "weighted_sum"       # 传统加权平均
    SVD_SMART = "svd_smart"             # SVD 智能融合
    ORTHOGONALIZE = "orthogonalize"     # 正交化冲突特征


@dataclass
class LayerSpectrum:
    """单层谱分析结果"""
    name: str
    singular_values_a: np.ndarray
    singular_values_b: np.ndarray
    similarity_score: float  # 0-1, 越高越相似
    conflict_score: float    # 0-1, 越高冲突越大
    structure_ratio_a: float # 模型 A 的结构贡献 (低秩)
    detail_ratio_a: float    # 模型 A 的细节贡献 (高秩)
    stable_rank_a: float = 0.0
    stable_rank_b: float = 0.0


@dataclass
class MergeAnalysis:
    """融合分析结果"""
    layers: List[LayerSpectrum]
    overall_similarity: float
    overall_conflict: float
    recommended_mode: MergeMode
    sample_layer_names: List[str]


class SVDMerger:
    """
    SVD 智能模型融合器
    
    使用方式:
        merger = SVDMerger()
        analysis = merger.analyze(model_a_path, model_b_path)
        result = merger.merge(model_a_path, model_b_path, output_path, 
                              mode=MergeMode.SVD_SMART,
                              structure_bias=0.6, detail_bias=0.4)
    """
    
    # 代表性层 (用于快速分析, 避免分析全部层)
    SAMPLE_LAYERS = [
        "mid_block", "up_blocks.1", "down_blocks.2",
        "transformer.0", "transformer.4" , "transformer.10",
        "double_blocks.4", "double_blocks.10", "single_blocks.0"
    ]
    
    def __init__(
        self,
        use_gpu: bool = True,
        precision: str = "fp32"
    ):
        """
        Args:
            use_gpu: 使用 GPU 加速 (如果可用)
            precision: 计算精度 ("fp32" 或 "fp16")
        """
        self.use_gpu = use_gpu and HAS_TORCH and torch.cuda.is_available()
        self.precision = precision
        self.device = "cuda" if self.use_gpu else "cpu"
    
    def load_model(self, path: str) -> Dict[str, np.ndarray]:
        """加载 safetensors 模型并识别 LoRA 层"""
        from safetensors import safe_open
        
        tensors = {}
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                if self._is_lora_key(key):
                    tensors[key] = f.get_tensor(key)
        return tensors
    
    def _is_lora_key(self, key: str) -> bool:
        """判断是否为相关的权重键 (LoRA 或 Checkpoint UNet/DiT)"""
        key_lower = key.lower()
        # LoRA 关键字
        if any(x in key_lower for x in ["lora_down", "lora_up", "lora_a", "lora_b"]):
            return True
        # Checkpoint 关键字 (UNet/Transformer 权重)
        # 我们主要关注权重矩阵 (.weight)，过滤掉 bias, norm, layer_stats 等
        if ".weight" in key_lower:
            if any(x in key_lower for x in ["model.diffusion_model", "unet", "double_blocks", "single_blocks", "transformer"]):
                return True
        return False
    
    def _is_sample_layer(self, key: str) -> bool:
        """判断是否为代表性层 (用于快速分析)"""
        key_lower = key.lower()
        return any(sample in key_lower for sample in self.SAMPLE_LAYERS)
    
    def compute_svd(self, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算 SVD 分解
        
        AUDIT FIX: Added VRAM check before GPU computation to prevent OOM.
        Falls back to CPU if insufficient VRAM available.
        
        Args:
            weight: 2D 或 4D 权重矩阵
            
        Returns:
            U, S, Vh: SVD 分解结果
        """
        # 处理 4D 卷积权重
        if weight.ndim == 4:
            weight = weight.reshape(weight.shape[0], -1)
        elif weight.ndim != 2:
            raise ValueError(f"Unsupported weight dimension: {weight.ndim}")
        
        # 转换为 float32 以确保精度
        weight = weight.astype(np.float32)
        
        if HAS_TORCH and self.use_gpu:
            # AUDIT FIX: Check available VRAM before GPU computation
            weight_bytes = weight.nbytes
            try:
                free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                # Require at least 1.5x weight size free (for intermediate results)
                required_vram = int(weight_bytes * 1.5)
                
                if free_vram < required_vram:
                    logger.info(f"[SVDMerger] 显存不足 ({free_vram//1024//1024}MB < {required_vram//1024//1024}MB), 回退到 CPU")
                    return np.linalg.svd(weight, full_matrices=False)
                
                # GPU 加速
                w_tensor = torch.from_numpy(weight).to(self.device)
                U, S, Vh = torch.linalg.svd(w_tensor, full_matrices=False)
                result = (U.cpu().numpy(), S.cpu().numpy(), Vh.cpu().numpy())
                
                # AUDIT FIX: Immediately free GPU memory
                del w_tensor, U, S, Vh
                torch.cuda.empty_cache()
                
                return result
            except RuntimeError as e:
                # OOM or other CUDA error - fall back to CPU
                logger.warning(f"[SVDMerger] GPU 错误: {e}, 回退到 CPU")
                torch.cuda.empty_cache()
                return np.linalg.svd(weight, full_matrices=False)
        else:
            # CPU NumPy
            return np.linalg.svd(weight, full_matrices=False)
    
    def analyze_layer_spectrum(
        self, 
        weight_a: np.ndarray, 
        weight_b: np.ndarray,
        layer_name: str
    ) -> LayerSpectrum:
        """
        分析两个层的谱特征
        
        返回相似度、冲突度和结构/细节比例
        """
        _, S_a, _ = self.compute_svd(weight_a)
        _, S_b, _ = self.compute_svd(weight_b)
        
        # Stable Rank 计算: (sum(S))² / sum(S²)
        # 这是一个衡量矩阵"有效秩"的指标，越低表示过拟合风险越大
        stable_rank_a = float((S_a.sum() ** 2) / (np.square(S_a).sum() + 1e-8))
        stable_rank_b = float((S_b.sum() ** 2) / (np.square(S_b).sum() + 1e-8))
        
        # 归一化奇异值
        S_a_norm = S_a / (S_a.sum() + 1e-8)
        S_b_norm = S_b / (S_b.sum() + 1e-8)
        
        # 确保长度一致
        min_len = min(len(S_a_norm), len(S_b_norm))
        S_a_norm = S_a_norm[:min_len]
        S_b_norm = S_b_norm[:min_len]
        
        # 谱相似度 (Spectrum Similarity): 衡量"能量分布"是否一致
        spectrum_similarity = float(np.dot(S_a_norm, S_b_norm) / 
                          (np.linalg.norm(S_a_norm) * np.linalg.norm(S_b_norm) + 1e-8))
        
        # 方向冲突度 (Directional Conflict): 衡量"参数方向"是否冲突
        # 需要展平权重向量来计算余弦相似度
        w_a_flat = weight_a.flatten()
        w_b_flat = weight_b.flatten()
        dot_prod = np.dot(w_a_flat, w_b_flat)
        norm_prod = np.linalg.norm(w_a_flat) * np.linalg.norm(w_b_flat) + 1e-8
        cosine_sim = dot_prod / norm_prod
        
        # 如果方向相反 (cos < 0)，则认为冲突度高；如果同向，冲突度为0
        # Conflict Score: 0 (No Conflict) -> 1 (Max Conflict, -1 cosine)
        conflict = float(max(0, -cosine_sim)) 
        
        # 结构/细节分离 (以中点为界)
        mid = min_len // 2
        structure_a = float(S_a_norm[:mid].sum())
        detail_a = float(S_a_norm[mid:].sum())
        
        return LayerSpectrum(
            name=layer_name,
            singular_values_a=S_a,
            singular_values_b=S_b,
            similarity_score=spectrum_similarity,
            conflict_score=conflict,
            structure_ratio_a=structure_a,
            detail_ratio_a=detail_a,
            stable_rank_a=stable_rank_a,
            stable_rank_b=stable_rank_b
        )
    
    def analyze(
        self, 
        model_a_path: str, 
        model_b_path: str,
        quick_mode: bool = True
    ) -> MergeAnalysis:
        """
        分析两个模型的谱特征 (用于可视化)
        
        Args:
            model_a_path: 模型 A 路径
            model_b_path: 模型 B 路径
            quick_mode: 仅分析代表性层 (推荐)
            
        Returns:
            MergeAnalysis: 包含可视化数据的分析结果
        """
        tensors_a = self.load_model(model_a_path)
        tensors_b = self.load_model(model_b_path)
        
        # 找到共同的层
        common_keys = set(tensors_a.keys()) & set(tensors_b.keys())
        
        # 筛选要分析的层
        if quick_mode:
            analysis_keys = [k for k in common_keys if self._is_sample_layer(k)]
            if not analysis_keys:
                # 如果没有匹配的代表性层，取前 10 个
                analysis_keys = list(common_keys)[:10]
        else:
            analysis_keys = list(common_keys)
        
        # 分析每层
        layer_results: List[LayerSpectrum] = []
        for key in analysis_keys:
            try:
                spectrum = self.analyze_layer_spectrum(
                    tensors_a[key], tensors_b[key], key
                )
                layer_results.append(spectrum)
            except Exception as e:
                logger.warning(f"[SVDMerger] Skipping layer {key}: {e}")
                continue
        
        if not layer_results:
            raise ValueError("No valid layers to analyze")
        
        # 汇总统计
        overall_sim = np.mean([l.similarity_score for l in layer_results])
        overall_conflict = np.mean([l.conflict_score for l in layer_results])
        
        # 推荐模式
        if overall_conflict > 0.5:
            recommended = MergeMode.ORTHOGONALIZE
        elif overall_sim > 0.8:
            recommended = MergeMode.WEIGHTED_SUM
        else:
            recommended = MergeMode.SVD_SMART
        
        return MergeAnalysis(
            layers=layer_results,
            overall_similarity=float(overall_sim),
            overall_conflict=float(overall_conflict),
            recommended_mode=recommended,
            sample_layer_names=analysis_keys
        )
    
    def merge(
        self,
        model_a_path: str,
        model_b_path: str,
        output_path: str,
        alpha_a: float = 0.5,
        alpha_b: float = 0.5,
        mode: MergeMode = MergeMode.SVD_SMART,
        structure_bias: float = 0.5,
        detail_bias: float = 0.5,
        output_precision: str = "fp16",
        layer_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        执行模型融合
        
        Args:
            model_a_path: 模型 A 路径
            model_b_path: 模型 B 路径
            output_path: 输出路径
            alpha_a: 模型 A 权重
            alpha_b: 模型 B 权重
            mode: 融合模式
            structure_bias: 结构偏好 (0-1, 倾向低秩特征)
            detail_bias: 细节偏好 (0-1, 倾向高秩特征)
            output_precision: 输出精度 ("fp16" 或 "fp32")
            
        Returns:
            融合统计信息
        """
        from safetensors.numpy import save_file
        
        tensors_a = self.load_model(model_a_path)
        tensors_b = self.load_model(model_b_path)
        
        merged = {}
        stats = {
            "mode": mode.value,
            "layers_merged": 0,
            "layers_skipped": 0
        }
        
        # 获取所有键 (包括可能只在一个模型中的)
        all_keys = set(tensors_a.keys()) | set(tensors_b.keys())
        
        for key in all_keys:
            try:
                # 确定当前层的 alpha
                # 如果提供了 layer_weights，尝试匹配
                current_alpha_b = alpha_b # 默认
                current_alpha_a = alpha_a
                
                if layer_weights:
                    # 尝试精确匹配或前缀匹配
                    # 这里假设 layer_weights 的 key 是层名的一部分 (如 "down_blocks.0")
                    # 或者完整的 key
                    matched_weight = None
                    if key in layer_weights:
                        matched_weight = layer_weights[key]
                    else:
                        # 前缀匹配逻辑 (简单版)
                        for k, w in layer_weights.items():
                            if k in key:
                                matched_weight = w
                                break
                    
                    if matched_weight is not None:
                        # layer_weights 通常定义的是 "Model B 的比重" (0=A, 1=B)
                        # 或者 "混合比例"
                        # 这里我们假设 input 是 alpha_b
                        current_alpha_b = matched_weight
                        current_alpha_a = 1.0 - matched_weight

                if key in tensors_a and key in tensors_b:
                    # 两个模型都有此层
                    if mode == MergeMode.WEIGHTED_SUM:
                        merged[key] = self._merge_weighted_sum(
                            tensors_a[key], tensors_b[key], current_alpha_a, current_alpha_b
                        )
                    elif mode == MergeMode.SVD_SMART:
                        merged[key] = self._merge_svd_smart(
                            tensors_a[key], tensors_b[key], 
                            current_alpha_a, current_alpha_b,
                            structure_bias, detail_bias
                        )
                    elif mode == MergeMode.ORTHOGONALIZE:
                        merged[key] = self._merge_orthogonalize(
                            tensors_a[key], tensors_b[key], current_alpha_a, current_alpha_b
                        )
                    stats["layers_merged"] += 1
                elif key in tensors_a:
                    merged[key] = tensors_a[key] * current_alpha_a
                else:
                    merged[key] = tensors_b[key] * current_alpha_b
                    
            except Exception as e:
                logger.error(f"[SVDMerger] Error merging {key}: {e}")
                # 回退到加权平均
                if key in tensors_a and key in tensors_b:
                    merged[key] = tensors_a[key] * alpha_a + tensors_b[key] * alpha_b
                elif key in tensors_a:
                    merged[key] = tensors_a[key]
                else:
                    merged[key] = tensors_b[key]
                stats["layers_skipped"] += 1
        
        # 转换精度
        if output_precision == "fp16":
            merged = {k: v.astype(np.float16) for k, v in merged.items()}
        
        # 保存
        save_file(merged, output_path)
        
        stats["output_path"] = output_path
        stats["output_size_mb"] = Path(output_path).stat().st_size / (1024 * 1024)
        
        return stats
    
    def _merge_weighted_sum(
        self, 
        w_a: np.ndarray, 
        w_b: np.ndarray, 
        alpha_a: float, 
        alpha_b: float
    ) -> np.ndarray:
        """传统加权平均"""
        return w_a.astype(np.float32) * alpha_a + w_b.astype(np.float32) * alpha_b
    
    def _merge_svd_smart(
        self,
        w_a: np.ndarray,
        w_b: np.ndarray,
        alpha_a: float,
        alpha_b: float,
        structure_bias: float,
        detail_bias: float
    ) -> np.ndarray:
        """
        SVD 智能融合
        
        在谱域中独立控制结构(低秩)和细节(高秩)的融合比例
        """
        original_shape = w_a.shape
        
        # 处理 4D
        if w_a.ndim == 4:
            w_a_2d = w_a.reshape(w_a.shape[0], -1)
            w_b_2d = w_b.reshape(w_b.shape[0], -1)
        else:
            w_a_2d = w_a
            w_b_2d = w_b
        
        # SVD 分解
        U_a, S_a, Vh_a = self.compute_svd(w_a_2d)
        U_b, S_b, Vh_b = self.compute_svd(w_b_2d)
        
        # 确定秩一致性
        rank = min(len(S_a), len(S_b))
        mid = rank // 2
        
        # 结构层 (前半部分) 和细节层 (后半部分) 的权重
        S_merged = np.zeros(rank, dtype=np.float32)
        
        # 低秩 (结构): 使用 structure_bias 加权
        alpha_a_struct = alpha_a * structure_bias + alpha_b * (1 - structure_bias)
        alpha_b_struct = 1 - alpha_a_struct
        S_merged[:mid] = S_a[:mid] * alpha_a_struct + S_b[:mid] * alpha_b_struct
        
        # 高秩 (细节): 使用 detail_bias 加权
        alpha_a_detail = alpha_a * detail_bias + alpha_b * (1 - detail_bias)
        alpha_b_detail = 1 - alpha_a_detail
        S_merged[mid:rank] = S_a[mid:rank] * alpha_a_detail + S_b[mid:rank] * alpha_b_detail
        
        # 基向量融合: 使用 Procrustes 对齐 (简化版: 加权平均)
        U_merged = U_a[:, :rank] * alpha_a + U_b[:, :rank] * alpha_b
        Vh_merged = Vh_a[:rank, :] * alpha_a + Vh_b[:rank, :] * alpha_b
        
        # 正交化 U 和 Vh (保持结构)
        U_merged, _ = np.linalg.qr(U_merged)
        Vh_merged_T, _ = np.linalg.qr(Vh_merged.T)
        Vh_merged = Vh_merged_T.T
        
        # 重建
        merged = U_merged @ np.diag(S_merged) @ Vh_merged
        
        # 恢复形状
        if len(original_shape) == 4:
            merged = merged.reshape(original_shape)
        
        return merged.astype(np.float32)
    
    def _merge_orthogonalize(
        self,
        w_a: np.ndarray,
        w_b: np.ndarray,
        alpha_a: float,
        alpha_b: float
    ) -> np.ndarray:
        """
        正交化融合: 移除 B 中与 A 平行的分量后再融合
        
        适用于高冲突场景，避免特征覆盖
        """
        original_shape = w_a.shape
        
        if w_a.ndim == 4:
            w_a_flat = w_a.reshape(-1)
            w_b_flat = w_b.reshape(-1)
        else:
            w_a_flat = w_a.flatten()
            w_b_flat = w_b.flatten()
        
        w_a_flat = w_a_flat.astype(np.float32)
        w_b_flat = w_b_flat.astype(np.float32)
        
        # 计算 B 在 A 方向上的投影
        norm_a = np.linalg.norm(w_a_flat)
        dot_prod = np.dot(w_b_flat, w_a_flat)
        
        # PCGrad 核心逻辑:
        # 只有在方向发生冲突 (dot_prod < 0) 时，才执行投影去冲突
        # 否则 (方向一致或正交)，保留原始梯度，允许"建设性干涉"
        if dot_prod < 0 and norm_a > 1e-8:
            proj_b_on_a = (dot_prod / (norm_a ** 2)) * w_a_flat
            w_b_orth = w_b_flat - proj_b_on_a
        else:
            w_b_orth = w_b_flat
        
        # 融合: A 的全部 + B 的正交分量
        merged_flat = w_a_flat * alpha_a + w_b_orth * alpha_b
        
        return merged_flat.reshape(original_shape)
    
    def get_visualization_data(self, analysis: MergeAnalysis) -> Dict[str, Any]:
        """
        生成前端可视化所需的 JSON 数据
        
        Returns:
            包含图表数据的字典
        """
        # 谱柱状图数据
        spectrum_charts = []
        for layer in analysis.layers[:5]:  # 最多 5 层
            # 归一化到 0-100
            max_val = max(layer.singular_values_a.max(), layer.singular_values_b.max())
            s_a = (layer.singular_values_a / max_val * 100).tolist()[:50]  # 前 50 个
            s_b = (layer.singular_values_b / max_val * 100).tolist()[:50]
            
            spectrum_charts.append({
                "layer": layer.name,
                "model_a": s_a,
                "model_b": s_b,
                "similarity": round(layer.similarity_score, 3),
                "conflict": round(layer.conflict_score, 3)
            })
        
        return {
            "spectrum_charts": spectrum_charts,
            "overall": {
                "similarity": round(analysis.overall_similarity, 3),
                "conflict": round(analysis.overall_conflict, 3),
                "recommended_mode": analysis.recommended_mode.value
            },
            "insights": self._generate_insights(analysis)
        }
    
    def _generate_insights(self, analysis: MergeAnalysis) -> List[str]:
        """生成人类可读的洞察"""
        insights = []
        
        if analysis.overall_similarity > 0.8:
            insights.append("两个模型高度相似，简单加权平均即可有效融合")
        elif analysis.overall_similarity < 0.3:
            insights.append("两个模型差异显著，建议使用 SVD 智能融合以保留各自特征")
        
        if analysis.overall_conflict > 0.5:
            insights.append("检测到较高冲突度，部分特征方向存在竞争，推荐正交化融合")
        
        # 查找高冲突层
        high_conflict_layers = [l.name for l in analysis.layers if l.conflict_score > 0.6]
        if high_conflict_layers:
            insights.append(f"高冲突层: {', '.join(high_conflict_layers[:3])}")
        
        return insights


# 便捷函数
def quick_analyze(model_a: str, model_b: str) -> Dict[str, Any]:
    """快速分析两个模型的兼容性"""
    merger = SVDMerger()
    analysis = merger.analyze(model_a, model_b, quick_mode=True)
    return merger.get_visualization_data(analysis)


def smart_merge(
    model_a: str, 
    model_b: str, 
    output: str,
    ratio_a: float = 0.5,
    structure_bias: float = 0.5,
    detail_bias: float = 0.5
) -> Dict[str, Any]:
    """执行智能融合"""
    merger = SVDMerger()
    return merger.merge(
        model_a, model_b, output,
        alpha_a=ratio_a, alpha_b=1-ratio_a,
        mode=MergeMode.SVD_SMART,
        structure_bias=structure_bias,
        detail_bias=detail_bias
    )
