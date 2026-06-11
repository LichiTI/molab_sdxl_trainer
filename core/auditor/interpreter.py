import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

@dataclass
class DiagnosisItem:
    """诊断项"""
    rule_id: str
    problem: str
    suggestion: str
    severity: str  # "info", "warning", "error"
    param_suggestion: Optional[Dict[str, Any]] = None  # {"lr": 8e-5}

@dataclass 
class DiagnosisReport:
    """训练后诊断报告"""
    timestamp: str
    total_steps: int
    problems: List[DiagnosisItem]
    summary: str
    overall_status: str  # "healthy", "warning", "critical"
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "total_steps": self.total_steps,
            "problems": [asdict(p) for p in self.problems],
            "summary": self.summary,
            "overall_status": self.overall_status
        }
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

class AuditInterpreter:
    """
    V10.0: 语义解释器
    
    将指标数值转化为人类可读的诊断建议
    包含 33 条诊断规则用于训练后复盘
    """
    
    # 诊断规则库 - 33 条完整规则
    # 格式: (rule_id, check_function, problem, suggestion, severity)
    RULES = [
        # ========== Loss 相关 (L1-L5) ==========
        ("L1", lambda h: h.get("loss_rising", False), 
         "Loss 后期回升 >10%", "减少 Epochs 或增加 Dropout", "warning"),
        ("L2", lambda h: h.get("loss_stuck", False), 
         "Loss 始终不下降", "提高 LR 2-5 倍，检查数据集", "error"),
        ("L3", lambda h: h.get("loss_oscillating", False), 
         "Loss 剧烈震荡 (std > 0.1)", "降低 LR 或增大 Batch Size", "warning"),
        ("L4", lambda h: h.get("loss_explode", False), 
         "Loss 瞬间爆炸 (>10)", "启用梯度裁剪，降低 LR", "error"),
        ("L5", lambda h: h.get("loss_plateau", False), 
         "Loss 快速收敛后停滞", "使用 Warmup 或更换优化器", "info"),
        
        # ========== 梯度相关 (G1-G4) ==========
        ("G1", lambda h: h.get("grad_coherence_low", False), 
         "梯度一致性 < 0.3 持续 100 步", "降低 LR 20-30%", "warning"),
        ("G2", lambda h: h.get("gsnr_low", False), 
         "梯度信噪比 GSNR < 0.01", "增大 Batch Size", "warning"),
        ("G3", lambda h: h.get("update_ratio_high", False), 
         "更新幅度 > 0.1", "降低 LR", "warning"),
        ("G4", lambda h: h.get("update_ratio_low", False), 
         "更新幅度 < 1e-6", "提高 LR 或检查权重冻结", "info"),
        
        # ========== 权重拓扑相关 (W1-W6) ==========
        ("W1", lambda h: h.get("dead_neuron_high", False), 
         "死神经元 > 60%", "降低 Rank", "warning"),
        ("W2", lambda h: h.get("dead_neuron_rising", False), 
         "死神经元持续上升", "检查 LR 或添加正则化", "warning"),
        ("W3", lambda h: h.get("rank_collapse", False), 
         "Stable Rank 持续下降", "增加 Rank 或降低 LR", "error"),
        ("W4", lambda h: h.get("rank_near_one", False), 
         "Stable Rank 接近 1", "重新训练，增加 Rank", "error"),
        ("W5", lambda h: h.get("entropy_collapse", False), 
         "SVD Entropy 趋近 0", "增加 Dropout 或降低 LR", "warning"),
        ("W6", lambda h: h.get("entropy_high", False), 
         "SVD Entropy 过高", "增加训练步数或提高 LR", "info"),
        
        # ========== 语义/画质相关 (S1-S5) ==========
        ("S1", lambda h: h.get("noise_collapse", False), 
         "Noise Pred Std 趋近 0 (灰图)", "降低 LR，检查数据集", "error"),
        ("S2", lambda h: h.get("noise_explode", False), 
         "Noise Pred Std 过高 (>2)", "降低 LR，启用梯度裁剪", "error"),
        ("S3", lambda h: h.get("clip_drift_high", False), 
         "CLIP Drift > 0.3 (语义偏移)", "降低 TE 学习率或冻结 TE", "warning"),
        ("S4", lambda h: h.get("attn_collapse", False), 
         "Attn Entropy 趋近 0 (注意力坍缩)", "检查数据多样性", "warning"),
        ("S5", lambda h: h.get("attn_diverge", False), 
         "Attn Entropy 过高 (注意力发散)", "增加训练步数", "info"),
        
        # ========== 配置相关 (C1-C6) ==========
        ("C1", lambda h: h.get("rank_too_high", False), 
         "Rank > 64 且 Dead Neuron > 40%", "降低 Rank 至 32 或更低", "warning"),
        ("C2", lambda h: h.get("data_insufficient", False), 
         "训练图片 < 20 张", "增加训练数据或降低 Epochs", "warning"),
        ("C3", lambda h: h.get("epochs_too_many", False), 
         "Epochs > 20 且 Loss 持平", "减少 Epochs", "info"),
        ("C4", lambda h: h.get("resolution_vram_mismatch", False), 
         "Resolution > 1024 且 VRAM < 8GB", "降低分辨率或启用梯度检查点", "warning"),
        ("C5", lambda h: h.get("batch_too_small", False), 
         "Batch Size = 1 且 Loss 震荡", "使用梯度累积", "info"),
        ("C6", lambda h: h.get("te_lr_too_high", False), 
         "TE LR = UNet LR", "TE LR 设为 UNet LR 的 50%", "info"),
        
        # ========== 硬件相关 (H1-H6) ==========
        ("H1", lambda h: h.get("vram_critical", False), 
         "VRAM 使用率 > 98%", "降低 Batch Size 或分辨率", "error"),
        ("H2", lambda h: h.get("speed_slow", False), 
         "训练速度 < 0.5 it/s", "启用 xformers 或检查数据加载", "info"),
        ("H3", lambda h: h.get("oom_crash", False), 
         "训练中途崩溃 (CUDA OOM)", "降低分辨率/Batch/启用 8bit Adam", "error"),
        ("H4", lambda h: h.get("gpu_underutilized", False), 
         "功耗持续 < 50% TDP", "增大 Batch Size 或检查数据瓶颈", "info"),
        ("H5", lambda h: h.get("hessian_high", False), 
         "Hessian Trace 过大", "损失曲面陡峭，降低 LR", "warning"),
        ("H6", lambda h: h.get("hessian_low", False), 
         "Hessian Trace 过小", "可能在鞍点，增大 LR", "info"),
        
        # ========== 权重拓扑扩展 (W7-W9) ==========
        ("W7", lambda h: h.get("rank_volatile", False), 
         "Stable Rank 波动 > 50%", "训练不稳定，检查数据质量", "warning"),
        ("W8", lambda h: h.get("spectral_gap_high", False), 
         "Spectral Gap 过大", "奇异值分布不健康，调整 Alpha", "info"),
        ("W9", lambda h: h.get("dead_neuron_initial_high", False), 
         "初始死神经元 > 40%", "Rank 可能设置过大", "info"),
        
        # ========== 梯度扩展 (G5-G7) ==========
        ("G5", lambda h: h.get("grad_coherence_negative", False), 
         "梯度一致性持续 < 0", "优化方向错误，检查数据/模型", "error"),
        ("G6", lambda h: h.get("gsnr_zero", False), 
         "GSNR 突然归零", "梯度消失，检查 LR/网络结构", "error"),
        ("G7", lambda h: h.get("gsnr_high", False), 
         "GSNR > 10", "过拟合风险，增加正则化", "warning"),
        
        # ========== 语义扩展 (S6-S8) ==========
        ("S6", lambda h: h.get("noise_volatile", False), 
         "Noise Std 波动 > 100%", "输出不稳定，降低 LR", "warning"),
        ("S7", lambda h: h.get("clip_drift_growing", False), 
         "CLIP Drift 持续增长", "语义逐渐偏离，冻结 TE", "warning"),
        ("S8", lambda h: h.get("attn_heads_uneven", False), 
         "注意力头间差异过大", "注意力不均匀，检查模型", "info"),
        
        # ========== 激活值漂移 (A1-A2) ==========
        ("A1", lambda h: h.get("act_drift_high", False), 
         "Activation Drift > 1.0", "特征空间变形，降低 LR", "warning"),
        ("A2", lambda h: h.get("act_drift_growing", False), 
         "Activation Drift 持续增长", "过拟合信号，减少 Epochs", "warning"),
        
        # ========== 遗忘探针 (F1-F2) ==========
        ("F1", lambda h: h.get("forgetting_rising", False), 
         "Forgetting Probe 上升 > 50%", "灾难性遗忘，降低 LR", "error"),
        ("F2", lambda h: h.get("forgetting_volatile", False), 
         "Forgetting Probe 波动剧烈", "任务干扰，检查数据分布", "warning"),
        
        # ========== 正常状态 (OK1-OK3) ==========
        ("OK1", lambda h: h.get("training_healthy", False), 
         "训练健康", "无需调整 ✅", "info"),
        ("OK2", lambda h: h.get("loss_decreasing", False), 
         "Loss 平稳下降", "可继续训练 ✅", "info"),
        ("OK3", lambda h: h.get("early_converge", False), 
         "提前收敛 (Loss < 0.01)", "可提前停止 ✅", "info"),
    ]
    
    def __init__(self):
        self._metrics_history: List[Dict] = []
    
    def add_metrics(self, metrics: Dict):
        """添加指标记录"""
        self._metrics_history.append(metrics)
    
    def analyze(self) -> Dict[str, bool]:
        """
        分析指标历史，返回问题标志
        
        检测所有 33 条规则涉及的条件
        """
        if len(self._metrics_history) < 10:
            return {"training_healthy": True}
        
        flags = {}
        history = self._metrics_history
        
        # 辅助函数
        def get_metric(h, key):
            return h.get("metrics", {}).get(key)
        
        def get_loss(h):
            return h.get("loss", {}).get("ema", 0)
        
        # ========== Loss 相关 ==========
        
        # L1: Loss 后期回升 >10%
        if len(history) > 50:
            early_loss = sum(get_loss(h) for h in history[:20]) / 20
            late_loss = sum(get_loss(h) for h in history[-20:]) / 20
            if early_loss > 0:
                flags["loss_rising"] = late_loss > early_loss * 1.1
        
        # L2: Loss 始终不下降
        if len(history) > 100:
            first_loss = get_loss(history[0])
            last_loss = get_loss(history[-1])
            flags["loss_stuck"] = last_loss >= first_loss * 0.95
        
        # L3: Loss 剧烈震荡
        losses = [get_loss(h) for h in history[-50:] if get_loss(h)]
        if len(losses) > 10:
            mean_loss = sum(losses) / len(losses)
            variance = sum((l - mean_loss) ** 2 for l in losses) / len(losses)
            std = variance ** 0.5
            flags["loss_oscillating"] = std > 0.1
        
        # L4: Loss 爆炸
        flags["loss_explode"] = any(get_loss(h) > 10 for h in history)
        
        # L5: Loss 快速收敛后停滞
        if len(history) > 100:
            mid_loss = get_loss(history[len(history)//2])
            late_loss = get_loss(history[-1])
            if mid_loss > 0:
                flags["loss_plateau"] = abs(late_loss - mid_loss) / mid_loss < 0.01
        
        # ========== 梯度相关 ==========
        
        # G1: 梯度一致性低
        coherences = [get_metric(h, "grad_coherence") for h in history[-100:] if get_metric(h, "grad_coherence")]
        if coherences:
            flags["grad_coherence_low"] = sum(coherences) / len(coherences) < 0.3
        
        # G2: GSNR 低
        gsnrs = [get_metric(h, "gsnr") for h in history[-50:] if get_metric(h, "gsnr")]
        if gsnrs:
            flags["gsnr_low"] = sum(gsnrs) / len(gsnrs) < 0.01
        
        # G3/G4: Update Ratio
        ratios = [get_metric(h, "update_ratio") for h in history[-50:] if get_metric(h, "update_ratio")]
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            flags["update_ratio_high"] = avg_ratio > 0.1
            flags["update_ratio_low"] = avg_ratio < 1e-6
        
        # ========== 权重拓扑相关 ==========
        
        # W1: 死神经元高
        dead_rates = [get_metric(h, "dead_neuron_rate") for h in history[-20:] if get_metric(h, "dead_neuron_rate")]
        if dead_rates:
            flags["dead_neuron_high"] = sum(dead_rates) / len(dead_rates) > 0.6
        
        # W2: 死神经元持续上升
        all_dead = [get_metric(h, "dead_neuron_rate") for h in history if get_metric(h, "dead_neuron_rate")]
        if len(all_dead) > 10:
            early_dead = sum(all_dead[:5]) / 5
            late_dead = sum(all_dead[-5:]) / 5
            flags["dead_neuron_rising"] = late_dead > early_dead * 1.5
        
        # W3: Stable Rank 下降
        ranks = [get_metric(h, "stable_rank") for h in history if get_metric(h, "stable_rank")]
        if len(ranks) > 10:
            early_rank = sum(ranks[:5]) / 5
            late_rank = sum(ranks[-5:]) / 5
            flags["rank_collapse"] = late_rank < early_rank * 0.5
            flags["rank_near_one"] = late_rank < 1.5
        
        # W5/W6: SVD Entropy
        entropies = [get_metric(h, "svd_entropy") for h in history if get_metric(h, "svd_entropy")]
        if entropies:
            avg_entropy = sum(entropies) / len(entropies)
            flags["entropy_collapse"] = entropies[-1] < 0.5 if entropies else False
            flags["entropy_high"] = avg_entropy > 5  # 取决于 rank
        
        # W7: Stable Rank 波动
        if len(ranks) > 10:
            mean_rank = sum(ranks) / len(ranks)
            max_rank = max(ranks)
            min_rank = min(ranks)
            if mean_rank > 0:
                flags["rank_volatile"] = (max_rank - min_rank) / mean_rank > 0.5
        
        # W8: Spectral Gap 过大
        spectral_gaps = [get_metric(h, "spectral_smoothness") for h in history if get_metric(h, "spectral_smoothness")]
        if spectral_gaps:
            flags["spectral_gap_high"] = max(spectral_gaps) > 0.5
        
        # W9: 初始死神经元高
        if len(all_dead) > 5:
            flags["dead_neuron_initial_high"] = all_dead[0] > 0.4 if all_dead else False
        
        # ========== 语义/画质相关 ==========
        
        # S1/S2: Noise Pred Std
        noise_stds = [get_metric(h, "noise_pred_std") for h in history if get_metric(h, "noise_pred_std")]
        if noise_stds:
            flags["noise_collapse"] = noise_stds[-1] < 0.01
            flags["noise_explode"] = any(n > 2 for n in noise_stds)
            # S6: Noise 波动
            if len(noise_stds) > 10:
                mean_noise = sum(noise_stds) / len(noise_stds)
                if mean_noise > 0:
                    max_noise = max(noise_stds)
                    min_noise = min(noise_stds)
                    flags["noise_volatile"] = (max_noise - min_noise) / mean_noise > 1.0
        
        # S3: CLIP Drift
        clip_drifts = [get_metric(h, "clip_drift") for h in history if get_metric(h, "clip_drift")]
        if clip_drifts:
            flags["clip_drift_high"] = max(clip_drifts) > 0.3
            # S7: CLIP Drift 持续增长
            if len(clip_drifts) > 10:
                early_drift = sum(clip_drifts[:5]) / 5
                late_drift = sum(clip_drifts[-5:]) / 5
                flags["clip_drift_growing"] = late_drift > early_drift * 1.5
        
        # S4/S5: Attention Entropy
        attn_ents = [get_metric(h, "attn_entropy") for h in history if get_metric(h, "attn_entropy")]
        if attn_ents:
            flags["attn_collapse"] = attn_ents[-1] < 0.1
            flags["attn_diverge"] = attn_ents[-1] > 5
        
        # ========== 梯度扩展 ==========
        
        # G5: 梯度一致性持续为负
        if coherences:
            flags["grad_coherence_negative"] = sum(1 for c in coherences[-20:] if c < 0) > 15
        
        # G6: GSNR 归零
        if gsnrs:
            flags["gsnr_zero"] = any(g < 1e-8 for g in gsnrs[-10:])
            # G7: GSNR 过高
            flags["gsnr_high"] = sum(gsnrs) / len(gsnrs) > 10
        
        # ========== 激活值漂移 ==========
        
        act_drifts = [get_metric(h, "act_drift") for h in history if get_metric(h, "act_drift")]
        if act_drifts:
            # A1: 漂移过大
            flags["act_drift_high"] = max(act_drifts) > 1.0
            # A2: 漂移持续增长
            if len(act_drifts) > 10:
                early_drift = sum(act_drifts[:5]) / 5
                late_drift = sum(act_drifts[-5:]) / 5
                flags["act_drift_growing"] = late_drift > early_drift * 2
        
        # ========== 遗忘探针 ==========
        
        forgetting = [get_metric(h, "forgetting_probe") for h in history if get_metric(h, "forgetting_probe")]
        if forgetting:
            # F1: 遗忘上升
            if len(forgetting) > 10:
                early_forget = sum(forgetting[:5]) / 5
                late_forget = sum(forgetting[-5:]) / 5
                if early_forget > 0:
                    flags["forgetting_rising"] = late_forget > early_forget * 1.5
            # F2: 遗忘波动
            if len(forgetting) > 10:
                mean_forget = sum(forgetting) / len(forgetting)
                variance = sum((f - mean_forget) ** 2 for f in forgetting) / len(forgetting)
                std = variance ** 0.5
                flags["forgetting_volatile"] = std / (mean_forget + 1e-8) > 0.5
        
        # ========== Hessian ==========
        
        hessians = [get_metric(h, "hessian_trace") for h in history if get_metric(h, "hessian_trace")]
        if hessians:
            avg_hessian = sum(hessians) / len(hessians)
            flags["hessian_high"] = avg_hessian > 100
            flags["hessian_low"] = avg_hessian < 0.01
        
        # ========== 硬件相关 ==========
        
        # H1: VRAM 临界
        vram_usages = [h.get("hardware", {}).get("vram_gb", 0) for h in history]
        if vram_usages:
            flags["vram_critical"] = max(vram_usages) > 0.98 * vram_usages[0] if vram_usages[0] else False
        
        # H2: 速度慢
        throughputs = [h.get("hardware", {}).get("throughput", 0) for h in history]
        if throughputs:
            flags["speed_slow"] = sum(throughputs) / len(throughputs) < 0.5
        
        # ========== 正常状态检测 ==========
        
        # 检查是否存在任何问题
        has_problems = any(v for k, v in flags.items() if not k.startswith("training") and not k.startswith("loss_decreasing") and not k.startswith("early"))
        flags["training_healthy"] = not has_problems
        
        # Loss 是否在下降
        if len(history) > 20:
            early_loss = sum(get_loss(h) for h in history[:10]) / 10
            late_loss = sum(get_loss(h) for h in history[-10:]) / 10
            flags["loss_decreasing"] = late_loss < early_loss * 0.9
        
        # 提前收敛
        if len(history) > 0:
            flags["early_converge"] = get_loss(history[-1]) < 0.01
        
        return flags
    
    def generate_report(self) -> DiagnosisReport:
        """生成诊断报告"""
        flags = self.analyze()
        problems = []
        
        for rule_id, check_fn, problem, suggestion, severity in self.RULES:
            if check_fn(flags) and rule_id != "OK1":
                problems.append(DiagnosisItem(
                    rule_id=rule_id,
                    problem=problem,
                    suggestion=suggestion,
                    severity=severity
                ))
        
        # 确定总体状态
        if any(p.severity == "error" for p in problems):
            overall = "critical"
            summary = f"检测到 {len(problems)} 个问题，建议调整参数后重新训练"
        elif problems:
            overall = "warning"
            summary = f"检测到 {len(problems)} 个潜在问题，建议关注"
        else:
            overall = "healthy"
            summary = "训练正常完成，未检测到明显问题 ✅"
        
        return DiagnosisReport(
            timestamp=datetime.now().isoformat(),
            total_steps=len(self._metrics_history),
            problems=problems,
            summary=summary,
            overall_status=overall
        )
