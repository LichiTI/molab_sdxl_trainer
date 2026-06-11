"""
Diagnostic Card Generator (诊断卡生成器)

生成可分享的模型诊断报告图片，包含:
- 模型健康评分 (S/A/B/C/CORRUPTED)
- 3D 神经网络图截图
- 关键指标数据
- Lulynx 品牌水印

用于社区传播的"病毒式增长"策略。
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import hashlib
import io
import base64

# 延迟导入，确保依赖可选
PIL_AVAILABLE = False
QRCODE_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    pass

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    pass


# === 评级系统 ===

GRADES = {
    "S": {"label": "S-RANK", "color": "#FFD700", "min_score": 90, "desc": "完美健康"},
    "A": {"label": "A-RANK", "color": "#B300FF", "min_score": 70, "desc": "优秀"},
    "B": {"label": "B-RANK", "color": "#4A90D9", "min_score": 50, "desc": "良好"},
    "C": {"label": "C-RANK", "color": "#808080", "min_score": 30, "desc": "需要调整"},
    "F": {"label": "CORRUPTED", "color": "#FF004D", "min_score": 0, "desc": "训练失败"},
}


def get_grade(score: float) -> str:
    """根据分数获取评级"""
    for grade, info in GRADES.items():
        if score >= info["min_score"]:
            return grade
    return "F"


# === 颜色配置 ===

COLORS = {
    "bg": "#0D0D0D",
    "bg_secondary": "#1A1A1A",
    "text": "#E0E0E0",
    "text_muted": "#666666",
    "accent": "#B300FF",  # Lulynx 紫
    "warning": "#FF004D",
    "success": "#4ADE80",
    "border": "#333333",
}


def create_diagnostic_card(
    model_name: str,
    health_score: float,
    neural_map_image: Optional[bytes] = None,
    metrics: Optional[Dict[str, Any]] = None,
    issues: Optional[List[str]] = None,
    version: str = "1.0.0",
    output_path: Optional[str] = None,
    project_url: str = "https://github.com/lulynx/lulynx",
) -> Optional[bytes]:
    """
    生成诊断卡片
    
    Args:
        model_name: 模型名称
        health_score: 健康评分 (0-100)
        neural_map_image: 3D 神经网络图的 PNG 字节数据
        metrics: 关键指标 {"gsnr": 10.7, "stable_rank": 18, ...}
        issues: 问题列表 ["IN04 过拟合", "Dead Neuron 过高"]
        version: 软件版本号
        output_path: 输出路径 (可选)
        project_url: 项目 URL (用于二维码)
    
    Returns:
        图片的 PNG 字节数据，或 None (如果缺少依赖)
    """
    if not PIL_AVAILABLE:
        print("[Diagnostic Card] PIL/Pillow 未安装，无法生成诊断卡")
        return None
    
    # 默认值
    metrics = metrics or {}
    issues = issues or []
    
    # 计算评级
    grade = get_grade(health_score)
    grade_info = GRADES[grade]
    
    # 生成唯一 ID (基于模型名和时间)
    uid = hashlib.md5(f"{model_name}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
    
    # === 尺寸配置 ===
    card_width = 800
    header_height = 160
    body_height = 400
    metrics_height = 100
    issues_height = min(60 + len(issues) * 40, 200) if issues else 0
    footer_height = 100
    
    total_height = header_height + body_height + metrics_height + issues_height + footer_height
    
    # === 创建画布 ===
    img = Image.new("RGB", (card_width, total_height), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    
    # === 加载字体 ===
    try:
        font_large = ImageFont.truetype("arialbd.ttf", 48)
        font_medium = ImageFont.truetype("arial.ttf", 24)
        font_small = ImageFont.truetype("arial.ttf", 16)
        font_mono = ImageFont.truetype("consola.ttf", 14)
        font_grade = ImageFont.truetype("arialbd.ttf", 72)
    except Exception:
        # 降级到默认字体
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large
        font_mono = font_large
        font_grade = font_large
    
    y_cursor = 0
    
    # === Header: 评级 + 模型名 ===
    # 背景
    draw.rectangle([0, 0, card_width, header_height], fill=COLORS["bg_secondary"])
    
    # 评级徽章
    badge_x = 40
    badge_y = 40
    draw.text((badge_x, badge_y), grade, fill=grade_info["color"], font=font_grade)
    draw.text((badge_x + 70, badge_y + 50), "RANK", fill=grade_info["color"], font=font_small)
    
    # 模型名
    draw.text((180, 50), model_name, fill=COLORS["text"], font=font_large)
    
    # 日期
    date_str = f"DIAGNOSTIC REPORT // {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    draw.text((180, 110), date_str, fill=COLORS["text_muted"], font=font_mono)
    
    y_cursor = header_height
    
    # === Body: 神经网络图 ===
    if neural_map_image:
        try:
            neural_img = Image.open(io.BytesIO(neural_map_image))
            # 缩放适应卡片
            neural_img = neural_img.convert("RGB")
            aspect = neural_img.width / neural_img.height
            new_width = card_width - 80
            new_height = int(new_width / aspect)
            if new_height > body_height - 40:
                new_height = body_height - 40
                new_width = int(new_height * aspect)
            neural_img = neural_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 居中粘贴
            paste_x = (card_width - new_width) // 2
            paste_y = y_cursor + (body_height - new_height) // 2
            img.paste(neural_img, (paste_x, paste_y))
        except Exception as e:
            # 占位符
            draw.rectangle([40, y_cursor + 20, card_width - 40, y_cursor + body_height - 20], 
                          outline=COLORS["border"], width=2)
            draw.text((card_width // 2 - 100, y_cursor + body_height // 2), 
                     "Neural Map Unavailable", fill=COLORS["text_muted"], font=font_medium)
    else:
        # 占位符
        draw.rectangle([40, y_cursor + 20, card_width - 40, y_cursor + body_height - 20], 
                      outline=COLORS["border"], width=2)
        draw.text((card_width // 2 - 80, y_cursor + body_height // 2), 
                 "[3D Neural Map]", fill=COLORS["text_muted"], font=font_medium)
    
    y_cursor += body_height
    
    # === Metrics: 关键指标 ===
    draw.line([(40, y_cursor), (card_width - 40, y_cursor)], fill=COLORS["border"], width=1)
    y_cursor += 20
    
    metric_items = [
        ("GSNR", metrics.get("gsnr", "--")),
        ("Stable Rank", metrics.get("stable_rank", "--")),
        ("Dead Neuron %", f"{metrics.get('dead_neuron', '--')}%"),
        ("Score", f"{health_score:.0f}/100"),
    ]
    
    metric_width = (card_width - 80) // len(metric_items)
    for i, (label, value) in enumerate(metric_items):
        x = 40 + i * metric_width
        draw.text((x, y_cursor), label, fill=COLORS["text_muted"], font=font_small)
        draw.text((x, y_cursor + 25), str(value), fill=COLORS["text"], font=font_medium)
    
    y_cursor += metrics_height - 20
    
    # === Issues: 问题列表 ===
    if issues:
        draw.line([(40, y_cursor), (card_width - 40, y_cursor)], fill=COLORS["border"], width=1)
        y_cursor += 15
        
        for issue in issues[:4]:  # 最多显示 4 个
            # 警告框
            draw.rectangle([40, y_cursor, 400, y_cursor + 35], fill="#330000", outline=COLORS["warning"])
            draw.text((55, y_cursor + 8), f"⚠ {issue}", fill=COLORS["warning"], font=font_small)
            y_cursor += 45
        
        y_cursor += 15
    
    # === Footer: 水印 ===
    footer_y = total_height - footer_height
    draw.line([(40, footer_y), (card_width - 40, footer_y)], fill=COLORS["border"], width=2)
    
    # Lulynx 品牌
    draw.text((40, footer_y + 20), "GENERATED BY", fill=COLORS["text_muted"], font=font_small)
    draw.text((40, footer_y + 45), "LULYNX LAB", fill=COLORS["accent"], font=font_large)
    
    # 版本和 UID
    draw.text((340, footer_y + 30), f"VER: {version}", fill=COLORS["text_muted"], font=font_mono)
    draw.text((340, footer_y + 50), f"UID: {uid}", fill="#444444", font=font_mono)
    
    # 二维码
    if QRCODE_AVAILABLE:
        try:
            qr = qrcode.make(project_url)
            qr = qr.resize((70, 70))
            qr = qr.convert("RGB")
            img.paste(qr, (card_width - 110, footer_y + 15))
        except Exception:
            pass
    
    # === 输出 ===
    output = io.BytesIO()
    img.save(output, format="PNG", quality=95)
    png_bytes = output.getvalue()
    
    # 保存到文件
    if output_path:
        Path(output_path).write_bytes(png_bytes)
    
    return png_bytes


def create_diagnostic_card_base64(
    model_name: str,
    health_score: float,
    **kwargs
) -> Optional[str]:
    """生成诊断卡片并返回 Base64 编码"""
    png_bytes = create_diagnostic_card(model_name, health_score, **kwargs)
    if png_bytes:
        return base64.b64encode(png_bytes).decode("utf-8")
    return None


# === 依赖检查 ===

def check_dependencies() -> Dict[str, bool]:
    """检查依赖状态"""
    return {
        "pillow": PIL_AVAILABLE,
        "qrcode": QRCODE_AVAILABLE,
    }


def get_missing_dependencies() -> List[str]:
    """获取缺失的依赖"""
    missing = []
    if not PIL_AVAILABLE:
        missing.append("pillow")
    if not QRCODE_AVAILABLE:
        missing.append("qrcode")
    return missing


# === 测试 ===

if __name__ == "__main__":
    # 测试生成
    deps = check_dependencies()
    print(f"Dependencies: {deps}")
    
    if PIL_AVAILABLE:
        png = create_diagnostic_card(
            model_name="MyFurryLora_v3",
            health_score=78,
            metrics={
                "gsnr": 10.77,
                "stable_rank": 18.05,
                "dead_neuron": 3.5,
            },
            issues=["IN04 层过拟合风险", "建议降低学习率"],
            output_path="test_card.png",
        )
        print(f"Generated: {len(png) if png else 0} bytes")
    else:
        print("Cannot test: Pillow not installed")
