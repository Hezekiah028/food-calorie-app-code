# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.checks import check_requirements


def _maybe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_food_config(raw: dict) -> dict[str, dict[str, Any]]:
    """Parse JSON: legacy number per class, or {kcal_per_100g, thickness_cm, density_g_per_cm3, ...}."""
    out: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if str(k).startswith("_"):
            continue
        key = _normalize_key(str(k))
        if isinstance(v, (int, float)):
            out[key] = {
                "legacy_kcal": float(v),
                "kcal_per_100g": None,
                "thickness_cm": None,
                "density_g_per_cm3": None,
            }
        elif isinstance(v, dict):
            out[key] = {
                "legacy_kcal": _maybe_float(v.get("legacy_kcal")),
                "kcal_per_100g": _maybe_float(v.get("kcal_per_100g")),
                "thickness_cm": _maybe_float(v.get("thickness_cm")),
                "density_g_per_cm3": _maybe_float(v.get("density_g_per_cm3")),
            }
    return out


def _safe_json_load_raw(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("calorie mapping JSON must be an object.")
    return data


def _normalize_key(s: str) -> str:
    # Keep it simple: lowercase and strip whitespace for better matching.
    return str(s).strip().lower()


@lru_cache(maxsize=4)
def _load_model_cached(model_path: str) -> YOLO:
    return YOLO(model_path)


def _effective_kcal_per_100g(cfg: dict[str, Any] | None, unknown_kcal: float, default_kcal_per_100g: float) -> float:
    """Resolve kcal/100g: prefer JSON; else legacy_kcal -> assume 200g portion; else global default."""
    if cfg is None:
        return float(default_kcal_per_100g)
    k100 = cfg.get("kcal_per_100g")
    if k100 is not None:
        return float(k100)
    leg = cfg.get("legacy_kcal")
    if leg is not None:
        return float(leg) / 200.0 * 100.0
    return float(default_kcal_per_100g)


def _effective_portion_grams(cfg: dict[str, Any] | None, size_bucket: str, default_portion_m: float) -> float:
    """Resolve S/M/L portion grams from JSON, fallback to default M with simple multipliers."""
    if cfg is None:
        base_m = float(default_portion_m)
        return base_m * (0.75 if size_bucket == "S" else 1.25 if size_bucket == "L" else 1.0)

    g_m = cfg.get("portion_grams_m")
    base_m = float(g_m) if g_m is not None else float(default_portion_m)
    g_s = cfg.get("portion_grams_s")
    g_l = cfg.get("portion_grams_l")
    if size_bucket == "S":
        return float(g_s) if g_s is not None else base_m * 0.75
    if size_bucket == "L":
        return float(g_l) if g_l is not None else base_m * 1.25
    return base_m


def _effective_legacy_kcal(cfg: dict[str, Any] | None, unknown_kcal: float) -> float:
    if cfg is None:
        return float(unknown_kcal)
    leg = cfg.get("legacy_kcal")
    if leg is not None:
        return float(leg)
    k100 = cfg.get("kcal_per_100g")
    if k100 is not None:
        return float(k100) / 100.0 * 200.0
    return float(unknown_kcal)


def _estimate_calories(
    res: Any,
    class_names: list[str],
    food_config: dict[str, dict[str, Any]],
    unknown_kcal: float,
    unknown_kcal_per_100g: float,
    portion_mode: str,
    area_power: float,
    conf_weight: bool,
    scene_width_cm: float,
    global_thickness_cm: float,
    global_density_g_per_cm3: float,
    sml_small_thresh: float,
    sml_large_thresh: float,
    default_portion_m_grams: float,
):
    """
    portion_mode:
      - count / bbox_area / mask_area: legacy or rough portion
      - portion_sml: classify bbox area ratio into S/M/L, then use per-class S/M/L grams and kcal/100g
      - volume_calibrated: pixel width -> cm, bbox area * thickness * density -> g, then kcal from kcal/100g
    """
    if res.boxes is None or len(res.boxes) == 0:
        return []

    h, w = res.orig_img.shape[:2]
    img_pixels = float(h * w)

    boxes = res.boxes
    xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
    cls_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)
    confs = boxes.conf.cpu().numpy().astype(float)  # (N,)

    masks = None
    if portion_mode == "mask_area" and getattr(res, "masks", None) is not None and res.masks.data is not None:
        masks = res.masks.data.cpu().numpy()

    cm_per_px = float(scene_width_cm) / float(w) if w > 0 else 0.0

    dets = []
    for i in range(len(cls_ids)):
        cls_id = int(cls_ids[i])
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        nk = _normalize_key(cls_name)
        cfg = food_config.get(nk)

        x1, y1, x2, y2 = xyxy[i].tolist()
        x1 = max(0.0, min(float(x1), float(w - 1)))
        y1 = max(0.0, min(float(y1), float(h - 1)))
        x2 = max(0.0, min(float(x2), float(w - 1)))
        y2 = max(0.0, min(float(y2), float(h - 1)))

        bbox_pixels = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        bbox_area_ratio = float(bbox_pixels / img_pixels)

        grams: float | None = None
        kcal = 0.0

        size_bucket: str | None = None
        if portion_mode == "portion_sml":
            if bbox_area_ratio < float(sml_small_thresh):
                size_bucket = "S"
            elif bbox_area_ratio > float(sml_large_thresh):
                size_bucket = "L"
            else:
                size_bucket = "M"
            grams = _effective_portion_grams(cfg, size_bucket, default_portion_m_grams)
            k100 = _effective_kcal_per_100g(cfg, unknown_kcal, unknown_kcal_per_100g)
            kcal = grams * (k100 / 100.0)
        elif portion_mode == "volume_calibrated":
            k100 = _effective_kcal_per_100g(cfg, unknown_kcal, unknown_kcal_per_100g)
            thickness = (
                float(cfg["thickness_cm"])
                if cfg and cfg.get("thickness_cm") is not None
                else float(global_thickness_cm)
            )
            density = (
                float(cfg["density_g_per_cm3"])
                if cfg and cfg.get("density_g_per_cm3") is not None
                else float(global_density_g_per_cm3)
            )
            bw_cm = max(0.0, (x2 - x1) * cm_per_px)
            bh_cm = max(0.0, (y2 - y1) * cm_per_px)
            area_cm2 = bw_cm * bh_cm
            volume_cm3 = area_cm2 * max(0.05, thickness)
            grams = volume_cm3 * max(0.05, density)
            kcal = grams * (k100 / 100.0)
        else:
            base_kcal = _effective_legacy_kcal(cfg, unknown_kcal)
            portion_factor = 1.0
            if portion_mode == "count":
                portion_factor = 1.0
            elif portion_mode == "bbox_area":
                portion_factor = max(0.0, bbox_area_ratio) ** float(area_power)
            elif portion_mode == "mask_area" and masks is not None:
                mask = masks[i]
                mask_area_ratio = float(mask.astype(bool).sum() / img_pixels)
                portion_factor = max(0.0, mask_area_ratio) ** float(area_power)
            else:
                portion_factor = max(0.0, bbox_area_ratio) ** float(area_power)

            kcal = base_kcal * portion_factor

        if conf_weight:
            kcal *= float(confs[i])

        row: dict[str, Any] = {
            "class_name": cls_name,
            "conf": float(confs[i]),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "kcal": float(kcal),
        }
        if grams is not None:
            row["grams_est"] = float(grams)
        if size_bucket is not None:
            row["size_bucket"] = size_bucket
        dets.append(row)
    return dets


def _draw_kcal_labels(img_bgr: np.ndarray, dets: list[dict[str, Any]]):
    # Draw per-detection kcal near the top-left corner of each bbox.
    for d in dets:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        conf = d["conf"]
        name = d["class_name"]
        kcal = d["kcal"]
        g = d.get("grams_est")
        text = f"{name} {conf:.2f} ~{kcal:.0f}kcal"
        if g is not None:
            text += f" ~{g:.0f}g"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        tx = max(0, x1)
        ty = max(0, y1 - 3)
        bg1 = (tx, ty - th - 6)
        bg2 = (tx + tw + 6, ty)

        cv2.rectangle(img_bgr, bg1, bg2, (0, 0, 0), -1)
        cv2.putText(
            img_bgr,
            text,
            (tx + 3, ty - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
    return img_bgr


def _extract_first_json(text: str) -> dict[str, Any]:
    """Extract first JSON object from model text output."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("AI output does not contain JSON object.")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("AI output JSON is not an object.")
    return obj


def _generate_ai_nutrition_plan(
    payload: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Call Doubao(Ark) via OpenAI-compatible API and return structured JSON."""
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"openai package is required: {e}")

    if not api_key:
        raise RuntimeError("Missing API key.")
    if not model:
        raise RuntimeError("Missing model/endpoint id.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    system_prompt = (
        "你是专业营养建议助手。根据识别到的食物和估算热量，给出可执行的饮食建议。"
        "仅返回一个JSON对象，不要输出任何额外文本。"
        "JSON必须包含键：meal_assessment, risk_flags, immediate_actions, next_meal_plan, daily_plan, disclaimer。"
        "其中 risk_flags/immediate_actions 为字符串数组；其余为简洁中文字符串。"
    )
    user_prompt = (
        "请基于以下数据生成饮食建议，注意建议应保守、可执行，避免医疗诊断语气：\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    content = (resp.choices[0].message.content or "").strip()
    return _extract_first_json(content)


def main() -> None:
    check_requirements("streamlit>=1.29.0")
    import streamlit as st

    st.set_page_config(page_title="Food Calorie Estimator", layout="wide")
    st.title("Food detection + calorie estimation (Ultralytics YOLO)")
    with st.expander("关于热量「准确度」说明（作业里建议写进报告）", expanded=False):
        st.markdown(
            """
- 若你没有真实克数标签，建议优先使用 **S/M/L 份量模式**：按检测框在图中的相对大小分成小/中/大，再用每类 S/M/L 参考克数估算热量。
- **像素本身没有厘米刻度**。要估计克数，必须做**尺度标定**：告诉程序「这张照片的宽度对应真实世界多少厘米」。
- **标定 + 体积粗估**：检测框在标定后得到面积（cm²）× 假设厚度（cm）× 密度（g/cm³）→ **估算克数**，再按 **kcal/100g** 换算热量。堆叠、遮挡、形状不规则时误差仍较大，属于**可解释的工程近似**。
- **识别越准、标定越认真、营养表越可靠，结果越可信**：请保证训练轮数足够，并按类食物调整 JSON 里的 `kcal_per_100g` / `thickness_cm` / `density_g_per_cm3`。
            """
        )

    script_dir = os.path.dirname(__file__)
    default_mapping_path = os.path.join(script_dir, "food_calories_template.json")

    with st.sidebar:
        st.header("Configuration")

        model_path = st.text_input(
            "Model path (.pt / .onnx / ...)",
            value=r"runs\detect\food_v1_full\weights\best.pt",
            help="训练完成后替换为你的 best.pt；试跑可用 runs\\detect\\food_v1_try\\weights\\best.pt",
        )

        conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
        iou = st.slider("IoU threshold (NMS)", 0.0, 1.0, 0.45, 0.01)

        st.divider()
        st.subheader("Calories mapping")
        mapping_upload = st.file_uploader("Upload mapping JSON (optional)", type=["json"])
        unknown_kcal = st.number_input(
            "unknown_kcal（简单模式：未匹配类别的一份参考热量）",
            min_value=0.0,
            value=150.0,
            step=10.0,
            help="仅用于 count / bbox_area 等简单模式；体积模式见下方 unknown_kcal_per_100g。",
        )
        unknown_kcal_per_100g = st.number_input(
            "unknown_kcal_per_100g（体积模式：未匹配类别）",
            min_value=1.0,
            value=220.0,
            step=10.0,
            help="体积标定模式下，未知类别默认按每 100 克多少千卡计算。",
        )

        MODE_LABELS = {
            "portion_sml": "S/M/L 份量估算（无真实克数时推荐）",
            "volume_calibrated": "标定宽度 + 体积粗估（推荐，可估克数）",
            "count": "固定份量（每框一份参考热量）",
            "bbox_area": "按检测框占整图比例缩放",
            "mask_area": "掩码面积（需分割模型）",
        }
        portion_mode = st.selectbox(
            "热量估算方式",
            list(MODE_LABELS.keys()),
            index=0,
            format_func=lambda k: MODE_LABELS[k],
        )
        area_power = st.slider("面积缩放指数（bbox/mask）", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        conf_weight = st.checkbox("用置信度加权热量", value=False)
        sml_small_thresh = 0.04
        sml_large_thresh = 0.12
        default_portion_m_grams = 180.0
        if portion_mode == "portion_sml":
            st.caption("基于检测框面积占整图比例，把每个目标归为 S/M/L 份量。")
            sml_small_thresh = float(
                st.number_input(
                    "S阈值（面积占比，小于该值算S）",
                    min_value=0.001,
                    max_value=0.5,
                    value=0.04,
                    step=0.005,
                    format="%.3f",
                )
            )
            sml_large_thresh = float(
                st.number_input(
                    "L阈值（面积占比，大于该值算L）",
                    min_value=0.005,
                    max_value=0.9,
                    value=0.12,
                    step=0.01,
                    format="%.3f",
                )
            )
            default_portion_m_grams = float(
                st.number_input(
                    "默认M份量克数（JSON缺失时使用）",
                    min_value=50.0,
                    max_value=600.0,
                    value=180.0,
                    step=10.0,
                )
            )

        scene_width_cm = 28.0
        global_thickness_cm = 2.5
        global_density_g_per_cm3 = 0.85
        if portion_mode == "volume_calibrated":
            st.caption("请先估「整张照片在真实世界里有多宽（厘米）」——例如整盘菜在画面里从左到右约 25cm，就填 25。")
            scene_width_cm = float(
                st.number_input(
                    "画面宽度（厘米）",
                    min_value=1.0,
                    value=28.0,
                    step=0.5,
                    help="对应图像水平像素宽度 w 的真实尺寸；越准则克数估计越准。",
                )
            )
            global_thickness_cm = float(
                st.number_input(
                    "默认食物厚度（厘米）",
                    min_value=0.5,
                    value=2.5,
                    step=0.1,
                    help="某类未单独写 thickness_cm 时用此值。",
                )
            )
            global_density_g_per_cm3 = float(
                st.number_input(
                    "默认密度（克/立方厘米）",
                    min_value=0.1,
                    value=0.85,
                    step=0.05,
                    help="菜、饭、肉差异大；JSON 里可按类覆盖。",
                )
            )

        st.divider()
        with st.expander("AI饮食建议（豆包）", expanded=False):
            enable_ai = st.checkbox("启用豆包建议", value=False)
            ark_api_key = st.text_input(
                "ARK_API_KEY",
                value=os.getenv("ARK_API_KEY", ""),
                type="password",
                help="建议通过环境变量设置。不要把密钥写进代码或提交到仓库。",
            )
            ark_base_url = st.text_input(
                "ARK_BASE_URL",
                value=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            )
            ark_model = st.text_input(
                "ARK_ENDPOINT_ID（作为model）",
                value=os.getenv("ARK_ENDPOINT_ID", os.getenv("LLM_MODEL", "ep-20260410144429-tjgm4")),
                help="火山方舟接入点ID，例如 ep-xxxxxx。",
            )
            ai_goal = st.selectbox("目标", ["维持体重", "减脂", "增肌"], index=0)
            ai_activity = st.selectbox("活动水平", ["低", "中", "高"], index=1)
            ai_notes = st.text_input("忌口/备注（可选）", value="")

    if "uploaded_image" not in st.session_state:
        st.session_state["uploaded_image"] = None

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if uploaded_image is None:
        st.info("Upload an image to run detection.")
        return

    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not decode the uploaded image.")
        return

    calories_path = default_mapping_path
    food_config: dict[str, dict[str, Any]] = {}
    try:
        if mapping_upload is not None:
            raw = json.loads(mapping_upload.read().decode("utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Uploaded JSON must be an object.")
            food_config = _parse_food_config(raw)
        else:
            food_config = _parse_food_config(_safe_json_load_raw(calories_path))
    except Exception as e:
        st.error(f"Failed to load calories mapping: {e}")
        return

    try:
        model = _load_model_cached(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Ensure class_names are in index order.
    class_names_dict = model.names
    class_names = [class_names_dict[i] for i in sorted(class_names_dict.keys())]
    with st.sidebar.expander("Detected classes (from model.names)", expanded=False):
        if not class_names:
            st.write("No classes found in the model.")
        else:
            st.write(class_names)

    missing_cal = [
        class_names[i]
        for i in range(len(class_names))
        if _normalize_key(class_names[i]) not in food_config
    ]
    if missing_cal:
        st.sidebar.warning(
            "热量 JSON 未覆盖这些类别（将使用 unknown 默认值）：\n" + ", ".join(missing_cal)
        )

    with st.spinner("Running YOLO inference..."):
        res_list = model(img_bgr, conf=conf, iou=iou)
    if not res_list:
        st.warning("No results returned by the model.")
        return

    res = res_list[0]
    dets = _estimate_calories(
        res=res,
        class_names=class_names,
        food_config=food_config,
        unknown_kcal=float(unknown_kcal),
        unknown_kcal_per_100g=float(unknown_kcal_per_100g),
        portion_mode=portion_mode,
        area_power=float(area_power),
        conf_weight=bool(conf_weight),
        scene_width_cm=float(scene_width_cm),
        global_thickness_cm=float(global_thickness_cm),
        global_density_g_per_cm3=float(global_density_g_per_cm3),
        sml_small_thresh=float(sml_small_thresh),
        sml_large_thresh=float(sml_large_thresh),
        default_portion_m_grams=float(default_portion_m_grams),
    )

    total_kcal = float(sum(d["kcal"] for d in dets))

    # Annotate image: use Ultralytics plot + our extra kcal text.
    annotated = res.plot()
    annotated = _draw_kcal_labels(annotated, dets)

    st.subheader("Results")
    left, right = st.columns(2)
    with left:
        st.image(img_bgr, channels="BGR", caption="Original")
    with right:
        st.image(annotated, channels="BGR", caption="Annotated + kcal labels")

    st.metric("Total estimated calories", f"{total_kcal:.0f} kcal")

    # Summary by class
    kcal_by_class = defaultdict(float)
    count_by_class = defaultdict(int)
    for d in dets:
        kcal_by_class[d["class_name"]] += float(d["kcal"])
        count_by_class[d["class_name"]] += 1

    st.subheader("Detection summary")
    if len(dets) == 0:
        st.warning("No detections above conf threshold.")
        return

    rows = []
    for cls_name, kcal_sum in sorted(kcal_by_class.items(), key=lambda x: x[1], reverse=True):
        rows.append(
            {
                "class_name": cls_name,
                "count": count_by_class[cls_name],
                "estimated_kcal": int(round(kcal_sum)),
            }
        )
    st.table(rows)

    st.subheader("Per-detection details")
    detail_rows = []
    for d in sorted(dets, key=lambda x: x["kcal"], reverse=True):
        row = {
            "class_name": d["class_name"],
            "conf": round(d["conf"], 3),
            "kcal": int(round(d["kcal"])),
            "bbox": [int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])],
        }
        if d.get("grams_est") is not None:
            row["grams_est"] = int(round(d["grams_est"]))
        if d.get("size_bucket") is not None:
            row["size_bucket"] = d["size_bucket"]
        detail_rows.append(row)
    st.table(detail_rows)

    if enable_ai:
        st.subheader("AI 饮食建议（豆包）")
        if st.button("生成AI饮食建议"):
            ai_payload = {
                "goal": ai_goal,
                "activity_level": ai_activity,
                "notes": ai_notes,
                "total_estimated_kcal": round(total_kcal, 1),
                "items": [
                    {
                        "class_name": d["class_name"],
                        "kcal": round(float(d["kcal"]), 1),
                        "conf": round(float(d["conf"]), 3),
                        "grams_est": round(float(d["grams_est"]), 1) if d.get("grams_est") is not None else None,
                        "size_bucket": d.get("size_bucket"),
                    }
                    for d in dets
                ],
                "summary_by_class": rows,
            }
            try:
                advice = _generate_ai_nutrition_plan(
                    payload=ai_payload,
                    api_key=ark_api_key.strip(),
                    base_url=ark_base_url.strip(),
                    model=ark_model.strip(),
                    temperature=0.2,
                )
                st.success("AI建议生成成功")
                st.write("**本餐评估：**", advice.get("meal_assessment", ""))
                st.write("**风险提示：**")
                for x in advice.get("risk_flags", []):
                    st.write(f"- {x}")
                st.write("**即时建议：**")
                for x in advice.get("immediate_actions", []):
                    st.write(f"- {x}")
                st.write("**下一餐建议：**", advice.get("next_meal_plan", ""))
                st.write("**当日建议：**", advice.get("daily_plan", ""))
                st.caption(advice.get("disclaimer", "仅供参考，不替代专业医生或营养师建议。"))
                with st.expander("原始JSON", expanded=False):
                    st.json(advice)
            except Exception as e:
                st.error(f"AI建议生成失败：{e}")


if __name__ == "__main__":
    main()

