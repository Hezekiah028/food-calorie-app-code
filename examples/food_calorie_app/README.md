## Food detection + calorie estimation (Streamlit)

### 完整流程（推荐按顺序做）

1. **训练检测模型（识别食物类别）**  
   编辑 `train_food_detect.py` 中的 `data_yaml`、`epochs`、`run_name`，在项目根目录执行：  
   `python examples/food_calorie_app/train_food_detect.py`  
   得到例如：`runs/detect/food_v1_full/weights/best.pt`。

2. **核对热量表 JSON 与数据集类别名一致**  
   本仓库已提供 `food_calories_template.json`，其 **key 需与 `data.yaml` 里 `names` 完全一致**（例如 `rice`、`fried_eggs`）。  
   若 key 写错，页面会一直用侧边栏的 **unknown_kcal**，看起来就像「全是同一个数」。

3. **启动网页**（Windows 建议）：  
   `python -m streamlit run examples/food_calorie_app/app.py`  
   在 **Model path** 填你的 `best.pt` 路径；上传图片测试。

4. **热量模式选择**（侧边栏）  
   - **`S/M/L 份量估算（无真实克数时推荐）`**：按检测框面积占比把目标分成小/中/大，结合每类 `portion_grams_s/m/l` 与 `kcal_per_100g` 计算热量。  
   - `标定宽度 + 体积粗估`：你能提供画面真实宽度（厘米）时再用，适合做更细致估算。  
   - `count` / `bbox_area`：更简化的基线模式。  

### 为什么以前热量全是 200？

旧版模板里的类别名（如 `noodles`）**不是你的数据集类别**，匹配失败就会走 **unknown_kcal**（默认约 200）。请使用与 `model.names` 一致的 key。

### Calories mapping format

JSON 示例：

```json
{
  "class_name_1": 123,
  "class_name_2": 456
}
```

`food_calories_template.json` 中的数值为**常见份量级 kcal 参考**，可按需要替换为查表数据，并在报告中注明来源与假设。

### Notes on “calorie estimation”

单张照片无法从像素得到真实克数，因此热量为**可解释的估算**：类别参考值 +（可选）面积缩放。若要更接近真实，需要额外信息（称重、分割体积估计、或营养数据库按克查询）。

### 快速校准指南

已提供 `CALIBRATION_10_IMAGES.md`，按“10张图快速校准流程”可把你本机拍摄场景下的误差明显收敛。

### 豆包AI饮食建议（可选）

应用已支持通过火山方舟（Doubao/Ark）OpenAI兼容接口生成饮食建议：

1. 安装依赖：`python -m pip install openai`
2. 设置环境变量（PowerShell）：
   - `$env:ARK_API_KEY="你的key"`
   - `$env:ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"`
   - `$env:ARK_ENDPOINT_ID="ep-xxxxxx"`
3. 启动应用后在侧边栏展开 **AI饮食建议（豆包）**，启用并生成建议。

注意：请勿把 API Key 写入代码或提交到仓库。

