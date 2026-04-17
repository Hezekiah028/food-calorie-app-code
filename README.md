# food-calorie-app-code
CV＋LLM 本项目面向同济大学课程实践场景，完成了一个“食物识别 + 热量估算 + AI建议”的可视化系统。
CV＋LLM 本项目面向同济大学课程实践场景，完成了一个“食物识别 + 热量估算 + AI建议”的可视化系统。

1. Project Overview
系统功能：

使用 YOLO11n 进行食物目标检测（类别、框、置信度）
基于类别参数进行热量估算（总热量 + 分项热量）
可选接入豆包（Ark API）生成饮食建议 核心代码位于：
examples/food_calorie_app/app.py
examples/food_calorie_app/train_food_detect.py
examples/food_calorie_app/food_calories_template.json
examples/food_calorie_app/CALIBRATION_10_IMAGES.md
2. Dataset
使用数据集：chinese food.v1i.yolov8（YOLO格式）

train: 919
valid: 31
test: 21 说明：数据集来自公开平台（Roboflow导出），数据网址(https://universe.roboflow.com/hit-pwt8p/chinese-food)
3. Environment
建议环境：

Python 3.8+
ultralytics
streamlit
opencv-python
openai（可选，用于豆包建议） 安装依赖：
python -m pip install ultralytics streamlit opencv-python openai
4. Train
python examples/food_calorie_app/train_food_detect.py
训练完成后得到 best.pt，在页面 Model path 中填写实际路径。

5. Run Web App
python -m streamlit run examples/food_calorie_app/app.py
页面可完成：

图片上传与识别可视化
热量估算结果输出
可选豆包建议生成
6. Doubao (Optional)
如需启用豆包建议，设置环境变量：

PowerShell:

$env:ARK_API_KEY="your_key"
$env:ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
$env:ARK_ENDPOINT_ID="ep-xxxxxx"
7. Notes
本项目热量为可解释估算，不等同于医疗/营养称重级精度。
API Key、模型权重、完整训练数据集未上传到仓库。
