from ultralytics import YOLO


def main():
    # 1) 改成你自己的 data.yaml 绝对路径
    data_yaml = r"C:\Users\DELL\Desktop\ultralytics-main\chinese food.v1i.yolov8\data.yaml"

    # 2) 训练轮数：试跑可改 10～15；正式建议 40～50（CPU 较慢，可过夜跑）
    epochs = 50

    # 3) 每次实验用不同名字，避免覆盖上一次 weights
    run_name = "food_v1_final50"

    # 4) 首次训练建议用小模型，速度快，适合入门
    model = YOLO("yolo11n.pt")

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        project="runs/detect",
        name=run_name,
    )

    # 5) 训练后自动验证最佳模型
    best_pt = f"runs/detect/{run_name}/weights/best.pt"
    model = YOLO(best_pt)
    model.val(data=data_yaml)


if __name__ == "__main__":
    main()

