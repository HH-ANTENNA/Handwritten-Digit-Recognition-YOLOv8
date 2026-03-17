from ultralytics import YOLO
import tkinter as tk
from tkinter import Canvas, Label, Frame, Button
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import os
import warnings
import torch  # 必须导入 torch
import sys

warnings.filterwarnings("ignore")

# ===================== 1. 基础配置 =====================
target_dir = r"D:\desktop\yolo_digit_plate"
os.makedirs(target_dir, exist_ok=True)  # 自动创建文件夹
MODEL_PATH = os.path.join(target_dir, "simple_digit_model.pt")
MNIST_SIZE = 32
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
# GPU传输优化 
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # 指定使用第0块GPU
    torch.backends.cudnn.enabled = True  # 开启cudnn，加速GPU计算/传输

# ===================== 2. 自动检测设备 (GPU优先) =====================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===================== 3. 训练模型（带早停） =====================
def train_simple_model():
    """训练模型：带patience早停"""
    if os.path.exists(MODEL_PATH):
        print("✅ 检测到已有训练好的模型，跳过训练")
        return YOLO(MODEL_PATH).to(DEVICE)
    
    print("🚀 开始训练（CPU+GPU 加速 | patience=5自动早停）")
    print("💡 提示：训练中按Ctrl+C可安全停止，模型会自动保存")
    model = YOLO("yolov8n-cls.yaml").to(DEVICE)
    # 新增：训练前清理GPU缓存，提升数据传输效率
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True  # 开启cudnn优化，加速GPU数据传输
    try:
        # 核心优化：增加训练轮数、批大小和线程数，并启用早停
        model.train(
            data="mnist",
            epochs=10,          # 最大轮数，早停会提前终止
            imgsz=MNIST_SIZE,
            batch=64,          # 大批量更稳定
            device=DEVICE,
            lr0=0.001,          # 低学习率更稳定
            pretrained=False,
            patience=5,         # 连续5轮loss不下降则自动早停
            workers=8,          # 多线程加载数据
            weight_decay=0.0005 # 防过拟合
        )
        model.save(MODEL_PATH)
        print(f"✅ 训练完成！模型已保存到：{MODEL_PATH}")
    except KeyboardInterrupt:
        # 手动按Ctrl+C停止时，安全保存当前模型
        model.save(MODEL_PATH)
        print(f"\n✅ 训练已手动停止！当前模型已保存到：{MODEL_PATH}")
    return model

# ===================== 4. 加载模型（仅加载，不训练） =====================
def load_trained_model():
    """仅加载已有模型，不训练"""
    if not os.path.exists(MODEL_PATH):
        print("❌ 未找到训练好的模型，请先选择「训练+识别」！")
        sys.exit(1)
    print(f"✅ 成功加载模型：{MODEL_PATH}")
    return YOLO(MODEL_PATH).to(DEVICE)

# ===================== 5. 手写识别界面 =====================
class SimpleDigitApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("手写数字识别 ")
        self.root.geometry("2500x2000") # 此处可更改窗口尺寸，自由决定绘画空间

        self.canvas = Canvas(root, width=2000, height=1200, bg="white") # 画布尺寸可调整，越大绘画空间越大，但可能需要更高性能的GPU支持
        self.canvas.pack(pady=20)
        self.res_label = Label(root, text="识别结果：无", font=("Arial", 24), fg="red")
        self.res_label.pack(pady=10)
        
        btn_frame = Frame(root)
        btn_frame.pack()
        Button(btn_frame, text="清空", command=self.clear, font=("Arial", 16)).grid(row=0, column=0, padx=20)
        Button(btn_frame, text="识别", command=self.recognize, font=("Arial", 16)).grid(row=0, column=1, padx=20)

        self.img = Image.new("L", (2000, 1200), 255)  # 与画布尺寸一致
        self.draw = ImageDraw.Draw(self.img)
        self.last_xy = None
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "last_xy", None))

    def draw_line(self, e):
        if self.last_xy:
            self.canvas.create_line(self.last_xy[0], self.last_xy[1], e.x, e.y, width=20, fill="black", capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_xy, (e.x, e.y)], fill=0, width=20)
        self.last_xy = (e.x, e.y)

    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (2000, 1200), 255)    # 与画布尺寸一致
        self.draw = ImageDraw.Draw(self.img)
        self.res_label.config(text="识别结果：无")

    def preprocess_image(self, img):
        """将单个数字图像转换为模型友好的 MNIST 风格输入。"""
        # 1. 转灰度并去噪
        img = img.convert("L").filter(ImageFilter.GaussianBlur(radius=1))

        # 2. 自动阈值（Otsu）二值化，得到白底黑字
        img_np = np.array(img, dtype=np.uint8)
        hist, _ = np.histogram(img_np, bins=256, range=(0, 255))
        total = img_np.size
        sum_total = np.dot(np.arange(256), hist)
        sum_b = 0
        w_b = 0
        max_var = 0
        threshold = 0
        for i in range(256):
            w_b += hist[i]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += i * hist[i]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * (m_b - m_f) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = i
        bin_img = (img_np < threshold).astype(np.uint8) * 255

        # 3. 提取数字紧密边界框，用于归一化大小/居中
        coords = np.column_stack(np.where(bin_img > 0))
        if coords.size == 0:
            crop = bin_img
        else:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            crop = bin_img[y0 : y1 + 1, x0 : x1 + 1]

        # 4. 等比缩放到 MNIST 尺寸并在黑底上居中
        h, w = crop.shape
        if h == 0 or w == 0:
            crop_img = Image.fromarray(bin_img)
        else:
            scale = (MNIST_SIZE - 4) / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            crop_img = Image.fromarray(crop).resize((new_w, new_h), Image.Resampling.LANCZOS)
        padded = Image.new("L", (MNIST_SIZE, MNIST_SIZE), 0)
        paste_x = (MNIST_SIZE - crop_img.width) // 2
        paste_y = (MNIST_SIZE - crop_img.height) // 2
        padded.paste(crop_img, (paste_x, paste_y))

        # 5. 转为3通道RGB（YOLO 期望输入）
        img_np = np.array(padded, dtype=np.uint8)
        img_np = np.stack([img_np, img_np, img_np], axis=-1)
        return img_np

    def _segment_digits(self, img):
        """基于连通域分割画布上的每个数字区域。支持多行手写数字。"""
        gray = img.convert("L")
        np_img = np.array(gray, dtype=np.uint8)
        # 简易阈值：把明显笔迹提取出来
        bin_img = (np_img < 200).astype(np.uint8)

        h, w = bin_img.shape
        labels = np.zeros_like(bin_img, dtype=np.int32)
        bboxes = []
        cur_label = 0

        # 4-连通域标记（含对角），提取每个连通区域的边界框
        for y in range(h):
            for x in range(w):
                if bin_img[y, x] and labels[y, x] == 0:
                    cur_label += 1
                    labels[y, x] = cur_label
                    stack = [(y, x)]
                    min_x = max_x = x
                    min_y = max_y = y
                    while stack:
                        yy, xx = stack.pop()
                        for dy in (-1, 0, 1):
                            ny = yy + dy
                            if ny < 0 or ny >= h:
                                continue
                            for dx in (-1, 0, 1):
                                nx = xx + dx
                                if nx < 0 or nx >= w:
                                    continue
                                if bin_img[ny, nx] and labels[ny, nx] == 0:
                                    labels[ny, nx] = cur_label
                                    stack.append((ny, nx))
                                    min_x = min(min_x, nx)
                                    max_x = max(max_x, nx)
                                    min_y = min(min_y, ny)
                                    max_y = max(max_y, ny)
                    bboxes.append((min_x, min_y, max_x + 1, max_y + 1))

        # 过滤掉噪声（过小连通域）
        bboxes = [b for b in bboxes if (b[2] - b[0]) * (b[3] - b[1]) > 50]

        # 如果无法分割出多个数字，则直接返回整个画布的预处理结果
        if not bboxes:
            return [self.preprocess_image(img)]

        # 先按 y 排序，再按 x 排序，保证多行数字按阅读顺序排列
        # 这里将连通域按行分组：先按 y 排序，再按行间距聚类，确保每行从左到右输出
        bboxes.sort(key=lambda b: (b[1], b[0]))
        heights = [b[3] - b[1] for b in bboxes]
        avg_h = np.mean(heights) if heights else 0
        row_gap = max(20, avg_h * 0.6)

        rows = []  # [(row_center_y, [bboxes...]), ...]
        for b in bboxes:
            y_center = (b[1] + b[3]) / 2
            if not rows or y_center - rows[-1][0] > row_gap:
                rows.append([y_center, [b]])
            else:
                rows[-1][1].append(b)

        ordered_bboxes = []
        for _, row in rows:
            row_sorted = sorted(row, key=lambda b: b[0])  # 按 x 轴从左到右
            ordered_bboxes.extend(row_sorted)

        digit_imgs = []
        for xmin, ymin, xmax, ymax in ordered_bboxes:
            pad = 5
            xmin2 = max(0, xmin - pad)
            ymin2 = max(0, ymin - pad)
            xmax2 = min(w, xmax + pad)
            ymax2 = min(h, ymax + pad)
            crop = img.crop((xmin2, ymin2, xmax2, ymax2))
            digit_imgs.append(self.preprocess_image(crop))

        return digit_imgs

    def recognize(self):
        digit_inputs = self._segment_digits(self.img)
        results = []
        for idx, digit_input in enumerate(digit_inputs, start=1):
            res = self.model(digit_input, imgsz=MNIST_SIZE, augment=False, device=DEVICE)
            digit = res[0].probs.top1
            conf = round(float(res[0].probs.top1conf), 2)
            results.append(f"{digit}({conf})")

        self.res_label.config(text="识别结果: " + " ".join(results))

# ===================== 6. 启动菜单（功能拆分核心） =====================
def show_start_menu():
    """启动菜单：选择训练/仅识别/退出"""
    print("\n" + "="*50)
    print("🎯 手写数字识别-启动菜单")
    print("="*50)
    print("1. 训练模型 + 启动识别（首次运行选这个）")
    print("2. 仅启动识别（已有模型，跳过训练）")
    print("3. 退出程序")
    print("="*50)
    
    while True:
        choice = input("请输入选择（1/2/3）：")
        if choice == "1":
            model = train_simple_model()  # 训练+加载
            break
        elif choice == "2":
            model = load_trained_model()  # 仅加载
            break
        elif choice == "3":
            print("👋 程序已退出")
            sys.exit(0)
        else:
            print("❌ 输入错误，请输入1/2/3！")
    return model

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 第一步：显示启动菜单，选择功能
    model = show_start_menu()
    # 第二步：启动手写识别界面
    root = tk.Tk()
    app = SimpleDigitApp(root, model)
    root.mainloop()