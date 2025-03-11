import os
import pdfplumber
import cv2
from paddleocr import PaddleOCR


# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

def preprocess_image(image_path):
    """
    图像预处理：灰度化 -> 二值化 -> 降噪
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

def pdf_to_text(pdf_path, output_path):
    """
    主函数：读取PDF并通过OCR提取文字
    """
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # 提取PDF页面为图像（设置分辨率300dpi提高识别精度）
            image = page.to_image(resolution=300)
            temp_image_path = f'temp_page_{page_num}.png'

            # 保存图像到临时文件
            image.save(temp_image_path)

            # 预处理图像
            processed_image = preprocess_image(temp_image_path)
            cv2.imwrite(temp_image_path, processed_image)

            # 执行OCR识别
            results = ocr.ocr(temp_image_path)
            thisStep="".join([line[1][0] for line in results[0]])
            print(str(page_num) +".")
            # 累加识别结果
            all_text.append(thisStep)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_text))

    # 清理临时文件
    for page_num in range(1, len(pdf.pages) + 1):
        temp_path = f'temp_page_{page_num}.png'
        if os.path.exists(temp_path):
            os.remove(temp_path)
if __name__ == "__main__":
    pdf_path = "input.pdf"    # 输入PDF路径
    output_path = "output.txt"# 输出文本路径
    pdf_to_text(pdf_path, output_path)
