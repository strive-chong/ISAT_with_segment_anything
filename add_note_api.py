import os
import json
from pathlib import Path
import re
import dashscope
import argparse

dashscope.api_key = "your_api_key_here"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["32b", "72b"], default="32b")
    parser.add_argument("--input_folder", type=str, required=True,
                       help="Path to the folder containing images")
    return parser.parse_args()




MODEL = "qwen2.5-vl-32b-instruct"
# MODEL = "qwen2.5-vl-72b-instruct"

INPUT_FOLDER = Path("the_path_to_your_images_folder")



def extract_outputs(text):
    print(f"Raw model output: {text}")
    
    eng_patterns = [
        r"Put the.*?into the trash can\.",
        r"English instruction:\s*\"?Put the.*?into the trash can\.\"?",
        r"Put the.*?in(to)? the trash\.",
        r"Put the.*?trash.*?"
    ]
    
    zh_patterns = [
        r"把.*?放进垃圾桶里。",
        r"中文指令：\s*\"?把.*?放进垃圾桶里。\"?",
        r"把.*?放进垃圾桶。",
        r"把.*?丢进垃圾桶里。"
    ]
    
    eng_instruction = ""
    for pattern in eng_patterns:
        eng_match = re.search(pattern, text, re.DOTALL)
        if eng_match:
            eng_instruction = eng_match.group(0).replace("English instruction:", "").replace('"', '').strip()
            break
    
    zh_instruction = ""
    for pattern in zh_patterns:
        zh_match = re.search(pattern, text, re.DOTALL)
        if zh_match:
            zh_instruction = zh_match.group(0).replace("中文指令：", "").replace('"', '').strip()
            break
    
    print(f"Extracted English: {eng_instruction}")
    print(f"Extracted Chinese: {zh_instruction}")
    
    return eng_instruction, zh_instruction

def process_image(image_path, model_use):
    image_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "You are a professional organizer. I will provide you with an image of a cluttered desk. Your job is to select one piece of trash and output exactly two lines:\n English instruction: “Put the {description of the item} {position relative to the desk} into the trash can.” \n中文指令：“把{相对于桌面的位置信息} {物品描述}放进垃圾桶里。”\n -{the item} / {物品} should be a specific object that often appear in daily life (e.g. “can”, “waste paper”, “bottle” / “易拉罐”, “废纸”, “瓶子”).\n Please don't just say “object” or other abstract expressions. \n- {description of the item} / {物品描述} should mention color, shape, or a distinctive feature (e.g. “lying red can” / “倒着的红色罐子”).\n- {position relative to the desk} / {相对于桌面的位置信息} should specify where it sits (e.g. “on the upper left corner of the desk” / “在桌面的左上角”). \n {example} / {例子}: Put the red can on the upper right corner of the table into the trash can. \n 把桌子右上角的红色易拉罐丢进垃圾桶里.\n ```Output only these two lines—nothing else.```"},
                {
                    "type": "image",
                    "image": str(image_path),
                    "min_pixels": 1280 * 960,
                    "max_pixels": 1280 * 960,
                },
            ],
        },
    ]

    response = dashscope.MultiModalConversation.call(model=model_use, messages=image_messages)
    output_text = response["output"].choices[0].message["content"][0]["text"]
    
    return extract_outputs(output_text)

def main():
    args = parse_arguments()
    MODEL = f"qwen2.5-vl-{args.model}-instruct"
    INPUT_FOLDER = Path(args.input_folder)
    english_outputs = {}
    SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')#定义支持的图片格式

    for img_file in os.listdir(INPUT_FOLDER):
        if img_file.lower().endswith(SUPPORTED_IMAGE_FORMATS):
            img_path = INPUT_FOLDER / img_file
            print(f"\n\nProcessing {img_file}...")
            eng_instruction, zh_instruction = process_image(img_path, MODEL)
            
            if not zh_instruction and eng_instruction:
                print("WARNING: No Chinese instruction found. Using translated English instruction.")
                zh_instruction = "把" + eng_instruction.replace("Put the", "").replace("into the trash can.", "放进垃圾桶里。")
            
            chinese_output = {
                "info": {
                    "description": "ISAT",
                    "name": img_file,
                    "note": zh_instruction
                }
            }
            
            output_path = INPUT_FOLDER / f"{os.path.splitext(img_file)[0]}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chinese_output, f, ensure_ascii=False, indent=4)
            print(f"Saved Chinese output to {output_path}")
            
            if eng_instruction:
                english_outputs[img_file] = eng_instruction
    
    eng_output_path = INPUT_FOLDER / "english_note.json"
    with open(eng_output_path, "w", encoding="utf-8") as f:
        json.dump(english_outputs, f, ensure_ascii=False, indent=4)
    print(f"\nProcessing complete. English results saved to {eng_output_path}")

if __name__ == "__main__":
    main()