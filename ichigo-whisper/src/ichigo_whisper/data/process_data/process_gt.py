import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


def load_llama_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    return model, tokenizer, device


def process_text_with_llama(text, model, tokenizer, device, timeout=30):
    prompt = f"""You are a Vietnamese text formatter. Your ONLY task is to format Vietnamese text following these EXACT rules:
    1. Capitalize the first letter of each sentence.
    2. Add proper punctuation throughout the text, including commas, periods (.), exclamation marks (!), and question marks (?) where appropriate to enhance readability and conform to standard Vietnamese grammar.
    3. Do not add or remove any words, except for:
    - Correcting spelling errors.
    - Converting written numbers into digits where appropriate.
    4. Preserve and capitalize proper nouns and terms, including:
    - English names (e.g., Nina, Alfred Jarry)
    - Numbers and dates (e.g., 2 giờ 20, 11 năm 2023)
    - Special terms (e.g., Warriors, Derringer 44)
    - Location names (e.g., Việt Nam, Pháp)
    - Person names (e.g., Trần Đình Hưng)

    Examples:

    Input: và mọi chuyện thì chưa dừng lại ở đó  
    Output: Và mọi chuyện thì chưa dừng lại ở đó.

    Input: anh tuấn sinh ngày bảy tháng tám năm hai nghìn lẻ hai  
    Output: Anh Tuấn sinh ngày 7 tháng 8 năm 2002.

    Input: trong bầu không khí nóng nhiệt của rạp hát hùng thủ rút khẩu súng ra ông derringer bốn tư ly, đi sát đầu tổng thống mà nã đạn  
    Output: Trong bầu không khí nóng nhiệt của rạp hát, hung thủ rút khẩu súng Derringer 44 ly, đi sát đầu Tổng thống mà nã đạn.

    Input: srimad bhagavad gita xuất hiện trong mahabharata một trong hai sử thi thiên liêng vĩ đại của ấn độ  
    Output: Srimad Bhagavad Gita xuất hiện trong Mahabharata, một trong hai sử thi thiêng liêng vĩ đại của Ấn Độ.

    Input: alfred jarry một tám bảy ba - một chín tám bảy hợp những nhà văn đối tiếng như  
    Output: Alfred Jarry 1873-1987, hợp những nhà văn nổi tiếng như.

    Input: đến hết tháng mười một năm hai nghìn không trăm hai ba tỷ lệ giải ngân vốn đầu tư công đạt trên sáu bảy phần trăm kế hoạch vốn giao  
    Output: Đến hết tháng 11 năm 2023, tỷ lệ giải ngân vốn đầu tư công đạt trên 67% kế hoạch vốn giao.

    Input: kế hoạch là vậy nhưng cả pháp lẫn tây ban nha vẫn chưa xác định được nơi nào phù hợp để mở màn chiến dịch  
    Output: Kế hoạch là vậy, nhưng cả Pháp lẫn Tây Ban Nha vẫn chưa xác định được nơi nào phù hợp để mở màn chiến dịch.

    Input: bà cho rằng các cuộc xung đột giữa hồi giáo và phật giáo không phải là một cuộc thanh lọc sắc tộc như giới chức quốc tế nhận định  
    Output: Bà cho rằng các cuộc xung đột giữa Hồi giáo và Phật giáo không phải là một cuộc thanh lọc sắc tộc như giới chức quốc tế nhận định.

    Input: thuật ngữ musou xuất phát từ tên tiếng nhật của series warriors đã trở thành từ dùng để gọi các tựa game được tạo ra với cùng một công thức như trên  
    Output: Thuật ngữ Musou, xuất phát từ tên tiếng Nhật của series Warriors, đã trở thành từ dùng để gọi các tựa game được tạo ra với cùng một công thức như trên.

    Input: tôi thích ăn phở nhưng hôm nay tôi muốn thử ăn bánh mì  
    Output: Tôi thích ăn phở, nhưng hôm nay tôi muốn thử ăn bánh mì.

    Input: nếu trời mưa chúng ta sẽ ở nhà còn nếu trời nắng chúng ta sẽ đi dã ngoại  
    Output: Nếu trời mưa, chúng ta sẽ ở nhà, còn nếu trời nắng, chúng ta sẽ đi dã ngoại.

    Input: {text}  
    Output: """

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,  # Reduced since we expect similar length output
                temperature=0.1,  # Low temperature for more deterministic output
                do_sample=False,  # We want deterministic output
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )

        improved_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        improved_text = improved_text.split("Output:")[-1].strip()

        # Verify no extra words were added (allowing for some flexibility due to number conversion)
        input_words = len(text.split())
        output_words = len(improved_text.split())

        if output_words < input_words * 0.8 or output_words > input_words * 1.2:
            print(
                f"Warning: Output length ({output_words}) significantly different from input length ({input_words}). Using original text with basic fixes."
            )
            improved_text = text[0].upper() + text[1:]
            if not improved_text[-1] in ".!?":
                improved_text += "."

        return improved_text

    except Exception as e:
        print(f"Error in text processing: {e}")
        return text


def process_batch(batch, model, tokenizer, device):
    improved_transcriptions = []
    for text in batch["transcription"]:
        try:
            improved_text = process_text_with_llama(text, model, tokenizer, device)
            improved_transcriptions.append(improved_text)
            print(f"Original: {text}")
            print(f"Improved: {improved_text}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing text: {e}")
            improved_transcriptions.append(text)
    return improved_transcriptions


def process_dataset_split(
    dataset_split, model, tokenizer, device, batch_size=32
):  # Reduced batch size
    def collate_fn(batch):
        return {
            "audio": [item["audio"] for item in batch],
            "transcription": [item["transcription"] for item in batch],
        }

    dataloader = DataLoader(
        dataset_split, batch_size=batch_size, collate_fn=collate_fn, num_workers=2
    )
    llama_transcriptions = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        transcriptions = process_batch(batch, model, tokenizer, device)
        llama_transcriptions.extend(transcriptions)

    new_dataset = dataset_split.add_column(
        "llama_improved_transcription", llama_transcriptions
    )
    return new_dataset


def main():
    # Load model
    print("Loading Llama model...")
    model, tokenizer, device = load_llama_model()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("linhtran92/viet_bud500")

    # Process each split
    processed_splits = {}
    for split in dataset.keys():
        print(f"Processing {split} split...")
        processed_splits[split] = process_dataset_split(
            dataset[split], model, tokenizer, device
        )

    # Push to hub
    print("Pushing to hub...")
    processed_dataset = Dataset.from_dict(processed_splits)
    processed_dataset.push_to_hub("jan-hq/viet_bud500_final")
    print("Done!")


if __name__ == "__main__":
    main()
