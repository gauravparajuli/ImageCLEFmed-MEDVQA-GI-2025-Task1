from transformers import AutoModelForCausalLM
from datasets import load_dataset
from transformers import AutoProcessor
import torch
import json
import time
from tqdm import tqdm
import subprocess
import platform
import sys

from evaluate import load

bleu = load("bleu")
rouge = load("rouge")
meteor = load("meteor")

val_dataset = load_dataset("SimulaMet/Kvasir-VQA-test", split="validation")

predictions = []  # List to store predictions

gpu_name = torch.cuda.get_device_name(
    0) if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_mem(): return torch.cuda.memory_allocated(device) / \
    (1024 ** 2) if torch.cuda.is_available() else 0


initial_mem = get_mem()

# ✏️✏️--------EDIT SECTION 1: SUBMISISON DETAILS and MODEL LOADING --------✏️✏️#

SUBMISSION_INFO = {
    # 🔹 TODO: PARTICIPANTS MUST ADD PROPER SUBMISSION INFO FOR THE SUBMISSION 🔹
    # This will be visible to the organizers
    # DONT change the keys, only add your info
    "Participant_Names": "Gaurav Parajuli",
    "Affiliations": "Johannes Kepler Universität Linz",
    "Contact_emails": ["parajuligaurav007@gmail.com", "k12455655@students.jku.at"],
    # But, the first email only will be used for correspondance
    "Team_Name": "MedPixel",
    "Country": "Austria",
    "Notes_to_organizers": '''
        This is a test submission.

        Model was trained on only 100% of available data.

        Reponame is in the format {base_model_name}_{effective_batch_size}_{lora_rank}
        '''
}
# 🔹 TODO: PARTICIPANTS MUST LOAD THEIR MODEL HERE, EDIT AS NECESSARY FOR YOUR MODEL 🔹
# can add necessary library imports here
from peft import PeftModel, PeftConfig

adapter_model_id = "gauravparajuli/florence2_4_r16"
peft_config = PeftConfig.from_pretrained(adapter_model_id)
base_model_id = peft_config.base_model_name_or_path
print(base_model_id)

# Load base model and processor
model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

# Load adapter on top of the base model
model_hf = PeftModel.from_pretrained(model, adapter_model_id).to(device)
model_hf.eval()  # Ensure model is in evaluation mode
# 🏁----------------END  SUBMISISON DETAILS and MODEL LOADING -----------------🏁#

start_time, post_model_mem = time.time(), get_mem()
total_time, final_mem = round(
    time.time() - start_time, 4), round(get_mem() - post_model_mem, 2)
model_mem_used = round(post_model_mem - initial_mem, 2)

for idx, ex in enumerate(tqdm(val_dataset, desc="Validating")):
    question = ex["question"]
    image = ex["image"].convert(
        "RGB") if ex["image"].mode != "RGB" else ex["image"]
    # you have access to 'question' and 'image' variables for each example

# ✏️✏️___________EDIT SECTION 2: ANSWER GENERATION___________✏️✏️#
    # 🔹 TODO: PARTICIPANTS CAN MODIFY THIS TOKENIZATION STEP IF NEEDED 🔹
    inputs = processor(text=[question], images=[image],
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()
              if k not in ['labels', 'attention_mask']}

    # 🔹 TODO: PARTICIPANTS CAN MODIFY THE GENERATION AND DECODING METHOD HERE 🔹
    with torch.no_grad():
        output = model_hf.generate(**inputs)
        # output = model_hf(**inputs)
    answer = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    # make sure 'answer' variable will hold answer (sentence/word) as str
# 🏁________________ END ANSWER GENERATION ________________🏁#

# ⛔ DO NOT EDIT any lines below from here, can edit only upto decoding step above as required. ⛔
    # Ensures answer is a string
    assert isinstance(
        answer, str), f"Generated answer at index {idx} is not a string"
    # Appends prediction
    predictions.append(
        {"index": idx, "img_id": ex["img_id"], "question": ex["question"], "answer": answer})

# Ensure all predictions match dataset length
assert len(predictions) == len(
    val_dataset), "Mismatch between predictions and dataset length"

total_time, final_mem = round(
    time.time() - start_time, 4), round(get_mem() - post_model_mem, 2)
model_mem_used = round(post_model_mem - initial_mem, 2)

# caulcualtes metrics
references = [[e] for e in val_dataset['answer']]
preds = [pred['answer'] for pred in predictions]

bleu_result = bleu.compute(predictions=preds, references=references)
rouge_result = rouge.compute(predictions=preds, references=references)
meteor_result = meteor.compute(predictions=preds, references=references)
bleu_score = round(bleu_result['bleu'], 2)
rouge1_score = round(float(rouge_result['rouge1']), 2)
rouge2_score = round(float(rouge_result['rouge2']), 2)
rougeL_score = round(float(rouge_result['rougeL']), 2)
meteor_score = round(float(meteor_result['meteor']), 2)

public_scores = {
    'bleu': bleu_score,
    'rouge1': rouge1_score,
    'rouge2': rouge2_score,
    'rougeL': rougeL_score,
    'meteor': meteor_score
}
print("✨Public scores: ", public_scores)

############################### categorywise score calculation #################################
from collections import defaultdict

category_public_scores = dict()

ques2cats = {
    'Have all polyps been removed?': 'yes_no',
    'Is this finding easy to detect?': 'yes_no',
    'Is there a green/black box artefact?': 'yes_no',
    'Is there text?': 'yes_no',
    'Does this image contain any finding?': 'yes_no',
    'What type of polyp is present?': 'choice(single)',
    'What type of procedure is the image taken from?':  'choice(single)',
    'What is the size of the polyp?': 'choice(single)', 
    'Are there any abnormalities in the image? Check all that are present.': 'choice(multiple)',
    'Are there any anatomical landmarks in the image? Check all that are present.': 'choice(multiple)',
    'Are there any instruments in the image? Check all that are present.': 'choice(multiple)',
    'What color is the abnormality? If more than one separate with ;': 'choice(color)',
    'What color is the anatomical landmark? If more than one separate with ;': 'choice(color)',
    'Where in the image is the instrument?': 'location',
    'Where in the image is the abnormality?': 'location',
    'Where in the image is the anatomical landmark?': 'location',
    'How many findings are present?': 'numerical_count',
    'How many polyps are in the image?': 'numerical_count',
    'How many instrumnets are in the image?': 'numerical_count'
}

def question_category(example):
    return {'question_category': ques2cats[example['question']]}

val_dataset = val_dataset.map(question_category)

question_categories = val_dataset['question_category']

# Add question categories to predictions
for i, pred in enumerate(predictions):
    pred['question_category'] = question_categories[i]

cat2preds = defaultdict(list)
cat2refs = defaultdict(list)
for i, pred in enumerate(predictions):
    cat = pred['question_category']
    cat2preds[cat].append(pred['answer'])
    cat2refs[cat].append([val_dataset[i]['answer']])  # reference must be wrapped in list



for cat in cat2preds:
    bleu_result = bleu.compute(predictions=cat2preds[cat], references=cat2refs[cat])
    rouge_result = rouge.compute(predictions=cat2preds[cat], references=cat2refs[cat])
    meteor_result = meteor.compute(predictions=cat2preds[cat], references=cat2refs[cat])

    category_public_scores[cat] = {
        'bleu': round(bleu_result['bleu'], 2),
        'rouge1': round(float(rouge_result['rouge1']), 2),
        'rouge2': round(float(rouge_result['rouge2']), 2),
        'rougeL': round(float(rouge_result['rougeL']), 2),
        'meteor': round(float(meteor_result['meteor']), 2)
    }

with open("cat2preds.json", "w") as f:
    json.dump(cat2preds, f, indent=4)

with open("cat2refs.json", "w") as f:
    json.dump(cat2refs, f, indent=4)

print("\n📊 Metrics by Question Category Type:")
for cat, scores in category_public_scores.items():
    print(f"\n🔹 Category: {cat}")
    for metric, val in scores.items():
        print(f"  {metric}: {val}")

################################################################################################


# Saves predictions to a JSON file

output_data = {"submission_info": SUBMISSION_INFO, "public_scores": public_scores,
               "predictions": predictions, "total_time": total_time, "time_per_item": total_time / len(val_dataset),
               "memory_used_mb": final_mem, "model_memory_mb": model_mem_used, "gpu_name": gpu_name,
               "debug": {
                   "packages": json.loads(subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"])),
                   "system": {
                       "python": platform.python_version(),
                       "os": platform.system(),
                       "platform": platform.platform(),
                       "arch": platform.machine()
                   }}}

output_data['category_public_scores'] = category_public_scores # scores category wise

with open("predictions_1.json", "w") as f:
    json.dump(output_data, f, indent=4)
print(f"Time: {total_time}s | Mem: {final_mem}MB | Model Load Mem: {model_mem_used}MB | GPU: {gpu_name}")
print("✅ Scripts Looks Good! Generation process completed successfully. Results saved to 'predictions_1.json'.")
print("Next Step:\n 1) Upload this submission_task1.py script file to HuggingFace model repository.")
print('''\n 2) Make a submission to the competition:\n Run:: medvqa validate_and_submit --competition=gi-2025 --task=1 --repo_id=...''')