import os
import warnings
import logging
import sqlite3
from PIL import Image

# Reduce noisy warnings/logs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

import torch
import librosa
import whisper
import ollama

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import ClapModel, ClapProcessor

from audio_labels import CLAP_CANDIDATE_LABELS


DB_PATH = "catlib.db"

MOONDREAM_MODEL_ID = "vikhyatk/moondream2"
OLLAMA_MODEL = "llama3.2:latest"
WHISPER_MODEL_SIZE = "base"
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_CONFIDENCE_THRESHOLD = 0.5


def get_connection():
    return sqlite3.connect(DB_PATH)


def clean_comma_list(text, max_words=4):
    text = (text or "").replace("\n", ",")
    parts = text.split(",")

    cleaned = []
    seen = set()

    bad_prefixes = [
        "here are",
        "the tags are",
        "expanded tags",
        "improved tags",
        "caption:",
        "tags:",
        "vibe:",
        "note:",
        "i included",
        "i'd be happy",
        "id be happy",
        "sure",
        "certainly",
        "clap:",
        "clap audio label",
        "audio label",
        "audio labels",
    ]

    bad_fragments = [
        "return only",
        "comma-separated",
        "i would",
        "i can",
        "this image",
        "this video",
        "based on",
        "the following",
    ]

    for part in parts:
        tag = part.strip().lower().strip(" .;:-\"'")

        for prefix in ["clap:", "tags:", "vibe:", "caption:"]:
            if tag.startswith(prefix):
                tag = tag.replace(prefix, "", 1).strip()

        if not tag:
            continue

        if any(tag.startswith(prefix) for prefix in bad_prefixes):
            continue

        if any(fragment in tag for fragment in bad_fragments):
            continue

        # Limit tag length.
        if len(tag.split()) > max_words:
            continue

        # Remove sentence-like outputs.
        if tag.endswith(".") or " i " in f" {tag} ":
            continue

        if tag not in seen:
            seen.add(tag)
            cleaned.append(tag)

    return cleaned


def merge_comma_lists(*items):
    merged = []
    seen = set()

    for item in items:
        for tag in clean_comma_list(item):
            if tag not in seen:
                seen.add(tag)
                merged.append(tag)

    return ", ".join(merged)


def get_empty_metadata_items():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, file_path, file_name, media_type, caption, tags, vibe
        FROM media_items
        WHERE caption = ''
           OR tags = ''
           OR vibe = ''
        ORDER BY id;
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def update_metadata(item_id, caption, tags, vibe):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE media_items
        SET caption = ?,
            tags = ?,
            vibe = ?
        WHERE id = ?;
    """, (caption, tags, vibe, item_id))

    conn.commit()
    conn.close()


def load_image_from_file(file_path):
    return Image.open(file_path).convert("RGB")


def load_image_from_video(file_path):
    import cv2

    video = cv2.VideoCapture(file_path)

    if not video.isOpened():
        raise ValueError(f"Could not open video: {file_path}")

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = frame_count // 2 if frame_count > 0 else 0

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = video.read()
    video.release()

    if not success:
        raise ValueError(f"Could not read frame from video: {file_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def ask_ollama(prompt):
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip()


def generate_visual_metadata(model, tokenizer, image):
    enc_image = model.encode_image(image)

    caption = model.answer_question(
        enc_image,
        (
            "Describe the full image in one detailed sentence. "
            "Include the main subject, setting, action, and mood. "
            "If visible text appears, include the exact text too. "
            "Do not only transcribe the text; describe what is visually happening."
        ),
        tokenizer
    ).strip()

    tags = model.answer_question(
        enc_image,
        (
            "Generate concise comma-separated tags for visible objects, "
            "setting, text, actions, and concepts. Return only comma-separated tags."
        ),
        tokenizer
    ).strip()

    vibe = model.answer_question(
        enc_image,
        (
            "Describe the emotional/social vibe using 3 to 5 "
            "comma-separated words. Return only comma-separated words."
        ),
        tokenizer
    ).strip()

    return caption, tags, vibe


def transcribe_with_whisper(whisper_model, file_path):
    result = whisper_model.transcribe(
        file_path,
        fp16=False,
        verbose=False
    )

    return result.get("text", "").strip()


def classify_audio_with_clap(clap_model, clap_processor, file_path, top_n=8):
    waveform, sample_rate = librosa.load(
        file_path,
        sr=48000,
        mono=True
    )

    inputs = clap_processor(
        text=CLAP_CANDIDATE_LABELS,
        audios=waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = clap_model(**inputs)

    probs = outputs.logits_per_audio[0].softmax(dim=0)

    ranked = sorted(
        zip(CLAP_CANDIDATE_LABELS, probs.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n]


def format_clap_labels(clap_results, threshold=CLAP_CONFIDENCE_THRESHOLD):
    return ", ".join(
        label
        for label, score in clap_results
        if score >= threshold
    )


def extract_transcript_tags_with_ollama(transcript):
    if not transcript.strip():
        return ""

    prompt = f"""
You are extracting searchable tags from an audio transcript for a cat meme database.

Transcript:
{transcript}

Task:
Generate comma-separated tags from the transcript.

Rules:
- Return ONLY comma-separated tags.
- Do NOT copy full sentences.
- Extract important names, places, topics, emotions, actions, and meme concepts.
- Keep tags short.
- Include important words like names, locations, repeated phrases, and strong concepts.
- No explanations.
- No bullet points.

Example:
Transcript: "I was going to the worst place in the world and I didn't even know it yet"
Tags: worst place, doomed, regret, dramatic, narration
"""

    return ask_ollama(prompt)


def refine_metadata_with_ollama(
    original_caption,
    visual_tags,
    visual_vibe,
    transcript="",
    transcript_tags="",
    clap_labels=""
):
    prompt = f"""
You are generating metadata for a searchable cat meme database.

Current visual caption:
{original_caption}

Visual tags:
{visual_tags}

Visual vibe:
{visual_vibe}

Audio transcript from Whisper, for context only:
{transcript}

Short tags extracted from the transcript:
{transcript_tags}

Audio labels from CLAP:
{clap_labels}

Task:
Return improved metadata.

Rules:
- Return exactly this format:
CAPTION: ...
TAGS: ...
VIBE: ...
- Caption should be one sentence.
- Caption should describe the visual scene, not only the text.
- Tags should be comma-separated.
- Vibe should be comma-separated.
- Tags should include literal objects, visible text concepts, audio concepts, meme meanings, and likely search terms.
- Do not copy full transcript sentences into tags.
- Use transcript only to create short search tags.
- CLAP audio labels should be used as separate tags when relevant.
- Vibe should describe emotional/social reaction energy.
- Do not explain your reasoning.

Good tag examples:
revolt, busy, distraction, avoidance, satire, car horn, angry meow, chaos, refusal
"""

    response = ask_ollama(prompt)

    caption = original_caption
    tags = visual_tags
    vibe = visual_vibe

    for line in response.splitlines():
        line = line.strip()

        if line.lower().startswith("caption:"):
            caption = line.split(":", 1)[1].strip()

        elif line.lower().startswith("tags:"):
            tags = line.split(":", 1)[1].strip()

        elif line.lower().startswith("vibe:"):
            vibe = line.split(":", 1)[1].strip()

    return caption, tags, vibe


def main():
    items = get_empty_metadata_items()

    if not items:
        print("No empty metadata items found.")
        return

    print(f"Found {len(items)} items with empty metadata.")

    print("Loading Moondream...")
    tokenizer = AutoTokenizer.from_pretrained(
        MOONDREAM_MODEL_ID,
        trust_remote_code=True
    )

    moondream_model = AutoModelForCausalLM.from_pretrained(
        MOONDREAM_MODEL_ID,
        trust_remote_code=True,
        device_map="auto"
    )

    print("Loading Whisper...")
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

    print("Loading CLAP...")
    clap_model = ClapModel.from_pretrained(CLAP_MODEL_ID)
    clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)

    for index, row in enumerate(items, start=1):
        item_id, file_path, file_name, media_type, old_caption, old_tags, old_vibe = row

        print("\n" + "=" * 80)
        print(f"[{index}/{len(items)}] ID {item_id}: {file_name}")
        print(f"Type: {media_type}")

        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue

        try:
            if media_type == "image":
                image = load_image_from_file(file_path)

                md_caption, md_tags, md_vibe = generate_visual_metadata(
                    moondream_model,
                    tokenizer,
                    image
                )

                final_caption, refined_tags, refined_vibe = refine_metadata_with_ollama(
                    original_caption=old_caption.strip() or md_caption,
                    visual_tags=old_tags.strip() or md_tags,
                    visual_vibe=old_vibe.strip() or md_vibe
                )

                final_tags = old_tags.strip() or merge_comma_lists(
                    md_tags,
                    refined_tags
                )

                final_vibe = old_vibe.strip() or merge_comma_lists(
                    md_vibe,
                    refined_vibe
                )

            elif media_type == "video":
                image = load_image_from_video(file_path)

                md_caption, md_tags, md_vibe = generate_visual_metadata(
                    moondream_model,
                    tokenizer,
                    image
                )

                transcript = transcribe_with_whisper(
                    whisper_model,
                    file_path
                )

                transcript_tags = extract_transcript_tags_with_ollama(
                    transcript
                )

                clap_results = classify_audio_with_clap(
                    clap_model,
                    clap_processor,
                    file_path
                )

                clap_labels = format_clap_labels(clap_results)

                final_caption, refined_tags, refined_vibe = refine_metadata_with_ollama(
                    original_caption=old_caption.strip() or md_caption,
                    visual_tags=old_tags.strip() or md_tags,
                    visual_vibe=old_vibe.strip() or md_vibe,
                    transcript=transcript,
                    transcript_tags=transcript_tags,
                    clap_labels=clap_labels
                )

                final_tags = old_tags.strip() or merge_comma_lists(
                    md_tags,
                    transcript_tags,
                    clap_labels,
                    refined_tags
                )

                final_vibe = old_vibe.strip() or merge_comma_lists(
                    md_vibe,
                    refined_vibe
                )

                print("Transcript:", transcript)
                print("Transcript tags:", transcript_tags)
                print("CLAP labels:", clap_labels)

            else:
                print(f"Skipping unsupported media type: {media_type}")
                continue

            update_metadata(
                item_id=item_id,
                caption=final_caption,
                tags=final_tags,
                vibe=final_vibe
            )

            print("Caption:", final_caption)
            print("Tags:", final_tags)
            print("Vibe:", final_vibe)
            print("Saved.")

        except Exception as e:
            print(f"Failed on ID {item_id}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()