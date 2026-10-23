import os
import warnings
import logging
import sqlite3
import json
import re
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
from faster_whisper import WhisperModel
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

        for prefix in ["clap:", "tags:", "caption:"]:
            if tag.startswith(prefix):
                tag = tag.replace(prefix, "", 1).strip()

        if not tag:
            continue

        if any(tag.startswith(prefix) for prefix in bad_prefixes):
            continue

        if any(fragment in tag for fragment in bad_fragments):
            continue

        if len(tag.split()) > max_words:
            continue

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


def parse_json_response(raw):
    """Parse JSON even if Ollama adds a small amount of extra text."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def clean_example_contexts(contexts, limit=5):
    """Keep only short, usable situation prompts returned by Ollama."""
    if not isinstance(contexts, list):
        return []

    cleaned = []
    seen = set()

    for context in contexts:
        context = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", str(context))
        context = " ".join(context.strip().split())

        if not context:
            continue

        key = context.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(context)

        if len(cleaned) >= limit:
            break

    return cleaned


def parse_stored_contexts(text):
    """Read quoted comma-separated contexts or older line-separated contexts."""
    text = (text or "").strip()

    if not text:
        return []

    try:
        parsed = json.loads(f"[{text}]")
        if isinstance(parsed, list):
            return clean_example_contexts(parsed)
    except json.JSONDecodeError:
        pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return clean_example_contexts(parsed)
    except json.JSONDecodeError:
        pass

    return clean_example_contexts(text.splitlines())


def format_example_contexts(contexts):
    """Store contexts as quoted, comma-separated sentences."""
    return ", ".join(
        json.dumps(context, ensure_ascii=False)
        for context in clean_example_contexts(contexts)
    )


def get_empty_metadata_items():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, file_path, file_name, media_type, caption, tags,
               example_context
        FROM media_items
        WHERE caption = ''
           OR tags = ''
           OR example_context = ''
        ORDER BY id;
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def update_metadata(item_id, caption, tags, example_context):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE media_items
        SET caption = ?,
            tags = ?,
            example_context = ?
        WHERE id = ?;
    """, (caption, tags, example_context, item_id))

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


def normalize_ocr_text(text):
    text = (text or "").strip()

    if not text:
        return ""

    no_text_values = {
        "no_text",
        "no text",
        "no visible text",
        "none",
        "n/a",
        "there is no text visible",
        "there is no visible text",
    }

    if text.strip().lower().strip(".") in no_text_values:
        return ""

    return text


def generate_visual_metadata(model, tokenizer, image):
    enc_image = model.encode_image(image)

    ocr_text = model.answer_question(
        enc_image,
        (
            "Read every piece of visible text in this image. "
            "Include meme captions, subtitles, signs, labels, usernames, "
            "watermarks, speech bubbles, logos containing words, and background text. "
            "Do not summarize. Transcribe exactly. "
            "If there are multiple text regions, include them all. "
            "If no text exists, respond with exactly: NO_TEXT"
        ),
        tokenizer
    ).strip()

    scene_description = model.answer_question(
        enc_image,
        (
            "Describe what is visually happening in the image in one detailed sentence. "
            "Include the main subject, setting, action, expressions, and mood. "
            "Do not transcribe or mention any visible text."
        ),
        tokenizer
    ).strip()

    ocr_text = normalize_ocr_text(ocr_text)

    if ocr_text:
        caption = f"{scene_description} Visible text: {ocr_text}"
    else:
        caption = scene_description

    tags = model.answer_question(
        enc_image,
        (
            "Generate concise comma-separated tags for visible objects, "
            "setting, text, actions, and concepts. Return only comma-separated tags."
        ),
        tokenizer
    ).strip()

    return caption, tags


def transcribe_with_whisper(whisper_model, file_path):
    segments, info = whisper_model.transcribe(
        file_path,
        beam_size=5
    )

    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    )


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
    transcript="",
    transcript_tags="",
    clap_labels="",
    existing_context=""
):
    prompt = f"""
You are generating metadata for a searchable cat meme database.

Current visual caption:
{original_caption}

Visual tags:
{visual_tags}

Audio transcript from Whisper, for context only:
{transcript}

Short tags extracted from the transcript:
{transcript_tags}

Audio labels from CLAP:
{clap_labels}

Task:
Return improved metadata and five short, broad, reusable texting prompts.

Rules:
- Return ONLY valid JSON with exactly these keys:
  "caption": one string,
  "tags": an array of short lowercase strings,
  "example_context": an array containing exactly five strings.
- Caption should be one sentence.
- Caption should preserve important visible text if included.
- Caption should describe the visual scene, not only the text.
- Tags should be lower case and short.
- Tags should include literal objects, visible text concepts, audio concepts, meme meanings, and likely search terms.
- Tags should also include overall emotion, setting, or action.
- Do not copy full transcript sentences into tags.
- Use transcript only to create short search tags.
- CLAP audio labels should be used as separate tags when relevant.
- Example contexts should be setups, emotional statements, questions,
  greetings, or broad topic references—not complete jokes or punchlines.
- Keep each prompt between 2 and 10 words when possible.
- Prefer vague, reusable language over specific narratives.
- Use only information supported by the caption, tags, transcript, or audio labels.
- Do not invent events, backstories, locations, objects, or actions.
- Do not mention objects that are not visible or explicitly supported.
- Do not imply hidden events such as cooking, going to work, arriving home,
  losing something, or having an argument.
- Do not make every prompt about the cat itself.
- Do not use generic words such as cat, kitten, pet, cute, or meme.
- Preserve the apparent emotional intensity. Do not turn mild sadness into grief.
- Do not require prompts to begin with "When...", "That feeling when...",
  or "Me when...".
- Make the five prompts meaningfully different when possible.
- Do not include numbering, markdown, or explanations outside the JSON.

Existing example context, if any:
{existing_context}

Good tag examples:
["revolt", "busy", "distraction", "avoidance", "satire", "car horn", "angry meow", "chaos", "refusal"]
"""

    response = ask_ollama(prompt)

    caption = original_caption
    tags = visual_tags
    example_context = parse_stored_contexts(existing_context)

    try:
        data = parse_json_response(response)

        if isinstance(data, dict):
            caption = str(data.get("caption", caption)).strip() or caption

            returned_tags = data.get("tags", [])
            if isinstance(returned_tags, list):
                cleaned_tags = clean_comma_list(
                    ", ".join(str(tag) for tag in returned_tags)
                )
                if cleaned_tags:
                    tags = ", ".join(cleaned_tags)

            returned_context = clean_example_contexts(
                data.get("example_context", [])
            )
            if returned_context:
                example_context = returned_context

    except (json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"Warning: Ollama returned invalid metadata JSON: {error}")
        print(response)

    return caption, tags, example_context


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
    whisper_model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )

    print("Loading CLAP...")
    clap_model = ClapModel.from_pretrained(CLAP_MODEL_ID)
    clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)

    for index, row in enumerate(items, start=1):
        (
            item_id,
            file_path,
            file_name,
            media_type,
            old_caption,
            old_tags,
            old_context,
        ) = row

        print("\n" + "=" * 80)
        print(f"[{index}/{len(items)}] ID {item_id}: {file_name}")
        print(f"Type: {media_type}")

        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue

        try:
            old_caption = old_caption or ""
            old_tags = old_tags or ""
            old_context = old_context or ""

            if media_type == "image":
                if old_caption.strip() and old_tags.strip():
                    # The image has already been labeled. Only generate the
                    # new example contexts instead of re-running Moondream.
                    md_caption, md_tags = old_caption, old_tags
                else:
                    image = load_image_from_file(file_path)

                    md_caption, md_tags = generate_visual_metadata(
                        moondream_model,
                        tokenizer,
                        image
                    )

                final_caption, refined_tags, example_context = refine_metadata_with_ollama(
                    original_caption=old_caption.strip() or md_caption,
                    visual_tags=old_tags.strip() or md_tags,
                    existing_context=old_context.strip()
                )

                final_tags = old_tags.strip() or merge_comma_lists(
                    md_tags,
                    refined_tags
                )

            elif media_type == "video":
                if old_caption.strip() and old_tags.strip():
                    # Existing caption/tags are enough to create contexts.
                    # Avoid decoding the video and running Whisper/CLAP again.
                    md_caption, md_tags = old_caption, old_tags
                    transcript = ""
                    transcript_tags = ""
                    clap_labels = ""
                else:
                    image = load_image_from_video(file_path)

                    md_caption, md_tags = generate_visual_metadata(
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

                final_caption, refined_tags, example_context = refine_metadata_with_ollama(
                    original_caption=old_caption.strip() or md_caption,
                    visual_tags=old_tags.strip() or md_tags,
                    transcript=transcript,
                    transcript_tags=transcript_tags,
                    clap_labels=clap_labels,
                    existing_context=old_context.strip()
                )

                final_tags = old_tags.strip() or merge_comma_lists(
                    md_tags,
                    transcript_tags,
                    clap_labels,
                    refined_tags
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
                example_context=format_example_contexts(example_context)
            )

            print("Caption:", final_caption)
            print("Tags:", final_tags)
            print("Example context:")
            for context in example_context:
                print(f"- {context}")
            print("Saved.")

        except Exception as e:
            print(f"Failed on ID {item_id}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
