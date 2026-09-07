import base64
import os
import re
import subprocess
import tempfile
import time
from io import BytesIO

from dotenv import load_dotenv
from groq import Groq
from PIL import Image

load_dotenv()


def extract_frame_from_video(video_filepath):
    """Extract a clear representative frame from the uploaded video using ffmpeg."""
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"video_frame_{int(time.time()*1000)}.jpg")
    try:
        # Try to capture a frame around 1 second mark
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            "00:00:01",
            "-i",
            video_filepath,
            "-vframes",
            "1",
            "-q:v",
            "2",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            # Fallback to very first frame (0s)
            cmd_fallback = [
                "ffmpeg",
                "-y",
                "-i",
                video_filepath,
                "-vframes",
                "1",
                "-q:v",
                "2",
                output_path,
            ]
            subprocess.run(cmd_fallback, capture_output=True, text=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except Exception as e:
        print(f"Failed to extract frame from video: {e}")
    return None


def encode_image_for_groq(filepath):
    image = Image.open(filepath)
    image.thumbnail((1024, 1024))

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=75)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def brain_of_the_doctor(patient_text, image_filepath=None, video_filepath=None):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in .env or environment")

    visual_filepath = image_filepath

    # If no image provided, extract frame from uploaded video
    if not visual_filepath and video_filepath:
        visual_filepath = extract_frame_from_video(video_filepath)

    if not visual_filepath:
        raise ValueError("Please upload a valid skin image or video for analysis.")

    image_data = encode_image_for_groq(visual_filepath)

    prompt = (
        "You are a confident, natural doctor specializing in skin care. Speak with the reassurance, clarity, and authority of a real doctor. "
        "Limit your entire response to two or three sentences maximum. "
        "Do not use any special characters, symbols, asterisks, or markdown formatting in your response because it will be converted directly to audio.\n\n"
        f"Patient text: {patient_text}"
    )

    if video_filepath and not image_filepath:
        prompt += "\nThe patient provided a skin video, and you are reviewing frames captured from their video."
    elif video_filepath and image_filepath:
        prompt += "\nThe patient provided both an image and a video of their skin concern."

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b"),
        max_completion_tokens=400,
        reasoning_effort="none",
        messages=[
            {
                "role": "system",
                "content": "You are a careful skin care assistant. Give general information, not a diagnosis.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                        },
                    },
                ],
            },
        ],
    )

    content = response.choices[0].message.content or ""
    # Strip any potential reasoning tags or extra symbols
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


# OLD CODE KEPT FOR REFERENCE
# import base64
# import os
# from io import BytesIO
#
# from dotenv import load_dotenv
# from groq import Groq
# from PIL import Image
#
#
# folder = os.path.dirname(__file__)
# env_path = os.path.join(folder, ".env")
# load_dotenv(env_path)
#
# api_key = os.environ.get("GROQ_API_KEY")
# if not api_key:
#     raise ValueError("Missing GROQ_API_KEY in .env or environment")
#
#
# image_path = os.path.join(folder, "sample-image.png")
#
# image = Image.open(image_path)
# image.thumbnail((1024, 1024))
#
# buffer = BytesIO()
# image.convert("RGB").save(buffer, format="JPEG", quality=75)
# image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
#
# client = Groq(api_key=api_key)
#
# response = client.chat.completions.create(
#     model=os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
#     max_completion_tokens=1000,
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful medical assistant. Give general information, not a diagnosis.",
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "What do you see in this image? Give general skin care advice, not a diagnosis.",
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{image_data}",
#                     },
#                 },
#             ],
#         },
#     ],
# )
#
# print(response.choices[0].message.content)
