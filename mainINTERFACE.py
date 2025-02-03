import os
import time
import torch
import gradio as gr
import shutil
import warnings
import numpy as np
from decord import VideoReader, cpu
import copy
import requests
from PIL import Image
import sys

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

warnings.filterwarnings("ignore")

device = "cuda"
device_map = "auto"
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"

print("Loading model... (ใช้เวลาสักครู่)")

tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained,
    None,
    model_name,
    torch_dtype="bfloat16",
    device_map=device_map
)
model.eval()

SERVER_VIDEOS_DIR = "server_videos"
os.makedirs(SERVER_VIDEOS_DIR, exist_ok=True)

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def refresh_file_list():
    files = [f for f in os.listdir(SERVER_VIDEOS_DIR) if f.lower().endswith(".mp4")]
    files.sort()
    return files

def upload_video(user_file, custom_filename):
    if user_file is None:
        return "No file uploaded.", gr.Dropdown.update()

    _, ext = os.path.splitext(user_file.name)
    if ext.lower() != ".mp4":
        raise gr.Error("Only .mp4 files are allowed!")
    
    try:
        if not custom_filename.strip():
            custom_filename = os.path.basename(user_file.name)

        if not custom_filename.lower().endswith(".mp4"):
            custom_filename += ".mp4"

        base, ext = os.path.splitext(custom_filename)
        counter = 1
        final_filename = custom_filename
        while os.path.exists(os.path.join(SERVER_VIDEOS_DIR, final_filename)):
            final_filename = f"{base}_{counter}{ext}"
            counter += 1

        target_path = os.path.join(SERVER_VIDEOS_DIR, final_filename)
        shutil.copyfile(user_file.name, target_path)

        msg = f"Uploaded as {final_filename} successfully!"
        
        files = refresh_file_list()
        dropdown_update = gr.Dropdown.update(choices=files, value=final_filename)
        return msg, dropdown_update

    except Exception as e:
        return f"Error uploading file: {str(e)}", gr.Dropdown.update()

def update_filename(user_file):
    if user_file is None:
        return ""
    filename = os.path.basename(user_file.name)
    base, _ = os.path.splitext(filename)
    return base

def refresh_videos():
    files = refresh_file_list()
    return gr.Dropdown.update(choices=files)

def show_video(selected_filename):
    if not selected_filename:
        return None
    video_path = os.path.join(SERVER_VIDEOS_DIR, selected_filename)
    if not os.path.exists(video_path):
        return None
    return video_path

def process_video(server_video_name, user_input, max_tokens, temperature, top_p, top_k):
    if not server_video_name:
        raise gr.Error("No video selected.")

    video_path = os.path.join(SERVER_VIDEOS_DIR, server_video_name)
    if not os.path.exists(video_path):
        raise gr.Error(f"File not found: {video_path}")

    max_frames_num = 18
    video_frames, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)

    video_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
    video_tensor = video_tensor.to(device, dtype=torch.bfloat16)
    video = [video_tensor]

    conv_template = "qwen_2"
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."

    question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{user_input}"

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(device)

    with torch.inference_mode():
        cont = model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_tokens,
        )

    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return text_outputs

def main():
    preset_questions = [
        "Please summarize this video in detail.",
        "I want to know the name of the movie.",
        "What platform is this movie released on?"
    ]
    with gr.Blocks() as demo:
        gr.Markdown("## LLaVA-NeXT Video Summarization (Qwen)")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1) Upload video")
                uploader = gr.File(
                    label="Choose a video from your PC",
                    file_types=[".mp4"],
                    type="file"
                )
                filename_input = gr.Textbox(
                    label="Enter filename (without extension)",
                    placeholder="Enter desired filename",
                    interactive=True
                )
                upload_btn = gr.Button("Upload to Server")
                upload_status = gr.Textbox(label="Upload Status")

                uploader.change(
                    fn=update_filename,
                    inputs=[uploader],
                    outputs=[filename_input]
                )

            with gr.Column():
                gr.Markdown("### 2) Select a video from server_videos/")
                video_list = gr.Dropdown(
                    label="Available videos",
                    choices=[],
                    value=None
                )
                refresh_btn = gr.Button("Refresh File List")
                video_player = gr.Video(
                    label="Preview",
                    type="filepath"
                )

                refresh_btn.click(
                    fn=refresh_videos,
                    inputs=[],
                    outputs=[video_list]
                )
                upload_btn.click(
                    fn=upload_video,
                    inputs=[uploader, filename_input],
                    outputs=[upload_status, video_list]
                )
                video_list.change(
                    fn=show_video,
                    inputs=[video_list],
                    outputs=[video_player]
                )

        gr.Markdown("### 3) Ask your question / Summarize")

        user_prompt = gr.Dropdown(
            choices=preset_questions,
            allow_custom_value=True,
            label="Enter your prompt or select from preset",
            value=""
        )
        style_radio = gr.Radio(
            choices=["Creative", "Balanced", "Strict", "Custom"],
            label="Answer Style",
            value="Balanced"
        )
        max_tokens_slider = gr.Slider(
            minimum=1,
            maximum=4096,
            value=512,
            step=1,
            label="max_tokens"
        )
        temperature_slider = gr.Slider(
            minimum=0.0,
            maximum=1.99,
            value=0.5,
            step=0.01,
            label="temperature"
        )
        top_p_slider = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.75,
            step=0.01,
            label="top_p"
        )
        top_k_slider = gr.Slider(
            minimum=1,
            maximum=100,
            value=40,
            step=1,
            label="top_k"
        )
        def style_change(selected_style):
            if selected_style == "Creative":
                return (
                    gr.update(value=1024),
                    gr.update(value=1.5),
                    gr.update(value=0.9),
                    gr.update(value=40)
                )
            elif selected_style == "Balanced":
                return (
                    gr.update(value=512),
                    gr.update(value=0.5),
                    gr.update(value=0.75),
                    gr.update(value=40)
                )
            elif selected_style == "Strict":
                return (
                    gr.update(value=256),
                    gr.update(value=0.2),
                    gr.update(value=0.5),
                    gr.update(value=30)
                )
            else:
                return (
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True)
                )

        style_radio.change(
            fn=style_change,
            inputs=[style_radio],
            outputs=[max_tokens_slider, temperature_slider, top_p_slider, top_k_slider]
        )

        submit_btn = gr.Button("Generate")
        text_output = gr.Textbox(label="Model Output", lines=5)

        submit_btn.click(
            fn=process_video,
            inputs=[video_list, user_prompt, max_tokens_slider, temperature_slider, top_p_slider, top_k_slider],
            outputs=[text_output]
        )

        demo.load(
            fn=refresh_videos,
            inputs=[],
            outputs=[video_list]
        )

        demo.launch(server_name="192.168.10.234", server_port=8887)

if __name__ == "__main__":
    main()