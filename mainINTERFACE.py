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
import magic

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

warnings.filterwarnings("ignore")

device = "cuda"
device_map = "auto"
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"

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

preset_questions = [
    "Please summarize this video in detail.",
    "What main topic."
]

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
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
    return spare_frames, frame_time, video_time

def refresh_file_list():
    files = [f for f in os.listdir(SERVER_VIDEOS_DIR) if f.lower().endswith(".mp4")]
    files.sort()
    return files

def upload_video(user_file, custom_filename):
    try:
        if user_file is None:
            raise gr.Error("Empty!, Did you forget to insert the file?")
        _, ext = os.path.splitext(user_file.name)
        if ext.lower() != ".mp4":
            raise gr.Error("It not mp4 file format!, Please insert only mp4 file format!")

        mime = magic.Magic(mime=True)
        file_mime_type = mime.from_file(user_file.name)
        if not file_mime_type.startswith("video/"):
            raise gr.Error("Upload file is not a valid video file based on it MIME type!, Try other file!")

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
        return f"{str(e)}", gr.Dropdown.update()

def update_filename(user_file):
    if user_file is None:
        return ""
    filename = os.path.basename(user_file.name)
    base, _ = os.path.splitext(filename)
    return base

def refresh_videos():
    files = refresh_file_list()
    return gr.Dropdown.update(choices=files, value=None)

def show_video(selected_filename):
    if not selected_filename:
        return None
    video_path = os.path.join(SERVER_VIDEOS_DIR, selected_filename)
    if not os.path.exists(video_path):
        return None
    return video_path

def process_video(server_video_name, user_input, max_tokens, temperature, top_p, top_k, chat_history, custom_prompt_history, style_radio_value):
    yield (
        chat_history,    
        chat_history,         
        gr.Button.update(interactive=False), 
        gr.Dropdown.update(),
        custom_prompt_history,      
        gr.Button.update(interactive=False), 
        gr.Button.update(interactive=False),  
        gr.Button.update(interactive=False), 
        gr.Radio.update(interactive=False), 
        gr.Slider.update(interactive=False),  
        gr.Slider.update(interactive=False),
        gr.Slider.update(interactive=False),
        gr.Slider.update(interactive=False),  
        gr.Dropdown.update(interactive=False),
        gr.Button.update(interactive=False)
    )
    
    if not server_video_name:
        gr.Info("No video selected.")
        yield (
            chat_history, chat_history,
            gr.Button.update(interactive=True),
            gr.Dropdown.update(),
            custom_prompt_history,
            gr.Button.update(interactive=True),
            gr.Button.update(interactive=True),
            gr.Button.update(interactive=True),
            gr.Radio.update(interactive=True),
            gr.Slider.update(interactive=True),
            gr.Slider.update(interactive=True),
            gr.Slider.update(interactive=True),
            gr.Slider.update(interactive=True),
            gr.Dropdown.update(interactive=True),
            gr.Button.update(interactive=True)
        )
        return

    if not user_input or not user_input.strip():
        gr.Info("No text in prompt. Please enter text.")
        yield (
            chat_history, chat_history,
            gr.Button.update(interactive=True),
            gr.Dropdown.update(),
            custom_prompt_history,
            gr.Button.update(interactive=True),
            gr.Button.update(interactive=True),
            gr.Button.update(interactive=True),
            gr.Radio.update(interactive=True),
            gr.Slider.update(interactive=True),
            gr.Slider.update(interactive=True),
            gr.Slider.update(interactive=True),
            gr.Slider.update(interactive=True),
            gr.Dropdown.update(interactive=True), 
            gr.Button.update(interactive=True)
        )
        return

    if user_input not in preset_questions and user_input not in custom_prompt_history:
        custom_prompt_history.append(user_input)
    updated_prompt_list = preset_questions + custom_prompt_history

    new_pair = [f"<b>User:</b>\n {user_input}", ""]
    updated_history = chat_history + [new_pair]
    yield (
        updated_history, updated_history,
        gr.Button.update(interactive=False),
        gr.Dropdown.update(choices=updated_prompt_list, value=user_input),
        custom_prompt_history,
        gr.Button.update(interactive=False),
        gr.Button.update(interactive=False),
        gr.Button.update(interactive=False),
        gr.Radio.update(interactive=False),
        gr.Slider.update(interactive=False),
        gr.Slider.update(interactive=False),
        gr.Slider.update(interactive=False),
        gr.Slider.update(interactive=False),
        gr.Dropdown.update(interactive=False), 
        gr.Button.update(interactive=False)
    )

    video_path = os.path.join(SERVER_VIDEOS_DIR, server_video_name)
    if not os.path.exists(video_path):
        raise gr.Error(f"File not found: {video_path}")

    max_frames_num = 1
    video_frames, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
    video_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
    video_tensor = video_tensor.to(device, dtype=torch.bfloat16)
    video = [video_tensor]

    conv_template = "qwen_2"
    time_instruction = (
        f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. "
        f"These frames are located at {frame_time}. Please answer the following questions related to this video."
    )
    question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{user_input}"
    
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
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_tokens,
        )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

    tokens = text_outputs.split()
    streaming_text = ""
    for token in tokens:
        streaming_text += token + " "
        new_pair[1] = f"<b>Assistant:</b>\n {streaming_text.strip()}"
        updated_history[-1] = new_pair
        yield (
            updated_history, updated_history,
            gr.Button.update(interactive=False),
            gr.Dropdown.update(choices=updated_prompt_list, value=user_input),
            custom_prompt_history,
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            gr.Radio.update(interactive=False),
            gr.Slider.update(interactive=False),
            gr.Slider.update(interactive=False),
            gr.Slider.update(interactive=False),
            gr.Slider.update(interactive=False),
            gr.Dropdown.update(interactive=False), 
            gr.Button.update(interactive=False)
        )
        time.sleep(0.1)
    
    if style_radio_value == "Custom":
        slider_update = gr.Slider.update(interactive=True)
    else:
        slider_update = gr.Slider.update(interactive=False)
    
    yield (
        updated_history, updated_history,
        gr.Button.update(interactive=True),
        gr.Dropdown.update(choices=updated_prompt_list, value=user_input),
        custom_prompt_history,
        gr.Button.update(interactive=True), 
        gr.Button.update(interactive=True), 
        gr.Button.update(interactive=True),
        gr.Radio.update(interactive=True),  
        slider_update,
        slider_update,  
        slider_update,
        slider_update,  
        gr.Dropdown.update(interactive=True), 
        gr.Button.update(interactive=True)
    )



def show_confirm(selected_file):
    if not selected_file:
        return gr.update(visible=False), "No file selected. Please select a video first."
    return gr.update(visible=True), f"Are you sure you want to delete '{selected_file}'?"

def perform_deletion(selected_file):
    if not selected_file:
        return "No file selected.", gr.update(visible=False), gr.Dropdown.update(choices=refresh_file_list(), value=None)
    video_path = os.path.join(SERVER_VIDEOS_DIR, selected_file)
    if os.path.exists(video_path):
        os.remove(video_path)
        msg = f"'{selected_file}' deleted successfully!"
        gr.Info(f"{msg}")
    else:
        msg = f"File not found: '{selected_file}'"
        gr.Info(f"{msg}")
    files = refresh_file_list()

    return gr.update(visible=False), gr.Dropdown.update(choices=files, value=None)

def cancel_deletion():
    return gr.update(visible=False)

def show_tooltip(message):
    gr.Info(message)

def clear_history():
    return [], []
    
def main():

    custom_css = """
    .message.user {
        background-color: #0047e1 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        padding: 8px 16px !important;
        margin: 4px 20px 4px auto !important;
        max-width: fit-content !important;
        display: inline-block !important;
        float: right !important;
        clear: both !important;
    }
    
    .message.bot {
        background-color: #e1e1e1 !important;
        color: #002d87 !important;
        border-radius: 12px !important;
        padding: 8px 16px !important;
        margin: 4px auto 4px 20px !important;
        max-width: fit-content !important;
        display: inline-block !important;
        float: left !important;
        clear: both !important;
    }
    
    .message-wrap {
        display: flow-root !important;
        margin-bottom: 8px !important;
    }

    /* CSS สำหรับให้ gr.Chatbot ขยายความสูงอัตโนมัติ */
    .custom-chatbot .chatbox {
        height: auto !important;
        max-height: none !important;
        overflow-y: visible !important;
    }
    """

    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("## LLaVA-NeXT Chat with video")
        chat_history = gr.State([])
        custom_prompt_history = gr.State([])
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1) Upload video")
                uploader = gr.File(
                    label="Choose a video from your device",
                    file_types=[".mp4"],
                    type="file", scale=0
                )
                filename_input = gr.Textbox(
                    label="Enter filename (with or without file format)",
                    placeholder="Enter desired filename",
                    interactive=True, scale=0
                )
                upload_btn = gr.Button("Upload to Server")
                upload_status = gr.Textbox(label="Upload Status", scale=0)
                uploader.change(
                    fn=update_filename,
                    inputs=[uploader],
                    outputs=[filename_input]
                )
            with gr.Column():
                gr.Markdown("### 2) Select video in server")
                video_list = gr.Dropdown(
                    label="Available videos",
                    choices=[],
                    value=None, scale=0
                )
                refresh_btn = gr.Button("Refresh Video List")
                delete_btn = gr.Button("Delete Select Video")
                with gr.Column(visible=False) as confirm_container:
                    confirm_msg = gr.Markdown("")
                    with gr.Row():
                        confirm_yes = gr.Button("Yes")
                        confirm_no = gr.Button("No")
                
                delete_btn.click(
                    fn=show_confirm,
                    inputs=[video_list],
                    outputs=[confirm_container, confirm_msg]
                )
                confirm_yes.click(
                    fn=perform_deletion,
                    inputs=[video_list],
                    outputs=[confirm_container, video_list]
                )
                confirm_no.click(
                    fn=cancel_deletion,
                    inputs=[],
                    outputs=[confirm_container]
                )
                video_player = gr.Video(
                    label="Preview",
                    type="filepath", scale=2
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
            with gr.Column():
                gr.Markdown("### 3) Ask")
                user_prompt = gr.Dropdown(
                    choices=preset_questions,
                    allow_custom_value=True,
                    label="Enter your prompt or select from dropdown",
                    value="", scale=0
                )
                with gr.Accordion("Advanced Configuration", open=False):
                    style_radio = gr.Radio(
                        choices=["Creative", "Balanced", "Strict", "Custom"],
                        label="Answer Style",
                        value="Balanced", scale=0
                    )
                    with gr.Row():
                        max_tokens_slider = gr.Slider(
                            minimum=1,
                            maximum=4096,
                            value=512,
                            step=1,
                            label="max_tokens",
                            interactive=False,
                        )
                        max_tokens_tooltip = gr.Button("❓", size="sm", min_width=0.1, scale=0.1)
                        max_tokens_tooltip.click(
                            fn=lambda: show_tooltip("max_tokens คือจำนวนโทเค็นสูงสุดที่โมเดลจะสร้างขึ้นในการตอบกลับ"),
                        )

                    with gr.Row():
                        temperature_slider = gr.Slider(
                            minimum=0.01,
                            maximum=1.99,
                            value=0.5,
                            step=0.01,
                            label="temperature",
                            interactive=False,
                        )
                        temp_tooltip = gr.Button("❓", size="sm", min_width=0.1, scale=0.1)
                        temp_tooltip.click(
                            fn=lambda: show_tooltip("temperature คือค่าที่ควบคุมความหลากหลายและความคิดสร้างสรรค์ของการตอบกลับ"),
                        )
                    
                    with gr.Row():
                        top_p_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.75,
                            step=0.01,
                            label="top_p",
                            interactive=False,
                        )
                        top_p_tooltip = gr.Button("❓", size="sm", min_width=0.1, scale=0.1)
                        top_p_tooltip.click(
                            fn=lambda: show_tooltip("top_p คือวิธีการเลือกโทเค็นสำหรับการสร้างข้อความ โดยพิจารณาจากความน่าจะเป็นสะสม"),
                        )
                    
                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=40,
                            step=1,
                            label="top_k",
                            interactive=False,
                        )
                        top_k_tooltip = gr.Button("❓", size="sm", min_width=0.1, scale=0.1)
                        top_k_tooltip.click(
                            fn=lambda: show_tooltip("top_k คือวิธีการเลือกโทเค็นสำหรับการสร้างข้อความ โดยพิจารณาจากโทเค็นที่มีความน่าจะเป็นสูงสุด K อันดับแรก"),
                        )
                    def style_change(selected_style):
                        if selected_style == "Creative":
                            return (
                                gr.update(value=1024, interactive=False),
                                gr.update(value=1.5, interactive=False),
                                gr.update(value=0.9, interactive=False),
                                gr.update(value=40, interactive=False)
                            )
                        elif selected_style == "Balanced":
                            return (
                                gr.update(value=512, interactive=False),
                                gr.update(value=0.5, interactive=False),
                                gr.update(value=0.75, interactive=False),
                                gr.update(value=40, interactive=False)
                            )
                        elif selected_style == "Strict":
                            return (
                                gr.update(value=256, interactive=False),
                                gr.update(value=0.2, interactive=False),
                                gr.update(value=0.5, interactive=False),
                                gr.update(value=30, interactive=False)
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
                clear_btn = gr.Button("Clear History")
        gr.Markdown("### 4) Result")
        chat_output = gr.Chatbot(label="Model Output",elem_classes="custom-chatbot")
        clear_btn.click(fn=clear_history, inputs=[], outputs=[chat_output, chat_history])

        submit_btn.click(
            fn=process_video,
            inputs=[video_list, user_prompt, max_tokens_slider, temperature_slider, top_p_slider, top_k_slider, chat_history, custom_prompt_history, style_radio],
            outputs=[
                chat_output,       
                chat_history, 
                submit_btn, 
                user_prompt,  
                custom_prompt_history,
                upload_btn,    
                refresh_btn,       
                delete_btn,
                style_radio,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                video_list,
                clear_btn
            ]
        )



        demo.load(
            fn=refresh_videos,
            inputs=[],
            outputs=[video_list]
        )
        demo.launch(server_name="192.168.10.234", server_port=8887, enable_queue=True)

if __name__ == "__main__":
    main()