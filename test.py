from tools.vamba_chat import Vamba

model = Vamba(model_path="ckpts/Vamba-Qwen2-VL-7B", device="cuda")

image_input = [
    {
        "type": "image",
        "content": "assets/old_man.png",
        "metadata": {}
    },
    {
        "type": "text",
        "content": "<image> Describe this image."
    }
]

print(">>> image_input >>>")
print(model(image_input))
print(">>> image_input >>>")

video_input = [
    {
        "type": "video",
        "content": "assets/magic.mp4",
        "metadata": {
            "video_num_frames": 128,
            "video_sample_type": "middle",
            "img_longest_edge": 640,
            "img_shortest_edge": 256,
        }
    },
    {
        "type": "text",
        "content": "<video> Describe the magic trick."
    }
]

print(">>> video_input >>>")
print(model(video_input))
print(">>> video_input >>>")