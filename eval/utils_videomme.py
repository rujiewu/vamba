import torch
import cv2
import copy
import os.path as osp
import pysubs2
import numpy as np
import os

from PIL import Image
from pandas import read_parquet
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from train.data import get_resize_output_image_size

def uniform_indices(num_frames: int, total_frames: int) -> list[int]:
    """Get uniform indices 

    Args:
        num_frames (int): number of frames
        total_frames (int): total number of frames

    Returns:
        list[int]: Output frame indices
    """
    if num_frames < total_frames:
        splits = torch.linspace(0, total_frames, num_frames+1, dtype=int)
        indices = ((splits[:-1] + splits[1:]) // 2).tolist()
    else:
        indices = list(range(total_frames))

    return indices


def fps_indices(input_fps: float, total_frames: int, output_fps: float = None, max_num_frames: int = -1) -> list[int]:
    """Get indices according to the output_fps

    Args:
        input_fps (float): input fps
        total_frames (int): total number of frames
        output_fps (float, optional): output fps. Defaults to None, means output_fps==input_fps.
        max_num_frames (int, optional): max number of frames. Defaults to -1, means no limitation.

    Returns:
        list[int]: Output frame indices
    """
    delta = 1 if output_fps is None else input_fps / output_fps
    indices = torch.arange(0, total_frames, delta).round().to(int)
    indices = [e for e in indices if e < total_frames]
    if 0 < max_num_frames < len(indices):
        indices = indices[:max_num_frames]

    return indices

def load_subtitle(sub_path: str, indices: list[int], fps: float) -> str:
    """Load subtitle related to given indices

    Args:
        sub_path (str): subtitle path
        indices (list[int]): frame indices
        fps (float): video average fps

    Returns:
        str: subtitle
    """
    subs = pysubs2.load(sub_path, encoding='utf-8')
    subtitles = []
    for idx in indices:
        sub_text = []
        cur_time = pysubs2.make_time(fps=fps, frames=idx)
        for sub in subs:
            if sub.end < cur_time:
                continue
            elif sub.start < cur_time:
                sub_text.append(sub.text.replace('\\N', ' '))
                break   # in accordance with the official Video-MME Benchmark
            else:
                break
        sub_text = ' '.join(sub_text)
        if sub_text.strip():
            subtitles.append(sub_text)
    subtitles = '\n'.join(subtitles)

    return subtitles
    

def load_decord(src_path: str, sample_type: str, sub_path: str = None, **kwargs) -> list[Image.Image]:
    """Load video using decord, optionally load subtitles

    Args:
        src_path (str): video path
        sample_type (str): 'uniform' or 'fps'
        sub_path (str): subtitle path, .srt
        kwargs: for 'uniform', require 'num_frames'; for 'fps', optionally require 'output_fps' and 'max_num_frames'

    Returns:
        list[Image.Image] | tuple[list[Image.Image], str]: frame list, subtitle str (optional)
    """
    vr = VideoReader(src_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    do_resize = kwargs.pop('do_resize', False)
    if sample_type == 'uniform':
        num_frames = kwargs.pop('num_frames')
        width = height = None
        if num_frames == "auto":
            model_patch_size = kwargs['model_patch_size']
            img_shortest_edge = kwargs['img_shortest_edge']
            img_longest_edge = kwargs['img_longest_edge']
            max_img_seq_len = kwargs['max_img_seq_len']
            vid = cv2.VideoCapture(src_path)
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            vid.release()
            height, width = get_resize_output_image_size(height, width, img_shortest_edge, img_longest_edge)
            num_patches = int((height // model_patch_size) * (width // model_patch_size))
            num_frames = int(max_img_seq_len // num_patches)
        elif do_resize:
            vid = cv2.VideoCapture(src_path)
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            vid.release()
            img_shortest_edge = kwargs['img_shortest_edge']
            img_longest_edge = kwargs['img_longest_edge']
            height, width = get_resize_output_image_size(height, width, img_shortest_edge, img_longest_edge)
        input_fps = float(vr.get_avg_fps())
        indices = uniform_indices(num_frames, total_frames)
        durations = [idx / input_fps for idx in indices]
        frames = vr.get_batch(indices).asnumpy()        # (T, H, W, C), np.uint8
        frames = [Image.fromarray(frame).resize((int(width), int(height)), resample=3) if width and height else Image.fromarray(frame) for frame in frames]
    elif sample_type == 'fps':
        input_fps = float(vr.get_avg_fps())
        output_fps = kwargs.pop('output_fps', None)
        max_num_frames = kwargs.pop('max_num_frames', -1)
        indices = fps_indices(input_fps, total_frames, output_fps, max_num_frames)
        durations = [idx / input_fps for idx in indices]
        frames = vr.get_batch(indices).asnumpy()        # (T, H, W, C), np.uint8
        frames = [Image.fromarray(frame) for frame in frames]
    else:
        raise ValueError(f'Do not support {sample_type} sample type')

    if sub_path is None:
        return frames, durations
    elif osp.exists(sub_path):
        subtitles = load_subtitle(sub_path, indices=indices, fps=float(vr.get_avg_fps()))
        return frames, durations, subtitles
    else:
        return frames, durations, ''


def load_folder(src_path: str, sample_type: str, video_path: str = None, sub_path: str = None, **kwargs) -> list[Image.Image]:
    """Load video using decord, optionally load subtitles

    Args:
        src_path (str): video path
        sample_type (str): 'uniform' or 'fps'
        sub_path (str): subtitle path, .srt
        kwargs: for 'uniform', require 'num_frames'; for 'fps', optionally require 'output_fps' and 'max_num_frames'

    Returns:
        list[Image.Image] | tuple[list[Image.Image], str]: frame list, subtitle str (optional)
    """
    # list all images in the folder
    img_list = sorted([osp.join(src_path, f) for f in os.listdir(src_path) if f.endswith('.jpg')])
    total_frames = len(img_list)
    img_list = [osp.join(src_path, f"frame_{i}.jpg") for i in range(total_frames)]
    do_resize = kwargs.pop('do_resize', False)
    durations = []
    if sample_type == 'uniform':
        num_frames = kwargs.pop('num_frames')
        width = height = None
        if num_frames == "auto":
            model_patch_size = kwargs['model_patch_size']
            img_shortest_edge = kwargs['img_shortest_edge']
            img_longest_edge = kwargs['img_longest_edge']
            max_img_seq_len = kwargs['max_img_seq_len']
            vid = Image.open(img_list[0])
            width, height = vid.size
            height, width = get_resize_output_image_size(height, width, img_shortest_edge, img_longest_edge)
            num_patches = int((height // model_patch_size) * (width // model_patch_size))
            num_frames = int(max_img_seq_len // num_patches)
        if do_resize:
            vid = Image.open(img_list[0])
            width, height = vid.size
            img_shortest_edge = kwargs['img_shortest_edge']
            img_longest_edge = kwargs['img_longest_edge']
            height, width = get_resize_output_image_size(height, width, img_shortest_edge, img_longest_edge)
        indices = uniform_indices(num_frames, total_frames)
        frames = [np.array(Image.open(os.path.join(src_path, img_list[idx]))) for idx in indices]
        frames = np.array(frames)        # (T, H, W, C), np.uint8
        frames = [Image.fromarray(frame).resize((int(width), int(height)), resample=3) if width and height else Image.fromarray(frame) for frame in frames]
    else:
        raise ValueError(f'Do not support {sample_type} sample type')
    
    if sub_path is None:
        return frames, durations
    elif osp.exists(sub_path):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        input_fps = float(vr.get_avg_fps())
        subtitles = load_subtitle(sub_path, indices=indices, fps=input_fps)
        return frames, durations, subtitles
    else:
        return frames, durations, ''


class VideoMMEDataset(Dataset):
    """Video-MME dataset. By default, videos are saved in `data/` and subtitles are saved in `subtitle/`
    """

    def __init__(self, dataset_path: str, sample_config: dict, frames_path: str = None, use_subtitle: bool = False):
        super().__init__()
        self.dataset_path = dataset_path
        self.frames_path = frames_path
        self.sample_config = sample_config
        self.use_subtitle = use_subtitle

        self.data_dict = {}
        value_keys = ['question_id', 'duration', 'task_type', 'question', 'options', 'answer']
        if 'filtered' not in sample_config or sample_config['filtered'] != True:
            df = read_parquet(osp.join(dataset_path, 'test-00000-of-00001.parquet'))
        else:
            df = read_parquet(osp.join(dataset_path, 'test-00000-of-00001-filtered.parquet'))
        df['options'] = df['options'].apply(list)
        for _, data in df.iterrows():
            key = data['videoID']
            value = data[value_keys].to_dict()
            if key in self.data_dict:
                self.data_dict[key].append(value)
            else:
                self.data_dict[key] = [value]
        
        self.video_list = list(self.data_dict.keys())

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx) -> dict:
        # Assume self.use_subtitle == False

        video_path = self.video_list[idx]
        if self.frames_path is not None:
            frames, durations = load_folder(
                src_path=osp.join(self.frames_path, video_path),
                **self.sample_config
            )
        else:
            frames, durations = load_decord(
                src_path=osp.join(self.dataset_path, 'videos', video_path + '.mp4'),
                **self.sample_config
            )

        question_data = self.data_dict[video_path]
        for i in range(len(question_data)):
            text = "\n".join([
                "Select the best answer to the following multiple-choice question based on the video.",
                "Respond with only the letter (A, B, C, or D) of the correct option.",
                question_data[i]["question"],
                " ".join(question_data[i]["options"]),
            ])
            question_data[i]["text"] = text

        return dict(
            video_path=video_path,
            video=frames,
            durations=durations,
            questions=question_data,
        )