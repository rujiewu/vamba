import os
import json
import logging
import torch
import torch.distributed as dist
import fire
from pathlib import Path
from tqdm import tqdm
from eval.utils_videomme import VideoMMEDataset


def setup_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return local_rank, world_size


def compute_summary(core_data: dict) -> dict:
    all_items = [item for answers in core_data.values() for item in answers]

    task_stats = {}
    duration_stats = {}
    for item in all_items:
        tt = item["task_type"]
        task_stats.setdefault(tt, {"correct": 0, "total": 0})
        task_stats[tt]["total"] += 1
        if item["correct"]:
            task_stats[tt]["correct"] += 1

        dur = item["duration"]
        duration_stats.setdefault(dur, {"correct": 0, "total": 0})
        duration_stats[dur]["total"] += 1
        if item["correct"]:
            duration_stats[dur]["correct"] += 1

    correct_all = sum(v["correct"] for v in task_stats.values())
    total_all   = sum(v["total"]   for v in task_stats.values())

    logging.info("=== Per-task Accuracy ===")
    for tt, v in task_stats.items():
        acc = v["correct"] / v["total"]
        logging.info(f"  {tt}: {v['correct']}/{v['total']} = {acc:.4f}")

    logging.info("=== Per-duration Accuracy (s) ===")
    for dur, v in duration_stats.items():
        acc = v["correct"] / v["total"]
        logging.info(f"  {dur}s: {v['correct']}/{v['total']} = {acc:.4f}")

    logging.info(f"=== Overall Accuracy: {correct_all}/{total_all} = {correct_all/total_all:.4f}")

    return {
        "per_task_accuracy": {
            tt: {"correct": v["correct"], "total": v["total"], "accuracy": v["correct"]/v["total"]}
            for tt, v in task_stats.items()
        },
        "per_duration_accuracy": {
            dur: {"correct": v["correct"], "total": v["total"], "accuracy": v["correct"]/v["total"]}
            for dur, v in duration_stats.items()
        },
        "overall_accuracy": {"correct": correct_all, "total": total_all, "accuracy": correct_all/total_all}
    }


def main(
    model_type="vamba",
    model_name_or_path="ckpts/Vamba-Qwen2-VL-7B",
    data_dir="/home/wurujie/workspace/dataset/videomme",
    frames_dir=None,
    num_frames=512,
    img_shortest_edge=256,
    img_longest_edge=480,
    max_img_seq_len=120000,
    do_resize=True,
    use_subtitle=False,
    results_dir="res/videomme",
    # generation config
    max_new_tokens=512,
    do_sample=False,
    top_k=None,
    top_p=0.9,
    temperature=0.6,
):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    torch.manual_seed(42)
    local_rank, world_size = setup_dist()

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "vamba":
        from tools.vamba_chat import Vamba
        model = Vamba(model_name_or_path)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    sample_config = {
        'num_frames': num_frames,
        'sample_type': 'uniform',
        'model_patch_size': model.patch_size,
        'img_shortest_edge': img_shortest_edge,
        'img_longest_edge': img_longest_edge,
        'max_img_seq_len': max_img_seq_len,
        'do_resize': do_resize,
    }

    dataset = VideoMMEDataset(
        data_dir,
        frames_path=frames_dir,
        sample_config=sample_config,
        use_subtitle=use_subtitle
    )

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
    }
    
    local_data = {}
    total = len(dataset)
    rank_indices = range(local_rank, total, world_size)
    
    if local_rank == 0:
        rank_iter = tqdm(rank_indices, desc=f"Rank {local_rank}")
    else:
        rank_iter = rank_indices
    
    for idx in rank_iter:
        entry = dataset[idx]
        vid_path = entry["video_path"]
        if vid_path in local_data:
            continue

        frames = entry["video"]
        questions = entry["questions"]
        answers = []

        for q in questions:
            msgs = [
                {"type": "pil_video", "content": frames},
                {"type": "text", "content": f"<video> {q['text']}"}
            ]
            with torch.no_grad():
                resp = model(msgs, generation_config)

            opt = next((c for c in resp if c.isalpha()), "")
            correct = (opt.upper() == q["answer"].upper())
            q["response"] = resp
            q["correct"] = correct
            answers.append(q)

        local_data[vid_path] = answers
        
        del frames, answers, msgs, resp, q, opt, correct
        torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()

    part_dir = results_dir
    part_dir.mkdir(parents=True, exist_ok=True)
    part_file = part_dir / f"{Path(model_name_or_path).name}_part{local_rank}.json"
    with open(part_file, "w") as pf:
        json.dump(local_data, pf, indent=4)

    local_data.clear()
    torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()

    if local_rank == 0:
        merged = {}
        for r in range(world_size):
            pf = part_dir / f"{Path(model_name_or_path).name}_part{r}.json"
            with open(pf) as f:
                merged.update(json.load(f))
            pf.unlink()

        results_file = part_dir / f"{Path(model_name_or_path).name}.json"
        with open(results_file, "w") as wf:
            json.dump(merged, wf, indent=4)

        summary = compute_summary(merged)
        summary_file = results_file.with_name(results_file.stem + "_summary.json")
        with open(summary_file, "w") as sf:
            json.dump(summary, sf, indent=4)

    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    fire.Fire(main)