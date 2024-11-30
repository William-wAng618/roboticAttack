import argparse
import os
import subprocess
import copy
import time
import queue
from concurrent.futures import ThreadPoolExecutor
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, default="", required=False)
    parser.add_argument('--cudaid', type=int, default=1, required=False)
    parser.add_argument('--trials', type=int, default=10, required=False)
    parser.add_argument('--max_concurrent_tasks', type=int, default=10, required=False)
    return parser.parse_args()

data = [
        {"dataset":"libero_spatial", "checkpoints":"openvla/openvla-7b-finetuned-libero-spatial", "x": 120, "y": 160, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_spatial", "checkpoints":"openvla/openvla-7b-finetuned-libero-spatial", "x": 120, "y": 163, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_spatial", "checkpoints":"openvla/openvla-7b-finetuned-libero-spatial", "x": 120, "y": 157, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_spatial", "checkpoints":"openvla/openvla-7b-finetuned-libero-spatial", "x": 123, "y": 160, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_spatial", "checkpoints":"openvla/openvla-7b-finetuned-libero-spatial", "x": 123, "y": 163, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_spatial", "checkpoints":"openvla/openvla-7b-finetuned-libero-spatial", "x": 123, "y": 157, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_spatial", "checkpoints":"openvla/openvla-7b-finetuned-libero-spatial", "x": 117, "y": 160, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_spatial", "checkpoints":"openvla/openvla-7b-finetuned-libero-spatial", "x": 117, "y": 163, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_spatial", "checkpoints":"openvla/openvla-7b-finetuned-libero-spatial", "x": 117, "y": 157, "angle": 0, "shx": 0, "shy": 0},

        {"dataset":"libero_object", "checkpoints":"openvla/openvla-7b-finetuned-libero-object", "x": 30, "y": 150, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_object", "checkpoints":"openvla/openvla-7b-finetuned-libero-object", "x": 30, "y": 153, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_object", "checkpoints":"openvla/openvla-7b-finetuned-libero-object", "x": 30, "y": 147, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_object", "checkpoints":"openvla/openvla-7b-finetuned-libero-object", "x": 33, "y": 150, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_object", "checkpoints":"openvla/openvla-7b-finetuned-libero-object", "x": 33, "y": 153, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_object", "checkpoints":"openvla/openvla-7b-finetuned-libero-object", "x": 33, "y": 147, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_object", "checkpoints":"openvla/openvla-7b-finetuned-libero-object", "x": 27, "y": 150, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_object", "checkpoints":"openvla/openvla-7b-finetuned-libero-object", "x": 27, "y": 153, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_object", "checkpoints":"openvla/openvla-7b-finetuned-libero-object", "x": 27, "y": 147, "angle": 0, "shx": 0, "shy": 0},

        {"dataset":"libero_goal", "checkpoints":"openvla/openvla-7b-finetuned-libero-goal", "x": 15, "y": 158, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_goal", "checkpoints":"openvla/openvla-7b-finetuned-libero-goal", "x": 15, "y": 161, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_goal", "checkpoints":"openvla/openvla-7b-finetuned-libero-goal", "x": 15, "y": 155, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_goal", "checkpoints":"openvla/openvla-7b-finetuned-libero-goal", "x": 18, "y": 158, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_goal", "checkpoints":"openvla/openvla-7b-finetuned-libero-goal", "x": 18, "y": 161, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_goal", "checkpoints":"openvla/openvla-7b-finetuned-libero-goal", "x": 18, "y": 155, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_goal", "checkpoints":"openvla/openvla-7b-finetuned-libero-goal", "x": 12, "y": 158, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_goal", "checkpoints":"openvla/openvla-7b-finetuned-libero-goal", "x": 12, "y": 161, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_goal", "checkpoints":"openvla/openvla-7b-finetuned-libero-goal", "x": 12, "y": 155, "angle": 0, "shx": 0, "shy": 0},

        {"dataset":"libero_10", "checkpoints":"openvla/openvla-7b-finetuned-libero-10", "x": 5, "y": 160, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_10", "checkpoints":"openvla/openvla-7b-finetuned-libero-10", "x": 5, "y": 163, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_10", "checkpoints":"openvla/openvla-7b-finetuned-libero-10", "x": 5, "y": 157, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_10", "checkpoints":"openvla/openvla-7b-finetuned-libero-10", "x": 8, "y": 160, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_10", "checkpoints":"openvla/openvla-7b-finetuned-libero-10", "x": 8, "y": 163, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_10", "checkpoints":"openvla/openvla-7b-finetuned-libero-10", "x": 8, "y": 157, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_10", "checkpoints":"openvla/openvla-7b-finetuned-libero-10", "x": 2, "y": 160, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_10", "checkpoints":"openvla/openvla-7b-finetuned-libero-10", "x": 2, "y": 163, "angle": 0, "shx": 0, "shy": 0},
        {"dataset":"libero_10", "checkpoints":"openvla/openvla-7b-finetuned-libero-10", "x": 2, "y": 157, "angle": 0, "shx": 0, "shy": 0},
]
def organize_exp(exp_path,args):
    task_list = []
    adv_experiemnt = os.listdir(exp_path)
    for i in range(len(adv_experiemnt)):
        iter_filename = os.listdir(os.path.join(exp_path, adv_experiemnt[i]))[0]
        iter_filepath = os.path.join(exp_path, adv_experiemnt[i], iter_filename)
        pt_filepath = os.path.join(iter_filepath, "patch.pt")
        for j in range(len(data)):
            run_id_note=f'{str(data[j]["x"])}_{str(data[j]["y"])}_{str(data[j]["angle"])}_{str(data[j]["shx"])}_{str(data[j]["shy"])}'
            exp = copy.deepcopy(data[j])
            if os.path.exists(os.path.join(iter_filepath, f"EVAL-{exp['dataset']}--{run_id_note}.txt")):
                print(f"exclude: EVAL-{exp['dataset']}--{run_id_note}.txt")
                continue
            exp["run_id_note"] = run_id_note
            exp["patchroot"] = pt_filepath
            exp["local_log_dir"] = iter_filepath
            exp["exp_name"] =str(os.path.join(os.path.join(iter_filepath,exp['dataset'],f'{str(data[j]["x"])}_{str(data[j]["y"])}_{str(data[j]["angle"])}_{str(data[j]["shx"])}_{str(data[j]["shy"])}')))
            os.makedirs(os.path.join(iter_filepath, exp['dataset'],
                                     f'{str(data[j]["x"])}_{str(data[j]["y"])}_{str(data[j]["angle"])}_{str(data[j]["shx"])}_{str(data[j]["shy"])}'),
                        exist_ok=True)
            task_list.append([exp,os.path.join(iter_filepath, f"EVAL-{exp['dataset']}--{run_id_note}.txt")])
            with open(os.path.join(iter_filepath,f"{exp['dataset']}.txt"), "w") as file:
                a=1
    return task_list

def run_task(task):
    if os.path.exists(task[1]):
        print(f"exclude: {task[1]}")
        return 0
    else:
        print(task[0])
        print(task)
        process = subprocess.Popen(task[0], shell=True)
        process.wait()
        return process.returncode

def main():
    args = parse_args()
    origin_task_list = organize_exp(args.exp_path, args)
    random.shuffle(origin_task_list)
    task_list = []
    for item in origin_task_list:
        train_cmd0 = f"python experiments/robot/libero/run_libero_eval_args_geo_batch.py --exp_name {item[0]['exp_name']} --pretrained_checkpoint {item[0]['checkpoints']} --task_suite_name {item[0]['dataset']} --num_trials_per_task {args.trials} --run_id_note {item[0]['run_id_note']} --local_log_dir {item[0]['local_log_dir']} --patchroot {item[0]['patchroot']} --cudaid {args.cudaid} --x {item[0]['x']} --y {item[0]['y']} --angle {item[0]['angle']} --shx {item[0]['shx']} --shy {item[0]['shy']} "
        task_list.append([train_cmd0,item[1]])
    max_concurrent_tasks = args.max_concurrent_tasks
    task_queue = queue.Queue()

    for task in task_list:
        task_queue.put(task)

    with ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        futures = []
        while not task_queue.empty() or any([f.running() for f in futures]):
            if len(futures) < max_concurrent_tasks and not task_queue.empty():
                task = task_queue.get()
                futures.append(executor.submit(run_task, task))

            # Remove completed tasks
            futures = [f for f in futures if not f.done()]
            time.sleep(0.5)



if __name__ == "__main__":
    main()
