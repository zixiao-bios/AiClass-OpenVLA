class Config:
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: str = "/mnt/workspace/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
    load_in_8bit: bool = True                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    
    # 每个任务重复执行的次数
    num_trials_per_task: int = 2                    # Number of rollouts per task
    max_steps: int = 220

    #################################################################################################################
    # Utils
    #################################################################################################################
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs


""" libero 中各个任务的最长步数
if cfg.task_suite_name == "libero_spatial":
    max_steps = 220  # longest training demo has 193 steps
elif cfg.task_suite_name == "libero_object":
    max_steps = 280  # longest training demo has 254 steps
elif cfg.task_suite_name == "libero_goal":
    max_steps = 300  # longest training demo has 270 steps
elif cfg.task_suite_name == "libero_10":
    max_steps = 520  # longest training demo has 505 steps
elif cfg.task_suite_name == "libero_90":
    max_steps = 400  # longest training demo has 373 steps
"""