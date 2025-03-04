from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
from Unsloth import get_model_and_tokenizer
from Reward import correct_reward_func, length_reward_func, xml_count_reward_func, strict_format_reward_func, soft_format_reward_func, action_format_reward_func
from Reward import get_maze_map

training_args = GRPOConfig(
    use_vllm = False,
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 6,
    gradient_accumulation_steps = 1,
    num_generations = 6,
    max_prompt_length = 256,
    max_completion_length = 200,
    max_steps = 250,
    save_steps = 50,
    max_grad_norm = 0.1,
    report_to = "wandb",
    output_dir = "outputs"
)

model, tokenizer = get_model_and_tokenizer()
dataset = get_maze_map()

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs = [
        correct_reward_func,
        length_reward_func,
        xml_count_reward_func,
        strict_format_reward_func,
        soft_format_reward_func,
        action_format_reward_func
    ],
    args = training_args,
    train_dataset = dataset,
)

trainer.train()
