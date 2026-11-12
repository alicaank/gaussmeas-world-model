import torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args=None,
):
    """Flow-matching SFT training loop with optional W&B logging."""
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    use_wandb = args is not None and getattr(args, "wandb_project", None) is not None
    if use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    global_step = 0
    # The dataset uses these to drive curriculum sampling.
    if hasattr(dataset, "num_epochs"):
        dataset.num_epochs = num_epochs
    for epoch_id in range(num_epochs):
        if hasattr(dataset, "current_epoch"):
            dataset.current_epoch = epoch_id
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
                global_step += 1

                if use_wandb and accelerator.is_main_process:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/epoch": epoch_id,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    }, step=global_step)

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

    if use_wandb and accelerator.is_main_process:
        wandb.finish()
