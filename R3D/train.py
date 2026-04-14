if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import copy
import random
import time
import threading
import datetime
import pathlib
import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast
from torch.utils.data import DataLoader
import dill
import wandb
import tqdm
import numpy as np
import shutil
from termcolor import cprint
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from r3d.policy.dp3 import DP3
from r3d.dataset.base_dataset import BaseDataset
from r3d.env_runner.base_runner import BaseRunner
from r3d.common.checkpoint_util import TopKCheckpointManager
from r3d.common.pytorch_util import dict_apply, optimizer_to
from r3d.model.diffusion.ema_model import EMAModel
from r3d.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

def setup_ddp(rank, world_size):
    """Initialize DDP environment"""
    # Note: MASTER_ADDR and MASTER_PORT should be set by torchrun
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(hours=10))
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up DDP environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if current process is the main process"""
    return not dist.is_initialized() or dist.get_rank() == 0

def _copy_to_cpu(state_dict):
    """Copy state dict to CPU"""
    cpu_state_dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state_dict[key] = value.cpu()
        else:
            cpu_state_dict[key] = value
    return cpu_state_dict

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        # DDP setup
        self.use_ddp = cfg.training.get('use_ddp', False)
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0)) if self.use_ddp else 0
        self.world_size = int(os.environ.get('WORLD_SIZE', 1)) if self.use_ddp else 1

        self.use_bfloat16 = cfg.training.get('use_bfloat16', False)
        self.autocast_dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
        
        if self.use_ddp:
            setup_ddp(self.local_rank, self.world_size)
            if is_main_process():
                print(f"DDP initialized: rank {self.local_rank}/{self.world_size}")
        if self.use_bfloat16 and is_main_process():
            cprint(f"[Training] Using bfloat16 mixed precision training", "green")
        
        # set seed (add rank to seed for different random states across processes)
        seed = cfg.training.seed + (self.local_rank if self.use_ddp else 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:  # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = True # reduce time cost
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                # Increment epoch since checkpoint contains the completed epoch
                # We should start training from the next epoch
                print(f"Checkpoint loaded: epoch {self.epoch} completed")
                self.epoch += 1
                self.global_step += 1
                print(f"Will resume training from epoch {self.epoch}")

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        
        # Configure data loaders with DDP support
        train_sampler = None
        val_sampler = None
        if self.use_ddp:
            train_sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.local_rank)
            val_dataset = dataset.get_validation_dataset()
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.local_rank)

            # Remove shuffle from dataloader config when using DistributedSampler
            train_dataloader_cfg = dict(cfg.dataloader)
            train_dataloader_cfg['shuffle'] = False
            train_dataloader_cfg['batch_size'] = cfg.dataloader['batch_size'] // self.world_size
            val_dataloader_cfg = dict(cfg.val_dataloader)
            val_dataloader_cfg['shuffle'] = False
            val_dataloader_cfg['batch_size'] = cfg.val_dataloader['batch_size'] // self.world_size

            if is_main_process():
                print(f"Rank {self.local_rank}: Train batch size = {train_dataloader_cfg['batch_size']}")
                print(f"Rank {self.local_rank}: Val batch size = {val_dataloader_cfg['batch_size']}")

            train_dataloader = DataLoader(dataset, sampler=train_sampler, **train_dataloader_cfg)
            val_dataloader = DataLoader(val_dataset, sampler=val_sampler, **val_dataloader_cfg)
        else:
            train_dataloader = DataLoader(dataset, **cfg.dataloader)
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        normalizer = dataset.get_normalizer()

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            
        # configure env
        env_runner = None
        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        
        # configure logging (only on main process)
        wandb_run = None
        if is_main_process():
            # cfg.logging.name = str(cfg.task.name)
            cprint("-----------------------------", "yellow")
            cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
            cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
            cprint("-----------------------------", "yellow")
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                },
                allow_val_change=True
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        if self.use_ddp:
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            device = torch.device(cfg.training.device)
        
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Wrap model with DDP
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            # EMA model doesn't need DDP wrapping

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            # Set epoch for DistributedSampler
            if self.use_ddp and train_sampler is not None:
                train_sampler.set_epoch(self.epoch)

            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            
            # Only show progress bar on main process
            if is_main_process():
                tepoch = tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec)
            else:
                tepoch = train_dataloader
            
            for batch_idx, batch in enumerate(tepoch):
                t1 = time.time()
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                if train_sampling_batch is None:
                    train_sampling_batch = batch
            
                # compute loss
                t1_1 = time.time()
                with autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.use_bfloat16):
                    if self.use_ddp:
                        raw_loss, loss_dict = self.model.module.compute_loss(batch)
                    else:
                        raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                loss.backward()
                
                t1_2 = time.time()

                # step optimizer
                if self.global_step % cfg.training.gradient_accumulate_every == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                t1_3 = time.time()
                # update ema
                if cfg.training.use_ema:
                    if self.use_ddp:
                        # For DDP, update EMA with the underlying model (without DDP wrapper)
                        ema.step(self.model.module)
                    else:
                        ema.step(self.model)
                t1_4 = time.time()
                # logging
                raw_loss_cpu = raw_loss.item()
                if is_main_process():
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                train_losses.append(raw_loss_cpu)
                step_log = {
                    'train_loss': raw_loss_cpu,
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                    'lr': lr_scheduler.get_last_lr()[0]
                }
                t1_5 = time.time()
                step_log.update(loss_dict)
                t2 = time.time()
                
                if verbose and is_main_process():
                    print(f"total one step time: {t2-t1:.3f}")
                    print(f" compute loss time: {t1_2-t1_1:.3f}")
                    print(f" step optimizer time: {t1_3-t1_2:.3f}")
                    print(f" update ema time: {t1_4-t1_3:.3f}")
                    print(f" logging time: {t1_5-t1_4:.3f}")

                is_last_batch = (batch_idx == (len(train_dataloader)-1))
                if not is_last_batch:
                        # log of last step is combined with validation and rollout
                    if is_main_process() and wandb_run is not None:
                        wandb_run.log(step_log, step=self.global_step)
                    self.global_step += 1

                if (cfg.training.max_train_steps is not None) \
                    and batch_idx >= (cfg.training.max_train_steps-1):
                    break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model.module if self.use_ddp else self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout (only on main process)
            if (self.epoch % cfg.training.rollout_every) == 0 and (self.epoch != 0) and RUN_ROLLOUT and env_runner is not None and is_main_process():
                t3 = time.time()
                task_config = getattr(cfg, 'setting', None)
                runner_log = env_runner.run(policy, self.epoch, task_config=task_config)
                t4 = time.time()
                step_log.update(runner_log)

            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    
                    if is_main_process():
                        val_tepoch = tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec)
                    else:
                        val_tepoch = val_dataloader
                    
                    for batch_idx, batch in enumerate(val_tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        with autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.use_bfloat16):
                            if self.use_ddp:
                                loss, loss_dict = self.model.module.compute_loss(batch)
                            else:
                                loss, loss_dict = self.model.compute_loss(batch)
                        val_losses.append(loss)
                        if (cfg.training.max_val_steps is not None) \
                            and batch_idx >= (cfg.training.max_val_steps-1):
                            break
                    
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()

                        # Synchronize validation loss across all processes if using DDP
                        if self.use_ddp:
                            val_loss_tensor = torch.tensor(val_loss, device=device)
                            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                            val_loss = (val_loss_tensor / self.world_size).item()

                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    
                    with autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.use_bfloat16):
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse

            if env_runner is None:
                step_log['test_mean_score'] = - train_loss
                
            # checkpoint (only save on main process)
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt and is_main_process():
                if not cfg.policy.use_pc_color:
                    # Build checkpoint save path
                    base_checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
                    save_path = os.path.join(base_checkpoint_dir, f'{self.epoch}.ckpt')
                else:
                    # Build checkpoint save path (with rgb)
                    base_checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
                    save_path = os.path.join(base_checkpoint_dir, f'{self.epoch}.ckpt')
                self.save_checkpoint(save_path)

            # Synchronize all processes after checkpoint saving
            if self.use_ddp:
                dist.barrier()
                
            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            if is_main_process() and wandb_run is not None:
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

        # Clean up DDP
        if self.use_ddp:
            cleanup_ddp()

    def eval(self):
        # load the latest checkpoint
        cfg = copy.deepcopy(self.cfg)

        # Initialize wandb for evaluation
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB Eval] group: {cfg.logging.group}_eval", "yellow")
        cprint(f"[WandB Eval] name: {cfg.logging.name}_eval", "yellow")
        cprint("-----------------------------", "yellow")
        
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            group=f"{cfg.logging.group}_eval",
            name=f"{cfg.logging.name}_eval",
            project="maniskill eval",
            tags=cfg.logging.get('tags', []) + ['evaluation']
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
                "eval_mode": True
            }
        )

    
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)

        start_epoch = 50
        end_epoch = 150
        epoch_interval = 50

        epochs_to_eval = list(range(start_epoch, end_epoch + 1, epoch_interval))
        
        # Store all results
        all_results = {}
        
        for epoch_tag in epochs_to_eval:
            cprint(f"\n{'='*60}", 'cyan')
            cprint(f"Evaluating checkpoint: {epoch_tag}", 'cyan')
            cprint(f"{'='*60}", 'cyan')
            
            # Load checkpoint
            lastest_ckpt_path = self.get_checkpoint_path(tag=str(epoch_tag))
            
            if not lastest_ckpt_path.is_file():
                cprint(f"Checkpoint {lastest_ckpt_path} not found, skipping...", 'red')
                continue
                
            cprint(f"Loading checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
            
            # Prepare policy
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()
            policy.cuda()
            
            # Run evaluation
            runner_log = env_runner.run(policy)
            
            # Print results for this checkpoint
            cprint(f"\n---------------- Eval Results (Epoch {epoch_tag}) --------------", 'magenta')
            for key, value in runner_log.items():
                if isinstance(value, float):
                    cprint(f"{key}: {value:.4f}", 'magenta')
            
            # Store results
            all_results[str(epoch_tag)] = runner_log
            
            # Log to wandb
            wandb_log_dict = {
                'eval_epoch': epoch_tag,
            }
            for key, value in runner_log.items():
                if isinstance(value, (int, float)):
                    wandb_log_dict[f'eval/{key}'] = value
            
            wandb_run.log(wandb_log_dict, step=epoch_tag)
        
        # Print summary
        cprint(f"\n{'='*60}", 'green')
        cprint(f"Evaluation Summary", 'green')
        cprint(f"{'='*60}", 'green')
        for epoch_tag, results in all_results.items():
            cprint(f"\nEpoch {epoch_tag}:", 'yellow')
            for key, value in results.items():
                if isinstance(value, float):
                    cprint(f"  {key}: {value:.4f}", 'yellow')
        
        # Create summary table for wandb
        summary_data = []
        for epoch_tag, results in all_results.items():
            row = {'epoch': int(epoch_tag)}
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    row[key] = value
            summary_data.append(row)
        
        # Log summary table to wandb
        if summary_data:
            import pandas as pd
            summary_df = pd.DataFrame(summary_data)
            wandb_run.log({"eval_summary_table": wandb.Table(dataframe=summary_df)})
            cprint(f"\nLogged evaluation summary to wandb", 'green')
        
        # Finish wandb run
        wandb_run.finish()
        cprint(f"\nWandB run finished", 'green')

    def get_policy(self, cfg, checkpoint_num=3000):
        # load the latest checkpoint
        cfg = copy.deepcopy(self.cfg)

        ckpt_file = self.get_checkpoint_path(tag=str(checkpoint_num))
        assert ckpt_file.is_file(), f"ckpt file doesn't exist, {ckpt_file}"
        
        if ckpt_file.is_file():
            cprint(f"Resuming from checkpoint {ckpt_file}", 'magenta')
            self.load_checkpoint(path=ckpt_file)
        
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()
        return policy

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(self, path=None, tag='latest',
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        print('saved in ', path)
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    state_dict = value.state_dict()

                    # Remove 'module.' prefix from DDP wrapped models
                    if key == 'model' and self.use_ddp and hasattr(value, 'module'):
                        # Create a new state dict without 'module.' prefix
                        new_state_dict = {}
                        for k, v in state_dict.items():
                            if k.startswith('module.'):
                                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                            else:
                                new_state_dict[k] = v
                        state_dict = new_state_dict

                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(state_dict)
                    else:
                        payload['state_dicts'][key] = state_dict
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag='latest'):
        if tag:
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag == 'best':
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])

    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(cls, path,
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'r3d', 'config'))
)
def main(cfg):
    # Handle DDP environment variables
    if cfg.training.get('use_ddp', False):
        # Set local rank from environment variable
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        cfg.training.local_rank = local_rank

        # Adjust device setting for DDP
        if cfg.training.device == "cuda":
            cfg.training.device = f"cuda:{local_rank}"

    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
