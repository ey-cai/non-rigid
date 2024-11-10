if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler

from diffusion_policy_3d.policy.tax3d_goalPC import TAX3D


OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
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
            cfg.training.rollout_every = 1 # 20
            cfg.training.checkpoint_every = 1# 20 # 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
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
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        
        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        
        if cfg.enable_wandb:
            cfg.logging.name = str(cfg.logging.name)
            cprint("-----------------------------", "yellow")
            cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
            cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
            cprint("-----------------------------", "yellow")
            # configure logging
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    t1_1 = time.time()
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
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
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
                    
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if cfg.enable_wandb:
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
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout
            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                t3 = time.time()
                # runner_log = env_runner.run(policy, dataset=dataset)
                runner_log = env_runner.run(policy)
                t4 = time.time()
                # print(f"rollout time: {t4-t3:.3f}")
                # log all
                step_log.update(runner_log)

                
            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    
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

            # checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                
                # We can't copy the last checkpoint here
                # since save_checkpoint uses threads.
                # therefore at this point the file might have been empty!
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

    def eval(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy)
        
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
    def eval_datasets(self):
        dataset_cfg = self.cfg.task.dataset
        cloth_geometry = dataset_cfg.cloth_geometry
        cloth_pose = dataset_cfg.cloth_pose
        anchor_geometry = dataset_cfg.anchor_geometry
        anchor_pose = dataset_cfg.anchor_pose
        hole = dataset_cfg.hole
        dataset_name = (
            f'cloth={cloth_geometry}-{cloth_pose} ' + \
            f'anchor={anchor_geometry}-{anchor_pose} ' + \
            f'hole={hole}'
        )
        dataset_dir = os.path.join(
            os.path.expanduser(dataset_cfg.root_dir),
            dataset_name
        )

        # based on the task name, load specific split of the dataset
        # and evaluate the model on it
        import torch.utils.data as data
        # dataset_dir = os.path.expanduser("/home/ktsim/Projects/non-rigid/data/proccloth/cloth=single-fixed anchor=single-random hole=single/")
        class DedoDataset(data.Dataset):
            def __init__(self, dir):
                self.dir = dir
                self.num_demos = int(len(os.listdir(self.dir)))

            def __len__(self):
                return self.num_demos
            
            def __getitem__(self, index):
                demo = np.load(f"{self.dir}/demo_{index}.npz", allow_pickle=True)
                return demo

        train_dataset = DedoDataset(dataset_dir + "/train_tax3d")
        val_dataset = DedoDataset(dataset_dir + "/val_tax3d")
        val_ood_dataset = DedoDataset(dataset_dir + "/val_ood_tax3d")
        
        # load the latest checkpoint

        cfg = copy.deepcopy(self.cfg)
        seed = cfg.task.inference.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        split = cfg.task.inference.split
        # dataset = DedoDataset(dataset_dir + f"/{split}_tax3d")

        lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        print(lastest_ckpt_path)
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir, viz=True)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()



        # cprint(f"EVALUATING ON SETTING: {dataset_name}", 'magenta')

        # runner_log = env_runner.run_dataset(policy, dataset, split)

        # cprint(f"---------------- Eval Results for {split} --------------", 'magenta')
        # for key, value in runner_log.items():
        #     if isinstance(value, float):
        #         cprint(f"{key}: {value:.4f}", 'magenta')
        

        runner_log = env_runner.run_dataset(policy, train_dataset, 'train')

        cprint(f"---------------- Eval Results for Train --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')

        runner_log = env_runner.run_dataset(policy, val_dataset, 'val')

        cprint(f"---------------- Eval Results for Val. --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
            
        runner_log = env_runner.run_dataset(policy, val_ood_dataset, 'val_ood')

        cprint(f"---------------- Eval Results for Val. OOD --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')


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
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
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
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
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
    


class EvalTAX3DWorkspace:
    # include_keys = ['global_step', 'epoch']
    # exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        device = cfg.training.device
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        # TODO: implement TAX3D
        #self.model: TAX3D = hydra.utils.instantiate(cfg.policy)

        # don't need to worry about ema, optimizer, global step, or epoch


        # checkpoint_run_id = cfg.exp_name.split("-")[-1]
        # project_dir = f"{cfg.logging.entity}/{cfg.logging.project}"
        # # grabbing config from wandb
        # api = wandb.Api()
        # run_config = api.run(f"{project_dir}/{checkpoint_run_id}").config
        # self.run_config = OmegaConf.create(run_config)


        # # grabbing checkpoint reference from exp_name
        # checkpoint_reference = f"{project_dir}/model-{checkpoint_run_id}:v0"
        # artifact_dir = cfg.checkpoint.artifact_dir
        # artifact = api.artifact(checkpoint_reference, type='model')
        # ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
        # breakpoint()
        # self.model: TAX3D = TAX3D(ckpt_file, device, cfg, self.run_config)


        # load default model and dataset config from TAX3D - this ensures config structure is up-to-date
        checkpoint_run_id = cfg.exp_name.split("-")[-1]
        tax3d_cfg_dir = pathlib.Path(__file__).parent.parent.parent.parent.resolve() / 'configs'
        dataset_cfg = OmegaConf.load(tax3d_cfg_dir / 'dataset' / 'proc_cloth.yaml')
        model_cfg = OmegaConf.load(tax3d_cfg_dir / 'model' / 'df_base.yaml')
        base_cfg = OmegaConf.create({'mode': 'eval', 'dataset': dataset_cfg, 'model': model_cfg})

        # TODO: eventually, may need to read task overrides here? e.g. if we want to eval 
        # on a different dataset than what we trained on
        self.run_cfg = load_checkpoint_config_from_wandb(base_cfg, [], cfg.logging.entity, cfg.logging.project, checkpoint_run_id)
        
        # also grabbing checkpoint file
        project_dir = f"{cfg.logging.entity}/{cfg.logging.project}"
        checkpoint_reference = f"{project_dir}/model-{checkpoint_run_id}:v0"
        artifact = wandb.Api().artifact(checkpoint_reference, type='model')
        ckpt_file = artifact.get_path("model.ckpt").download(root=cfg.checkpoint.artifact_dir)
        self.model: TAX3D = TAX3D(ckpt_file, device, cfg, self.run_cfg)

    def eval_datasets(self):


        # TODO: OVERRIDE THESE VALUES BASED ON THE RUN CONFIG - MODEL SHOULD ALWAYS 
        # EVALUATE ON THE DATASET TYPE THAT IT WAS TRAINED ON

        dataset_cfg = self.cfg.task.dataset
        cloth_geometry = dataset_cfg.cloth_geometry
        cloth_pose = dataset_cfg.cloth_pose
        anchor_geometry = dataset_cfg.anchor_geometry
        anchor_pose = dataset_cfg.anchor_pose
        hole = dataset_cfg.hole
        dataset_name = (
            f'cloth={cloth_geometry}-{cloth_pose} ' + \
            f'anchor={anchor_geometry}-{anchor_pose} ' + \
            f'hole={hole}'
        )
        dataset_dir = os.path.join(
            os.path.expanduser(dataset_cfg.root_dir),
            dataset_name
        )

        import torch.utils.data as data

        class DedoDataset(data.Dataset):
            def __init__(self, dir):
                self.dir = dir
                self.num_demos = int(len(os.listdir(self.dir)))

            def __len__(self):
                return self.num_demos
            
            def __getitem__(self, index):
                demo = np.load(f"{self.dir}/demo_{index}.npz", allow_pickle=True)
                return demo
            
        train_dataset = DedoDataset(dataset_dir + "/train_tax3d")
        val_dataset = DedoDataset(dataset_dir + "/val_tax3d")
        val_ood_dataset = DedoDataset(dataset_dir + "/val_ood_tax3d")

        

        cfg = copy.deepcopy(self.cfg)
        split = cfg.task.inference.split
        dataset = DedoDataset(dataset_dir + f"/{split}_tax3d")

        # configure env
        # for TAX3D, horizon is 1
        cfg.task.env_runner.n_obs_steps = 1
        cfg.task.env_runner.n_action_steps = 1

        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir,
            tax3d=True, viz=True)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        policy.eval()
        policy.cuda()

        # cprint(f"EVALUATING ON SETTING: {dataset_name}", 'magenta')

        # runner_log = env_runner.run_dataset(policy, dataset, split)

        # cprint(f"---------------- Eval Results for {split} --------------", 'magenta')
        # for key, value in runner_log.items():
        #     if isinstance(value, float):
        #         cprint(f"{key}: {value:.4f}", 'magenta')


        pc1 = env_runner.run_dataset(policy, train_dataset, 'train')

        # cprint(f"---------------- Eval Results for Train --------------", 'magenta')
        # for key, value in runner_log.items():
        #     if isinstance(value, float):
        #         cprint(f"{key}: {value:.4f}", 'magenta')

        # pc2 = env_runner.run_dataset(policy, val_dataset, 'val')

        # cprint(f"---------------- Eval Results for Val. --------------", 'magenta')
        # for key, value in runner_log.items():
        #     if isinstance(value, float):
        #         cprint(f"{key}: {value:.4f}", 'magenta')
            
        # pc3 = env_runner.run_dataset(policy, val_ood_dataset, 'val_ood')

        # cprint(f"---------------- Eval Results for Val. OOD --------------", 'magenta')
        # for key, value in runner_log.items():
        #     if isinstance(value, float):
        #         cprint(f"{key}: {value:.4f}", 'magenta')

        # final = pc1 + pc2 + pc3
        torch.save(pc1, 'tensor_list_train.pt')
        # torch.save(pc2, 'tensor_list_val.pt')
        # torch.save(pc3, 'tensor_list_val_ood.pt')

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        output_dir = os.path.expanduser(output_dir)
        return output_dir




@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
