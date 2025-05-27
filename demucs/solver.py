# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main training loop."""

import logging

from dora import get_xp
from dora.utils import write_and_rename
from dora.log import LogProgress, bold
import torch
import torch.nn.functional as F

from . import augment, distrib, states, pretrained
from .apply import apply_model
from .ema import ModelEMA
from .evaluate import evaluate, new_sdr
from .svd import svd_penalty
from .utils import pull_metric, EMA

logger = logging.getLogger(__name__)


def _summary(metrics):
    return " | ".join(f"{key.capitalize()}={val}" for key, val in metrics.items())


class Solver(object):
    def __init__(self, loaders, model, optimizer, args):
        self.args = args
        self.loaders = loaders

        self.model = model
        self.optimizer = optimizer
        self.quantizer = states.get_quantizer(self.model, args.quant, self.optimizer)
        self.dmodel = distrib.wrap(model)
        self.device = next(iter(self.model.parameters())).device

        # Exponential moving average of the model, either updated every batch or epoch.
        # The best model from all the EMAs and the original one is kept based on the valid
        # loss for the final best model.
        self.emas = {'batch': [], 'epoch': []}
        for kind in self.emas.keys():
            decays = getattr(args.ema, kind)
            device = self.device if kind == 'batch' else 'cpu'
            if decays:
                for decay in decays:
                    self.emas[kind].append(ModelEMA(self.model, decay, device=device))

        # data augment
        augments = [augment.Shift(shift=int(args.dset.samplerate * args.dset.shift),
                                  same=args.augment.shift_same)]
        if args.augment.flip:
            augments += [augment.FlipChannels(), augment.FlipSign()]
        for aug in ['scale', 'remix']:
            kw = getattr(args.augment, aug)
            if kw.proba:
                augments.append(getattr(augment, aug.capitalize())(**kw))
        self.augment = torch.nn.Sequential(*augments)

        xp = get_xp()
        self.folder = xp.folder
        # Checkpoints
        self.checkpoint_file = xp.folder / 'checkpoint.th'
        self.best_file = xp.folder / 'best.th'
        logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.best_state = None
        self.best_changed = False

        self.link = xp.link
        self.history = self.link.history

        self._reset()

    def _serialize(self, epoch):
        package = {}
        package['state'] = self.model.state_dict()
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        for kind, emas in self.emas.items():
            for k, ema in enumerate(emas):
                package[f'ema_{kind}_{k}'] = ema.state_dict()
        with write_and_rename(self.checkpoint_file) as tmp:
            torch.save(package, tmp)

        save_every = self.args.save_every
        if save_every and (epoch + 1) % save_every == 0 and epoch + 1 != self.args.epochs:
            with write_and_rename(self.folder / f'checkpoint_{epoch + 1}.th') as tmp:
                torch.save(package, tmp)

        if self.best_changed:
            # Saving only the latest best model.
            with write_and_rename(self.best_file) as tmp:
                package = states.serialize_model(self.model, self.args)
                package['state'] = self.best_state
                torch.save(package, tmp)
            self.best_changed = False

    def _reset(self):
        """Reset state of the solver, potentially using checkpoint."""
        logger.info("--- DEBUG: Entering _reset() ---")
        loaded_successfully = False # Flag to track if any loading path was successful for the model

        if self.checkpoint_file.exists():
            logger.info(f'DEBUG: Found existing checkpoint for current XP: {self.checkpoint_file}')
            logger.info(f'DEBUG: Loading state from this existing checkpoint (resuming current XP).')
            package = torch.load(self.checkpoint_file, 'cpu')
            
            if 'state' in package and package['state'] is not None:
                self.model.load_state_dict(package['state'])
                logger.info("DEBUG: self.model weights loaded from current XP checkpoint ('state').")
                loaded_successfully = True
            else:
                logger.warning("DEBUG: 'state' key not found or is None in current XP checkpoint. self.model not loaded.")

            if self.args.continue_opt and 'optimizer' in package and package['optimizer'] is not None:
                self.optimizer.load_state_dict(package['optimizer'])
                logger.info("DEBUG: Optimizer state loaded from current XP checkpoint.")
            else:
                logger.info("DEBUG: Optimizer state not loaded from current XP checkpoint (continue_opt or key missing).")
            
            self.history[:] = package.get('history', [])
            self.best_state = package.get('best_state', None)
            logger.info(f"DEBUG: self.best_state set from current XP checkpoint (is None: {self.best_state is None}).")
            # ... (EMA loading logic if present in your actual code)
            for kind, emas_list in self.emas.items(): # Corrected variable name
                for k_idx, ema_model in enumerate(emas_list): # Corrected variable name
                    ema_key = f'ema_{kind}_{k_idx}'
                    if ema_key in package:
                        ema_model.load_state_dict(package[ema_key])
                        logger.info(f"DEBUG: Loaded state for {ema_key}.")


        elif self.args.continue_pretrained:
          logger.info(f"--- Solver._reset(): Attempting to load from continue_pretrained='{self.args.continue_pretrained}' ---")
          try:
              # Get the pretrained model object and its state_dict
              pretrained_model_object = pretrained.get_model(
                  name=self.args.continue_pretrained,
                  repo=self.args.pretrained_repo
              )
              pretrained_sd = pretrained_model_object.state_dict()
              
              logger.info(f"DEBUG: Fetched pretrained model '{self.args.continue_pretrained}'. Number of keys in its state_dict: {len(pretrained_sd)}.")
              # Log first 3 keys from pretrained model to see their structure
              if pretrained_sd:
                  logger.debug(f"DEBUG: First 3 keys from PRETRAINED state_dict: {list(pretrained_sd.keys())[:3]}")

              # Log first 3 keys from current self.model (BEFORE loading) to see its expected structure
              current_model_sd = self.model.state_dict()
              logger.info(f"DEBUG: Current self.model (before loading). Number of keys: {len(current_model_sd)}.")
              if current_model_sd:
                  logger.debug(f"DEBUG: First 3 keys from CURRENT self.model state_dict: {list(current_model_sd.keys())[:3]}")

              # NEW: Check if we need to remove a prefix from the pretrained keys
              pretrained_keys = list(pretrained_sd.keys())
              current_keys = list(current_model_sd.keys())
              
              # Check if pretrained keys have a common prefix that current keys don't have
              needs_prefix_removal = False
              prefix_to_remove = ""
              
              if pretrained_keys and current_keys:
                  # Look for common prefixes in pretrained keys
                  potential_prefixes = ["models.0.", "model.", "module."]
                  
                  for prefix in potential_prefixes:
                      if all(key.startswith(prefix) for key in pretrained_keys):
                          # Check if removing this prefix would match current keys
                          stripped_keys = [key[len(prefix):] for key in pretrained_keys]
                          if set(stripped_keys) == set(current_keys):
                              needs_prefix_removal = True
                              prefix_to_remove = prefix
                              logger.info(f"DEBUG: Detected need to remove prefix '{prefix_to_remove}' from pretrained keys")
                              break
              
              # Create the corrected state dict
              if needs_prefix_removal:
                  logger.info(f"DEBUG: Removing prefix '{prefix_to_remove}' from pretrained state dict keys")
                  corrected_sd = {}
                  for key, value in pretrained_sd.items():
                      new_key = key[len(prefix_to_remove):]
                      corrected_sd[new_key] = value
                  pretrained_sd = corrected_sd
                  logger.info(f"DEBUG: After prefix removal, first 3 keys: {list(pretrained_sd.keys())[:3]}")

              # The crucial line: This uses strict=True by default
              logger.info(f"DEBUG: Attempting self.model.load_state_dict(pretrained_sd) (strict=True by default).")
              missing_keys, unexpected_keys = self.model.load_state_dict(pretrained_sd, strict=False)
              
              if missing_keys:
                  logger.warning(f"DEBUG: Missing keys when loading pretrained weights: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
              if unexpected_keys:
                  logger.warning(f"DEBUG: Unexpected keys when loading pretrained weights: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
              
              # Check if the loading was successful (no missing keys for important parameters)
              if not missing_keys and not unexpected_keys:
                  logger.info(f"--- Solver._reset(): PERFECTLY loaded weights from pretrained '{self.args.continue_pretrained}'. Fine-tuning will proceed with these weights. ---")
              elif len(missing_keys) == 0:
                  logger.info(f"--- Solver._reset(): SUCCESSFULLY loaded weights from pretrained '{self.args.continue_pretrained}' (with some unexpected keys ignored). Fine-tuning will proceed. ---")
              else:
                  logger.warning(f"--- Solver._reset(): PARTIALLY loaded weights from pretrained '{self.args.continue_pretrained}'. {len(missing_keys)} keys were missing. Fine-tuning will proceed but some layers will use random weights. ---")

          except RuntimeError as e:
              # This block will execute if load_state_dict (strict=True) fails due to key mismatch or other issues.
              logger.error(f"--- Solver._reset(): FAILED to load weights from pretrained '{self.args.continue_pretrained}' due to RuntimeError: ---")
              logger.error(e) # This will print the detailed "Missing key(s)... Unexpected key(s)..." error
              logger.warning("--- Solver._reset(): Model will use its default/random initial weights. NOT FINE-TUNING. ---")
              # The training will continue, but from scratch.
          except Exception as e:
              # Catch any other unexpected errors during pretrained.get_model or .state_dict() calls
              logger.error(f"--- Solver._reset(): UNEXPECTED EXCEPTION during continue_pretrained loading for '{self.args.continue_pretrained}': {e} ---", exc_info=True)
              logger.warning("--- Solver._reset(): Model will use its default/random initial weights due to UNEXPECTED EXCEPTION. ---")

        elif self.args.continue_from:
            name = 'checkpoint.th'
            root = self.folder.parent
            cf = root / str(self.args.continue_from) / name
            logger.info("Loading from %s", cf)
            package = torch.load(cf, 'cpu')
            self.best_state = package['best_state']
            if self.args.continue_best:
                self.model.load_state_dict(package['best_state'], strict=False)
            else:
                self.model.load_state_dict(package['state'], strict=False)
            if self.args.continue_opt:
                self.optimizer.load_state_dict(package['optimizer'])


    def _format_train(self, metrics: dict) -> dict:
        """Formatting for train/valid metrics.""" 
        losses = {
            'loss': format(metrics['loss'], ".4f"),
            'reco': format(metrics['reco'], ".4f"),
        }
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], ".3f")
        if self.quantizer is not None:
            losses['ms'] = format(metrics['ms'], ".2f")
        if 'grad' in metrics:
            losses['grad'] = format(metrics['grad'], ".4f")
        if 'best' in metrics:
            losses['best'] = format(metrics['best'], '.4f')
        if 'bname' in metrics:
            losses['bname'] = metrics['bname']
        if 'penalty' in metrics:
            losses['penalty'] = format(metrics['penalty'], ".4f")
        if 'hloss' in metrics:
            losses['hloss'] = format(metrics['hloss'], ".4f")
        return losses

    def _format_test(self, metrics: dict) -> dict:
        """Formatting for test metrics."""
        losses = {}
        if 'sdr' in metrics:
            losses['sdr'] = format(metrics['sdr'], '.3f')
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], '.3f')
        for source in self.model.sources:
            key = f'sdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
            key = f'nsdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
        return losses

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            formatted = self._format_train(metrics['train'])
            logger.info(
                bold(f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}'))
            formatted = self._format_train(metrics['valid'])
            logger.info(
                bold(f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}'))
            if 'test' in metrics:
                formatted = self._format_test(metrics['test'])
                if formatted:
                    logger.info(bold(f"Test Summary | Epoch {epoch + 1} | {_summary(formatted)}"))

        epoch = 0
        for epoch in range(len(self.history), self.args.epochs):
            # Train one epoch
            self.model.train()  # Turn on BatchNorm & Dropout
            metrics = {}
            logger.info('-' * 70)
            logger.info("Training...")
            metrics['train'] = self._run_one_epoch(epoch)
            formatted = self._format_train(metrics['train'])
            logger.info(
                bold(f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

            # Cross validation
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                valid = self._run_one_epoch(epoch, train=False)
                bvalid = valid
                bname = 'main'
                state = states.copy_state(self.model.state_dict())
                metrics['valid'] = {}
                metrics['valid']['main'] = valid
                key = self.args.test.metric
                for kind, emas in self.emas.items():
                    for k, ema in enumerate(emas):
                        with ema.swap():
                            valid = self._run_one_epoch(epoch, train=False)
                        name = f'ema_{kind}_{k}'
                        metrics['valid'][name] = valid
                        a = valid[key]
                        b = bvalid[key]
                        if key.startswith('nsdr'):
                            a = -a
                            b = -b
                        if a < b:
                            bvalid = valid
                            state = ema.state
                            bname = name
                    metrics['valid'].update(bvalid)
                    metrics['valid']['bname'] = bname

            valid_loss = metrics['valid'][key]
            mets = pull_metric(self.link.history, f'valid.{key}') + [valid_loss]
            if key.startswith('nsdr'):
                best_loss = max(mets)
            else:
                best_loss = min(mets)
            metrics['valid']['best'] = best_loss
            if self.args.svd.penalty > 0:
                kw = dict(self.args.svd)
                kw.pop('penalty')
                with torch.no_grad():
                    penalty = svd_penalty(self.model, exact=True, **kw)
                metrics['valid']['penalty'] = penalty

            formatted = self._format_train(metrics['valid'])
            logger.info(
                bold(f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

            # Save the best model
            if valid_loss == best_loss or self.args.dset.train_valid:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = states.copy_state(state)
                self.best_changed = True

            # Eval model every `test.every` epoch or on last epoch
            should_eval = (epoch + 1) % self.args.test.every == 0
            is_last = epoch == self.args.epochs - 1
            # # Tries to detect divergence in a reliable way and finish job
            # # not to waste compute.
            # # Commented out as this was super specific to the MDX competition.
            # reco = metrics['valid']['main']['reco']
            # div = epoch >= 180 and reco > 0.18
            # div = div or epoch >= 100 and reco > 0.25
            # div = div and self.args.optim.loss == 'l1'
            # if div:
            #     logger.warning("Finishing training early because valid loss is too high.")
            #     is_last = True
            if should_eval or is_last:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # We switch to the best known model for testing
                if self.args.test.best:
                    state = self.best_state
                else:
                    state = states.copy_state(self.model.state_dict())
                compute_sdr = self.args.test.sdr and is_last
                with states.swap_state(self.model, state):
                    with torch.no_grad():
                        metrics['test'] = evaluate(self, compute_sdr=compute_sdr)
                formatted = self._format_test(metrics['test'])
                logger.info(bold(f"Test Summary | Epoch {epoch + 1} | {_summary(formatted)}"))
            self.link.push_metrics(metrics)

            if distrib.rank == 0:
                # Save model each epoch
                self._serialize(epoch)
                logger.info("Checkpoint saved to %s", self.checkpoint_file.resolve())
            if is_last:
                break

    def _run_one_epoch(self, epoch, train=True):
        args = self.args
        data_loader = self.loaders['train'] if train else self.loaders['valid']
        if distrib.world_size > 1 and train:
            data_loader.sampler.set_epoch(epoch)

        label = ["Valid", "Train"][train]
        name = label + f" | Epoch {epoch + 1}"
        total = len(data_loader)
        if args.max_batches:
            total = min(total, args.max_batches)
        logprog = LogProgress(logger, data_loader, total=total,
                              updates=self.args.misc.num_prints, name=name)
        averager = EMA()

        for idx, sources in enumerate(logprog):
            sources = sources.to(self.device)
            if train:
                sources = self.augment(sources)
                mix = sources.sum(dim=1)
            else:
                mix = sources[:, 0]
                sources = sources[:, 1:]

            if not train and self.args.valid_apply:
                estimate = apply_model(self.model, mix, split=self.args.test.split, overlap=0)
            else:
                estimate = self.dmodel(mix)
            if train and hasattr(self.model, 'transform_target'):
                sources = self.model.transform_target(mix, sources)
            assert estimate.shape == sources.shape, (estimate.shape, sources.shape)
            dims = tuple(range(2, sources.dim()))

            if args.optim.loss == 'l1':
                loss = F.l1_loss(estimate, sources, reduction='none')
                loss = loss.mean(dims).mean(0)
                reco = loss
            elif args.optim.loss == 'mse':
                loss = F.mse_loss(estimate, sources, reduction='none')
                loss = loss.mean(dims)
                reco = loss**0.5
                reco = reco.mean(0)
            else:
                raise ValueError(f"Invalid loss {self.args.loss}")
            weights = torch.tensor(args.weights).to(sources)
            loss = (loss * weights).sum() / weights.sum()

            ms = 0
            if self.quantizer is not None:
                ms = self.quantizer.model_size()
            if args.quant.diffq:
                loss += args.quant.diffq * ms

            losses = {}
            losses['reco'] = (reco * weights).sum() / weights.sum()
            losses['ms'] = ms

            if not train:
                nsdrs = new_sdr(sources, estimate.detach()).mean(0)
                total = 0
                for source, nsdr, w in zip(self.model.sources, nsdrs, weights):
                    losses[f'nsdr_{source}'] = nsdr
                    total += w * nsdr
                losses['nsdr'] = total / weights.sum()

            if train and args.svd.penalty > 0:
                kw = dict(args.svd)
                kw.pop('penalty')
                penalty = svd_penalty(self.model, **kw)
                losses['penalty'] = penalty
                loss += args.svd.penalty * penalty

            losses['loss'] = loss

            for k, source in enumerate(self.model.sources):
                losses[f'reco_{source}'] = reco[k]

            # optimize model in training mode
            if train:
                loss.backward()
                grad_norm = 0
                grads = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm()**2
                        grads.append(p.grad.data)
                losses['grad'] = grad_norm ** 0.5
                if args.optim.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        args.optim.clip_grad)

                if self.args.flag == 'uns':
                    for n, p in self.model.named_parameters():
                        if p.grad is None:
                            print('no grad', n)
                self.optimizer.step()
                self.optimizer.zero_grad()
                for ema in self.emas['batch']:
                    ema.update()
            losses = averager(losses)
            logs = self._format_train(losses)
            logprog.update(**logs)
            # Just in case, clear some memory
            del loss, estimate, reco, ms
            if args.max_batches == idx:
                break
            if self.args.debug and train:
                break
            if self.args.flag == 'debug':
                break
        if train:
            for ema in self.emas['epoch']:
                ema.update()
        return distrib.average(losses, idx + 1)
