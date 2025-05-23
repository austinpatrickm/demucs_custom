# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test time evaluation, either using the original SDR from [Vincent et al. 2006]
or the newest SDR definition from the MDX 2021 competition (this one will
be reported as `nsdr` for `new sdr`).
"""

from concurrent import futures
import logging

from dora.log import LogProgress
import numpy as np
import musdb
import museval
import torch as th

from .apply import apply_model
from .audio import convert_audio, save_audio
from . import distrib
from .utils import DummyPoolExecutor


logger = logging.getLogger(__name__)


def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = th.sum(th.square(references), dim=(2, 3))
    den = th.sum(th.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * th.log10(num / den)
    return scores


def eval_track(references, estimates, win, hop, compute_sdr=True):
    
    references = references.transpose(1, 2).double()
    estimates = estimates.transpose(1, 2).double()

    new_scores = new_sdr(references.cpu()[None], estimates.cpu()[None])[0]

    logger.info(f"New_sdr computed successfully. Scores: {new_scores.tolist()}") # .tolist() for cleaner log

    if not compute_sdr:
        return None, new_scores
    else:
        references = references.numpy()
        estimates = estimates.numpy()
        scores = museval.metrics.bss_eval(
            references, estimates,
            compute_permutation=False,
            window=win,
            hop=hop,
            framewise_filters=False,
            bsseval_sources_version=False)[:-1]
        return scores, new_scores


def evaluate(solver, compute_sdr=False):
    """
    Evaluate model using museval.
    compute_sdr=False means using only the MDX definition of the SDR, which
    is much faster to evaluate.
    """

    args = solver.args

    output_dir = solver.folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = solver.folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)


    # --- FORCING SERIAL EXECUTION ---
    original_workers_setting = args.test.workers # Store the original setting
    logger.info(f"[DEBUG] Original args.test.workers from config: {original_workers_setting}")
    # We will use this to decide which executor to instantiate,
    # and if ProcessPoolExecutor, what max_workers to pass.
    # For DummyPoolExecutor, we don't pass max_workers.
    use_serial_execution = True # Explicitly set to True for this debugging phase
    if use_serial_execution:
        logger.info(f"[DEBUG] Forcing serial execution for debugging (will use DummyPoolExecutor).")
        current_workers_to_use = 0 
    else:
        logger.info(f"[DEBUG] Using configured worker count: {original_workers_setting}.")
        current_workers_to_use = original_workers_setting
    # --- END FORCING SERIAL ---

    # we load tracks from the original musdb set
    if args.test.nonhq is None:
        test_set = musdb.DB(args.dset.musdb, subsets=["test"], is_wav=True)
    else:
        test_set = musdb.DB(args.test.nonhq, subsets=["test"], is_wav=False)
    src_rate = args.dset.musdb_samplerate

    eval_device = 'cpu'

    model = solver.model
    win = int(1. * model.samplerate)
    hop = int(1. * model.samplerate)

    indexes = range(distrib.rank, len(test_set), distrib.world_size)
    indexes = LogProgress(logger, indexes, updates=args.misc.num_prints,
                          name='Eval')
    
    main_loop_progress = LogProgress(logger, list(indexes), updates=args.misc.num_prints, name='Eval')

    pendings = []

    # Determine which pool executor to use and how to instantiate it
    if current_workers_to_use > 0:
        logger.info(f"[DEBUG] Instantiating ProcessPoolExecutor with max_workers={current_workers_to_use}.")
        pool_executor_instance = futures.ProcessPoolExecutor(max_workers=current_workers_to_use)
    else:
        logger.info(f"[DEBUG] Instantiating DummyPoolExecutor (no max_workers argument).")
        pool_executor_instance = DummyPoolExecutor() # No max_workers here

    with pool_executor_instance as pool:
        for loop_idx, track_list_idx in enumerate(main_loop_progress): # main_loop_progress yields actual indices for test_set.tracks
            track = test_set.tracks[track_list_idx]
            logger.info(f"--- [EVAL ITEM {loop_idx + 1}/{len(test_set.tracks)}] START --- Track: {track.name} (index {track_list_idx}) ---")


            mix = th.from_numpy(track.audio).t().float()
            if mix.dim() == 1:
                mix = mix[None]
            mix = mix.to(solver.device)
            ref = mix.mean(dim=0)  # mono mixture
            mix = (mix - ref.mean()) / ref.std()
            mix = convert_audio(mix, src_rate, model.samplerate, model.audio_channels)
            estimates = apply_model(model, mix[None],
                                    shifts=args.test.shifts, split=args.test.split,
                                    overlap=args.test.overlap)[0]
            estimates = estimates * ref.std() + ref.mean()
            estimates = estimates.to(eval_device)

            references = th.stack(
                [th.from_numpy(track.targets[name].audio).t() for name in model.sources])
            if references.dim() == 2:
                references = references[:, None]
            references = references.to(eval_device)
            references = convert_audio(references, src_rate,
                                       model.samplerate, model.audio_channels)
            if args.test.save:
                folder = solver.folder / "wav" / track.name
                folder.mkdir(exist_ok=True, parents=True)
                for name, estimate in zip(model.sources, estimates):
                    save_audio(estimate.cpu(), folder / (name + ".mp3"), model.samplerate)

            pendings.append((track.name, pool.submit(
                eval_track, references, estimates, win=win, hop=hop, compute_sdr=compute_sdr)))

        pendings = LogProgress(logger, pendings, updates=args.misc.num_prints,
                               name='Eval (BSS)')
        tracks = {}
        for track_name, pending in pendings:
            pending = pending.result()
            scores, nsdrs = pending
            tracks[track_name] = {}
            for idx, target in enumerate(model.sources):
                tracks[track_name][target] = {'nsdr': [float(nsdrs[idx])]}
            if scores is not None:
                (sdr, isr, sir, sar) = scores
                for idx, target in enumerate(model.sources):
                    values = {
                        "SDR": sdr[idx].tolist(),
                        "SIR": sir[idx].tolist(),
                        "ISR": isr[idx].tolist(),
                        "SAR": sar[idx].tolist()
                    }
                    tracks[track_name][target].update(values)

        all_tracks = {}
        for src in range(distrib.world_size):
            all_tracks.update(distrib.share(tracks, src))

        result = {}
        metric_names = next(iter(all_tracks.values()))[model.sources[0]]
        for metric_name in metric_names:
            avg = 0
            avg_of_medians = 0
            for source in model.sources:
                medians = [
                    np.nanmedian(all_tracks[track][source][metric_name])
                    for track in all_tracks.keys()]
                mean = np.mean(medians)
                median = np.median(medians)
                result[metric_name.lower() + "_" + source] = mean
                result[metric_name.lower() + "_med" + "_" + source] = median
                avg += mean / len(model.sources)
                avg_of_medians += median / len(model.sources)
            result[metric_name.lower()] = avg
            result[metric_name.lower() + "_med"] = avg_of_medians
        return result
