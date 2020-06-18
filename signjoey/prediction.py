#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

import logging
import numpy as np
import pickle as pickle
import time
import torch.nn as nn

from typing import List
from torchtext.data import Dataset
from signjoey.loss import XentLoss
from signjoey.helpers import (
    bpe_postprocess,
    load_config,
    get_latest_checkpoint,
    load_checkpoint,
)
from signjoey.metrics import bleu, chrf, rouge, wer_list
from signjoey.model import build_model, SignModel
from signjoey.batch import Batch
from signjoey.data import load_data, make_data_iter
from signjoey.vocabulary import PAD_TOKEN, SIL_TOKEN
from signjoey.phoenix_utils.phoenix_cleanup import (
    clean_phoenix_2014,
    clean_phoenix_2014_trans,
)


# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(
    model: SignModel,
    data: Dataset,
    batch_size: int,
    use_cuda: bool,
    sgn_dim: int,
    do_recognition: bool,
    recognition_loss_function: torch.nn.Module,
    recognition_loss_weight: int,
    do_translation: bool,
    translation_loss_function: torch.nn.Module,
    translation_loss_weight: int,
    translation_max_output_length: int,
    level: str,
    txt_pad_index: int,
    recognition_beam_size: int = 1,
    translation_beam_size: int = 1,
    translation_beam_alpha: int = -1,
    batch_type: str = "sentence",
    dataset_version: str = "phoenix_2014_trans",
    frame_subsampling_ratio: int = None,
) -> (
    float,
    float,
    float,
    List[str],
    List[List[str]],
    List[str],
    List[str],
    List[List[str]],
    List[np.array],
):
    """
    Generate translations for the given data.
    If `loss_function` is not None and references are given,
    also compute the loss.

    :param model: model module
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param use_cuda: if True, use CUDA
    :param translation_max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param translation_loss_function: translation loss function (XEntropy)
    :param recognition_loss_function: recognition loss function (CTC)
    :param recognition_loss_weight: CTC loss weight
    :param translation_loss_weight: Translation loss weight
    :param txt_pad_index: txt padding token index
    :param sgn_dim: Feature dimension of sgn frames
    :param recognition_beam_size: beam size for validation (recognition, i.e. CTC).
        If 0 then greedy decoding (default).
    :param translation_beam_size: beam size for validation (translation).
        If 0 then greedy decoding (default).
    :param translation_beam_alpha: beam search alpha for length penalty (translation),
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)
    :param do_recognition: flag for predicting glosses
    :param do_translation: flag for predicting text
    :param dataset_version: phoenix_2014 or phoenix_2014_trans
    :param frame_subsampling_ratio: frame subsampling ratio

    :return:
        - current_valid_score: current validation score [eval_metric],
        - valid_loss: validation loss,
        - valid_ppl:, validation perplexity,
        - valid_sources: validation sources,
        - valid_sources_raw: raw validation sources (before post-processing),
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    valid_iter = make_data_iter(
        dataset=data,
        batch_size=batch_size,
        batch_type=batch_type,
        shuffle=False,
        train=False,
    )

    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        all_gls_outputs = []
        all_txt_outputs = []
        all_attention_scores = []
        total_recognition_loss = 0
        total_translation_loss = 0
        total_num_txt_tokens = 0
        total_num_gls_tokens = 0
        total_num_seqs = 0
        for valid_batch in iter(valid_iter):
            batch = Batch(
                is_train=False,
                torch_batch=valid_batch,
                txt_pad_index=txt_pad_index,
                sgn_dim=sgn_dim,
                use_cuda=use_cuda,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )
            sort_reverse_index = batch.sort_by_sgn_lengths()

            batch_recognition_loss, batch_translation_loss = model.get_loss_for_batch(
                batch=batch,
                recognition_loss_function=recognition_loss_function
                if do_recognition
                else None,
                translation_loss_function=translation_loss_function
                if do_translation
                else None,
                recognition_loss_weight=recognition_loss_weight
                if do_recognition
                else None,
                translation_loss_weight=translation_loss_weight
                if do_translation
                else None,
            )
            if do_recognition:
                total_recognition_loss += batch_recognition_loss
                total_num_gls_tokens += batch.num_gls_tokens
            if do_translation:
                total_translation_loss += batch_translation_loss
                total_num_txt_tokens += batch.num_txt_tokens
            total_num_seqs += batch.num_seqs

            (
                batch_gls_predictions,
                batch_txt_predictions,
                batch_attention_scores,
            ) = model.run_batch(
                batch=batch,
                recognition_beam_size=recognition_beam_size if do_recognition else None,
                translation_beam_size=translation_beam_size if do_translation else None,
                translation_beam_alpha=translation_beam_alpha
                if do_translation
                else None,
                translation_max_output_length=translation_max_output_length
                if do_translation
                else None,
            )

            # sort outputs back to original order
            if do_recognition:
                all_gls_outputs.extend(
                    [batch_gls_predictions[sri] for sri in sort_reverse_index]
                )
            if do_translation:
                all_txt_outputs.extend(batch_txt_predictions[sort_reverse_index])
            all_attention_scores.extend(
                batch_attention_scores[sort_reverse_index]
                if batch_attention_scores is not None
                else []
            )

        if do_recognition:
            assert len(all_gls_outputs) == len(data)
            if (
                recognition_loss_function is not None
                and recognition_loss_weight != 0
                and total_num_gls_tokens > 0
            ):
                valid_recognition_loss = total_recognition_loss
            else:
                valid_recognition_loss = -1
            # decode back to symbols
            decoded_gls = model.gls_vocab.arrays_to_sentences(arrays=all_gls_outputs)

            # Gloss clean-up function
            if dataset_version == "phoenix_2014_trans":
                gls_cln_fn = clean_phoenix_2014_trans
            elif dataset_version == "phoenix_2014":
                gls_cln_fn = clean_phoenix_2014
            else:
                raise ValueError("Unknown Dataset Version: " + dataset_version)

            # Construct gloss sequences for metrics
            gls_ref = [gls_cln_fn(" ".join(t)) for t in data.gls]
            gls_hyp = [gls_cln_fn(" ".join(t)) for t in decoded_gls]
            assert len(gls_ref) == len(gls_hyp)

            # GLS Metrics
            gls_wer_score = wer_list(hypotheses=gls_hyp, references=gls_ref)

        if do_translation:
            assert len(all_txt_outputs) == len(data)
            if (
                translation_loss_function is not None
                and translation_loss_weight != 0
                and total_num_txt_tokens > 0
            ):
                # total validation translation loss
                valid_translation_loss = total_translation_loss
                # exponent of token-level negative log prob
                valid_ppl = torch.exp(total_translation_loss / total_num_txt_tokens)
            else:
                valid_translation_loss = -1
                valid_ppl = -1
            # decode back to symbols
            decoded_txt = model.txt_vocab.arrays_to_sentences(arrays=all_txt_outputs)
            # evaluate with metric on full dataset
            join_char = " " if level in ["word", "bpe"] else ""
            # Construct text sequences for metrics
            txt_ref = [join_char.join(t) for t in data.txt]
            txt_hyp = [join_char.join(t) for t in decoded_txt]
            # post-process
            if level == "bpe":
                txt_ref = [bpe_postprocess(v) for v in txt_ref]
                txt_hyp = [bpe_postprocess(v) for v in txt_hyp]
            assert len(txt_ref) == len(txt_hyp)

            # TXT Metrics
            txt_bleu = bleu(references=txt_ref, hypotheses=txt_hyp)
            txt_chrf = chrf(references=txt_ref, hypotheses=txt_hyp)
            txt_rouge = rouge(references=txt_ref, hypotheses=txt_hyp)

        valid_scores = {}
        if do_recognition:
            valid_scores["wer"] = gls_wer_score["wer"]
            valid_scores["wer_scores"] = gls_wer_score
        if do_translation:
            valid_scores["bleu"] = txt_bleu["bleu4"]
            valid_scores["bleu_scores"] = txt_bleu
            valid_scores["chrf"] = txt_chrf
            valid_scores["rouge"] = txt_rouge

    results = {
        "valid_scores": valid_scores,
        "all_attention_scores": all_attention_scores,
    }
    if do_recognition:
        results["valid_recognition_loss"] = valid_recognition_loss
        results["decoded_gls"] = decoded_gls
        results["gls_ref"] = gls_ref
        results["gls_hyp"] = gls_hyp

    if do_translation:
        results["valid_translation_loss"] = valid_translation_loss
        results["valid_ppl"] = valid_ppl
        results["decoded_txt"] = decoded_txt
        results["txt_ref"] = txt_ref
        results["txt_hyp"] = txt_hyp

    return results


# pylint: disable-msg=logging-too-many-args
def test(
    cfg_file, ckpt: str, output_path: str = None, logger: logging.Logger = None
) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )

    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"].get("batch_type", "sentence")
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    dataset_version = cfg["data"].get("version", "phoenix_2014_trans")
    translation_max_output_length = cfg["training"].get(
        "translation_max_output_length", None
    )

    # load the data
    _, dev_data, test_data, gls_vocab, txt_vocab = load_data(data_cfg=cfg["data"])

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # Data Augmentation Parameters
    frame_subsampling_ratio = cfg["data"].get("frame_subsampling_ratio", None)
    # Note (Cihan): we are not using 'random_frame_subsampling' and
    #   'random_frame_masking_ratio' in testing as they are just for training.

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        recognition_beam_sizes = cfg["testing"].get("recognition_beam_sizes", [1])
        translation_beam_sizes = cfg["testing"].get("translation_beam_sizes", [1])
        translation_beam_alphas = cfg["testing"].get("translation_beam_alphas", [-1])
    else:
        recognition_beam_sizes = [1]
        translation_beam_sizes = [1]
        translation_beam_alphas = [-1]

    if "testing" in cfg.keys():
        max_recognition_beam_size = cfg["testing"].get(
            "max_recognition_beam_size", None
        )
        if max_recognition_beam_size is not None:
            recognition_beam_sizes = list(range(1, max_recognition_beam_size + 1))

    if do_recognition:
        recognition_loss_function = torch.nn.CTCLoss(
            blank=model.gls_vocab.stoi[SIL_TOKEN], zero_infinity=True
        )
        if use_cuda:
            recognition_loss_function.cuda()
    if do_translation:
        translation_loss_function = XentLoss(
            pad_index=txt_vocab.stoi[PAD_TOKEN], smoothing=0.0
        )
        if use_cuda:
            translation_loss_function.cuda()

    # NOTE (Cihan): Currently Hardcoded to be 0 for TensorFlow decoding
    assert model.gls_vocab.stoi[SIL_TOKEN] == 0

    if do_recognition:
        # Dev Recognition CTC Beam Search Results
        dev_recognition_results = {}
        dev_best_wer_score = float("inf")
        dev_best_recognition_beam_size = 1
        for rbw in recognition_beam_sizes:
            logger.info("-" * 60)
            valid_start_time = time.time()
            logger.info("[DEV] partition [RECOGNITION] experiment [BW]: %d", rbw)
            dev_recognition_results[rbw] = validate_on_data(
                model=model,
                data=dev_data,
                batch_size=batch_size,
                use_cuda=use_cuda,
                batch_type=batch_type,
                dataset_version=dataset_version,
                sgn_dim=sum(cfg["data"]["feature_size"])
                if isinstance(cfg["data"]["feature_size"], list)
                else cfg["data"]["feature_size"],
                txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                # Recognition Parameters
                do_recognition=do_recognition,
                recognition_loss_function=recognition_loss_function,
                recognition_loss_weight=1,
                recognition_beam_size=rbw,
                # Translation Parameters
                do_translation=do_translation,
                translation_loss_function=translation_loss_function
                if do_translation
                else None,
                translation_loss_weight=1 if do_translation else None,
                translation_max_output_length=translation_max_output_length
                if do_translation
                else None,
                level=level if do_translation else None,
                translation_beam_size=1 if do_translation else None,
                translation_beam_alpha=-1 if do_translation else None,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )
            logger.info("finished in %.4fs ", time.time() - valid_start_time)
            if dev_recognition_results[rbw]["valid_scores"]["wer"] < dev_best_wer_score:
                dev_best_wer_score = dev_recognition_results[rbw]["valid_scores"]["wer"]
                dev_best_recognition_beam_size = rbw
                dev_best_recognition_result = dev_recognition_results[rbw]
                logger.info("*" * 60)
                logger.info(
                    "[DEV] partition [RECOGNITION] results:\n\t"
                    "New Best CTC Decode Beam Size: %d\n\t"
                    "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)",
                    dev_best_recognition_beam_size,
                    dev_best_recognition_result["valid_scores"]["wer"],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "del_rate"
                    ],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "ins_rate"
                    ],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "sub_rate"
                    ],
                )
                logger.info("*" * 60)

    if do_translation:
        logger.info("=" * 60)
        dev_translation_results = {}
        dev_best_bleu_score = float("-inf")
        dev_best_translation_beam_size = 1
        dev_best_translation_alpha = 1
        for tbw in translation_beam_sizes:
            dev_translation_results[tbw] = {}
            for ta in translation_beam_alphas:
                dev_translation_results[tbw][ta] = validate_on_data(
                    model=model,
                    data=dev_data,
                    batch_size=batch_size,
                    use_cuda=use_cuda,
                    level=level,
                    sgn_dim=sum(cfg["data"]["feature_size"])
                    if isinstance(cfg["data"]["feature_size"], list)
                    else cfg["data"]["feature_size"],
                    batch_type=batch_type,
                    dataset_version=dataset_version,
                    do_recognition=do_recognition,
                    recognition_loss_function=recognition_loss_function
                    if do_recognition
                    else None,
                    recognition_loss_weight=1 if do_recognition else None,
                    recognition_beam_size=1 if do_recognition else None,
                    do_translation=do_translation,
                    translation_loss_function=translation_loss_function,
                    translation_loss_weight=1,
                    translation_max_output_length=translation_max_output_length,
                    txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                    translation_beam_size=tbw,
                    translation_beam_alpha=ta,
                    frame_subsampling_ratio=frame_subsampling_ratio,
                )

                if (
                    dev_translation_results[tbw][ta]["valid_scores"]["bleu"]
                    > dev_best_bleu_score
                ):
                    dev_best_bleu_score = dev_translation_results[tbw][ta][
                        "valid_scores"
                    ]["bleu"]
                    dev_best_translation_beam_size = tbw
                    dev_best_translation_alpha = ta
                    dev_best_translation_result = dev_translation_results[tbw][ta]
                    logger.info(
                        "[DEV] partition [Translation] results:\n\t"
                        "New Best Translation Beam Size: %d and Alpha: %d\n\t"
                        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                        "CHRF %.2f\t"
                        "ROUGE %.2f",
                        dev_best_translation_beam_size,
                        dev_best_translation_alpha,
                        dev_best_translation_result["valid_scores"]["bleu"],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu1"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu2"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu3"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu4"
                        ],
                        dev_best_translation_result["valid_scores"]["chrf"],
                        dev_best_translation_result["valid_scores"]["rouge"],
                    )
                    logger.info("-" * 60)

    logger.info("*" * 60)
    logger.info(
        "[DEV] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: %d\n\t"
        "Best Translation Beam Size: %d and Alpha: %d\n\t"
        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_recognition_beam_size if do_recognition else -1,
        dev_best_translation_beam_size if do_translation else -1,
        dev_best_translation_alpha if do_translation else -1,
        dev_best_recognition_result["valid_scores"]["wer"] if do_recognition else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["del_rate"]
        if do_recognition
        else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["ins_rate"]
        if do_recognition
        else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["sub_rate"]
        if do_recognition
        else -1,
        dev_best_translation_result["valid_scores"]["bleu"] if do_translation else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu1"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu2"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu3"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu4"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["chrf"] if do_translation else -1,
        dev_best_translation_result["valid_scores"]["rouge"] if do_translation else -1,
    )
    logger.info("*" * 60)

    test_best_result = validate_on_data(
        model=model,
        data=test_data,
        batch_size=batch_size,
        use_cuda=use_cuda,
        batch_type=batch_type,
        dataset_version=dataset_version,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
        do_recognition=do_recognition,
        recognition_loss_function=recognition_loss_function if do_recognition else None,
        recognition_loss_weight=1 if do_recognition else None,
        recognition_beam_size=dev_best_recognition_beam_size
        if do_recognition
        else None,
        do_translation=do_translation,
        translation_loss_function=translation_loss_function if do_translation else None,
        translation_loss_weight=1 if do_translation else None,
        translation_max_output_length=translation_max_output_length
        if do_translation
        else None,
        level=level if do_translation else None,
        translation_beam_size=dev_best_translation_beam_size
        if do_translation
        else None,
        translation_beam_alpha=dev_best_translation_alpha if do_translation else None,
        frame_subsampling_ratio=frame_subsampling_ratio,
    )

    logger.info(
        "[TEST] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: %d\n\t"
        "Best Translation Beam Size: %d and Alpha: %d\n\t"
        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_recognition_beam_size if do_recognition else -1,
        dev_best_translation_beam_size if do_translation else -1,
        dev_best_translation_alpha if do_translation else -1,
        test_best_result["valid_scores"]["wer"] if do_recognition else -1,
        test_best_result["valid_scores"]["wer_scores"]["del_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["wer_scores"]["ins_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["wer_scores"]["sub_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["bleu"] if do_translation else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu1"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu2"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu3"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu4"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["chrf"] if do_translation else -1,
        test_best_result["valid_scores"]["rouge"] if do_translation else -1,
    )
    logger.info("*" * 60)

    def _write_to_file(file_path: str, sequence_ids: List[str], hypotheses: List[str]):
        with open(file_path, mode="w", encoding="utf-8") as out_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                out_file.write(seq + "|" + hyp + "\n")

    if output_path is not None:
        if do_recognition:
            dev_gls_output_path_set = "{}.BW_{:03d}.{}.gls".format(
                output_path, dev_best_recognition_beam_size, "dev"
            )
            _write_to_file(
                dev_gls_output_path_set,
                [s for s in dev_data.sequence],
                dev_best_recognition_result["gls_hyp"],
            )
            test_gls_output_path_set = "{}.BW_{:03d}.{}.gls".format(
                output_path, dev_best_recognition_beam_size, "test"
            )
            _write_to_file(
                test_gls_output_path_set,
                [s for s in test_data.sequence],
                test_best_result["gls_hyp"],
            )

        if do_translation:
            if dev_best_translation_beam_size > -1:
                dev_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "dev",
                )
                test_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "test",
                )
            else:
                dev_txt_output_path_set = "{}.BW_{:02d}.{}.txt".format(
                    output_path, dev_best_translation_beam_size, "dev"
                )
                test_txt_output_path_set = "{}.BW_{:02d}.{}.txt".format(
                    output_path, dev_best_translation_beam_size, "test"
                )

            _write_to_file(
                dev_txt_output_path_set,
                [s for s in dev_data.sequence],
                dev_best_translation_result["txt_hyp"],
            )
            _write_to_file(
                test_txt_output_path_set,
                [s for s in test_data.sequence],
                test_best_result["txt_hyp"],
            )

        with open(output_path + ".dev_results.pkl", "wb") as out:
            pickle.dump(
                {
                    "recognition_results": dev_recognition_results
                    if do_recognition
                    else None,
                    "translation_results": dev_translation_results
                    if do_translation
                    else None,
                },
                out,
            )
        with open(output_path + ".test_results.pkl", "wb") as out:
            pickle.dump(test_best_result, out)
