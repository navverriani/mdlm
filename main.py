import os

import ast
import gzip
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
    if "hf" in config.backbone:
        return diffusion.Diffusion(config, tokenizer=tokenizer).to("cuda")

    return diffusion.Diffusion.load_from_checkpoint(config.eval.checkpoint_path, tokenizer=tokenizer, config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if save_cfg:
        with fsspec.open("{}/config_tree.txt".format(config.checkpointing.save_dir), "w") as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [("train", train_ds), ("valid", valid_ds)]:
        print(f"Printing {dl_type} dataloader batch.")
        batch = next(iter(dl))
        print("Batch input_ids.shape", batch["input_ids"].shape)
        first = batch["input_ids"][0, :k]
        last = batch["input_ids"][0, -k:]
        first_attention_mask = batch["attention_mask"][0, :k]
        ids = batch["input_ids"]
        mask = batch["attention_mask"]

        # BOS at position 0 for all sequences
        assert (ids[:, 0] == tokenizer.bos_token_id).all()

        # EOS must occur in the non-padding region
        eos_in_mask = ((ids == tokenizer.eos_token_id) & (mask == 1)).any(dim=1)
        assert eos_in_mask.all()

        # No EOS inside padding
        assert not ((ids == tokenizer.eos_token_id) & (mask == 0)).any()
        print("Everything is fine!")

        print(f"First {k} tokens:", tokenizer.decode(first))
        print("ids:", first)
        print(f"First {k} attention mask tokens:", first_attention_mask)
        print(f"Last {k} tokens:", tokenizer.decode(last))
        print("ids:", last)


def generate_samples(config, logger, tokenizer):
    logger.info("Generating samples.")
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    model.gen_ppl_metric.reset()
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps,
            )
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
            text_samples = model.tokenizer.batch_decode(samples)
            model.compute_generative_perplexity(text_samples)
    print("Text samples:", text_samples)
    if not config.sampling.semi_ar:
        print("Generative perplexity:", model.gen_ppl_metric.compute())
    return text_samples


def _ppl_eval(config, logger, tokenizer):
    logger.info("Starting Zero Shot Eval.")

    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(config=omegaconf.OmegaConf.to_object(config), **config.wandb)
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    _, valid_ds = dataloader.get_dataloaders(config, tokenizer, skip_train=True, valid_seed=config.seed)
    trainer.validate(model, valid_ds)


def _get_scores(config, logger, tokenizer):
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        model.ema = None

    batch_size = config.rescore.batch_size

    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    lm_scores = {}

    if config.rescore.hypothesis_file.endswith(".gz"):
        with gzip.open(config.rescore.hypothesis_file, "rt") as f:
            content = f.read()
            hypotheses_dict = ast.literal_eval(content)
    else:
        with open(config.rescore.hypothesis_file, "r") as f:
            content = f.read()
            hypotheses_dict = ast.literal_eval(content)

    for utt_id, nbest_scores in hypotheses_dict.items():
        all_log_probs = []
        all_lengths = []
        hypotheses = [hyp for _, hyp in nbest_scores]

        vocab = tokenizer.get_vocab()

        hypotheses_ids = []
        pad_id = tokenizer.pad_token_id
        bos_id = tokenizer.bos_token_id  # <s>
        eos_id = tokenizer.eos_token_id  # </s>
        for hyp in hypotheses:
            tokens = hyp.split()
            token_ids = [vocab.get(token, tokenizer.unk_token_id) for token in tokens]
            token_ids = [bos_id] + token_ids + [eos_id]
            hypotheses_ids.append(token_ids)

        if config.rescore.fixed_padding:
            max_len = config.model.length
        else:
            max_len = min(max([len(token_ids) for token_ids in hypotheses_ids]), config.model.length)

        input_ids_list = []
        attention_masks = []

        for token_ids in hypotheses_ids:
            if len(token_ids) > max_len - 2:
                token_ids = token_ids[: max_len - 2]

            # Padding
            pad_len = max_len - len(token_ids)
            padded = token_ids + [pad_id] * pad_len
            mask = [1] * len(token_ids) + [0] * pad_len

            input_ids_list.append(padded)
            attention_masks.append(mask)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long).to("cuda")
        attention_mask = torch.tensor(attention_masks, dtype=torch.long).to("cuda")

        for i in range(config.rescore.sampling_number):
            seed = 42 + i
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            hyp_log_probs = []
            hyp_lengths = []
            with torch.no_grad():
                for j in range(0, len(input_ids), batch_size):
                    batch_ids = input_ids[j : j + batch_size]
                    batch_mask = attention_mask[j : j + batch_size]

                    loss_output = model._loss(batch_ids, batch_mask)

                    if config.rescore.full_length:
                        lengths = batch_mask.sum(dim=-1)
                    else:
                        non_zero_mask = (loss_output.nlls != 0.0).float()
                        lengths = (batch_mask * non_zero_mask).sum(dim=-1).clamp(min=1)

                    if not config.rescore.normalize_per_sample:
                        batch_log_prob = -loss_output.nlls.sum(dim=-1) / lengths
                    else:
                        batch_log_prob = -loss_output.nlls.sum(dim=-1)
                        hyp_lengths.append(lengths)
                    hyp_log_probs.append(batch_log_prob)

                log_probs = torch.cat(hyp_log_probs)
                all_log_probs.append(log_probs)
                if config.rescore.normalize_per_sample:
                    all_lengths.append(torch.cat(hyp_lengths))

        all_log_probs = torch.stack(all_log_probs)
        if config.rescore.normalize_per_sample:
            all_lengths = torch.stack(all_lengths)

            if not config.rescore.zero_include:
                non_zero_mask = all_log_probs != 0.0
                sum_log_probs = (all_log_probs * non_zero_mask).sum(dim=0)
                sum_lengths = (all_lengths * non_zero_mask).sum(dim=0)
                mean_log_probs = (
                    torch.where(sum_lengths > 0, sum_log_probs / sum_lengths, torch.zeros_like(sum_log_probs))
                    .cpu()
                    .tolist()
                )
            else:
                total_log_probs = all_log_probs.sum(dim=0)
                total_lengths = all_lengths.sum(dim=0)
                mean_log_probs = (total_log_probs / total_lengths).cpu().tolist()
        else:
            if not config.rescore.zero_include:
                non_zero_mask = all_log_probs != 0.0
                sum_log_probs = (all_log_probs * non_zero_mask).sum(dim=0)
                counts = non_zero_mask.sum(dim=0).float()
                mean_log_probs = (
                    torch.where(counts > 0, sum_log_probs / counts, torch.zeros_like(sum_log_probs)).cpu().tolist()
                )
            else:
                mean_log_probs = all_log_probs.mean(dim=0).cpu().tolist()
        scored_hypotheses = list(zip(mean_log_probs, hypotheses))
        lm_scores[utt_id] = scored_hypotheses
    # https://arxiv.org/abs/1905.06655
    # elif config.rescore.rescoring_method == "mdlm_elbo":

    output_file = os.path.join(config.rescore.output_dir, "lm_scores.py.gz")

    with gzip.open(output_file, "wt") as f:
        f.write("{\n")

        for utt_id, scored_hyps in lm_scores.items():
            f.write(f"{repr(utt_id)}: [\n")

            for score, hyp in scored_hyps:
                f.write(f"    ({score}, {repr(hyp)}),\n")

            f.write("],\n")

        f.write("}\n")


def _train(config, logger, tokenizer):
    logger.info("Starting Training.")
    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(config=omegaconf.OmegaConf.to_object(config), **config.wandb)

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    _print_batch(train_ds, valid_ds, tokenizer)

    model = diffusion.Diffusion(config, tokenizer=valid_ds.tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    if config.mode == "sample_eval":
        generate_samples(config, logger, tokenizer)
    elif config.mode == "ppl_eval":
        _ppl_eval(config, logger, tokenizer)
    elif config.mode == "rescore":
        _get_scores(config, logger, tokenizer)
    else:
        _train(config, logger, tokenizer)


if __name__ == "__main__":
    main()
