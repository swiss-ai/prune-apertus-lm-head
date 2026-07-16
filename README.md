# Prune Apertus output embeddings

This converter removes the image and audio code rows from `lm_head.weight` only.
It leaves `model.embed_tokens.weight` and `tokenizer.json` unchanged, because
multimodal prompts still use their original token IDs as input IDs.

For the supplied Apertus 1.5 checkpoint, the output head changes from
`266752 x 4096` to `131272 x 4096`. The retained output IDs are `0-131271`.
The removed suffix contains the image codes (`131272-262343`), audio codes
(`262344-266439`), and the unused padded rows after the audio codes.

The converted `config.json` records this separately as
`output_vocab_size: 131272`; its `vocab_size` remains `266752` for input
embedding lookup. This is a checkpoint conversion marker: the inference
runtime that loads the converted checkpoint must construct its output head
from `output_vocab_size` while continuing to use `vocab_size` for input
embedding lookup.

## Usage

Use the requested virtual environment:

```sh
/opt/venv/bin/python prune_apertus_output_embeddings/convert.py \
  --source /iopsstor/scratch/cscs/anunay/swissai/Apertus-1p5-8B-SFT-RL-DPO-SDPO-Low-mm-merged \
  --output /iopsstor/scratch/cscs/anunay/swissai/Apertus-1p5-8B-SFT-RL-DPO-SDPO-Low-mm-output-pruned
```

The output path must not already exist. By default unchanged files are hard
linked where the filesystem permits, so the conversion only consumes space for
the rewritten `lm_head` shard. Add `--copy-unchanged` to make independent
copies of all files, or `--dry-run` to validate the source without writing.
Positional source/output paths are also supported.
