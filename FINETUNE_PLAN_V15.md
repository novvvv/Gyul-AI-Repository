# Fish Speech 1.5 Emotion LoRA Fine-tuning Plan

## Why We Pivoted

The first Colab attempt used `openaudio-s1-mini`, but the available code snapshots and checkpoint/tokenizer format did not stay aligned across the fine-tuning path. The run reached preprocessing, VQ extraction, and protobuf packing, but LoRA training failed because the selected dataset/tokenizer stack expected incompatible semantic token conventions.

Observed blockers:

- `fish-speech` GitHub `main` expected tokenizer loading through `AutoTokenizer`, while `openaudio-s1-mini` ships `tokenizer.tiktoken`.
- The OpenAudio S1-mini Space snapshot did not have a Python package layout, so editable installation failed.
- New Colab packages exposed compatibility issues:
  - `torchaudio.list_audio_backends()` removed.
  - `protobuf` was downgraded by transitive dependencies.
- Missing Space dependencies had to be discovered one by one:
  - `descript-audiotools`
  - `descript-audio-codec`
- VQ extraction required code compatibility patches:
  - `model.spec_transform.sample_rate` vs `model.sample_rate`
- Training config and tokenizer conventions diverged:
  - config referenced a dataset class not present in that snapshot.
  - dataset expected `convert_tokens_to_ids`.
  - dataset expected `<|semantic|>`, while S1-mini tokenizer uses indexed semantic tokens.

The final `<|semantic|>` mismatch is a semantic-format mismatch, not a harmless missing method. Continuing with ad hoc patches could start training but produce an invalid or misleading fine-tuning result.

## New Baseline

Use the stable Fish Speech 1.5 fine-tuning path:

- Code: `fishaudio/fish-speech` tag `v1.5.1`
- Checkpoint: `fishaudio/fish-speech-1.5`
- VQ config: `firefly_gan_vq`
- VQ checkpoint:
  `firefly-gan-vq-fsq-8x1024-21hz-generator.pth`
- Fine-tune config:
  `text2semantic_finetune`
- LoRA:
  `r_8_alpha_16`

This follows the Fish Speech 1.5 documentation, where VQ extraction, protobuf packing, LoRA training, and LoRA merge are part of the same release/checkpoint family.

## Colab Notebook

Use:

```text
notebooks/colab_aihub_emotion_finetune_v15.ipynb
```

Run it in a fresh Colab runtime. Start with:

```python
RUN_MODE = "smoke"
```

After smoke succeeds, change:

```python
RUN_MODE = "full"
```

and rerun from dataset preparation onward.

## Important Data Note

Do not reuse S1-mini `.npy` VQ token files or protobuf files. Fish Speech 1.5 uses a different VQGAN/token format, so the notebook writes separate outputs:

```text
processed/fish15_smoke
processed/fish15_full
protos/fish15_smoke
protos/fish15_full
```

## Source References

- Fish Speech GitHub releases show `v1.5.1` as the last stable branch before the next model release and `v1.5.0` as the Fish Speech 1.5 release with inference and fine-tuning.
- Fish Speech 1.5 model files are hosted at `fishaudio/fish-speech-1.5`.
- The v1.5.1 fine-tuning docs use `fish-speech-1.5`, `firefly_gan_vq`, protobuf packing, `text2semantic_finetune`, and LoRA merge.
