# PCAS-Prompt: Visual Semantic FSCIL Research Repository

The default implementation on `dev` is **VSMR** (Visual Semantic Decomposition and Prototype Reconstruction), documented in [README_VSMR.md](README_VSMR.md). The previous APL-PRSN, PCAS, LSP and Prompt-Slot-OT implementations remain selectable as explicit historical baselines and are not part of the VSMR computation graph.

## Run the default method

```bash
python train.py --method vsmr -project base -dataset cub200 -vit \
  -dataroot "$FSCIL_DATA_ROOT" --index_list_root "$FSCIL_INDEX_LIST_ROOT" \
  -gpu 0 -seed 1 -epochs_base 50 -num_workers 4 -test_batch_size 64
```

See [README_VSMR.md](README_VSMR.md) for the architecture, full objective, incremental protocol, diagnostics, experiment settings, and test commands.


## Continual VSMR-Prompt V2.0

The separate `--method continual_vsmr` implementation on
`feature/continual-vsmr-peft` uses one shared Prompt, cached frozen ViT
features, global/local mean prototypes, and immutable historical representation
and prototype anchors. It does not use the legacy VSMR OT/reconstruction graph.
See [README_CONTINUAL_VSMR_V2.md](README_CONTINUAL_VSMR_V2.md) for its
architecture, runtime/memory plan, launcher, and acceptance tests.
