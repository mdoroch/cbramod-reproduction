# CBraMod vs EEG-SimpleConv on SHU-MI

Reproduction of [CBraMod](https://arxiv.org/abs/2412.07236) (ICLR 2025) on SHU-MI (left vs right hand motor imagery) and a comparison with [EEG-SimpleConv](https://github.com/elouayas/EEGSimpleConv), with shared data, split, training loop and metrics. Only the model differs.

## Commands

```bash
pip install -r requirements.txt
# SHU-MI from figshare 19228725 (1.4 GB). Entries are AES-encrypted: unzip fails, use bsdtar (libarchive-tools)
cd data && wget -O mat_files.zip https://ndownloader.figshare.com/files/36728994 \
  && bsdtar --passphrase shu-bci2022 -xf mat_files.zip && rm mat_files.zip && cd ..   # -> data/mat/ (125 files)
python prepare_data.py                    # data/mat/*.mat -> data/processed/*.npy (200 Hz)
bash scripts/download_weights.sh          # pretrained CBraMod weights from HuggingFace -> weights/

python train.py --model cbramod --seed 0      # configs/cbramod.yaml (paper recipe: one lr 1e-4) = E7
python train.py --model simpleconv --seed 0   # configs/simpleconv.yaml
python train.py --model cbramod --seed 0 --name my_run --set head=small epochs=10   # override config keys

bash scripts/reproduction/run_all.sh      # E1-E8 below, then results/summary.md
tensorboard --logdir runs --port 6006    # open http://localhost:6006
```

Device is picked automatically (CUDA > MPS > CPU), fp32 everywhere.

## Experiments (`scripts/reproduction/`)

| | Script | What | Seeds |
|---|---|---|---|
| E1 | `e1_cbramod.sh` | CBraMod, pretrained weights, full fine-tuning, **official repo lr** (head 5e-4) | 5 |
| E2 | `e2_simpleconv_lr.sh`, `e2_simpleconv.sh` | EEG-SimpleConv from scratch, lr chosen on val | 5 |
| E3 | `e3_cbramod_scratch.sh` | CBraMod from random init: does pretraining pay off? Repo lr, then one lr 1e-4 (as E7) | 3 + 3 |
| E4 | `e4_cbramod_small_head.sh` | Small head (201 params) vs the official 20.6M MLP head (E1), repo lr | 3 |
| E5 | `e5_low_data.sh` | 3 / 7 train subjects (seeded random subsets) for CBraMod (repo lr and one lr), CBraMod scratch (both), SimpleConv | 3 |
| E6 | `e6_cbramod_frozen.sh` | Pretrained backbone frozen, only the head is trained (paper Table 18), repo lr | 3 |
| **E7** | `e7_cbramod_single_lr.sh` | **CBraMod, paper recipe**: one lr 1e-4 for backbone and head (Table 6). Config default. Reproduces Table 3 | 5 |
| E8 | `e8_ea.sh` | Euclidean Alignment per subject, both models (CBraMod as E7, SimpleConv as E2) | 3 + 3 |

Every run selects the epoch with the best val AUROC and evaluates the test set once. It writes `results/{name}/seed{seed}.json` (config, val and test metrics, **test metrics per subject**) and `seed{seed}_test_preds.npz` (probabilities, labels, subject) for paired per-subject comparisons. With only 5 test subjects, the between-subject spread dominates; differences below ~0.02 AUROC are treated as noise.

## Does it match the original?

1. **Implementation parity** (exact, CPU, no training): `bash scripts/get_official_code.sh && python scripts/check_parity.py`. It checks that the data split and preprocessing equal the official `preprocessing_shu.py`, that the pretrained weights load strictly, and that our CBraMod and EEG-SimpleConv give the same logits as the official code with the same weights.
2. **Scores**: E7 (paper recipe) and E1 (repo default) vs paper Table 3 (BAcc 0.6370 ± 0.0151, AUC-PR 0.7139 ± 0.0088, AUROC 0.6988 ± 0.0068). EEG-SimpleConv has no published SHU-MI number, so only its implementation is checked.

## Results

Test metrics, mean ± std over seeds (ddof=0). Full table incl. per-subject AUROC: [`results/summary.md`](results/summary.md). "Repo lr" = official `multi_lr` (backbone 1e-4, head 5e-4); "one lr" = 1e-4 for all, as in the paper (Table 6) and the config default.

| Setting | Paper | | | Ours | | | |
|---|---|---|---|---|---|---|---|
| | BAcc | AUC-PR | AUROC | Exp. | BAcc | AUC-PR | AUROC |
| **CBraMod pretrained, one lr** (Table 3) | 0.637 ± 0.015 | 0.714 ± 0.009 | **0.699 ± 0.007** | **E7, 5 seeds** | **0.617 ± 0.010** | **0.681 ± 0.020** | **0.678 ± 0.020** |
| ↳ repo lr | | | | E1, 5 seeds | 0.588 ± 0.013 | 0.655 ± 0.010 | 0.632 ± 0.015 |
| ↳ repo lr, repo default seed 3407 | | | | 1 seed | 0.603 | 0.682 | 0.661 |
| w/o pretraining, one lr (Table 4) | 0.629 ± 0.018 | 0.703 ± 0.015 | 0.688 ± 0.017 | E3, 3 seeds | 0.611 ± 0.004 | 0.673 ± 0.014 | 0.672 ± 0.011 |
| ↳ repo lr | | | | E3, 3 seeds | 0.613 ± 0.022 | 0.689 ± 0.017 | 0.683 ± 0.023 |
| frozen backbone (Table 18), repo lr | 0.522 ± 0.025 | 0.530 ± 0.025 | 0.524 ± 0.032 | E6, 3 seeds | 0.582 ± 0.032 | 0.653 ± 0.006 | 0.650 ± 0.002 |
| low data: 30% (Table 19) / ours 7 of 15 subjects, one lr | 0.623 ± 0.020 | 0.690 ± 0.013 | 0.675 ± 0.011 | E5, 3 seeds | 0.614 ± 0.008 | 0.679 ± 0.007 | 0.681 ± 0.010 |
| CBraMod small head, repo lr (not in paper) | – | – | – | E4, 3 seeds | 0.535 ± 0.037 | 0.547 ± 0.056 | 0.550 ± 0.056 |
| **EEG-SimpleConv** (not in paper) | – | – | – | E2, 5 seeds, lr 3e-4 | 0.535 ± 0.040 | 0.651 ± 0.022 | **0.667 ± 0.022** |
| CBraMod + Euclidean Alignment, one lr | – | – | – | E8, 3 seeds | 0.615 ± 0.011 | 0.670 ± 0.013 | 0.671 ± 0.015 |
| EEG-SimpleConv + Euclidean Alignment | – | – | – | E8, 3 seeds | 0.579 ± 0.015 | 0.617 ± 0.018 | 0.622 ± 0.023 |

Low data (E5), test AUROC by number of train subjects (3 seeds each; 15 = E7 / E3 / E2):

| train subjects | CBraMod pretrained, one lr | CBraMod scratch, one lr | gain | CBraMod pretrained, repo lr | CBraMod scratch, repo lr | EEG-SimpleConv |
|---|---|---|---|---|---|---|
| 3 | 0.649 | 0.602 | **+0.047** | 0.650 | 0.607 | 0.570 |
| 7 | 0.681 | 0.664 | +0.017 | 0.672 | 0.665 | 0.647 |
| 15 | 0.678 (seeds 0-2: 0.690) | 0.672 | +0.006 (+0.018 on seeds 0-2) | 0.632 | 0.683 | 0.667 |

- **Reproduction.** With the paper's recipe (one lr 1e-4) CBraMod gets 0.678 ± 0.020 test AUROC vs 0.699 ± 0.007 (seeds 0-3: 0.683-0.695; seed 4 picks epoch 14 and gets 0.640). With the official repo default (head lr 5e-4) it gets 0.632 ± 0.015: **the head lr is the main cause of the gap**. Data, logits, optimizer, selection and metrics match the official code exactly.
- **Pretraining pays off only with little data**: +0.047 AUROC with 3 train subjects, +0.017 with 7, +0.006 with all 15 (within noise). Same trend as the paper (+0.011 on full data).
- **EEG-SimpleConv** (2.5M params, no pretraining, ~3 min) is 0.011 below CBraMod on full data, within noise. Its BAcc (0.535) is low because its logits are shifted on unseen subjects (threshold 0.5); compare models on AUROC.
- **Euclidean Alignment did not help**: CBraMod 0.671 vs 0.690 (same seeds 0-2), SimpleConv 0.622 vs 0.671. For SimpleConv it fixes the threshold (BAcc 0.513 → 0.579) but hurts ranking.
- **Test subjects 24 and 25 are at chance for every model and setting**; with 5 test subjects, differences < 0.02 are noise.

## Problems log

| Issue | What was missing | Our decision | Possible cost |
|---|---|---|---|
| SHU-MI preprocessing | Not described in the paper | Official repo: FFT resample to 200 Hz, `x / 100`, no filtering, no clipping | Input differs from pretraining (0.3-75 Hz band-pass, 60 Hz notch, samples > 100 µV dropped); 33% of SHU trials exceed 100 µV |
| Head learning rate | Paper gives one lr (1e-4); repo default `multi_lr` uses head lr 0.001·sqrt(64/256) = 5e-4 | Config default = paper (one lr, E7); repo lr kept as E1 (`--set head_lr=5.0e-4`) | **Main cause of the gap**: E1 0.632 vs E7 0.678 test AUROC |
| "±" in tables | Not stated what it is over | Seeds (5 for E1/E2, 3 otherwise) | Paper std may be over something else |
| QK LayerNorm | Described in the paper, absent in the code | Follow the code (the weights were trained with it) | — |
| Channels | 13 of 32 SHU channels are not among the 19 pretraining channels; ACPE depends on channel order | Use all 32 in file order | Unseen channels may hurt transfer |
| EEG-SimpleConv recipe | Not defined for SHU-MI | Official cross-subject BCIC-IV-2a architecture, shared loop, no EA / Mixup, lr tuned on val | May underestimate SimpleConv: its "key ingredients" add ~16 pts in CBraMod Table 23 |
| Seed values | "five random seeds", values not given; repo default 3407 | Seeds 0-4; 3407 run separately | Paper number may come from different seeds |
| Overfitting | Not discussed | Official recipe kept: train loss -> ~0.01 by epoch 20, val loss rises from epoch 3, val AUROC flat | ~90% of the 50 epochs add nothing; best-val-AUROC epoch is partly noise (val AUROC moves ±0.015 between epochs) |
| Checkpoint selection | "AUROC is the monitor score" | Best val AUROC epoch, test once | Val (16-20) and test (21-25) are different people: val AUROC ~0.73 vs test ~0.64 |
| BAcc at threshold 0.5 | Threshold not stated (repo: 0.5) | 0.5 for both models | Penalises EEG-SimpleConv: good ranking (AUROC) but shifted logits on unseen subjects -> BAcc ~0.5. Compare models on AUROC |
| Low-data protocol | Table 19: "30% of fine-tuning data", sampling not described | Seeded random subsets of 3 / 7 of the 15 train subjects (20% / 47%) | Not directly comparable to the paper's 30% |
| Frozen backbone | Table 18: AUROC 0.524 | Repo `--frozen`: backbone `requires_grad=False`, same head and recipe | Ours gives 0.650, far from the paper; cause unknown |
| CPU threads | — | `torch.set_num_threads(4)` in train.py | None on results; 2x faster when several runs share one GPU |
