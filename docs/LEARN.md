# Learn this codebase (mentor guide)

You don't need to understand everything at once. Follow the path, run the commands, do the exercises, and come back with questions. Asking "why is it done this way?" is the fastest way to learn.

## Part 1 · Read the repo in 30 minutes (in this order)

| # | File | What to notice |
|---|---|---|
| 1 | `waferscan/classes.py` | The 9 classes in one fixed order. Every model output column follows it. |
| 2 | `waferscan/data/synthetic.py` → `make_wafer` | Each defect *shape* in ~5 lines. Reading how a Donut is drawn teaches you what a Donut *is*. |
| 3 | `waferscan/data/preprocess.py` | A die map becomes a 2×64×64 tensor. `rotate()` explains why rotation never changes the label. |
| 4 | `waferscan/models/wafernet.py` | A small CNN plus the ClassHead. Read the docstring on why GAP + linear gives free heatmaps. |
| 5 | `waferscan/training/losses.py` | Focal loss is only 6 lines. Label smoothing is the `q =` line. |
| 6 | `waferscan/training/train.py` → `run()` | The cross-validation loop. Find the `assert` that proves no wafer is in both train and validation. |
| 7 | `waferscan/inference/policy.py` | Uncertainty → verification → decision. This is the "production brain". |
| 8 | `waferscan/inference/predictor.py` → `predict()` | The whole pipeline as the API runs it. |
| 9 | `waferscan/api/main.py` | Async endpoints; `run_in_threadpool` keeps the server responsive. |
| 10 | `web/app.js` → `boot()` | The landing page tries the API. If it's there, the page goes live; if not, it runs the original demo. |

## Part 2 · The ideas, explained simply

**Die map.** A wafer is cut into a grid of chips (dies). A probe tests each one: pass or fail. WM-811K stores that as a grid of numbers: 0 = no die there, 1 = pass, 2 = fail. The *pattern* of the 2s tells an engineer which machine misbehaved.

**Data leakage.** If the same wafer (or a rotated copy) is in both training and validation, the validation score measures memory, not skill. It's like practising with the exact exam questions. We found about 1 copy in 3 in the image dataset, so we split by *group* (wafer or lot), never by image.

**Cross-validation, out-of-fold (OOF).** Split the data into 5 parts. Train on 4, test on the 5th, and rotate. Every wafer gets exactly one prediction from a model that never saw it. Those OOF predictions are the only honest ones, so every metric and threshold in this repo is computed from them.

**Focal loss.** Normal cross-entropy keeps pushing on examples the model already gets right. Focal loss multiplies by (1 − p)^γ, so an easy example (p = 0.99) contributes almost nothing, and learning time goes to the hard wafers.

**Label smoothing.** Instead of training toward "100% Edge-Ring", train toward "90% Edge-Ring, a little of everything else". The model learns not to be arrogant. But we measured a side effect: together with focal loss, it made our model *too modest*. It was right 99% of the time but only about 72% sure. That's why calibration (below) exists.

**Calibration and temperature scaling.** A model is *calibrated* when "90% sure" really means right 9 times out of 10. ECE measures the gap: 0 is perfect. Our raw CNN scored ECE 0.276. The fix is one number, the temperature T. Divide the model's raw scores by T before softmax. T < 1 sharpens, T > 1 softens. We fit T = 0.37 on out-of-fold predictions, and ECE fell to 0.0025. The predicted class never changes, only the honesty of the confidence.

**Class-balanced weights.** WM-811K has 147,000 clean wafers and only 149 Near-full ones. Without weights, the model can score 85% by always saying "clean". The effective-number formula gives rare classes more weight, but not *absurdly* more.

**Attention (CBAM).** Two small learned masks. One says "these feature channels matter". The other says "look at this part of the wafer".

**CAM heatmap.** The last layer is a weighted sum of feature maps. Multiply the maps by the weights of the predicted class and you see *where* the network found evidence. No extra model needed.

**MC dropout.** Dropout randomly switches off features. Leave it on at prediction time and ask 30 times. If the 30 answers disagree, the model is unsure because it hasn't seen enough wafers like this one.

**Evidential learning.** A second head outputs "how much evidence for each class". Little total evidence means "I don't know", from a single pass.

**Deep ensemble.** The 5 fold models were trained on different data. When they disagree, be careful.

**Stacking.** A tiny second model (logistic regression) learns how much to trust each first-level model. We give it the CNN and a model built on hand-measured shape features. Its learned weights are in the model card (`meta_weight_share`).

**Verification.** After the CNN says "Edge-Ring", we check with plain geometry: is the wafer's edge actually failing all around? If not, the prediction gets a WARN or FAIL, with a reason a human can read.

**Wilson interval.** 12 failed dies out of 400 is 3%, but how sure are we? The Wilson interval gives a range (for example 1.7%–5.2%) that behaves well even for small counts.

**Moran's I.** A number for "do fails clump together?" It's about 0 for random speckle and approaches 1 for clusters. Random-class wafers score low; Center, Donut and Scratch score high.

**Changepoint test.** "Did the Edge-Ring rate jump somewhere in this sequence?" Try every split point, keep the best one, then shuffle the order 300 times. If shuffled data rarely produces a split that good, the jump is real.

## Part 3 · Exercises (easy → hard)

1. **See a leak with your own eyes.** Open `data/processed/rendered_wm811k_audit.json`. Which class has the most `images_with_renamed_copy`? *Hint:* the dataset author made extra copies of the rarest class.
2. **Break a wafer on purpose.** In Python:
   ```python
   from waferscan.data.synthetic import make_wafer; import numpy as np
   dm = make_wafer("Edge-Ring", np.random.default_rng(0))
   ```
   POST it to `/api/v1/predict`. Then set half the ring to pass (`dm[:, :26][dm[:, :26] == 2] = 1`) and POST again. Does the class change to Edge-Loc? What does `verification.reasons` say?
3. **Why rotate?** Set the augmentation to no rotation (edit `augment()` in `train.py` to return `x`). Train one fold with `--folds 0 --epochs 4`. Compare `val_acc` with and without rotation.
4. **Calibration.** In the model card, compare `ece` for `cnn_wafernet`, `cnn_wafernet_temperature_scaled` and `stacked`. Then open `test_temperature_scaling_recovers_known_temperature_and_fixes_ece` in `tests/test_logic.py`: it invents data with a known T and checks we find it again. Change `true_t` to 2.0 (an *over*confident model). Does the fit still find it? Why can't temperature scaling ever change accuracy?
5. **The coverage–accuracy trade-off.** Change `AUTO_ACCEPT_TARGET` in `training/stack.py` to 0.99, then to 0.999, and re-run `make stack`. Plot `coverage_auto_accepted` against the target. Explain to a friend why "100% accuracy" and "100% automation" can't both be true.
6. **Root cause detective.** `GET /api/v1/rootcause/demo`. Which chamber is guilty, and from which wafer does the drift start? Then read `simulate_fab()` and check yourself against the planted truth.
7. **GPU challenge.** Run `notebooks/train_gpu.ipynb` on Kaggle with Swin-T. Is a 28M-parameter transformer better than the 0.5M WaferNet on 64×64 maps? Report macro-F1 *and* training time.

## Part 4 · Questions judges will ask (and good answers)

- **"Why not 100% accuracy?"** Because some wafers are genuinely ambiguous, and their labels were assigned by people. Instead we measure which wafers we can decide alone (93.8% of them, at 99.73% accuracy, cross-fitted), and route the rest to an engineer. That's what fabs actually need.
- **"How do you know there's no data leakage?"** We grouped duplicate copies, including renamed rotated ones found by fingerprint, and used grouped 5-fold CV. The same model scores 99.29% with a random split and 98.81% with the grouped one. We report the grouped number.
- **"Why is the model so small?"** Wafer maps are about 52×52 dies, so a 66M-parameter ImageNet model is overkill. Our 0.5M model runs in milliseconds on a CPU and in INT8 on an offline station. The big backbones are one command away for a GPU comparison.
- **"Can it explain itself?"** Yes. There's a CAM heatmap of where it looked, a verification check with a plain-English reason, and the uncertainty numbers.
- **"Is the root cause real?"** It's an evidence-weighted ranking of associations, validated on a simulated fab where we planted the causes. On real data it tells engineers where to look first. It doesn't replace their experiment.
