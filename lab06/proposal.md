# Individual project proposal

**Student:** Pavel Slobodich, student ID 61993
**Course:** Machine Learning and Its Applications [CII6SE027CI], Group 01

## Problem

On-device traffic-sign recognition: given a cropped photo of a sign, say
which of 43 sign types it is (speed limit, no entry, give way and so on).
This is the classification step of a driver-assistance or dash-cam app, so
it must answer quickly and work offline. The final model will run in the
browser, and so on a phone, through ONNX Runtime Web, as a small demo page.
Finding the sign in a full road image is out of scope.

## Dataset (not the course default, so it needs approval)

**GTSRB, the German Traffic Sign Recognition Benchmark**, loaded with
`torchvision.datasets.GTSRB` (official train and test splits, about 276 MB).

- Colour photos of 43 sign classes, from 15×15 to 250×250 pixels, resized to
  32×32. The torchvision version has 26,640 training images (888 tracks of
  30 frames) plus the official test set.
- **Source:** Stallkamp et al., "The German Traffic Sign Recognition
  Benchmark: A multi-class classification competition", IJCNN 2011.
- **Licence:** none is stated; the authors ask to cite the paper. I will use
  it for study only, with that citation.
- **Known issues:**
  - Images come in tracks of 30 frames of the same physical sign, so the
    validation split will be made by track to avoid leakage.
  - The classes are imbalanced: from 150 to 1,500 training images per class.
  - All signs are German.

## Success metric

- Test accuracy ≥ 97 %, together with macro-F1 because the classes are
  imbalanced. The worst classes are reported separately.
- Optimisation should cost at most 1 point of accuracy against the
  unoptimised model.

## Deployment constraints

- **Latency ≤ 50 ms** per image (batch 1): the median of 100 runs after a
  warm-up, measured both in Python with ONNX Runtime on 1 CPU thread and in
  the browser demo.
- **Model size ≤ 5 MB** for the ONNX file. Peak memory is reported as well.

## Planned approach and optimisation

1. **Baseline.** A CNN of about 1 M parameters for 3×32×32 input, based on
   Lab 7, trained with the YAML and MLflow pipeline from Lab 6.
   - Training uses small rotations, shifts and brightness changes.
   - There are no horizontal flips: a flipped "turn left" sign would mean
     "turn right".
2. **Structured pruning.** Remove whole filters by L2 norm, then fine-tune. In
   Lab 7 this was the method that actually cut CPU latency: half of the
   filters cost 0.3 points and gave a 2.3× speed-up.
3. **INT8 quantization** of the pruned model (post-training first,
   quantization-aware training if accuracy drops by more than 1 point).
4. **Deployment.** Export to ONNX and build the browser demo with ONNX Runtime
   Web.
5. **Fallback.** Knowledge distillation into a smaller student if pruning and
   quantization lose too much accuracy.

**Deliverables:**

- one table comparing the baseline, the pruned model and the pruned and
  quantized model (accuracy, macro-F1, worst classes, size, latency, memory);
- the demo page;
- runs tracked in MLflow, with SHA256 hashes of the data and the models;
- a data card and a model card;
- in Lab 11, a robustness test on blurred, darker and noisy images.

## Risks

- **No stated licence.** Mitigation: citation and study use only. If the
  dataset is not approved, the same plan runs on the course default,
  Fashion-MNIST.
- **Leakage between near-identical frames.** Mitigation: a validation split by
  track, and the official test set kept untouched.
- **INT8 may not be faster in every ONNX Runtime backend.** Mitigation: report
  measured latency, and ship the float32 pruned model if INT8 is not faster in
  the browser.
