# Individual project proposal (revised)

**Student:** Pavel Slobodich, student ID 61993
**Course:** Machine Learning and Its Applications [CII6SE027CI], Group 01

## Problem

An AI technique coach: from a phone video of one repetition, check the
technique of the back squat, overhead press and barbell row, and name the
error (for example "knees inward"). It runs offline on the phone, between
sets. Motivation: I train in the gym and spent a long time fixing my own
squat technique.

## Data

**Main: Fitness-AQA** (Parmar, Gharat and Rhodin, "Domain
Knowledge-Informed Self-Supervised Representations for Workout Form
Assessment", ECCV 2022).

- Unscripted gym videos from YouTube and Instagram, with many people and
  camera angles; one sample is one repetition.
- Errors were labelled by two professional trainers:
  - squat: knees inward, knees forward, shallow depth;
  - overhead press: elbows, knees;
  - barbell row: lower back, torso angle.
- **Licence:** CC BY-NC 4.0. Study use only; no redistribution (the data
  stays on my computer, not on GitHub); faces blurred in anything shown.
  Access is by request through the authors' form.

**Start and second source: Squat Dataset** (Teng, Zenodo 2025, DOI
10.5281/zenodo.17558630, CC BY 4.0).

- 3,806 side-view photos in three classes: good, bad back, bad heel. The
  official split is 2,837 training and 969 test photos.
- From a sample I checked: bodyweight squats at home by a few people, as
  consecutive video frames; the test people seem to differ from the training
  people. It is available now, so the first pipeline and baseline use it.

**Own test set:** a few of my own gym repetitions filmed from the side,
labelled by me, with no other people in the frame.

## Approach

1. **Pose first.** A small pretrained pose estimator turns each frame into
   body keypoints (shoulders, hips, knees and so on). The classifier works on
   these, not on raw pixels, so clothes and background do not matter and
   no faces are stored.
2. **Error classifier.** A small network over the keypoints of one
   repetition (a 1D CNN over time), one output per error type. The Squat
   Dataset has single photos, so there it classifies the keypoints of one
   frame.
3. **Compression.** Structured pruning and INT8 quantization, as in Labs 7
   and 8, with distillation into a smaller student as a fallback.
4. **Deployment.** ONNX export and a phone or browser demo.

## Success metric

- **Main:** F1 per error type and macro-F1 on the held-out test data. The
  errors are imbalanced and some are rare, so accuracy alone would mislead.
- **Reference:** the F1 reported in the Fitness-AQA paper (for example
  0.87 for shallow squat and 0.53 for knees inward). The goal is to come
  close to it with a model small enough for a phone.
- **Squat Dataset:** macro-F1 on its official test split.
- **Own videos:** errors caught on my repetitions, as a small real-world
  test.

## Deployment constraints

- **Latency ≤ 50 ms** per frame for the pose estimator plus the classifier:
  the median of 100 runs after a warm-up, on the laptop CPU limited to
  1 thread as a stand-in for a phone.
- **Size ≤ 5 MB** for everything shipped: the pose model and the classifier.

## Milestones

- **Lab 8:** data pipeline in the style of Lab 4, a baseline on the Squat
  Dataset, and `metrics.json`.
- **Next:** Fitness-AQA once access is granted, then compression, ONNX and
  the demo.
- **Lab 11:** robustness to lighting, camera angle and partly hidden body
  parts.

## Risks

- **Fitness-AQA access may be slow or refused.** The project then runs on
  the Squat Dataset plus my own videos.
- **Few people in the Squat Dataset.** Mitigation: report results per person
  where possible, and test on my own videos.
- **The pretrained pose model may use most of the 5 MB.** Mitigation: choose
  the smallest pose model that is accurate enough, and quantize it too.
- **The pose estimator fails when plates or racks hide the body.**
  Mitigation: measure how often this happens, and report results with and
  without such frames.
