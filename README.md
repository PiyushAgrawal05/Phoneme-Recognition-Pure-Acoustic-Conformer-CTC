# Pure-Acoustic Conformer-CTC Phoneme Recognizer

## Project Overview
This repository contains an end-to-end, Pure-Acoustic Automatic Speech Recognition (ASR) pipeline built entirely from scratch in PyTorch. The model is designed to transcribe raw audio directly into CMU ARPAbet phoneme sequences without the reliance on external Language Models (LMs). A core focus of this project is a mathematically verified **Domain Adaptation strategy** that significantly improves phonetic decoding accuracy on non-native English accents.

## 
## 🏗️ Technical Architecture & Pipeline

```text
+-----------------------------------------------------------------------------------+
|                        1. MULTI-CORPUS DATA INGESTION                             |
|  • LibriSpeech (train.100)  --> Native English Baseline                           |
|  • SpeechOcean762 (train)   --> Non-Native Expansion                              |
|  • L2-Arctic (scripted)     --> Target L2 Domain (80% Train / 10% Val / 10% Test) |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                     2. AUDIO NORMALIZATION & FRONT-END                            |
|  • 16kHz Raw Waveform (<= 10s) -> 80-Channel Log-Mel Spectrograms                 |
|  • FP16 Logarithmic Clamping (min=1e-5) to prevent gradient underflow             |
|  • SpecAugment Regularization (FreqMask=27, TimeMask=40)                          |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                  3. PURE-ACOUSTIC CONFORMER SEQUENCE ENCODER                      |
|  • Linear Projection (80 -> 256)                                                  |
|  • 12x Conformer Blocks (4 Heads, 1024 FFN, Depthwise Conv Kernel = 31)           |
|  • CTC Classification Head (256 -> 41 Classes)                                    |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                   4. TWO-STAGE CURRICULUM TRAINING ROUTINE                        |
|  • Stage 1 (Pre-Training): 20,000 Steps on LibriSpeech + SO762 (lr=5e-4)          |
|  • Stage 2 (L2 Adaptation): 5,000 Steps on L2-Arctic (lr=5e-5)                    |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                   5. PURE-ACOUSTIC CTC GREEDY DECODING                            |
|  • Argmax selection per 20ms frame -> Collapse repeated tokens -> Strip <blank>   |
+-----------------------------------------------------------------------------------+
```

## Results & Ablation Study
The domain adaptation hypothesis was rigorously tested by evaluating unmasked acoustic features (bypassing SpecAugment during evaluation) on the validation and unseen test sets. 

* **Variant B (Baseline - Stage 1):** ~42% Phoneme Error Rate (PER)
* **Variant C (Domain Adapted - Stage 2):** ~32% Phoneme Error Rate (PER)

**Key Insight:** The 10% absolute reduction in PER was primarily driven by a sharp decrease in Deletion Errors. Fine-tuning allowed the model to confidently map ambiguous, non-native acoustic boundaries to actual phonemes rather than defaulting to the safe CTC `` token.

## Repository Structure
The pipeline is contained within a cleanly structured Google Colab notebook, divided into 8 core engineering phases:
1. **Environment Setup & Authentication:** Drive mounting and Hugging Face OAuth.
2. **Vocabulary & Data Ingestion:** 41-class deterministic mapping and stream partitioning.
3. **Model Architecture & Regularization:** Conformer initialization and SpecAugment.
4. **Curriculum Learning Stage 1:** 20k-step baseline mixed-precision training.
5. **Decoding & Quantitative Evaluation:** CTC greedy decoding and `jiwer` error analysis.
6. **Target Domain Adaptation (Stage 2):** 5k-step fine-tuning loop.
7. **Visualizations & Ablation Analysis:** Loss trajectory curves, ablation bar charts, and Spectrogram-to-Logit alignment graphs.
8. **Real-Time Inference:** Standalone audio processing and interactive UI.

## Interactive Inference
The project includes a standalone inference engine wrapped in a **Gradio v4** web interface. This allows for live microphone capture directly within the browser. The pipeline handles on-the-fly stereo-to-mono downmixing, 16 kHz resampling, feature extraction, and real-time CTC decoding for arbitrary `.wav` files or live voice recordings.

## Known Limitations
As a pure-acoustic, LM-free system, the model is highly sensitive to background noise and compression artifacts (e.g., standard web-audio compression). In live microphone testing, the model successfully identifies sharp consonant boundaries but exhibits aggressive deletion of continuous vowel sounds. Future iterations will require integrating an N-gram Language Model (e.g., KenLM) during the decoding phase to recover dropped vowels via linguistic context.
