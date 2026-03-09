# Elara-TTS: Egyptian Arabic Speech Synthesis Engine

Elara-TTS represents the core text-to-speech engine developed for the Elara educational platform. This project outlines the complete methodology and engineering process required to adapt the Spark-TTS (0.5B) architecture to fully grasp and generate the Egyptian Arabic dialect with high fidelity and natural intonation.

## Methodology & Pipeline Overview

Training a generalized TTS model on a specific, non-standardized dialect like Egyptian Arabic requires a rigorous pipeline. The process is heavily focused on data quality, linguistic normalization, and memory-optimized fine-tuning.

### Stage 1: Data Acquisition & Pre-processing
The foundation of Elara-TTS is a meticulously curated 14-hour dataset of natural Egyptian speech. The raw data underwent a strict processing pipeline:

1. Extraction & Standardization: Raw audio streams were extracted and standardized to 16kHz mono-channel WAV files to meet baseline acoustic requirements.
2. Intelligent Segmentation: Using `faster-whisper` (large-v3), the audio was transcribed and sliced. A Voice Activity Detection (VAD) filter was crucial here to trim extended silences (min_silence_duration_ms=500) and prevent the model from learning "dead air".
3. Duration Constraints: To ensure stable attention mechanisms during training, audio chunks were strictly filtered to be between 1.5 and 15.0 seconds. 
4. Metadata Generation: A pipe-separated (`|`) CSV format was chosen over standard commas to completely avoid tokenization conflicts with Arabic text punctuation.
<img width="649" height="675" alt="Screenshot 2026-03-09 085448" src="https://github.com/user-attachments/assets/4bc47aa2-69b5-4578-b78e-f4314c5df76e" />
<img width="700" height="692" alt="Screenshot 2026-03-09 085352" src="https://github.com/user-attachments/assets/2251ed69-223e-4a7e-9d0c-d77446b2974a" />


### Stage 2: The Egyptian Linguistic Normalization Engine
TTS mo![Uploading Screenshot 2026-03-09 085352.png…]()
dels synthesize exactly what they read. Since the Egyptian dialect contains heavy use of numbers, symbols, and foreign currencies that are spoken differently than written, a custom NLP normalization engine was developed:

1. Number Expansion (Grapheme-to-Phoneme adaptation): Digits were programmatically expanded into their Egyptian spoken equivalents based on complex scaling rules (e.g., converting "15" to "خمستاشر" and "200" to "ميتين").
2. Contextual Currency Pluralization: The engine identifies currencies (EGP, USD, EUR) and applies accurate Arabic pluralization logic based on the preceding number (singular, dual, or plural forms, such as "جنيه", "جنيهين", "قروش").
3. Temporal Formatting: Time and date formats (like 10:45) were normalized into natural conversational forms (e.g., "حداشر إلا ربع").
4. Audio Resampling for Tokenization: In the final dataset preparation, the normalized text was paired with audio that was upsampled to 24kHz to match the BiCodecTokenizer's native configuration.

### Stage 3: Neural Tokenization & Feature Extraction
Unlike traditional spectrogram-based TTS, Spark-TTS relies on discrete audio tokens. The Elara-TTS pipeline implements a specialized formatting process:

1. Wav2Vec2 Feature Extraction: The pipeline extracts hidden states from specific layers (11, 14, and 16) of a Wav2Vec2 model to capture the deep acoustic features of the Egyptian voice.
2. BiCodec Tokenization: The audio is compressed into two distinct token streams:
   - Semantic Tokens: Capturing the linguistic content and pronunciation.
   - Global Tokens: Capturing the acoustic environment, speaker timbre, and prosody.

### Stage 4: Fine-Tuning Strategy
To teach the 0.5B model the Egyptian dialect without catastrophic forgetting or hardware exhaustion, the following strategy was employed:

1. Full Fine-Tuning over LoRA: While LoRA is resource-efficient, Full Fine-Tuning was selected to deeply alter the model's phonetic representations, allowing it to natively generate the Egyptian dialect rather than just overlaying an accent.
2. Unsloth Optimization: The `FastModel` from Unsloth was utilized to aggressively reduce VRAM consumption, allowing larger batch processing on consumer-grade hardware.
3. Precision Stability: To counter the notorious NaN loss issues in TTS training, the pipeline is designed to leverage `bf16` precision where hardware permits, maintaining numerical stability in the extremely small weights of the audio tokens.
4. Prompt Formatting: The SFTTrainer was configured to feed the model a highly structured sequence combining the normalized Arabic text, global tokens, and semantic tokens inside specific control tags (e.g., `<|start_content|>`, `<|start_semantic_token|>`).

## Project Outcome
The resulting dataset (~18,900 samples) and the subsequent fine-tuning process successfully shift the Spark-TTS model's latent space to generate natural, educational-grade Egyptian Arabic speech for the Elara platform.
