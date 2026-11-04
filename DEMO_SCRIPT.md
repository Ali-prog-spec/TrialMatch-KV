# Demo video script (target 6:30, limit 5-7 min)

## Before you press record
1. Close other GPU programs (browser video, games) so timings are not noisy.
2. PowerShell: `cd "C:\Users\KENKINIKI\Desktop\New folder (2)"` then `python -m app.server`. Wait ~60 s until the status box says Ready.
3. Open http://127.0.0.1:8000 in a browser at 100% zoom. Make the window full screen.
4. Have this unseen test note ready to paste (it is not from the dataset):
   Note: `A 60-year-old man with type 2 diabetes. He smokes daily. No kidney disease.`
   Criteria:
   ```
   Inclusion criteria:
   - Type 2 diabetes
   Exclusion criteria:
   - Chronic kidney disease
   - Pregnancy
   ```
5. Record the screen with Win+G (Game Bar) or OBS, with your microphone on.

## 0:00-0:40  Introduction (show the GitHub repo page or the report title)
"Assalam o alaikum. I am Muhammad Ali, roll number 23i-0008, section B. This is my Generative AI assignment 2, TrialMatch-KV. The system reads a patient description and one clinical-trial criterion, decides whether the patient meets it, and cites the exact supporting sentences. I fine-tuned Qwen2.5-1.5B with LoRA on the TrialGPT criterion annotations, using a patient-grouped 37, 8, 8 split with seed 42, so no patient appears in two splits. I also implemented KV-cache decoding, tiled attention and speculative decoding by hand."

## 0:40-1:30  Start and disclaimer (show the app)
"The app runs locally in the browser. The banner at the top says this is an educational screening prototype, not a clinical enrollment decision. The status box shows the fine-tuned adapter and the draft model are loaded on my RTX 2060."

## 1:30-3:00  Screening a sample patient (Screening tab)
- Pick a sample from the dropdown, click Load, then Run screening.
"Here is a validation-split patient. For every criterion the app shows the decision, the exact evidence sentences with their IDs, and an explanation that comes only from the label and cited sentences."
- Point at one `included` row with its evidence sentence.
- Point at a `not enough information` row.
"This one is marked as missing information. The model never treats missing evidence as proof of eligibility or ineligibility. The overall outcome at the top is a rule I wrote, not a ground-truth label: any excluded or not-included criterion means likely ineligible, otherwise any missing information means needs more information."
- Click Download report (HTML) and open it briefly.
"The report includes provenance: model, adapter, decoding mode and timestamp."

## 3:00-3:50  Unseen case and invalid input
- Click Clear, paste the unseen note and criteria above, run it.
"This is a note and criteria the model has never seen."
- Point out that Pregnancy for a male patient is answered `not excluded`.
"This is one of the model's real weaknesses: the correct label is not applicable. I analyse this error type in the report."
- Clear everything and click Run with an empty note, then with a note but no criteria.
"Invalid input gives a clear error message instead of crashing."

## 3:50-5:00  Inference comparison tab
- Click Inference Comparison, set k to 4, click Run Live Comparison.
"This runs my three hand-written decoders on the same prompt: no cache, KV cache, and speculative decoding. The last column shows all three produce exactly the same tokens. The KV cache is faster than recomputing everything. Speculative decoding is token-identical but slower on my GPU, because the 0.5B draft model costs almost as much per step as the 1.5B target. In my benchmarks it never beat plain greedy decoding, and the report explains why."

## 5:00-5:40  Attention lab and saved experiments
- Attention Lab tab: T 512, block 64, Run.
"This compares reference attention, my tiled online-softmax attention and PyTorch's fused kernel. The error against the reference is tiny. My tiled version is slower because it is a Python loop, but it needs far less memory: about 85 times less extra memory at length 2048."
- Saved Experiments tab.
"Here are the saved test results and benchmarks."

## 5:40-6:30  Results and limitations (show the report's results table or the saved experiments tab)
"On the held-out test set of 8 patients and 206 criteria, the base model gets 22 percent accuracy. After fine-tuning, accuracy is 72 percent and macro-F1 is 0.49, with evidence F1 of 0.54. My first model always answered not excluded for exclusion criteria, so I retrained with class-balanced sampling, choosing the epoch on validation data only. The model is still over-confident when information is missing, and the test set is small. All code, the split manifest, logs and the adapter are on GitHub, linked in the report. Thank you."

## Tips
- Speak slowly; ~130 words per minute fits the time.
- If a run takes long, say "this takes a few seconds on my laptop GPU" and keep talking.
- Do the whole flow once without recording first.
- Upload as YouTube unlisted (or Google Drive with link sharing), then add the link to the README and the Classroom submission.
