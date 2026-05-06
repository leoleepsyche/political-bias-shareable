# Narrative Plan

Audience: research advisor / collaborator reviewing the current political-bias steering analysis.

Objective: create a clear research-story PPT that answers one primary question: whether hidden-state political ideology directions can causally steer Political Compass behavior, and whether this effect generalizes across models, steering strengths, and languages.

Primary research question:

Can political ideology directions extracted from model hidden states be used to causally steer LLM political behavior, and does this steering generalize across models, steering strengths, and languages?

Narrative arc:

1. Define the research problem as representation plus intervention: ideology may be detectable in hidden states, but the key question is whether that direction changes behavior.
2. Describe the method: train/extract left and right ideology directions from paired ideological examples, then inject those directions during generation.
3. Show the experimental design: three instruct models, five languages, left/right steering, five coefficients, 62 Political Compass statements per condition.
4. Result 1: ideology directions are highly detectable in hidden states.
5. Result 2: English Political Compass coordinates move under steering, but the effect is model-dependent.
6. Result 3: detection strength and behavioral steerability are separable; high AUC does not guarantee large compass movement.
7. Result 4: cross-lingual transfer is clearest for Llama-3; Qwen has important missing-coordinate caveats.
8. Use stance-distribution figures as mechanism/supplementary evidence, not as the main result.
9. End with the claim: internal ideology directions are real and sometimes causally useful, but detectability, steerability, and multilingual transfer are distinct properties.

Slide list:

1. Title and one-sentence thesis.
2. Research problem: representation is not the same as behavioral control.
3. Primary research question and three subquestions.
4. Experimental logic: paired ideology data -> hidden-state direction -> activation steering -> Political Compass shift.
5. Dataset and scope: models, languages, coefficients, steering sides, compass items.
6. Detection method: layer selection and left/right hidden-state directions.
7. Result 1: detection AUC by model and ideology direction.
8. Steering intervention: alpha as dose and alpha=0 as local baseline.
9. Result 2: English compass movement under steering, with alpha points and displacement.
10. Result 3: detection is not steerability; summarize AUC versus max shift.
11. Result 4: cross-lingual transfer heatmap.
12. Missing-data caveat: NA means no valid compass coordinate, not zero effect.
13. Mechanism/supplement: stance distribution across coefficients for a focused model/language.
14. Main empirical story by model: Llama-3 strong, Mistral weak, Qwen mixed/incomplete.
15. Limitations and next steps.
16. Takeaway slide.

Visual system: plain white slides, large black titles, minimal blue accent, no template. Existing plots are inserted as images; all explanation text remains editable.

Figure role map:

- Main: detection AUC bar chart from `outputs/figures_main/main_detection_data.csv`.
- Main: redesigned English compass displacement figure from `outputs/figures_main/main_shift_data.csv`.
- Main or synthesis: detection-versus-steerability summary from `outputs/figures_main/detection_vs_steering_data.csv`.
- Main: redesigned cross-lingual heatmap from `outputs/figures_main/main_shift_data.csv`.
- Supplementary/mechanism: focused stance distribution from `plot_instruct_stance_coefficients.py` outputs.
- Optional QC: parse-rate heatmap only if needed to explain missing coordinates.

Caption requirements:

- For displacement figures: "shift from alpha=0" means coordinate displacement relative to the local unsteered baseline.
- For cross-lingual heatmap: "NA means no valid compass coordinate was parsed; it is not zero effect."
- For stance-distribution figures: bars show answer-category proportions across 62 items, not compass coordinates.

Editability plan: slide text is editable PowerPoint text; figures are inserted as image assets from the local analysis outputs.

Detailed figure strategy is documented in `../../RESEARCH_QUESTION_AND_FIGURE_MAP.md`.
