# Error analysis (held-out test, fine-tuned model)

58 of 206 criterion predictions wrong.

| kind                                                  |   count |
|:------------------------------------------------------|--------:|
| unsupported eligibility (missing info treated as met) |      25 |
| over-abstention (info present)                        |      13 |
| missed not-applicable                                 |      10 |
| contradiction / wrong polarity                        |      10 |

- **[contradiction / wrong polarity]** ann 383 (inclusion) criterion: "Present with active SLE disease with at least one of the following symptoms: i) Arthritis - > 2 joints with pain and signs of inflammation (" -> gold *included* [0, 1], predicted *not included* [0, 1, 2]
- **[contradiction / wrong polarity]** ann 835 (inclusion) criterion: "Negative allergy skin test (AST)" -> gold *not enough information* [], predicted *not included* [1]
- **[contradiction / wrong polarity]** ann 691 (inclusion) criterion: "Experiences symptoms of OSA, including snoring and sleepiness" -> gold *included* [0, 2], predicted *not included* [0]
- **[missed not-applicable]** ann 683 (exclusion) criterion: "Previous tonsillectomy, including partial tonsillectomy" -> gold *not applicable* [], predicted *not excluded* []
- **[missed not-applicable]** ann 771 (exclusion) criterion: "Females of childbearing potential (ie, have reached menarche)" -> gold *not applicable* [0], predicted *not excluded* []
- **[missed not-applicable]** ann 751 (exclusion) criterion: "breast-feeding" -> gold *not applicable* [], predicted *not excluded* []
- **[over-abstention (info present)]** ann 187 (inclusion) criterion: "A respiratory rate of at least 20/min or a PaCO2 of 32mmHg or lower or the use of mechanical ventilation;" -> gold *included* [2], predicted *not enough information* [2]
- **[over-abstention (info present)]** ann 425 (exclusion) criterion: "APACHE II (health) score > 25" -> gold *not excluded* [], predicted *not enough information* []
- **[over-abstention (info present)]** ann 185 (inclusion) criterion: "Patients were included if they were diagnosed with Shock, Acute respiratory distress syndrome (ARDS), or both." -> gold *included* [2, 3], predicted *not enough information* [0, 1, 2]
- **[unsupported eligibility (missing info treated as met)]** ann 391 (exclusion) criterion: "Intravenous corticosteroids within 3 months prior to the first dose of study medication." -> gold *not enough information* [], predicted *not excluded* []
- **[unsupported eligibility (missing info treated as met)]** ann 315 (exclusion) criterion: "BMI> 40 kg/m2." -> gold *not enough information* [], predicted *not excluded* []
- **[unsupported eligibility (missing info treated as met)]** ann 412 (exclusion) criterion: "Vaccination within 4 weeks prior to the first dose of study medication." -> gold *not enough information* [], predicted *not excluded* []