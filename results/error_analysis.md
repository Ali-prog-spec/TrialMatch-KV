# Error analysis (held-out test, fine-tuned model)

57 of 206 criterion predictions wrong.

| kind                                                  |   count |
|:------------------------------------------------------|--------:|
| unsupported eligibility (missing info treated as met) |      32 |
| missed not-applicable                                 |      15 |
| over-abstention (info present)                        |       8 |
| contradiction / wrong polarity                        |       2 |

- **[contradiction / wrong polarity]** ann 164 (inclusion) criterion: "Patients undergoing surgical lower limb revascularisation (suprainguinal or infrainguinal)" -> gold *not included* [], predicted *included* [0, 1, 2]
- **[contradiction / wrong polarity]** ann 163 (inclusion) criterion: "Patients undergoing endovascular abdominal aneurysm repair" -> gold *not included* [], predicted *included* [0, 1, 2]
- **[missed not-applicable]** ann 422 (inclusion) criterion: "If female, using birth control" -> gold *not applicable* [3], predicted *not enough information* [3]
- **[missed not-applicable]** ann 750 (exclusion) criterion: "pregnancy" -> gold *not applicable* [], predicted *not excluded* []
- **[missed not-applicable]** ann 754 (exclusion) criterion: "alcoholism (drunk at least once a week)" -> gold *not applicable* [], predicted *not excluded* []
- **[over-abstention (info present)]** ann 188 (inclusion) criterion: "The use of vasopressors to maintain a systolic blood pressure of at least 90mmHg despite fluid resuscitation, low dose of dopamine (≤ 5 μg/k" -> gold *not included* [3], predicted *not enough information* []
- **[over-abstention (info present)]** ann 176 (inclusion) criterion: "No medical history of central vena catheterization." -> gold *included* [0], predicted *not enough information* []
- **[over-abstention (info present)]** ann 162 (inclusion) criterion: "Patients undergoing open abdominal aortic aneurysm repair" -> gold *not included* [], predicted *not enough information* []
- **[unsupported eligibility (missing info treated as met)]** ann 395 (exclusion) criterion: "Potent inhibitors or inducers of CYP3A4 intravenously or orally within 14 days prior to first dose of study medication." -> gold *not enough information* [], predicted *not excluded* []
- **[unsupported eligibility (missing info treated as met)]** ann 411 (exclusion) criterion: "Patients likely to receive oral or intravenous steroids or immunosuppressant for other non-SLE condition during the study duration, as this " -> gold *not enough information* [], predicted *not excluded* []
- **[unsupported eligibility (missing info treated as met)]** ann 394 (exclusion) criterion: "B-cell depletion therapy (such as treatment with Rituximab) within 12 months prior to the first dose of study medication." -> gold *not enough information* [], predicted *not excluded* []