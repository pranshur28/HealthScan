# Hey Doc! Meet Your New AI Assistant for Medical Diagnosis

Ever wished you had a smart assistant to help with tricky diagnoses? Well, here it is! This tool uses Random Forest algorithms (fancy machine learning stuff) to help you make better decisions by comparing your patient's data with similar cases.

## What Can This Thing Do?

- Make smart predictions about possible diagnoses (it's pretty accurate!)
- Fine-tune itself to get even better over time
- Show you which symptoms or test results actually matter most
- Find patients with similar conditions (great for comparative analysis)
- Explain its thinking so you're not left wondering "why this diagnosis?"
- Create nice charts and graphs to visualize what's going on

## Getting Started

1. Grab a copy of this code
2. Install the stuff it needs:

```bash
pip install -r requirements.txt
```

## How to Use It

### The Basics

```python
from medical_classifier import MedicalDiagnosticTool

# Fire it up
diagnostic_tool = MedicalDiagnosticTool()

# Teach it with your medical records
results = diagnostic_tool.train(X, y, feature_names, target_name)

# Ask what it thinks about a new patient
prediction = diagnostic_tool.predict(patient_data)
print(f"I think it might be: {prediction['prediction']}")
print(f"I'm about {prediction['confidence']*100}% sure")

# Find similar patients in your records
similar_cases = diagnostic_tool.find_similar_cases(patient_data, dataset)

# Save your trained model for next time
diagnostic_tool.save_model('medical_diagnostic_model.pkl')
```

### Try the Demo

Want to see it in action? Run this:

```bash
python demo.py
```

## The Science Bit

This tool uses Random Forest classifiers because they:

- Handle all kinds of messy data (just like real patient records!)
- Don't freak out over outliers or unusual values
- Tell you which factors matter most in making a diagnosis
- Work really well even without tons of tweaking
- Can handle lots of different patient measurements at once

## Important Note

This tool is meant to be your assistant, not your replacement! Always use your medical expertise and judgment. This is just a really smart second opinion.

