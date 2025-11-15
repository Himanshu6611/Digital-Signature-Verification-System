# Digital-Signature-Verification-System
This project implements a digital signature verification system that captures signatures using a touchscreen canvas, extracts features such as stroke shape, speed, and pressure, and compares two samples using DTW-based similarity. It enables enrollment and authentication for secure, real-time signature verification.

**Overview**


The system allows users to draw their signature on a touchscreen or mouse-enabled canvas. It extracts dynamic and static biometric features, converts them into numerical vectors, and compares new samples with stored templates to determine authenticity. The application is ideal for biometric practicals and verification demonstrations.

✨ Features

🔹 Signature Capture

-->Draw signatures on a digital canvas

-->Supports freehand strokes

-->Captures coordinates, timestamps, and pressure (if supported)

🔹 Feature Extraction

-->Stroke shape resampling

-->Shape normalization

-->Mean and variance of writing speed

-->Pressure statistics

-->Dynamic Time Warping (DTW) comparison

🔹 User Enrollment

-->Save user signature templates as .npz feature files

🔹 Authentication

-->Compare probe signature with stored template

-->Outputs similarity score

-->Displays ACCEPTED or REJECTED based on threshold




⚙️ Installation

->Install dependencies:

->pip install numpy


Tkinter comes built-in with most Python installations.

▶️ How to Run
-->python signature_app.py

-->Buttons in GUI

-->Enroll → Save signature template

-->Authenticate → Compare signature with stored user

-->Clear → Reset canvas

-->Quit → Exit app

🚀 Workflow

1.User draws a signature


2.System extracts shape, speed, and pressure metrics


3.Enroll user → saves .npz template


4.Authenticate user → compares new sample to stored template


5.Displays similarity score + final decision


🏁 Conclusion

This project provides a functional signature verification system using stroke dynamics and shape analysis. It demonstrates practical concepts of behavioral biometrics, ideal for academic labs, demos, and verification prototypes.
