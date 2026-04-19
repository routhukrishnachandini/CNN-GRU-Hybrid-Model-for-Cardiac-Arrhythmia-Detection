Cardiac Arrhythmia Detection using CNN–GRU

This project implements a hybrid deep learning model combining CNN (Convolutional Neural Networks) and GRU (Gated Recurrent Units) to detect cardiac arrhythmias from ECG signals.

Overview

The model uses CNN layers to automatically extract important features from ECG waveforms, while GRU layers capture temporal dependencies in the signal. This combination improves accuracy and efficiency in identifying abnormal heart rhythms.

Features
-Automated ECG feature extraction using CNN
-Temporal pattern learning with GRU
-Improved arrhythmia detection performance
-Suitable for real-time monitoring applications

Project Structure
-->app.py – Main application file
-->arrhythmia.ipynb – Model training and experimentation
-->email_helper.py – Email notification functionality
-->requirements_updated.txt – Required dependencies

Requirements
-->Python 3.x
-->TensorFlow / PyTorch
-->NumPy, Pandas, Matplotlib

Future Improvements
-->Deploy as a web application
-->Optimize model for edge devices
-->Expand dataset for better generalization
