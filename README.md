COMP576 Final Project
--------------------------------------------------------------------


Project Overview
-------------------
This project is studies the effectives of incorporating multimodal physiological data alongside video-based behavioral data for driver drowsiness detection. 
For this project, we have compare
	Unitmodel baseline model: Only trained on facial landmarks from video.
	Multimodel: Trained with facial landmarks from video + PSG signals.
Our goal is to quantritatively evaluate whether multimodel inputs improve performance pver video only approaches

*****************************************************************************************************************************************

Dataset
-------------------
In this project we use **ULg Multimodality Drowsiness Database (DROZY)**, which is a publicly available multimodal dataset designed for drowsiness detection research..

This DROZY dataset includes:
	- Near-infrared facial videos
	- 2D facial landmark annotations
	- Polysomnography (PSG) signals:
		- EEG
		- EOG
		- ECG
		- EMG
	- Subhective drowsiness labels based on the Karolinska Sleepiness Scales(KSS)

Drowsiness lables are defined as:
	- Alert: KSS <= 5
	- Drowsy: KSS >=6

#Dataset Source

	- The dataset and detail description are provide in the following link
	- https://orbi.uliege.be/handle/2268/191620

#Why Dataset is not Included in the Github Repository

	1. Dataset contains large video files and PSG recordings, which exceed GitHub's storage limits.
	2. Licensing and redistribution constraints
		a. Dataset is intended for research   use and shpuld be obtained directly from original authors's distribution platform
	
	This repository provides all code necessary tob preprocess the data, train the models, show result by graph, while require users to download the dataset independently.

****************************************************************************************************************************************

Model Description
-------------------

Baseline Model
	- Architecture: LSTM
	- Input: 2D facial landmarks extracted from video
	- Task: Binary classification
		- Alert: KSS <= 5
		- Drowsy: KSS >= 6

Multimodel
	- Architecture: CNN + LSTM hybrid
	- Inputs
		- Behavioral: Facial landmark extracted from video.
		- Physiological: PSG signals

****************************************************************************************************************************************

Run Code
-------------------

Requirements
	- pip install -r requirements.txt

Debug
	- python run_baseline.py --debug

Training
	- python run_baseline.py --epochs 50

***************************************************************************************************************************************

Code Avaliability
-------------------

All code used for data preprocessing, model training, and evaluation in this project is available in this reposity.

 



