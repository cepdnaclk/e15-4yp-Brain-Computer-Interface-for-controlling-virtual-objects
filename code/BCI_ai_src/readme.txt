BCI_3D_environment -  This is the virtual environment created using Unity game engine(v 2019.4.4.f1).
do not change folder arrangements of the project and open it as a unity project.

folder structure
BCI_3D_environment ->
	->Assets
		->Scripts (this folder contains behaviour scripts of scene)
		->Scenes (this folder contain the scene)

BCI_ai_src - This is the python program which is classify the thoughts and feed into virtual environment

folder structure

BCI_ai_src->

	->dataset		(this folder contains training data)

		->lsl_data  (directly recorded time series data)

			->left
			->right
			->none

		->simulation_lsl_data (data for simulation)

			->left
			->right
			->none

		->transformed_data	(transformed data)

			->fft_data
			->wavelet data

	->main_program	(this folder contains the main program and simulation program)
		-> run_v1.0.py file

	->models (this folder contains trained models as .joblib files)

	->preprocessors (contains FFT , WT converters =>> to dataset->transformed_data)

	->recorder (offline data recorder)

	->trainer_evaluator	(machine learning algorithms training and evaluation)

	->utilities (transform utilities for fft,wt,etc..)
-----------------------------------------------------------------------------------------------------------------
How to run the program.
turn on the openbci gui application(5.0.3v) and enable the lsl stream for time series data (bandpassed 5-50 and notch 50Hz)
execute the 
	->run_v1.0.py file
goto the virtual environment and play
press button (-)