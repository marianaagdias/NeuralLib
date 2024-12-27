from NeuralLib.architectures.upload_to_hugging import upload_production_model

"""
After training, testing, etc, to make a given model a Production Model, it is necessary that the weights, parameters and
metadata of the model are uploaded to hugging face.
1 - copy model_weights.pth, hparams.yaml, training_info.json in a local directory
2 - run "upload_production_model" as shown below. (adapt repo_name, token, model_name, description, local_dir)
3 - in hugging face, add to collection "NeuralLib: Deep Learning Models for Biosignals Processing"

to upload the model, in the terminal i logged in:
huggingface-cli login
and then paste the token (below)
"""

# TODO: retirar o token daqui (este é meu)
token = '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'
description = "GRU-based model for ECG peak detection"
upload_production_model(local_dir=r'C:\Users\Catia Bastos\dev\trained_models\ECGPeakDetector',
                        repo_name='marianaagdias/ecg_peak_detection',
                        token=token,
                        model_name='ECGPeakDetector',
                        description=description)


