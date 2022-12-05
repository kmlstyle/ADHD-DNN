# Predicting childhood and adolescent attention-deficit/hyperactivity disorder onset: a nationwide deep learning approach
Python code for the paper *"Predicting childhood and adolescent attention-deficit/hyperactivity disorder onset: a nationwide deep learning approach"* (2022) by Miguel Garcia-Argibay, Ph.D., Yanli Zhang-James, MD, Ph.D., Samuele Cortese, MD, Ph.D., Paul Lichtenstein, Ph.D., Henrik Larsson, Ph.D., Stephen V. Faraone, Ph.D.

`DNN.py` contains the Python code for the Deep Neural Network (DNN).

`ML.py` contains the code for the different machine learning models & ensemble model.

`model_weights\best.h5` includes the model weights. Use `tf.keras.models.load_model()` to load the weights. Note that the models expects an array with 22 features in the specified order (see Python code) and that features need to be scaled. After the model is loaded, use `model.predict(x = scaled_features, batch_size=100000, verbose=0)` to calculate the model predicted probabilities of having ADHD for each individual.

# Requirements
- Python 3.8.13
- Numpy & Pandas
- Scikit-learn 1.1.1 
- TensorFlow 2.8.0
- [Imbalanced-learn 0.9.1](https://imbalanced-learn.org/stable/)
- [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap)
- [Hyperopt: Distributed Hyperparameter Optimization](https://github.com/hyperopt/hyperopt)
- [Bayesian Optimization; Bayes_opt](https://github.com/fmfn/BayesianOptimization)
- [Keras tuner](https://keras.io/keras_tuner/)


# Data availability 

**The Public Access to Information and Secrecy Act in Sweden prohibits us from making individual level data publicly available. Researchers who are interested in replicating our work can apply for individual level data at [Statistics Sweden](https://www.scb.se/en/services/guidance-for-researchers-and-universities/)**.
