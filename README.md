# Predicting childhood and adolescent attention-deficit/hyperactivity disorder onset: a nationwide deep learning approach

Python code for the paper: Garcia-Argibay, M., Zhang-James, Y., Cortese, S., Lichtenstein, P., Larsson, H., & Faraone, S. V. (2023). Predicting childhood and adolescent attention-deficit/hyperactivity disorder onset: a nationwide deep learning approach. *Molecular Psychiatry*, 28(3), 1232-1239.
[[Link to paper]](https://www.nature.com/articles/s41380-022-01918-8)

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

# Citation

    @article{garcia2023predicting,
      title={Predicting childhood and adolescent attention-deficit/hyperactivity disorder onset: a nationwide deep learning approach},
      author={Garcia-Argibay, Miguel and Zhang-James, Yanli and Cortese, Samuele and Lichtenstein, Paul and Larsson, Henrik and Faraone, Stephen V},
      journal={Molecular Psychiatry},
      volume={28},
      number={3},
      pages={1232--1239},
      year={2023},
      publisher={Nature Publishing Group UK London}
    }
