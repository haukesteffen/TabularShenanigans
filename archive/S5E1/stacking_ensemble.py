import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class StackingEnsemble:
    """
    A high-level class for K-fold stacked ensembling.
    Assumes a regression task.
    """
    
    def __init__(
        self,
        base_models,       # list of callables or objects that implement 'fit' and 'predict'
        meta_model,        # the second-level learner; also implements 'fit' and 'predict'
        n_splits=5,
        random_state=42,
        shuffle=True,
        oof_total_training_time=1*60*60
    ):
        """
        Args:
            base_models        (list): Each element is something that can be trained on (X_train, y_train)
                                       and then predict on X_valid. E.g., AutoGluon, MLJAR, FLAML wrappers.
            meta_model               : Model for stacking, e.g. XGBRegressor, RandomForestRegressor, or a PyTorch wrapper.
            n_splits            (int): Number of folds for OOF.
            random_state        (int): Random seed for KFold.
            shuffle            (bool): Whether to shuffle data in KFold.
            total_training_time (int): Approximate number of seconds to train the l1 models
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.oof_total_training_time = oof_total_training_time
        self.oof_training_time = oof_total_training_time // (n_splits * len(base_models))

        # We'll store the fold-specific models here: 
        # self.fold_models[i][j] = base model j trained on fold i
        self.fold_models = []
        
        # After we produce OOF predictions, we can train final "full-data" base models
        self.final_base_models = []
        
        # We'll store the meta-model after training
        self.trained_meta_model = None

        # For convenience, we can store OOF predictions:
        self.oof_preds_ = None  # shape (n_samples, n_base_models)
    
    def fit(self, train_data, label):
        """
        Main entry point to:
         1) Generate out-of-fold predictions for each base model (KFold).
         2) Train the meta-model on these OOF predictions.
         3) (Optionally) Retrain each base model on ALL data for final deployment.
        """
        # 1) Build OOF predictions
        oof_preds = self._generate_oof_predictions(train_data, label)
        self.oof_preds_ = oof_preds  # store for reference/analysis if desired
        
        # 2) Train the meta-model using these OOF predictions
        self._fit_meta_model(oof_preds, train_data[label])
        
        # 3) Retrain each base model on the entire dataset so we have
        #    a single final model per base learner for inference
        self._fit_final_base_models(train_data, label)
        
        return self
    
    def _generate_oof_predictions(self, train_data, label):
        """
        For each fold:
          - Train each base model on the fold's training subset
          - Predict on the fold's validation subset (OOF)
          - Store those OOF predictions in the appropriate place
        Returns:
            oof_preds (np.array): shape (n_samples, n_base_models)
        """
        n_samples = len(train_data)
        n_base_models = len(self.base_models)
        
        # We'll create an empty array for OOF predictions
        oof_preds = np.zeros((n_samples, n_base_models), dtype=float)

        # Prepare to store fold-specific models
        # self.fold_models will become a list of length n_splits
        # Each element is a list of length n_base_models for that fold
        self.fold_models = [[] for _ in range(self.n_splits)]
        
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(train_data)):
            train, valid = train_data.iloc[train_idx], train_data.iloc[valid_idx]
            
            fold_models_list = []

            # Train each base model on the (X_train, y_train) of this fold
            for m_idx, base_model in enumerate(self.base_models):
                # We assume each base_model is a "fresh" object or a callable that returns a model.
                # If base_model is a class or function, we instantiate it here:
                model = self._init_model(base_model)
                
                # Fit on fold
                model.fit(train, label)
                
                # Predict on the validation portion
                preds_valid = model.predict(valid)
                
                # Store OOF predictions
                oof_preds[valid_idx, m_idx] = preds_valid
                
                # Store the model so we can do predictions for the next fold
                fold_models_list.append(model)
            
            # Save the list of base models for this fold
            self.fold_models[fold_idx] = fold_models_list
        
        return oof_preds
    
    def _init_model(self, base_model):
        """
        Helper to either clone/instantiate the given base_model. 
        - If base_model is already an instantiated object, you might need to clone it.
        - If it's a function or class, we instantiate here.
        For simplicity, we'll assume base_model is a callable that returns a fresh model.
        """
        if callable(base_model):
            # We assume calling it returns a fresh instance, e.g. base_model() -> new model
            return base_model(time_limit=self.oof_training_time)
        else:
            # If it's already an object, we might need to do something like copy.deepcopy
            import copy
            return copy.deepcopy(base_model)
    
    def _fit_meta_model(self, oof_preds, y):
        """
        Trains the meta-model on the out-of-fold predictions from the base models.
        """
        print('fitting meta learner.')
        self.meta_model.fit(oof_preds, y)
        self.trained_meta_model = self.meta_model
    
    def _fit_final_base_models(self, train_data, label):
        """
        Train each base model on ALL data to get a single final model
        for each base learner. This is typically done AFTER generating OOF preds.
        """
        print('fitting final base models.')
        self.final_base_models = []
        for base_model in self.base_models:
            model = self._init_model(base_model)
            model.fit(train_data, label)
            self.final_base_models.append(model)
    
    def predict(self, X):
        """
        Predict on new data:
          1) Use the final base models (each trained on full data) to get predictions
          2) Feed those predictions into the meta-model
        """        
        # 1) Gather base-model predictions for shape (n_samples, n_base_models)
        base_preds_list = []
        for model in self.final_base_models:
            preds = model.predict(X)            # shape (n_samples,)
            base_preds_list.append(preds)       # collect into list

        # Column-stack them so shape => (n_samples, n_base_models)
        base_preds = np.column_stack(base_preds_list)
        
        # 2) Get final stacked predictions from meta-model
        final_preds = self.trained_meta_model.predict(base_preds)
        return final_preds
    
    def get_oof_data(self):
        """
        Optionally retrieve the OOF predictions if you need them.
        """
        return self.oof_preds_
    
    def score(self, X, y, metric_func):
        """
        Evaluate the entire stack on a given metric, e.g. MSE, MAE, etc.
        (On a holdout set, typically.)
        """
        preds = self.predict(X)
        return metric_func(y, preds)
