import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
import category_encoders as ce

def colTypes(
        df:pd.DataFrame, 
        print_:bool=True
    ) -> tuple:
    """
    Arguments:
        df (pd.DataFrame): The input DataFrame to be analyzed for column types.
        print_ (bool): A flag to determine whether to print the identified 
            column lists to the console. Defaults to True.

    Resume:
        The function separates the DataFrame columns into two categories: 
        numerical and categorical. It identifies categorical columns (col_c) 
        based on two criteria: having an 'object' data type or having fewer 
        than 25 unique values. Any remaining columns are classified as 
        numerical (col_n).

    Output:
        tuple: A tuple containing two lists: (col_n, col_c), where col_n 
            represents numeric columns and col_c represents categorical columns.
    """

    col_c = df.columns[(df.dtypes == "object") | (df.nunique() < 25)].tolist()
    col_n = df.columns.difference(col_c).tolist()

    if print_:
        print(f"Numeric Cols: {col_n}")
        print(f"Categorical Cols: {col_c}")
        
    return col_n, col_c

def encodeCatVariables(
    df: pd.DataFrame, 
    ohe_cols: list = None, 
    ord_cols: list = None, 
    freq_cols: list = None,
    ord_categories: dict = None,
    ohe_encoder: OneHotEncoder = None,  # <--- optional pre-fitted encoder
) -> tuple[pd.DataFrame, OneHotEncoder]:
    
    ohe_cols = ohe_cols or []
    ord_cols = ord_cols or []
    freq_cols = freq_cols or []
    ord_categories = ord_categories or {}
    """
    Arguments:
        df (pd.DataFrame): The source DataFrame containing categorical features.
        ohe_cols (list, optional): Columns to be transformed using One-Hot Encoding.
        ord_cols (list, optional): Columns to be transformed using Ordinal Encoding.
        freq_cols (list, optional): Columns to be transformed based on value frequency.
        ord_categories (dict, optional): A dictionary mapping ordinal column names 
            to their specific category rankings (lists).
        ohe_encoder (OneHotEncoder, optional): An existing pre-fitted OneHotEncoder. 
            If None, a new encoder will be initialized and fitted.

    Resume:
        The function applies three different encoding strategies to specified 
        categorical columns. It handles One-Hot Encoding (with an option to use 
        pre-fitted encoders for pipeline consistency), Ordinal Encoding using 
        custom hierarchies, and Frequency Encoding. After transformation, it 
        removes the original categorical columns and concatenates the new 
        features back into the DataFrame, ensuring index alignment throughout.

    Output:
        tuple: A tuple containing:
            - pd.DataFrame: The transformed DataFrame with encoded variables.
            - OneHotEncoder: The encoder used for One-Hot transformation (useful 
              for applying the same mapping to test sets).
    """
    df_encoded = df.copy()

    # One-Hot Encoding
    if ohe_cols:
        if ohe_encoder is None:  # fit new encoder
            ohe_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            ohe_encoded = ohe_encoder.fit_transform(df[ohe_cols])
        else:                   # use pre-fitted encoder
            ohe_encoded = ohe_encoder.transform(df[ohe_cols])
        colnames = ohe_encoder.get_feature_names_out(ohe_cols)
        ohe_encoded_df = pd.DataFrame(ohe_encoded, columns=colnames, index=df.index)
    else:
        ohe_encoded_df = pd.DataFrame(index=df.index)

    # Ordinal Encoding
    if ord_cols:
        categories_list = [ord_categories.get(col, "auto") for col in ord_cols]
        ord_encoder = OrdinalEncoder(categories=categories_list)
        ord_encoded = ord_encoder.fit_transform(df[ord_cols])
        ord_encoded_df = pd.DataFrame(ord_encoded, columns=ord_cols, index=df.index)
    else:
        ord_encoded_df = pd.DataFrame(index=df.index)

    # Frequency Encoding
    if freq_cols:
        freq_encoded = pd.DataFrame(index=df.index)
        for col in freq_cols:
            count = df[col].value_counts()
            freq_encoded[col] = df[col].map(count)
    else:
        freq_encoded = pd.DataFrame(index=df.index)

    # Drop original categorical columns
    cat_cols = ohe_cols + ord_cols + freq_cols
    df_encoded.drop(columns=cat_cols, inplace=True, errors="ignore")

    # Concatenate all
    res = pd.concat([df_encoded, ohe_encoded_df, ord_encoded_df, freq_encoded], axis=1)

    return res, ohe_encoder


def scaleNumVariables(
        df: pd.DataFrame, 
        cols: list = None,
        method: str = None, 
        supervised: bool = None,
        scaler: object = None
    ) -> tuple[pd.DataFrame, object]:
    """
    Arguments:
        df (pd.DataFrame): The input DataFrame containing numerical features.
        cols (list): A list of column names to be scaled.
        method (str): The scaling technique to use. Must be one of 
            ["Standard", "MinMax", "Robust"].
        supervised (bool): If True, the function expects a 'train' column (bool) 
            in the df to separate training data for fitting and test data 
            for transformation only.
        scaler (object, optional): A pre-fitted scaler object. If provided, 
            this scaler will be used instead of initializing a new one.

    Resume:
        The function scales numerical variables using the specified Scikit-Learn 
        method. If 'supervised' is set to True, it prevents data leakage by 
        fitting the scaler strictly on the training subset (where df['train'] == True) 
        and applying that transformation to the test subset. If 'supervised' is 
        False, it fits and transforms the entire provided DataFrame. It also 
        supports using an external pre-fitted scaler for inference.

    Output:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame with the specified columns scaled.
            - object: The fitted scaler object (StandardScaler, MinMaxScaler, or RobustScaler).
    """
    df_scaled = df.copy()
    assert method in ["Standard", "MinMax", "Robust"]

    # Choose scaler
    if scaler:
       scaler = scaler
    else: 
        if method == "Standard":
            scaler = StandardScaler()
        elif method == "MinMax":
            scaler = MinMaxScaler()
        elif method == "Robust":
            scaler = RobustScaler()

    if supervised:
        train = df[df["train"] == True]
        test = df[df["train"] == False]

        scaler.fit(train[cols])
        train_scaled = scaler.transform(train[cols])
        test_scaled = scaler.transform(test[cols])

        df_scaled.loc[train.index, cols] = train_scaled
        df_scaled.loc[test.index, cols] = test_scaled

    else:
        scaler.fit(df[cols])
        df_scaled[cols] = scaler.transform(df[cols])
    
    # Return both scaled DataFrame and fitted scaler
    return df_scaled, scaler

class preProcessing():

    def __init__(self,
                 df
                ):
        self.df = df
        self.train = df[df["train"] == True]
        self.test =  df[df["train"] == False]
        
        self.imputed_df = self.df.copy()
        """
        Arguments:
            mask (pd.Series or array-like): A boolean mask where 'True' represents 
                rows to keep and 'False' represents outliers to be removed. This 
                mask should correspond to the indices of the training data.

        Resume:
            The function filters the training subset of the dataset using the 
            provided boolean mask while keeping the test subset completely intact. 
            It then reconstructs the main DataFrame (`self.df`) by concatenating 
            the filtered training data with the original test data, re-sorting by 
            index, and synchronizing all internal class attributes (`self.train`, 
            `self.test`, and `self.imputed_df`) to reflect the changes.

        Output:
            None: Updates the state of the class instance and prints the new
        """
    def remove_outliers(self, mask):
        """
        Applies a boolean mask to the training data only.
        The test data remains unchanged to ensure full evaluation capability.
        """
        # 1. Identify training and testing indices
        train_indices = self.df[self.df["train"] == True].index
        test_indices = self.df[self.df["train"] == False].index

        # 2. Apply the mask specifically to the training subset
        # This keeps only rows that are IN the mask AND are part of the train set
        train_filtered = self.df.loc[train_indices].loc[mask]
        
        # 3. Keep all test rows as they are
        test_data = self.df.loc[test_indices]

        # 4. Reconstruct the global dataframe
        self.df = pd.concat([train_filtered, test_data]).sort_index()

        # 5. Synchronize the internal train/test helpers
        self.train = self.df[self.df["train"] == True]
        self.test = self.df[self.df["train"] == False]
        
        # Also update imputed_df to reflect the removal of rows
        self.imputed_df = self.df.copy()
        
        print(f"Outliers removed. New train size: {len(self.train)}, Test size: {len(self.test)}")


    def impute_values(self, 
                            cols_impute_mode=None, 
                            cols_impute_mean=None, 
                            cols_impute_median=None, 
                            group_by_model=False):
            
            # 1. Calculate statistics using ONLY the training data (self.train)
            # This prevents data leakage from the test set.
            """
            Arguments:
                cols_impute_mode (list, optional): Columns where missing values will 
                    be replaced by the most frequent value (mode).
                cols_impute_mean (list, optional): Columns where missing values will 
                    be replaced by the average value (mean).
                cols_impute_median (list, optional): Columns where missing values 
                    will be replaced by the median.
                group_by_model (bool): If True, calculates statistics for each category 
                    within the 'model' column. If False, calculates global statistics.

            Resume:
                The function fills missing values in specified columns using statistics 
                derived exclusively from the training set (`self.train`). It supports 
                both global imputation and grouped imputation (by 'model'). In grouped 
                mode, if a specific model in the full dataset was not present in the 
                training set, the function automatically falls back to global training 
                averages. Finally, it synchronizes the class attributes (`self.train`, 
                `self.test`, and `self.imputed_df`) to reflect the changes.

            Output:
                None: Modifies the internal state of the class instance.
            """
            if group_by_model:
                # Grouped stats
                if cols_impute_mode:
                    mode_values = self.train.groupby('model')[cols_impute_mode].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
                if cols_impute_mean:
                    mean_values = self.train.groupby('model')[cols_impute_mean].mean()
                if cols_impute_median:
                    median_values = self.train.groupby('model')[cols_impute_median].median()
            
            # Global stats (used for global imputation OR as a fallback for missing models)
            global_modes = self.train[cols_impute_mode].mode().iloc[0] if cols_impute_mode else None
            global_means = self.train[cols_impute_mean].mean() if cols_impute_mean else None
            global_medians = self.train[cols_impute_median].median() if cols_impute_median else None

            # 2. Apply imputations to self.df
            if group_by_model:
                for model, group in self.df.groupby('model'):
                    # Handle models present in data but missing from training stats
                    if model in median_values.index:
                        if cols_impute_mode:
                            self.df.loc[group.index, cols_impute_mode] = group[cols_impute_mode].fillna(mode_values.loc[model])
                        if cols_impute_mean:
                            self.df.loc[group.index, cols_impute_mean] = group[cols_impute_mean].fillna(mean_values.loc[model])
                        if cols_impute_median:
                            self.df.loc[group.index, cols_impute_median] = group[cols_impute_median].fillna(median_values.loc[model])
                    else:
                        # FALLBACK: Model wasn't in training, use global training averages
                        if cols_impute_mode:
                            self.df.loc[group.index, cols_impute_mode] = group[cols_impute_mode].fillna(global_modes)
                        if cols_impute_mean:
                            self.df.loc[group.index, cols_impute_mean] = group[cols_impute_mean].fillna(global_means)
                        if cols_impute_median:
                            self.df.loc[group.index, cols_impute_median] = group[cols_impute_median].fillna(global_medians)
            else:
                # Simple global imputation
                if cols_impute_mode: self.df[cols_impute_mode] = self.df[cols_impute_mode].fillna(global_modes)
                if cols_impute_mean: self.df[cols_impute_mean] = self.df[cols_impute_mean].fillna(global_means)
                if cols_impute_median: self.df[cols_impute_median] = self.df[cols_impute_median].fillna(global_medians)

            # 3. Synchronize
            self.train = self.df[self.df["train"] == True]
            self.test = self.df[self.df["train"] == False]
            self.imputed_df = self.df.copy()

    def knnImputation(self, cols_impute_knn, col_n, col_ohe):
        """
        Arguments:
            cols_impute_knn (list): The specific columns containing missing values 
                to be filled using the K-Nearest Neighbors algorithm.
            col_n (list): The list of numerical columns required for distance 
                calculation and scaling.
            col_ohe (list): The list of categorical columns to be one-hot 
                encoded to facilitate distance calculation.

        Resume:
            The function performs model-specific KNN imputation for 'Focus' and 
            'C-Class' vehicles. It implements a rigorous pipeline for each group: 
            (1) robust scaling of numerical features, (2) one-hot encoding of 
            categorical features, and (3) fitting the KNNImputer strictly on 
            training rows to prevent leakage. After transforming all rows, it 
            performs an inverse scaling transformation to return the imputed 
            values to their original units before updating the global `imputed_df`.

        Output:
            None: Updates the `self.imputed_df` attribute with the newly 
                calculated values.
        """
        # 0 — Initialize imputed_df if not already
        if not hasattr(self, "imputed_df"):
            self.imputed_df = self.df.copy()

        # 1 — Split Focus and C-Class
        df_focus  = self.df[self.df["model"] == "Focus"].copy()
        df_cclass = self.df[self.df["model"] == "C-Class"].copy()

        # 2 — Scale numeric columns
        df_focus_scaled, df_focus_scaler = scaleNumVariables(df_focus, col_n, method="Robust", supervised=True)
        df_cclass_scaled, df_cclass_scaler = scaleNumVariables(df_cclass, col_n, method="Robust", supervised=True)

        # 3 — Keep only relevant columns
        df_focus_scaled  = df_focus_scaled[col_ohe + col_n + ["train"]].copy()
        df_cclass_scaled = df_cclass_scaled[col_ohe + col_n + ["train"]].copy() 

        # 4 — Encode categorical variables
        df_focus_scaled_encoded, _  = encodeCatVariables(df_focus_scaled, col_ohe)
        df_cclass_scaled_encoded, _ = encodeCatVariables(df_cclass_scaled, col_ohe)

        # 5 — Fit KNN imputers on training rows only
        train_focus_idx  = df_focus_scaled_encoded["train"] == True
        train_cclass_idx = df_cclass_scaled_encoded["train"] == True

        imputer_focus  = KNNImputer(n_neighbors=5, weights="uniform")
        imputer_cclass = KNNImputer(n_neighbors=5, weights="uniform")

        imputer_focus.fit(df_focus_scaled_encoded.loc[train_focus_idx])
        imputer_cclass.fit(df_cclass_scaled_encoded.loc[train_cclass_idx])

        # 6 — Transform all rows
        df_imputed_focus  = pd.DataFrame(imputer_focus.transform(df_focus_scaled_encoded),
                                        columns=df_focus_scaled_encoded.columns,
                                        index=df_focus_scaled_encoded.index)
        df_imputed_cclass = pd.DataFrame(imputer_cclass.transform(df_cclass_scaled_encoded),
                                        columns=df_cclass_scaled_encoded.columns,
                                        index=df_cclass_scaled_encoded.index)

        # 7 — Inverse scale numeric columns
        df_imputed_focus[col_n]  = df_focus_scaler.inverse_transform(df_imputed_focus[col_n])
        df_imputed_cclass[col_n] = df_cclass_scaler.inverse_transform(df_imputed_cclass[col_n])

        # 8 — Update global imputed_df
        self.imputed_df.loc[df_imputed_focus.index, cols_impute_knn]  = df_imputed_focus[cols_impute_knn]
        self.imputed_df.loc[df_imputed_cclass.index, cols_impute_knn] = df_imputed_cclass[cols_impute_knn]
    
    def create_newVariables(self):
            """
            Arguments:
                None: This method utilizes the internal 'self.imputed_df' 
                    already stored in the class instance.

            Resume:
                The function performs feature engineering to generate three new 
                variables: 'age', 'mileage_per_litre', and 'efficiency_ratio'. 
                It includes safety logic using `np.where` to handle potential 
                division by zero errors (e.g., electric vehicles or data errors 
                where engineSize is 0). The newly created features are then saved 
                back into the class's 'imputed_df' attribute.

            Output:
                None: Updates the 'self.imputed_df' attribute with the new 
                    engineered feature columns.
            """
            df_fe = self.imputed_df.copy()
            
            # 1. Base Age
            df_fe["age"] = 2022 - df_fe["year"]

            # 2. Existing Ratios with safety check
            df_fe["mileage_per_litre"] = np.where(
                df_fe["engineSize"] != 0,
                df_fe["mileage"] / df_fe["engineSize"],
                0
            )

            df_fe["efficiency_ratio"] = np.where(
                df_fe["engineSize"] != 0,
                df_fe["mpg"] / df_fe["engineSize"],
                0
            )
            
            self.imputed_df = df_fe.copy()

def preprocess_data(
    train_fold: pd.DataFrame, 
    val_fold: pd.DataFrame, 
    config: dict, # Dictionary holding all configuration variables
    knn_impute:bool = False,
    group_by_model_impute: bool = False
) -> tuple:
    """
    Applies the full preprocessing pipeline to separate training and validation folds 
    using explicit configuration arguments.
    
    The pipeline includes imputation, feature engineering, target encoding, 
    one-hot encoding, and scaling. Encoders/Scalers are fitted ONLY on the 
    training fold and applied to both.

    Parameters:
    - train_fold (pd.DataFrame): Training data and target.
    - val_fold (pd.DataFrame): Validation data and target.
    - config (dict): Dictionary of configuration variables (e.g., column lists).

    Returns:
    - tuple: (X_train_final, X_val_final, y_train, y_val) 
             The final feature and target arrays ready for model training.
    """
    
    # Unpack configuration variables for cleaner code
    target_name = config['target_name']
    col_n = config['col_n']
    cols_imputer_ohe = config['cols_imputer_ohe']
    mode_imputation_cols = config['mode_imputation_cols']
    mean_imputation_cols = config['mean_imputation_cols']
    median_imputation_cols = config['median_imputation_cols']
    col_target = config['col_target']
    cols_ohe = config['cols_ohe']
    frequency_cols = config["frequency_cols"]
    col_n_scale = config['col_n_scale']
    outliers_mask = config["outliers_mask"]
    scaler_name = config["scaler_name"]

    # 1. Combine for Imputation (using the 'train' flag)
    train_fold["train"] = True
    val_fold["train"] = False
    
    fold_df = pd.concat([train_fold, val_fold], axis=0)

    # 2. Imputation 
    # (Assuming preProcessing, knnImputation, imputeModeMean are defined externally)
    pre = preProcessing(fold_df)
    pre.remove_outliers(outliers_mask)

    if knn_impute:
        pre.knnImputation(["tax", "mpg"], col_n, cols_imputer_ohe)
    if group_by_model_impute:
        pre.impute_values(mode_imputation_cols, mean_imputation_cols, median_imputation_cols, group_by_model_impute)
        
    pre.impute_values(mode_imputation_cols, mean_imputation_cols, median_imputation_cols)
    # 3. Feature Engineering
    pre.create_newVariables()

    df_imputed_fe = pre.imputed_df
    
    # 3. Feature Engineering

    # Split back into training and validation sets
    train_fold_imputed = df_imputed_fe[df_imputed_fe["train"] == True].drop(columns=["train"]).copy()
    val_fold_imputed   = df_imputed_fe[df_imputed_fe["train"] == False].drop(columns=["train"]).copy()
    
    # Separate features and target
    y_train = train_fold_imputed[target_name]
    y_val = val_fold_imputed[target_name]
    
    X_train_imputed = train_fold_imputed.drop(columns=[target_name])
    X_val_imputed = val_fold_imputed.drop(columns=[target_name])

    # 4. Target Encoding (Fit on X_train, y_train)
    te = ce.TargetEncoder(cols=col_target, smoothing=1)
    
    # Target Encoder needs features AND target for fitting
    X_train_imputed[col_target] = te.fit_transform(X_train_imputed[col_target], y_train)
    X_val_imputed[col_target] = te.transform(X_val_imputed[col_target])

    # 5. One-Hot Encoding (Fit on X_train)
    ohe_cols_to_use = [c for c in cols_ohe if c not in col_target]
    
    # (Assuming encodeCatVariables is defined externally)
    X_train_cat, ohe_enc = encodeCatVariables(X_train_imputed, ohe_cols=ohe_cols_to_use, freq_cols=frequency_cols)
    X_val_cat, _ = encodeCatVariables(X_val_imputed, ohe_cols=ohe_cols_to_use, ohe_encoder=ohe_enc)
    
    # 6. Scaling
    X_train_scaled, scaler_train = scaleNumVariables(X_train_cat, col_n_scale, method=scaler_name, supervised=False)
    X_val_scaled, _ = scaleNumVariables(X_val_cat, col_n_scale, method=scaler_name, supervised=False, scaler=scaler_train)

    # 7. Final Output
    X_train_final = X_train_scaled.copy()
    X_val_final = X_val_scaled.copy()

    return X_train_final, X_val_final, y_train.values, y_val.values