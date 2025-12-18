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


def inertia_plot(df: pd.DataFrame, n_clusters:int):
    inertia = []

    range_clusters = range(1, n_clusters)
    for n_clus in range_clusters:  # iterate over desired ncluster range
        kmclust = KMeans(n_clusters=n_clus, init='k-means++', n_init=15, random_state=1)
        kmclust.fit(df)
        inertia.append(kmclust.inertia_)  # save the inertia of the given cluster solution
    
    fig, ax = plt.subplots(figsize=(9,5))

    ax.plot(range_clusters, inertia)
    ax.set_xticks(range_clusters)
    ax.set_ylabel("Inertia: SSw")
    ax.set_xlabel("Number of clusters")
    ax.set_title("Inertia plot over clusters", size=15)

    plt.show()


class preProcessing():

    def __init__(self,
                 df
                ):
        self.df = df
        self.train = df[df["train"] == True]
        self.test =  df[df["train"] == False]
        
        self.imputed_df = self.df.copy()

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
            # --- NEW VARIABLES ---

            # 3. Usage Intensity (Is it a fleet/commuter car or a weekend car?)
            # High mileage on a young car usually drops price faster.
            """df_fe["mileage_per_year"] = np.where(
                df_fe["age"] > 0,
                df_fe["mileage"] / df_fe["age"],
                df_fe["mileage"] # For cars less than 1 year old
            )

            # 4. Total Cost of Ownership Proxy
            # Combines Tax and Fuel efficiency to show how "expensive" it is to keep.
            # We invert MPG because lower MPG = higher cost.
            df_fe["running_cost_index"] = df_fe["tax"] + (100 / (df_fe["mpg"] + 1) * 1.5)

            # 5. Engine Displacement vs Age (The "Classic" vs "Modern Performance" split)
            # Large engines in old cars have a different value curve than large engines in new cars.
            df_fe["engine_age_interact"] = df_fe["engineSize"] * df_fe["age"]

            # 6. Mileage/Age Interaction (Non-linear depreciation)
            # Helps the model see that 50k miles on a 2yr old caris worse than 50k on a 10yr old car.
            df_fe["log_mileage"] = np.log1p(df_fe["mileage"])"""
            
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
    col_n_scale = config['col_n_scale']
    outliers_mask = config["outliers_mask"]

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
    X_train_cat, ohe_enc = encodeCatVariables(X_train_imputed, ohe_cols=ohe_cols_to_use)
    X_val_cat, _ = encodeCatVariables(X_val_imputed, ohe_cols=ohe_cols_to_use, ohe_encoder=ohe_enc)
    
    # 6. Scaling
    X_train_scaled, scaler_train = scaleNumVariables(X_train_cat, col_n_scale, method="Robust", supervised=False)
    X_val_scaled, _ = scaleNumVariables(X_val_cat, col_n_scale, method="Robust", supervised=False, scaler=scaler_train)

    # 7. Final Output
    X_train_final = X_train_scaled.copy()
    X_val_final = X_val_scaled.copy()

    return X_train_final, X_val_final, y_train.values, y_val.values