import re
import pandas as pd
from typing import Sequence, Mapping, Optional
import numpy as np
from rapidfuzz import fuzz, process

class FuzzyMatching:
    """
    A utility class for cleaning and standardizing categorical car dataset columns,
    including Brand, Model, Transmission, and Fuel Type. Uses string normalization,
    alias mapping, and fuzzy matching to correct inconsistent or incomplete values.
    Supports brand-aware model cleaning.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        brand_list: Sequence[str],
        model_by_brand: Mapping[str, Sequence[str]],
        brand_aliases: Optional[Mapping[str, str]] = None,
        transmission_list: Optional[Sequence[str]] = None,
        transmission_aliases: Optional[Sequence[str]] = None,
        fueltype_list: Optional[Sequence[str]] = None,
    ):
        """
        Initializes the FuzzyMatching class with the dataset and valid value lists.

        Args:
            df (pd.DataFrame): DataFrame containing car dataset.
            brand_list (Sequence[str]): List of canonical car brands.
            model_by_brand (Mapping[str, Sequence[str]]): Dictionary mapping brands to valid models.
            brand_aliases (Optional[Mapping[str, str]]): Optional dictionary mapping brand aliases to canonical names.
            transmission_list (Optional[Sequence[str]]): Optional list of valid transmission types.
            fueltype_list (Optional[Sequence[str]]): Optional list of valid fuel types.
        """
        self.df = df.copy()
        self.brand_list = brand_list
        self.model_by_brand = model_by_brand
        self.brand_aliases = brand_aliases or {}
        self.transmission_list = transmission_list or []
        self.transmission_aliases = transmission_aliases or []
        self.fueltype_list = fueltype_list or []

    @staticmethod
    def normalize_string(string: str) -> str:
        """
        Normalizes a string by converting to lowercase, stripping whitespace,
        and removing non-alphanumeric characters.

        Args:
            string (str): The string to normalize.

        Returns:
            str: The normalized string.
        """
        s = str(string).lower().strip()
        return re.sub(r"[^a-z0-9]+", "", s)

    def fuzzy_fix(
        self,
        raw_value: object,
        valid_values: Sequence[str],
        alias_map: Optional[Mapping[str, str]] = None
    ):
        """
        Cleans a single value using normalization, optional alias mapping,
        exact matching, and fuzzy matching.

        Args:
            raw_value (object): Original value to clean.
            valid_values (Sequence[str]): List of canonical valid values to match against.
            alias_map (Optional[Mapping[str, str]]): Optional mapping of aliases to canonical values.
            cutoff (int): Minimum fuzzy similarity score to accept a match.
            scorer: Fuzzy matching scoring function (default fuzz.WRatio).

        Returns:
            str or np.nan: Corrected canonical value or NaN if no match found.
        """
        if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
            return np.nan

        canonical_lookup = {self.normalize_string(v): v for v in valid_values}
        normalized = self.normalize_string(raw_value)

        # Alias override
        if alias_map:
            normalized_aliases = {self.normalize_string(k): v for k, v in alias_map.items()}
            if normalized in normalized_aliases:
                return normalized_aliases[normalized]

        # Direct match
        if normalized in canonical_lookup:
            return canonical_lookup[normalized]

        # Fuzzy match fallback
        match = process.extractOne(
            normalized,
            list(canonical_lookup.keys()),
            scorer= fuzz.WRatio,
            score_cutoff=80
        )
        return canonical_lookup[match[0]] if match else np.nan

    def clean_column(
            self,
            column_name: str
        ):
            """
            Cleans an entire column in-place using fuzzy_fix.
            """
            # Use stored valid values if not explicitly provided
            valid_values = None
            alias_map = None
            
            if column_name.lower() == "brand":
                valid_values = self.brand_list
                alias_map = self.brand_aliases

            elif column_name.lower() == "transmission":
                valid_values = self.transmission_list
                alias_map = self.transmission_aliases

            elif column_name.lower() == "fueltype":                
                valid_values = valid_values or self.fueltype_list

            self.df[column_name] = self.df[column_name].apply(
                lambda x: self.fuzzy_fix(
                    raw_value=x,
                    valid_values=valid_values,
                    alias_map=alias_map,
                )
            )
            return self.df

    def clean_modelCol(self, model_col="Model", brand_col="Brand"):
        """
        Cleans the Model column in a brand-aware manner. Uses the brand column
        to select valid models for each row and applies fuzzy matching.

        Args:
            model_col (str): Name of the Model column.
            brand_col (str): Name of the Brand column used for brand-aware matching.
            cutoff (int): Minimum fuzzy similarity score.

        Returns:
            pd.DataFrame: DataFrame with cleaned Model column (in-place).
        """
        all_models = sorted({m for lst in self.model_by_brand.values() for m in lst})

        def fix_model(row):
            raw_model = row[model_col]
            brand = row.get(brand_col)
            candidates = self.model_by_brand.get(brand, all_models)
            return self.fuzzy_fix(
                raw_model,
                valid_values=candidates
            )

        self.df[model_col] = self.df.apply(fix_model, axis=1)
        return self.df

"""
Some entries on our dataframe have the Brand collum with NA but they actually provide a model
What this method does is it identifies what is the brand is and fills it accordingly
"""
def findModel(df, MODEL_BY_BRAND):
    count = 0
    # get indices where Brand is missing but model is not
    model_noBrand_index = df[df["Brand"].isna() & df["model"].notna()].index

    for index in model_noBrand_index:
        model = df.at[index, "model"]
        keys_found = [brand for brand, lst in MODEL_BY_BRAND.items() if model in lst]

        if keys_found:
            df.at[index, "Brand"] = keys_found[0]
            count += 1

    print(f"A total of {count} models without the respective brand were found and filled.")
    return df