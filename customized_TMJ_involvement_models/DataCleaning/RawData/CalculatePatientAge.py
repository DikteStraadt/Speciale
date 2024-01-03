import pandas as pd

class CalculateAge:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        data['birth'] = pd.to_datetime(data['birth'], errors='coerce')

        for col in data.columns:
            if '_US' in col or 'first_visitation' in col:
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                    age_col_name = f'age_at_{col}'
                    mask_birth_notna = data['birth'].notna()
                    mask_col_notna = data[col].notna()
                    mask = mask_col_notna & mask_birth_notna
                    data[age_col_name] = (data[col] - data['birth']).dt.total_seconds() / (365.25 * 24 * 3600)
                    data[age_col_name] = data[age_col_name].where(mask)
                except pd.errors.ParserError:
                    print(f"Warning: Failed to parse dates in column {col}")

        print("Diff dates calculated for time slicing")

        return data