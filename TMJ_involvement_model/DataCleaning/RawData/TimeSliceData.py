import pandas as pd

class TimeSliceData:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        data['t0'] = pd.to_datetime(data['first_visitation'], errors='coerce')
        data['birth'] = pd.to_datetime(data['birth'], errors='coerce')

        for col in data.columns:
            if '_US' in col or 'first_visitation' in col:
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                    diff_col_name = f'diff_t0_{col}'
                    age_col_name = f'age_at_{col}'
                    mask_t0_notna = data['t0'].notna()
                    mask_birth_notna = data['birth'].notna()
                    mask_col_notna = data[col].notna()
                    mask = mask_t0_notna & mask_col_notna & mask_birth_notna
                    data[diff_col_name] = (data[col] - data['t0']).dt.total_seconds() / (365.25 * 24 * 3600)
                    data[age_col_name] = (data[col] - data['birth']).dt.total_seconds() / (365.25 * 24 * 3600)
                    data[diff_col_name] = data[diff_col_name].where(mask)
                    data[age_col_name] = data[age_col_name].where(mask)
                except pd.errors.ParserError:
                    print(f"Warning: Failed to parse dates in column {col}")

        print("Diff dates calculated for time slicing")

        return data

def filter_visitations(data, time_slice):

    metric = time_slice[0]
    slice_min = time_slice[1]
    slice_max = time_slice[2]

    if metric == "age":
        data = data[(data['ageatvisitation'] >= slice_min) & (data['ageatvisitation'] <= slice_max)]
    elif metric == "diff":
        data = data[(data['difftdate'] >= slice_min) & (data['difftdate'] <= slice_max)]
    else:
        print("ERROR choosing time slicing metric")

    return data.reset_index(drop=False)