import pandas as pd
import scipy.stats as stats

from typing import Tuple


def drop_duplicated_userid(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop incoherent data from the DataFrame.
    """
    data = data.drop_duplicates(subset="user_id", inplace=True)
    return data


def drop_mismatched_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop mismatched data from the DataFrame.
    """
    df = data[
        (data["group"] == "treatment") & (data["landing_page"] == "old_page")
        | (data["group"] == "control") & (data["landing_page"] == "new_page")
    ]
    return df


def drop_incoherent_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop incoherent data from the DataFrame.
    """
    data = drop_duplicated_userid(data)
    data = drop_mismatched_data(data)
    return data


def split_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Split the data into two groups.
    """
    treatment = data.query("group == 'treatment'")
    control = data.query("group == 'control'")
    return treatment, control


def compute_conversion_rate(data: pd.DataFrame) -> Tuple[float, float]:
    return data[data.converted == 1].shape[0] / data.shape[0]


def compute_z_statistic(treatment: pd.DataFrame, control: pd.DataFrame) -> float:
    """
    Compute the z-statistic of the z-test.
    """
    n_treatment = treatment.shape[0]
    n_control = control.shape[0]
    treatment_rate = compute_conversion_rate(treatment)
    control_rate = compute_conversion_rate(control)

    diff_rate = treatment_rate - control_rate

    p = (n_treatment * treatment_rate + n_control * control_rate) / (n_treatment + n_control)
    pool_stdev = (p * (1 - p) * (1 / n_treatment + 1 / n_control)) ** 0.5

    # z-statistic
    observed_z_score = diff_rate / pool_stdev
    critical_z_score = stats.norm.ppf(0.975)

    return observed_z_score, critical_z_score


if __name__ == "__main__":
    data = pd.read_csv("ab_data.csv")
    print(data.head())
    data = drop_incoherent_data(data)
    print(data.head())

    treatment, control = split_data(data)
    observed_z_score, critical_z_score = compute_z_statistic(treatment, control)
    p_value = 2 * (1 - stats.norm.cdf(observed_z_score))

    print(f"Z-score: {observed_z_score:.2f}")
    print(f"Critical Z-score: {critical_z_score:.2f}")

    if p_value < 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis")
