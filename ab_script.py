import pandas as pd
import scipy.stats as stats


def drop_duplicated_userid(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop incoherent data from the DataFrame.

    Parameters
    ----------
    raw_data : pd.DataFrame
        The DataFrame to clean.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    """
    df: pd.DataFrame = raw_data.drop_duplicates(subset="user_id")
    return df


def drop_mismatched_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop mismatched data from the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to clean.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    """
    df: pd.DataFrame = raw_data[
        (raw_data["group"] == "treatment") & (raw_data["landing_page"] == "old_page")
        | (raw_data["group"] == "control") & (raw_data["landing_page"] == "new_page")
    ]
    return df


def drop_incoherent_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop incoherent data from the DataFrame.
    """
    df: pd.DataFrame = drop_mismatched_data(raw_data)
    cleaned_data: pd.DataFrame = drop_duplicated_userid(df)
    return cleaned_data


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into two groups.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to split.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The treatment and control groups.
    """
    treatment: pd.DataFrame = data.query("group == 'treatment'")
    control: pd.DataFrame = data.query("group == 'control'")
    return treatment, control


def compute_conversion_rate(data: pd.DataFrame) -> tuple[float, float]:
    """
    Compute the conversion rate of the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to compute the conversion rate.

    Returns
    -------
    float
        The conversion rate.

    """
    return data[data.converted == 1].shape[0] / data.shape[0]


def compute_z_statistic(treatment: pd.DataFrame, control: pd.DataFrame) -> tuple[float, float]:
    """
    Compute the z-statistic of the z-test.

    Parameters
    ----------
    treatment : pd.DataFrame
        The DataFrame of the treatment group.
    control : pd.DataFrame
        The DataFrame of the control group.

    Returns
    -------
    tuple[float, float]
        The observed z-score and the critical z-score.
    """
    n_treatment: int = treatment.shape[0]
    n_control: int = control.shape[0]
    treatment_rate: float = compute_conversion_rate(treatment)
    control_rate: float = compute_conversion_rate(control)

    diff_rate: float = treatment_rate - control_rate

    p: float = (n_treatment * treatment_rate + n_control * control_rate) / (n_treatment + n_control)
    pool_stdev: float = (p * (1 - p) * (1 / n_treatment + 1 / n_control)) ** 0.5

    # z-statistic
    observed_z_score: float = diff_rate / pool_stdev
    critical_z_score: float = stats.norm.ppf(0.975)

    return observed_z_score, critical_z_score


if __name__ == "__main__":
    df: pd.DataFrame = pd.read_csv("ab_data.csv")
    cleaned_data: pd.DataFrame = drop_incoherent_data(df)

    treatment, control = split_data(cleaned_data)
    observed_z_score, critical_z_score = compute_z_statistic(treatment, control)
    p_value: float = 2 * (1 - stats.norm.cdf(observed_z_score))

    print(f"Z-score: {observed_z_score:.2f}")
    print(f"Critical Z-score: {critical_z_score:.2f}")

    if p_value < 0.05:
        print(f"Reject the null hypothesis, p-value: {p_value:.2f}")
    else:
        print(f"Fail to reject the null hypothesis, p-value: {p_value:.2f}")
