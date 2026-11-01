"""Exploratory analysis of the training set: summary tables and figures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from credit_scoring.config import Config, load_config
from credit_scoring.plotting import INK, SURFACE, diverging_cmap, figure, figure_path, series_color

# real values stop at 17 and nothing appears until 96. Thus, the codes 96 and 98 are not counts
DELINQUENCY_CODES = (96, 98)


def save_table(table: pd.DataFrame, name: str, config: Config | None = None) -> Path:
    """Write a table to the reports directory as CSV and return the path."""
    cfg = config or load_config()
    directory = cfg.paths.absolute(cfg.paths.tables_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    table.to_csv(path, index=False)
    return path


def structure_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return dtype, completeness and cardinality for every column."""
    return pd.DataFrame(
        {
            "feature": frame.columns,
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "non_null": frame.notna().sum().to_numpy(),
            "missing": frame.isna().sum().to_numpy(),
            "missing_pct": (frame.isna().mean() * 100).round(2).to_numpy(),
            "unique": frame.nunique().to_numpy(),
        }
    )


def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return count, mean, spread and quartiles, one row per feature."""
    return frame.describe().T.round(4).reset_index(names="feature")


def missing_per_row(frame: pd.DataFrame) -> pd.DataFrame:
    """Return how many rows carry 0, 1, 2 ... missing values at the same time.

    Missing values per column do not say whether they overlap. This does: if the
    count of rows with two gaps equals the gaps in one column, that column is
    never missing on its own.
    """
    counts = frame.isna().sum(axis=1).value_counts().sort_index()
    return pd.DataFrame(
        {
            "missing_in_row": counts.index,
            "rows": counts.to_numpy(),
            "rows_pct": (counts / len(frame) * 100).round(2).to_numpy(),
        }
    )


def outlier_report(frame: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Compare what Tukey's rule would flag against the treatment actually used.

    Tukey assumes a roughly symmetric distribution. Financial ratios have heavy
    right tails that carry real signal, so the comparison is what justifies
    winsorising at the extreme quantiles instead.
    """
    cfg = config or load_config()
    lower_q = cfg.preprocessing.winsorize.lower_quantile
    upper_q = cfg.preprocessing.winsorize.upper_quantile
    age_column = cfg.preprocessing.age_column

    rows = []
    for column in [*cfg.preprocessing.winsorize.columns, age_column]:
        if column not in frame.columns:
            continue
        values = frame[column].dropna()

        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        tukey_low, tukey_high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        tukey_n = int(((values < tukey_low) | (values > tukey_high)).sum())

        if column == age_column:
            treatment = "domain clip"
            applied_low = float(cfg.preprocessing.age_bounds.min)
            applied_high = float(cfg.preprocessing.age_bounds.max)
        else:
            treatment = f"winsorise [{lower_q:.0%}, {upper_q:.0%}]"
            applied_low = float(values.quantile(lower_q))
            applied_high = float(values.quantile(upper_q))
        applied_n = int(((values < applied_low) | (values > applied_high)).sum())

        rows.append(
            {
                "feature": column,
                "n_values": len(values),
                "tukey_lower": round(float(tukey_low), 2),
                "tukey_upper": round(float(tukey_high), 2),
                "tukey_flagged": tukey_n,
                "tukey_pct": round(100 * tukey_n / len(values), 2),
                "treatment": treatment,
                "applied_lower": round(applied_low, 4),
                "applied_upper": round(applied_high, 2),
                "applied_flagged": applied_n,
                "applied_pct": round(100 * applied_n / len(values), 2),
            }
        )
    return pd.DataFrame(rows)


def implausible_value_report(frame: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Count values that domain knowledge rules out, such as an age of zero."""
    cfg = config or load_config()
    age = cfg.preprocessing.age_column
    low, high = cfg.preprocessing.age_bounds.min, cfg.preprocessing.age_bounds.max

    rows = []
    if age in frame.columns:
        rows += [
            {"check": f"{age} == 0", "rows": int((frame[age] == 0).sum())},
            {"check": f"{age} < {low}", "rows": int((frame[age] < low).sum())},
            {"check": f"{age} > {high}", "rows": int((frame[age] > high).sum())},
        ]
    for column in cfg.preprocessing.delinquency.columns:
        if column in frame.columns:
            flagged = int(frame[column].isin(DELINQUENCY_CODES).sum())
            rows.append({"check": f"{column} in {DELINQUENCY_CODES}", "rows": flagged})

    table = pd.DataFrame(rows, columns=["check", "rows"])
    table["rows_pct"] = (table["rows"] / len(frame) * 100).round(4)
    return table


def counter_value_counts(frame: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Return how often every distinct value occurs in the delinquency counters.

    A counter runs upwards from zero without gaps. Listing the frequencies makes
    the break visible: they thin out, stop, and then two isolated values reappear
    far above the rest, which is what a code looks like and not what a count
    looks like.
    """
    cfg = config or load_config()
    columns = [c for c in cfg.preprocessing.delinquency.columns if c in frame.columns]
    table = pd.DataFrame({column: frame[column].value_counts() for column in columns})
    return table.fillna(0).astype(int).sort_index().reset_index(names="value")


def default_rate_by_group(
    frame: pd.DataFrame,
    config: Config | None = None,
    utilisation_column: str = "RevolvingUtilizationOfUnsecuredLines",
    utilisation_thresholds: Sequence[float] = (1.0, 10.0),
) -> pd.DataFrame:
    """Compare the default rate inside each suspect subgroup against the whole training set.

    Counting anomalies says how many there are, not whether they matter. A
    subgroup that defaults several times more often than the base rate carries
    signal and needs a flag of its own; one that matches the base rate is noise
    and can be cleaned away.
    """
    cfg = config or load_config()
    target = frame[cfg.data.target]
    baseline = target.mean()

    groups = {
        f"{column} missing": frame[column].isna()
        for column in frame.columns
        if frame[column].isna().any()
    }
    delinquency = [c for c in cfg.preprocessing.delinquency.columns if c in frame.columns]
    if delinquency:
        # One flag for the row, because the codes appear in all three counters together.
        groups[f"delinquency code in {DELINQUENCY_CODES}"] = (
            frame[delinquency].isin(DELINQUENCY_CODES).any(axis=1)
        )
    age = cfg.preprocessing.age_column
    if age in frame.columns:
        groups[f"{age} == 0"] = frame[age] == 0
    if utilisation_column in frame.columns:
        for limit in utilisation_thresholds:
            # A utilisation ratio reports drawn credit over the limit, so anything
            # above 1 is already unusual and anything far above it is not a ratio.
            groups[f"{utilisation_column} > {limit:g}"] = frame[utilisation_column] > limit

    rows = [
        {
            "group": "all applicants",
            "rows": len(frame),
            "rows_pct": 100.0,
            "default_rate": baseline,
            "lift": 1.0,
        }
    ]
    rows += [
        {
            "group": name,
            "rows": int(mask.sum()),
            "rows_pct": mask.mean() * 100,
            "default_rate": target[mask].mean(),
            # Times more often than the portfolio: 1.0 means the subgroup is ordinary.
            "lift": target[mask].mean() / baseline,
        }
        for name, mask in groups.items()
        if mask.any()
    ]
    return pd.DataFrame(rows)


def default_rate_by_band(
    frame: pd.DataFrame,
    column: str,
    edges: Sequence[float],
    config: Config | None = None,
) -> pd.DataFrame:
    """Return the default rate inside mutually exclusive value bands of one column.

    Cumulative thresholds hide where behaviour changes, because each one also
    contains every band above it. Exclusive bands show the point at which the
    default rate stops rising, which is where a genuine signal ends and
    corrupted values begin.
    """
    cfg = config or load_config()
    target = frame[cfg.data.target]
    # include_lowest keeps the zeros, which would otherwise fall outside every band.
    band = pd.cut(frame[column], bins=list(edges), include_lowest=True)

    table = (
        target.groupby(band, observed=False)
        .agg(rows="size", default_rate="mean")
        .reset_index()
        .rename(columns={column: "band"})
    )
    table["rows_pct"] = table["rows"] / len(frame) * 100
    table["lift"] = table["default_rate"] / target.mean()
    return table[["band", "rows", "rows_pct", "default_rate", "lift"]]


def debt_ratio_by_income_missingness(
    frame: pd.DataFrame, ratio_column: str = "DebtRatio", income_column: str = "MonthlyIncome"
) -> pd.DataFrame:
    """Describe the debt ratio separately for rows that do and do not report an income.

    A ratio and an absolute amount cannot share one column. If the two groups
    differ by orders of magnitude, the column is not measuring the same quantity
    in both, and imputing the income afterwards cannot repair it.
    """
    grouped = frame.groupby(frame[income_column].isna())[ratio_column]
    table = grouped.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    table = table.rename(index={False: "income reported", True: "income missing"})
    return table.reset_index(names="group")


def duplicate_report(frame: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Count repeated feature vectors and show why they repeat.

    Pandas treats two missing values as equal, so rows can match purely because
    income is absent in both. The last row separates those from true repeats.
    """
    cfg = config or load_config()
    target = cfg.data.target
    features = [column for column in frame.columns if column != target]

    repeated = frame[frame.duplicated(subset=features, keep=False)]
    grouped = repeated.groupby(features, dropna=False)[target]

    rows = [
        {"check": "rows repeating an earlier row", "count": len(repeated), "unit": "rows"},
        {"check": "distinct repeated vectors", "count": int(grouped.ngroups), "unit": "groups"},
        {
            "check": "vectors disagreeing on the target",
            "count": int(grouped.nunique().gt(1).sum()),
            "unit": "groups",
        },
        {
            "check": "repeats caused by a missing value",
            "count": int(repeated[features].isna().any(axis=1).sum()),
            "unit": "rows",
        },
    ]
    return pd.DataFrame(rows)


def correlation_with_target(frame: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Return each feature's linear correlation with the target, strongest first."""
    cfg = config or load_config()
    correlations = frame.corr(numeric_only=True)[cfg.data.target].drop(cfg.data.target)
    ordered = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    return pd.DataFrame({"feature": ordered.index, "correlation": ordered.round(4).to_numpy()})


def plot_target_balance(frame: pd.DataFrame, config: Config | None = None) -> Path:
    """Plot how many applicants fall in each class."""
    cfg = config or load_config()
    counts = frame[cfg.data.target].value_counts().sort_index()
    shares = counts / counts.sum() * 100

    with figure(
        "target_balance",
        title="Target class balance",
        xlabel=cfg.data.target,
        ylabel="Applicants",
        figsize=(6, 4.5),
        config=cfg,
    ) as ax:
        bars = ax.bar(
            [str(label) for label in counts.index],
            counts.to_numpy(),
            color=[series_color(0), series_color(1)],
            width=0.6,
        )
        for bar, share in zip(bars, shares, strict=True):
            ax.annotate(
                f"{share:.2f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                color=INK,
            )
        ax.margins(y=0.12)

    return figure_path("target_balance", cfg)


def plot_distributions(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    trim_quantile: float | None = None,
    config: Config | None = None,
) -> list[Path]:
    """Plot a histogram per column, optionally hiding the far tail.

    Trimming drops the values above the quantile instead of capping them:
    capping would pile them into the last bin and invent a spike that the data
    does not have. The title states that values were left out.
    """
    cfg = config or load_config()
    paths = []

    for column in columns:
        values = frame[column].dropna()
        title = f"Distribution of {column}"
        if trim_quantile is not None:
            cutoff = values.quantile(trim_quantile)
            hidden = int((values > cutoff).sum())
            values = values[values <= cutoff]
            title += f" (top {hidden:,} values above the {trim_quantile:.0%} quantile not shown)"

        name = f"{column}_histogram" + ("_trimmed" if trim_quantile else "")
        with figure(
            name, title=title, xlabel=column, ylabel="Applicants", figsize=(7, 4.5), config=cfg
        ) as ax:
            ax.hist(values, bins=40, color=series_color(0), edgecolor=SURFACE, linewidth=0.4)
        paths.append(figure_path(name, cfg))

    return paths


def plot_boxplots(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    zoom_quantile: float | None = None,
    config: Config | None = None,
) -> list[Path]:
    """Plot each column split by target class.

    Zooming limits the axis but leaves the data intact, so the quartiles and the
    median are still computed from every observation.
    """
    cfg = config or load_config()
    target = cfg.data.target
    labels = sorted(frame[target].unique())
    paths = []

    for column in columns:
        groups = [frame.loc[frame[target] == label, column].dropna() for label in labels]
        title = f"{column} by target class"
        if zoom_quantile is not None:
            title += f" (axis zoomed to the {zoom_quantile:.0%} quantile)"

        name = f"{column}_boxplot"
        with figure(
            name, title=title, xlabel=target, ylabel=column, figsize=(6.5, 4.5), config=cfg
        ) as ax:
            artists = ax.boxplot(
                groups, tick_labels=[str(label) for label in labels], patch_artist=True, widths=0.5
            )
            for index, box in enumerate(artists["boxes"]):
                box.set_facecolor(series_color(index))
                box.set_alpha(0.75)
                box.set_edgecolor(INK)
            for element in ("whiskers", "caps", "medians"):
                for artist in artists[element]:
                    artist.set_color(INK)
            for flier in artists["fliers"]:
                flier.set(marker="o", markersize=2.5, markerfacecolor="none", alpha=0.4)
            if zoom_quantile is not None:
                # Both limits are set: leaving the lower one to matplotlib lets
                # the far outliers stretch the axis downwards and flatten the
                # boxes into invisible lines.
                values = frame[column].dropna()
                bottom, top = values.min(), values.quantile(zoom_quantile)
                margin = 0.05 * (top - bottom)
                ax.set_ylim(bottom - margin, top + margin)
        paths.append(figure_path(name, cfg))

    return paths


def plot_correlation_matrix(frame: pd.DataFrame, config: Config | None = None) -> Path:
    """Plot the lower triangle of the correlation matrix."""
    cfg = config or load_config()
    matrix = frame.corr(numeric_only=True).round(4)
    labels = matrix.columns.tolist()

    # The matrix is symmetric, so the upper triangle repeats the lower one.
    # Dropping the first row and last column removes the labels that would then
    # sit beside an empty strip.
    hidden = np.triu(np.ones(matrix.shape, dtype=bool))[1:, :-1]
    values = matrix.to_numpy()[1:, :-1]
    row_labels, column_labels = labels[1:], labels[:-1]

    colormap = diverging_cmap()
    colormap.set_bad(SURFACE)

    with figure(
        "correlation_matrix", title="Correlation matrix", figsize=(9, 7.5), config=cfg
    ) as ax:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        image = ax.imshow(np.ma.masked_where(hidden, values), cmap=colormap, vmin=-1, vmax=1)

        ax.set_xticks(range(len(column_labels)), column_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)

        for row in range(len(row_labels)):
            for column in range(len(column_labels)):
                if hidden[row, column]:
                    continue
                value = values[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=SURFACE if abs(value) > 0.55 else INK,
                )

        colorbar = ax.figure.colorbar(image, ax=ax, shrink=0.8)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(labelsize=8)

    return figure_path("correlation_matrix", cfg)
