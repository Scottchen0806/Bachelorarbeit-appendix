import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/Users/chenxinyang/Desktop/bachelor_thesis_appendix/simulated_all_model_results.csv"
df = pd.read_csv(csv_path)

cols = ["Scenario", "Model", "Log-Likelihood", "AIC", "BIC"]
df = df[cols].copy()

scenario_order = ["indifference", "indecisiveness", "experimentation"]
model_order = ["APUM", "ARUM", "Luce", "RUM"]

agg = (
    df.groupby(["Scenario", "Model"], as_index=False)[["Log-Likelihood", "AIC", "BIC"]]
      .mean()
)
agg["Scenario"] = pd.Categorical(agg["Scenario"], categories=scenario_order, ordered=True)
agg["Model"] = pd.Categorical(agg["Model"], categories=model_order, ordered=True)
agg.sort_values(["Scenario", "Model"], inplace=True)

pivot_ll = agg.pivot(index="Scenario", columns="Model", values="Log-Likelihood").reindex(scenario_order)
pivot_aic = agg.pivot(index="Scenario", columns="Model", values="AIC").reindex(scenario_order)
pivot_bic = agg.pivot(index="Scenario", columns="Model", values="BIC").reindex(scenario_order)

name_map = {
    "indifference": "Indifference",
    "indecisiveness": "Indecisiveness",
    "experimentation": "Experimentation"
}

for pivot_df, ylabel, title in [
    (pivot_ll, "Log-Likelihood", "Model Comparison by Log-Likelihood"),
    (pivot_aic, "AIC", "Model Comparison by AIC"),
    (pivot_bic, "BIC", "Model Comparison by BIC")
]:
    ax = pivot_df.plot(kind="bar", rot=0)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Model", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticklabels([name_map.get(s, s) for s in pivot_df.index])
    plt.show()
