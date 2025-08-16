import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)
participants = range(0, 30)
choices = np.random.choice(['A', 'B', 'C'], size=(len(participants), 30))
df = pd.DataFrame(choices, columns=[f"Q{i}" for i in range(1, 31)])
df.insert(0, "ID", list(participants))  


def choice_to_index(choice_str):
    if isinstance(choice_str, str):
        choice_str = choice_str.strip().upper()
        if choice_str.startswith("A"):
            return 0
        elif choice_str.startswith("B"):
            return 1
        elif choice_str.startswith("C"):
            return 2
    return np.nan

scaler = StandardScaler()

indifference_qs = [
    [[50, 2.00], [51, 2.01], [49, 1.99]],
    [[500, 1.00], [505, 1.01], [495, 0.99]],
    [[1000, 2.00], [1010, 2.02], [990, 1.98]],
    [[200, 3.00], [202, 3.03], [198, 2.97]],
    [[100, 3.00], [102, 3.06], [98, 2.94]],
    [[700, 7.00], [705, 7.05], [695, 6.95]],
    [[10000, 20.00], [10100, 20.20], [9900, 19.80]],
    [[100, 1.00], [102, 1.02], [98, 0.98]],
    [[50, 2.50], [51, 2.55], [49, 2.45]],
    [[300, 15.00], [305, 15.25], [295, 14.75]],
]

def build_indifference_sets(df, participant_id):
    rows = []
    answers = df.iloc[participant_id, 1:11]  # Q1–Q10
    for i, answer in enumerate(answers):
        choice_idx = choice_to_index(answer)
        for j, (qty, price) in enumerate(indifference_qs[i]):
            rows.append({
                "quantity": qty,
                "price": price,
                "choice": 1 if j == choice_idx else 0,
                "question": i + 1,
                "participant": participant_id + 1
            })
    data = pd.DataFrame(rows)
    data[["quantity_scaled", "price_scaled"]] = scaler.fit_transform(data[["quantity", "price"]])
    return [g for _, g in data.groupby("question")], ["quantity_scaled", "price_scaled"]

experimenting_qs = [
    [[0, 30], [1, 35], [0, 32]],
    [[0, 6], [1, 12], [0, 13]],
    [[0, 9], [1, 10.5], [0, 8.5]],
    [[0, 5], [1, 6], [0, 5]],
    [[0, 50], [1, 55], [0, 48]],
    [[0, 5], [1, 4], [0, 3]],
    [[0, 40], [1, 45], [0, 38]],
    [[0, 1500], [1, 1400], [0, 1200]],
    [[0, 60], [1, 20], [0, 40]],
    [[0, 300], [1, 320], [0, 280]],
]

def build_experimentation_sets(df, participant_id):
    rows = []
    answers = df.iloc[participant_id, 21:31]  # Q21–Q30
    for i, answer in enumerate(answers):
        choice_idx = choice_to_index(answer)
        for j, (feat1, feat2) in enumerate(experimenting_qs[i]):
            rows.append({
                "feature1": feat1,
                "feature2": feat2,
                "choice": 1 if j == choice_idx else 0,
                "question": i + 1,
                "participant": participant_id + 1
            })
    data = pd.DataFrame(rows)
    data[["feature1_scaled", "feature2_scaled"]] = scaler.fit_transform(data[["feature1", "feature2"]])
    return [g for _, g in data.groupby("question")], ["feature1_scaled", "feature2_scaled"]

attribute_pool = [
    "brand_known", "brand_average", "brand_unknown",
    "battery_known", "battery_unknown",
    "service_known", "service_unknown",
    "camera_strong", "camera_weak",
    "flight_included", "flight_not_included",
    "hotel_high_quality", "hotel_unknown",
    "includes_hospital_visits", "outpatient_unclear", "coverage_incomplete",
    "full_coverage", "high_deductible",
    "famous_professor", "new_professor", "experienced_professor",
    "grading_policy_unknown", "syllabus_available", "graduation_rate_unclear",
    "insurance_unclear", "insurance_included", "partial_insurance",
    "fuel_policy_unclear", "full_info_provided",
    "great_location", "amenities_unclear", "amenities_full", "far_from_downtown",
    "close_to_downtown", "customer_service_unknown",
    "no_annual_fee", "cashback_unclear", "clear_cashback", "reward_policy_partial",
    "live_classes", "on_demand_unclear", "on_demand_detailed", "features_unspecified",
    "brand_new", "battery_unclear", "older_model", "full_specs",
    "stylish_design", "compatibility_unclear"
]

question_data = {
    11: [{"brand_known": 1, "battery_unknown": 1, "price": 1000},
         {"brand_average": 1, "battery_known": 1, "price": 1200},
         {"brand_unknown": 1, "battery_known": 1, "service_unknown": 1, "price": 800}],
    12: [{"brand_known": 1, "camera_strong": 1, "price": 500},
         {"brand_average": 1, "camera_weak": 1, "price": 450},
         {"brand_unknown": 1, "camera_strong": 1, "price": 480}],
    13: [{"flight_included": 1, "hotel_high_quality": 1, "price": 3000},
         {"flight_not_included": 1, "hotel_unknown": 1, "price": 3500},
         {"flight_included": 1, "hotel_unknown": 1, "price": 2800}],
    14: [{"includes_hospital_visits": 1, "outpatient_unclear": 1, "price": 300},
         {"full_coverage": 1, "high_deductible": 1, "price": 250},
         {"coverage_incomplete": 1, "price": 280}],
    15: [{"famous_professor": 1, "grading_policy_unknown": 1, "price": 500},
         {"new_professor": 1, "syllabus_available": 1, "price": 450},
         {"experienced_professor": 1, "graduation_rate_unclear": 1, "price": 480}],
    16: [{"insurance_unclear": 1, "price": 50},
         {"insurance_included": 1, "fuel_policy_unclear": 1, "price": 55},
         {"partial_insurance": 1, "full_info_provided": 1, "price": 45}],
    17: [{"great_location": 1, "amenities_unclear": 1, "price": 120},
         {"amenities_full": 1, "far_from_downtown": 1, "price": 130},
         {"close_to_downtown": 1, "customer_service_unknown": 1, "price": 110}],
    18: [{"no_annual_fee": 1, "cashback_unclear": 1, "price": 0},
         {"clear_cashback": 1, "price": 50},
         {"reward_policy_partial": 1, "price": 30}],
    19: [{"live_classes": 1, "on_demand_unclear": 1, "price": 100},
         {"on_demand_detailed": 1, "price": 90},
         {"features_unspecified": 1, "price": 95}],
    20: [{"brand_new": 1, "battery_unclear": 1, "price": 300},
         {"older_model": 1, "full_specs": 1, "price": 280},
         {"stylish_design": 1, "compatibility_unclear": 1, "price": 320}],
}


def build_indecisiveness_sets(df, participant_id):
    rows = []
    answers = df.iloc[participant_id, 11:21]  # Q11–Q20
    for i, qid in enumerate(range(11, 21)):
        choice_idx = choice_to_index(answers.iloc[i])
        for j, opt in enumerate(question_data[qid]):
            row = {attr: opt.get(attr, 0) for attr in attribute_pool}
            row["price"] = opt["price"]
            row["question"] = qid
            row["choice"] = 1 if j == choice_idx else 0
            row["participant"] = participant_id + 1
            rows.append(row)
    data = pd.DataFrame(rows)
    data[[f + "_scaled" for f in attribute_pool + ["price"]]] = scaler.fit_transform(data[attribute_pool + ["price"]])
    return [g for _, g in data.groupby("question")], [f + "_scaled" for f in attribute_pool + ["price"]]

def get_model_funcs(scaled_features):
    def luce_ll(beta, sets):
        ll = 0
        for g in sets:
            X = g[scaled_features].values
            u = X @ beta
            u -= np.max(u)
            try:
                ll += u[g['choice'] == 1][0] - np.log(np.sum(np.exp(u)))
            except:
                return np.inf
        return -ll
    def rum_ll(beta, sets):
        return -sum(np.log((np.exp((X := g[scaled_features].values @ beta / 0.01) - np.max(X)) / np.sum(np.exp(X - np.max(X))))[g['choice'].values.argmax()] + 1e-8) for g in sets)
    def arum_ll(beta, sets):
        return -sum(np.log((np.exp((X := g[scaled_features].values @ beta) - np.max(X)) / np.sum(np.exp(X - np.max(X))))[g['choice'].values.argmax()]) for g in sets)
    def apum_ll(beta, sets, lam=0.1):
        return -sum((np.log((p := np.exp((X := g[scaled_features].values @ beta) - np.max(X)) / np.sum(np.exp(X - np.max(X))))[g['choice'].values.argmax()]) + lam * -np.sum(p * np.log(p + 1e-8))) for g in sets)
    return luce_ll, rum_ll, arum_ll, apum_ll

def fit_models(choice_sets, features, participant_id, scenario_name):
    results = []
    luce_ll, rum_ll, arum_ll, apum_ll = get_model_funcs(features)
    models = {
        "Luce": (luce_ll, (choice_sets,)),
        "RUM": (rum_ll, (choice_sets,)),
        "ARUM": (arum_ll, (choice_sets,)),
        "APUM": (apum_ll, (choice_sets, 0.1)),
    }
    for name, (func, args) in models.items():
        try:
            res = minimize(func, np.random.randn(len(features)), args=args, method='BFGS')
            if not res.success or np.isnan(res.fun):
                raise ValueError("fail")
            ll = -res.fun
            k = len(res.x)
            n = len(choice_sets)
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
            beta = res.x.round(4).tolist()
        except:
            ll, aic, bic = np.nan, np.nan, np.nan
            beta = [np.nan] * len(features)

        results.append({
            "Participant": participant_id + 1,
            "Scenario": scenario_name,
            "Model": name,
            "Log-Likelihood": ll,
            "AIC": aic,
            "BIC": bic,
            "Beta": beta
        })
    return results
    
all_results = []

for pid in range(df.shape[0]):
    for scenario, builder in {
        "indifference": build_indifference_sets,
        "indecisiveness": build_indecisiveness_sets,
        "experimentation": build_experimentation_sets
    }.items():
        sets, feats = builder(df, pid)
        all_results.extend(fit_models(sets, feats, pid, scenario))

final_df = pd.DataFrame(all_results)

beta_records = []

for idx, row in final_df.iterrows():
    beta_list = row["Beta"]
    if isinstance(beta_list, list):
        for i, val in enumerate(beta_list):
            beta_records.append({
                "Participant": row["Participant"],
                "Scenario": row.get("Scenario", ""),
                "Model": row["Model"],
                "Log-Likelihood": row["Log-Likelihood"],
                "AIC": row["AIC"],
                "BIC": row["BIC"],
                "Beta_Index": f"Beta_{i+1}",
                "Beta_Value": val
            })

tidy_df = pd.DataFrame(beta_records)

print(tidy_df.to_string(index=False))
