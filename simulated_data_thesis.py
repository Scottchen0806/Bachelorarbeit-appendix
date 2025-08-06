import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
participants = range(1, 5)

# ===============================
# Indifference / Experimentation
# ===============================
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

# Each question has 3 options, and each option includes a quantity and a price (experimental condition).
experimenting_qs = [
    [[100, 30], [100, 35], [100, 32]],
    [[145, 6], [300, 12], [300, 13]],
    [[100, 9], [105, 10.5], [95, 8.5]],
    [[100, 5], [100, 6], [100, 5]],
    [[100, 50], [100, 55], [100, 48]],
    [[100, 5], [100, 4], [100, 3]],
    [[100, 40], [100, 45], [100, 38]],
    [[5, 1500], [6, 1400], [4, 1200]],
    [[60, 60], [20, 20], [30, 40]],
    [[100, 300], [100, 320], [100, 280]],
]

experimenting_innovation_flags = [
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
]

# ===============================
# Indecisiveness
# ===============================
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
    12: [{"brand_known": 1, "camera_strong": 1, "price": 5000},
         {"brand_average": 1, "camera_weak": 1, "price": 4500},
         {"brand_unknown": 1, "camera_strong": 1, "price": 4800}],
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


# ===============================
scaler = StandardScaler()

def simulate_choices(df, beta=None):
    if beta is None:
        beta = [0.2] * len(attribute_pool) + [-1.0]
    data = []
    for qid in df['question'].unique():
        df_q = df[df['question'] == qid].copy()
        X = df_q[attribute_pool + ['price']].values
        utilities = X @ beta + np.random.normal(0, 0.05, 3)
        probs = np.exp(utilities - np.max(utilities))
        probs /= np.sum(probs)
        choice = np.random.choice([0, 1, 2], p=probs)
        df_q['choice'] = 0
        df_q.iloc[choice, df_q.columns.get_loc('choice')] = 1
        data.append(df_q)
    return pd.concat(data, ignore_index=True)

def build_choice_sets(scenario_name, questions, participant_id):
    if scenario_name == "indecisiveness":
        rows = []
        for qid, options in question_data.items():
            for i, opt in enumerate(options):
                row = {attr: opt.get(attr, 0) for attr in attribute_pool}
                row['price'] = opt['price']
                row['question'] = qid
                row['option'] = ['A', 'B', 'C'][i]
                row['participant'] = participant_id
                rows.append(row)
        df = pd.DataFrame(rows)
        df = simulate_choices(df)
        df[[f + '_scaled' for f in attribute_pool + ['price']]] = scaler.fit_transform(df[attribute_pool + ['price']])
        return [g for _, g in df.groupby('question')], [f + '_scaled' for f in attribute_pool + ['price']]
    else:
        rows = []
        for qid, q in enumerate(questions, start=1):
            X = np.array(q)
            utilities = X @ np.array([0.5, -0.5]) + np.random.normal(0, 0.05, 3)
            probs = np.exp(utilities - np.max(utilities))
            probs /= np.sum(probs)
            choice = np.random.choice([0, 1, 2], p=probs)
            for i in range(3):
                rows.append({
                    'feature1': X[i, 0],
                    'feature2': X[i, 1],
                    'choice': 1 if i == choice else 0,
                    'question': qid,
                    'participant': participant_id
                })
        df = pd.DataFrame(rows)
        df[['feature1_scaled', 'feature2_scaled']] = scaler.fit_transform(df[['feature1', 'feature2']])
        return [g for _, g in df.groupby('question')], ['feature1_scaled', 'feature2_scaled']

# ===============================
# Model definition
# ===============================
def get_model_funcs(scaled_features):
    def luce_ll(beta, sets):
        ll = 0
        for g in sets:
            X = g[scaled_features].values
            u = X @ beta
            u = u - np.max(u)
            try:
                ll += u[g['choice'] == 1][0] - np.log(np.sum(np.exp(u)))
            except:
                return np.inf
        return -ll

    def rum_ll(beta, sets):
        ll = 0
        for g in sets:
            X = g[scaled_features].values
            u = X @ beta / 0.01
            probs = np.exp(u - np.max(u))
            probs /= np.sum(probs)
            chosen = g['choice'].values.argmax()
            ll += np.log(probs[chosen] + 1e-8)
        return -ll

    def arum_ll(beta, sets):
        ll = 0
        for g in sets:
            X = g[scaled_features].values
            u = X @ beta
            probs = np.exp(u - np.max(u))
            probs /= np.sum(probs)
            chosen = g['choice'].values.argmax()
            ll += np.log(probs[chosen])
        return -ll

    def apum_ll(beta, sets, lam=0.1):
        ll = 0
        for g in sets:
            X = g[scaled_features].values
            u = X @ beta
            probs = np.exp(u - np.max(u))
            probs /= np.sum(probs)
            chosen = g['choice'].values.argmax()
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            ll += np.log(probs[chosen]) + lam * entropy
        return -ll

    return luce_ll, rum_ll, arum_ll, apum_ll


def fit_models(scenario_name, sets, scaled_features, participant_id):
    results = []
    luce_ll, rum_ll, arum_ll, apum_ll = get_model_funcs(scaled_features)
    funcs = {
        'Luce': (luce_ll, (sets,)),
        'RUM': (rum_ll, (sets,)),
        'ARUM': (arum_ll, (sets,)),
        'APUM': (apum_ll, (sets, 0.1))
    }
    for name, (func, args) in funcs.items():
        try:
            res = minimize(func, x0=np.random.randn(len(scaled_features)), args=args, method='BFGS')
            if not res.success or np.isnan(res.fun):
                raise ValueError("Optimization failed")
            ll = -res.fun
            k = len(res.x)
            n = len(sets)
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
            beta = res.x[:5].round(4).tolist()
        except Exception as e:
            ll, aic, bic, beta = np.nan, np.nan, np.nan, np.nan
        results.append({
            'Participant': participant_id,
            'Scenario': scenario_name,
            'Model': name,
            'Log-Likelihood': ll,
            'AIC': aic,
            'BIC': bic,
            'Beta': beta
        })
    return results

# ===============================
# Main execution process: Each participant goes through 3 scenarios × 4 models.
# ===============================
scenarios = {
    "indifference": indifference_qs,
    "experimentation": experimenting_qs,
    "indecisiveness": question_data
}

all_results = []

for pid in participants:
    for scenario_name, questions in scenarios.items():
        choice_sets, features = build_choice_sets(scenario_name, questions, participant_id=pid)
        results = fit_models(scenario_name, choice_sets, features, participant_id=pid)
        all_results.extend(results)

final_df = pd.DataFrame(all_results)
print(final_df.to_string(index=False))
