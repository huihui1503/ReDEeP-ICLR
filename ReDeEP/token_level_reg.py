import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import pdb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(description='Script for processing data and models.')
parser.add_argument('--model_name', type=str, required=True, help='llama2-7b or llama2-13b')
parser.add_argument(
    '--dataset', 
    type=str, 
    default="ragtruth", 
    help='ragtruth, dolly'
)

args = parser.parse_args()


def construct_dataframe(response, number):
    # Create a dataframe to hold the combined information

    data_dict = {
        "identifier": [],
        **{f"external_similarity_{k}": [] for k in range(number)},
        **{f"parameter_knowledge_difference_{k}": [] for k in range(number)},
        "hallucination_label": []
    }

    for i, resp in enumerate(response):
        for j in range(len(resp["external_similarity"])):
            data_dict["identifier"].append(f"response_{i}_item_{j}")
            for k in range(number):
                data_dict[f"external_similarity_{k}"].append(resp["external_similarity"][j][k])
                data_dict[f"parameter_knowledge_difference_{k}"].append(resp["parameter_knowledge_difference"][j][k])

            label_list = resp.get("label")
            if label_list is not None:
                data_dict["hallucination_label"].append(1 if len(label_list) > 0 else 0)
            else:
                data_dict["hallucination_label"].append(resp["hallucination_label"][j])

    df = pd.DataFrame(data_dict)

    # print(df["hallucination_label"].value_counts(normalize=True))
    return df


def linear_regression(df):
    # Extract features and labels
    features = df.drop(columns=["identifier", "hallucination_label"])
    labels = df["hallucination_label"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(accuracy)
    print(report)


def calculate_auc_pcc(df, number):
    # Calculate AUC and Pearson correlation for each of the 64 values
    auc_external_similarity = []
    pearson_external_similarity = []

    auc_parameter_knowledge_difference = []
    pearson_parameter_knowledge_difference = []

    for k in range(number):
        # External similarity metrics
        auc_ext = roc_auc_score(1 - df['hallucination_label'], df[f'external_similarity_{k}'])
        pearson_ext, _ = pearsonr(df[f'external_similarity_{k}'], 1 - df['hallucination_label'])
        auc_external_similarity.append((auc_ext, f'external_similarity_{k}'))
        pearson_external_similarity.append((pearson_ext, f'external_similarity_{k}'))

        # Parameter knowledge difference metrics
        auc_param = roc_auc_score(df['hallucination_label'], df[f'parameter_knowledge_difference_{k}'])
        if df[f'parameter_knowledge_difference_{k}'].nunique() == 1:
            print(k)
        pearson_param, _ = pearsonr(df[f'parameter_knowledge_difference_{k}'], df['hallucination_label'])
        auc_parameter_knowledge_difference.append((auc_param, f'parameter_knowledge_difference_{k}'))
        pearson_parameter_knowledge_difference.append((pearson_param, f'parameter_knowledge_difference_{k}'))
    return auc_external_similarity, auc_parameter_knowledge_difference


def calculate_auc_pcc_32_32(df, top_n, top_k, alpha, auc_external_similarity, auc_parameter_knowledge_difference, m=1):
    collect_info = {}
    # Sort by AUC and select the top N features (for example, top 5)
    top_auc_external_similarity = sorted(auc_external_similarity, reverse=True)[:top_n]
    collect_info.update({"select_heads":[sorted_copy_heads[eval(name.split('_')[-1])] for _, name in top_auc_external_similarity]})

    top_auc_parameter_knowledge_difference = sorted(auc_parameter_knowledge_difference, reverse=True)[:top_k]
    if args.model_name == "llama2-13b":
        base_layer = 7
    else:
        base_layer = 0
    collect_info.update({"select_layers": [eval(name.split('_')[-1])+base_layer for _, name in top_auc_parameter_knowledge_difference]})

    # Sum the top N features for each type
    df['external_similarity_sum'] = df[[col for _, col in top_auc_external_similarity]].sum(axis=1)
    df['parameter_knowledge_difference_sum'] = df[[col for _, col in top_auc_parameter_knowledge_difference]].sum(axis=1)

    # Calculate AUC for the summed top N features
    final_auc_external_similarity = roc_auc_score(1 - df['hallucination_label'], df['external_similarity_sum'])
    final_auc_parameter_knowledge_difference = roc_auc_score(df['hallucination_label'], df['parameter_knowledge_difference_sum'])

    # Calculate Pearson correlation for the summed top N features
    final_pearson_external_similarity, _ = pearsonr(df['external_similarity_sum'], 1 - df['hallucination_label'])
    final_pearson_parameter_knowledge_difference, _ = pearsonr(df['parameter_knowledge_difference_sum'], df['hallucination_label'])

    results = {
        "Top N AUC External Similarity": final_auc_external_similarity,
        "Top N AUC Parameter Knowledge Difference": final_auc_parameter_knowledge_difference,
        "Top N Pearson Correlation External Similarity": final_pearson_external_similarity,
        "Top N Pearson Correlation Parameter Knowledge Difference": final_pearson_parameter_knowledge_difference
    }

    scaler = MinMaxScaler()
    # Normalize the columns
    df['external_similarity_sum_normalized'] = scaler.fit_transform(df[['external_similarity_sum']])
    external_similarity_sum_max_value = scaler.data_max_[0]
    external_similarity_sum_min_value = scaler.data_min_[0]
    collect_info.update({
        "head_max_min": [external_similarity_sum_max_value, external_similarity_sum_min_value],
    })
    df['parameter_knowledge_difference_sum_normalized'] = scaler.fit_transform(df[['parameter_knowledge_difference_sum']])
    parameter_knowledge_sum_max_value = scaler.data_max_[0]
    parameter_knowledge_sum_min_value = scaler.data_min_[0]
    collect_info.update({
        "layers_max_min": [parameter_knowledge_sum_max_value, parameter_knowledge_sum_min_value]
    })
    # Subtract the normalized columns
    df['difference_normalized'] = m*df['parameter_knowledge_difference_sum_normalized'] - alpha*df['external_similarity_sum_normalized']

    # Calculate AUC for the difference
    auc_difference_normalized = roc_auc_score(df['hallucination_label'], df['difference_normalized'])
    person_difference_normalized, _ = pearsonr(df['hallucination_label'], df['difference_normalized'])
    results.update({"Normalized Difference AUC": auc_difference_normalized})
    results.update({"Normalized Difference Pearson Correlation": person_difference_normalized})

    # Group by 'identifier' and calculate the sum of 'difference_normalized' and max of 'hallucination_label'
    df['response_group'] = df['identifier'].str.extract(r'(response_\d+)')

    # Group by 'response_group' and calculate the sum of 'difference_normalized' and max of 'hallucination_label'
    grouped_df = df.groupby('response_group').agg(
        difference_normalized_mean=('difference_normalized', 'mean'),
        hallucination_label=('hallucination_label', 'max')
    ).reset_index()

    min_val = grouped_df['difference_normalized_mean'].min()
    max_val = grouped_df['difference_normalized_mean'].max()
    collect_info.update({'final_max_min': [max_val, min_val]})
    # 进行归一化
    grouped_df['difference_normalized_mean_norm'] = (grouped_df['difference_normalized_mean'] - min_val) / (max_val - min_val)


    # Calculate AUC for the grouped means
    auc_difference_normalized = roc_auc_score(grouped_df['hallucination_label'], grouped_df['difference_normalized_mean_norm'])
    person_difference_normalized, _ = pearsonr(grouped_df['hallucination_label'], grouped_df['difference_normalized_mean_norm'])
    preds = (grouped_df['difference_normalized_mean_norm'] > 0.5).astype(int)
    balanced_acc = balanced_accuracy_score(grouped_df['hallucination_label'], preds)
    macro_f1 = f1_score(grouped_df['hallucination_label'], preds, average="macro")


    results.update({"Grouped means AUC": auc_difference_normalized})
    results.update({"Grouped means Pearson Correlation": person_difference_normalized})
    return auc_difference_normalized, person_difference_normalized, balanced_acc, macro_f1

def print_summary(metrics: dict, n_splits: int, n_repeats: int):
    n_splits = n_splits * n_repeats
    print(f"\n=== Summary (mean ± std across {n_splits} splits, {n_repeats} repeats) ===")
    for name, values in metrics.items():
        print(f"{name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

def cross_validate(
    X,
    y,
    i, j, k, m,
    number,
    n_splits=4,
    n_repeats=10,
    random_state=0,
):
    y = np.asarray(y)                      # list of 0/1 -> array
    X_dummy = np.zeros(len(y))             # stratification only needs y

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    metrics = {"auroc": [], "balanced_acc": [], "macro_f1": []}

    for split_num, (train_idx, test_idx) in enumerate(cv.split(X_dummy, y), start=1):
        repeat_num = (split_num - 1) // n_splits + 1
        fold_num = (split_num - 1) % n_splits + 1
        X_train = [X[i] for i in train_idx]     # list of objects
        X_test  = [X[i] for i in test_idx]
        # y_train, y_test = y[train_idx], y[test_idx]

        # --- your fit / predict here ---
        df_train = construct_dataframe(X_train, number)
        auc_external_similarity, auc_parameter_knowledge_difference = calculate_auc_pcc(df_train, number)

        df_test = construct_dataframe(X_test, number)
        auroc, person_difference_normalized, balanced_acc, macro_f1 = calculate_auc_pcc_32_32(df_test, i, j, k, auc_external_similarity, auc_parameter_knowledge_difference, m)

        metrics["auroc"].append(auroc)
        metrics["balanced_acc"].append(balanced_acc)
        metrics["macro_f1"].append(macro_f1)
        print(
            f"[Repeat {repeat_num}/{n_repeats} Fold {fold_num}/{n_splits}] "
            f"AUROC={auroc:.4f}  BalancedAcc={balanced_acc:.4f}  "
            f"MacroF1={macro_f1:.4f}"
        )
    
    print_summary(metrics, n_splits=n_splits, n_repeats=n_repeats)

def evaluate_on_test_data(
    X_train,
    X_test,
    i, j, k, m,
    number):
    df_train = construct_dataframe(X_train, number)
    auc_external_similarity, auc_parameter_knowledge_difference = calculate_auc_pcc(df_train, number)

    df_test = construct_dataframe(X_test, number)
    auroc, person_difference_normalized, balanced_acc, macro_f1 = calculate_auc_pcc_32_32(df_test, i, j, k, auc_external_similarity, auc_parameter_knowledge_difference, m)
    print(
        f"AUROC={auroc:.4f}  BalancedAcc={balanced_acc:.4f}  "
        f"MacroF1={macro_f1:.4f}"
    )

if __name__ == "__main__":
    if args.model_name == "llama2-7b":
        topk_head_path = "./ReDeEP/log/test_llama2_7B/topk_heads.json"
    elif args.model_name == "llama2-13b":
        topk_head_path = "./ReDeEP/log/test_llama2_13B/topk_heads.json"
    elif args.model_name == "llama3-8b":
        topk_head_path =  "./ReDeEP/log/test_llama3_8B/topk_heads.json"
    elif args.model_name == "mistral-7b":
        topk_head_path =  "./ReDeEP/log/test_mistral2_7B/topk_heads.json"
    else:
        print("model name error")
        exit(-1)

    with open(topk_head_path,'r') as f:
        # [(layer, head)...]
        copy_heads = json.load(f)
    sorted_copy_heads = sorted(copy_heads, key=lambda x: (x[0], x[1]))

    if args.model_name == "llama2-7b":
        if args.dataset == "ragtruth":
            data_path = "./ReDeEP/log/test_llama2_7B/llama2_7B_response_v1.json"
        elif args.dataset == "dolly":
            data_path = "./ReDeEP/log/test_llama2_7B/llama2_7B_response_v1_dolly.json"
        elif args.dataset == "hallurag":
            data_path = "./ReDeEP/log/test_llama2_7B/llama2_7B_response_v1_hallurag.json"
        number = 32
    elif args.model_name == "llama2-13b":
        if args.dataset == "ragtruth":
            data_path = "./ReDeEP/log/test_llama2_13B/llama2_13B_response_v1.json"
        elif args.dataset == "dolly":
            data_path = "./ReDeEP/log/test_llama2_13B/llama2_13B_response_v1_dolly.json"
        elif args.dataset == "hallurag":
            data_path = "./ReDeEP/log/test_llama2_13B/llama2_13B_response_v1_hallurag.json"
        number = 32
    elif args.model_name == "llama3-8b":
        if args.dataset == "ragtruth":
            data_path = "./ReDeEP/log/test_llama3_8B/llama3_8B_response_v1.json"
        elif args.dataset == "dolly":
            data_path = "./ReDeEP/log/test_llama3_8B/llama3_8B_response_v1_dolly.json"
        number = 32
    elif args.model_name == "mistral-7b":
        if args.dataset == "ragtruth":
            data_path = "./ReDeEP/log/test_mistral2_7B/mistral2_7B_response_v1.json"
        elif args.dataset == "hallurag":
            data_path = "./ReDeEP/log/test_mistral2_7B/mistral2_7B_response_hallurag.json"
        number = 32
    else:
        print("model name error")
        exit(-1)

    if args.model_name == "llama2-7b":
        if args.dataset == "ragtruth":
            i, j, k, m = 1, 10, 0.2, 1
        elif args.dataset == "dolly":
            i, j , k, m = 4, 3, 0.2, 1
        elif args.dataset == "hallurag":
            i, j, k, m = 1, 10, 0.2, 1

    elif args.model_name == "mistral-7b":
        if args.dataset == "ragtruth":
            i, j, k, m = 1, 10, 0.2, 1
        elif args.dataset == "hallurag":
            i, j, k, m = 1, 10, 0.2, 1

    elif args.model_name == "llama2-13b":
        if args.dataset == "ragtruth":
            i, j, k, m = 2, 17, 0.6, 1
        elif args.dataset == "dolly":
            i, j, k, m = 4, 5, 0.6, 1
        elif args.dataset == "hallurag":
            i, j, k, m = 2, 17, 0.6, 1
        
    elif args.model_name == "llama3-8b":
        if args.dataset == "ragtruth":
            i, j, k, m = 3, 30, 0.4, 1
        elif args.dataset == "dolly":
            i, j, k, m = 1, 1, 0.1, 1
    else:
        print("model name error")
        exit(-1)
    response = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                response.append(json.loads(line))
            
    response_train = [i for i in response if i["split"] == "train"]
    response_test = [i for i in response if i["split"] == "test"]

    if len(response_train) == 0:
        print(f"Run cross validation on {args.model_name} and {args.dataset}")
        labels = [1 if len(i["labels"]) > 0 else 0 for i in response_test]
        cross_validate(
            X=response_test,
            y=labels,
            i=i, j=j, k=k, m=m, number=number,
            n_splits=4,
            n_repeats=10,
            random_state=0,
        )
    else:
        print(f"Run evaluation on {args.model_name} and {args.dataset}")
        evaluate_on_test_data(
            X_train=response_train,
            X_test=response_test,
            i=i, j=j, k=k, m=m, number=number,
        )

    
    # auc_difference_normalized, person_difference_normalized, balanced_acc, macro_f1 = calculate_auc_pcc_32_32(df, i, j, k, auc_external_similarity, auc_parameter_knowledge_difference, m)
    # if args.model_name == "llama2-7b":
    #     save_path = "./ReDeEP/log/test_llama2_7B/ReDeEP(token).json"
    # elif args.model_name == "llama2-13b":
    #     save_path = "./ReDeEP/log/test_llama2_13B/ReDeEP(token).json"
    # elif args.model_name == "llama3-8b":
    #     save_path = "./ReDeEP/log/test_llama3_8B/ReDeEP(token).json"
    # else:
    #     print("model name error")
    #     exit(-1)
    # result_dict = {
    #     "auc":auc_difference_normalized,
    #     "pcc": person_difference_normalized,
    #     "acc": balanced_acc,
    #     "f1": macro_f1,
    #     }
    
    # print(result_dict)
    # with open(save_path, 'w') as f:
    #     json.dump(result_dict, f, ensure_ascii=False)