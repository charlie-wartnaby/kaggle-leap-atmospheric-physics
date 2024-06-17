# One-off quick hack to see if get better submission score from averaging
# results over catboost data subsets, than got by averaging models over
# the same data to give a single prediction

# Pandas seems better than polars for arithmetic over dataframes
import pandas as pd
import tqdm

submission_files = ["100k_0/submission.csv",
                    "100k_1/submission.csv",
                    "100k_2/submission.csv",
                    "100k_3/submission.csv",
                    "100k_4/submission.csv"]

avg_submission_df = None
num_files_in_avg = 0
for submission_file in tqdm.tqdm(submission_files):
    single_df = pd.read_csv(submission_file)
    num_files_in_avg += 1
    if avg_submission_df is not None:
        # Avoiding loading more than two 4 GB files into memory at any one time
        new_factor = 1.0 / num_files_in_avg
        old_factor = 1.0 - new_factor
        single_df = single_df.loc[:,single_df.columns[1:]] # leaving only numeric columns
        avg_submission_df = (avg_submission_df * old_factor) + (single_df.values * new_factor)
    else:
        test_ids_df       = single_df.loc[:,single_df.columns[:1]] # alphanumeric
        avg_submission_df = single_df.loc[:,single_df.columns[1:]] # rest all numeric

overall_df = pd.concat((test_ids_df,avg_submission_df), axis=1)

# Submission rejected as having too many cols unless suppress index col pandas inserts by default
overall_df.to_csv("averaged_submission.csv", index=False)
