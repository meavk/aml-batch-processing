# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import glob
import pandas as pd
import argparse
import mlflow
import time
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_output_folder", type=str)
    args, _ = parser.parse_known_args()

    global job_output_folder
    job_output_folder = args.job_output_folder


def run(input_data, mini_batch_context):
    mlflow.autolog()

    if not isinstance(input_data, pd.DataFrame):
        raise Exception("Not a valid DataFrame input.")
    
    print(f"=========Processing New batch=========+")
    print(f"partition_key_value = {mini_batch_context.partition_key_value}")
    print(f"number of rows = {input_data.shape[0]}")

    batch_suffix = f"{mini_batch_context.partition_key_value}"
    batch_suffix = batch_suffix.replace("{", "").replace("}", "").replace("'", "_").replace(":", "_").replace(" ", "")
    outfile_path = os.path.join(
        job_output_folder,
        f"part_{batch_suffix}.parquet",
    )

    results = process_batch(input_data)
    results.to_parquet(outfile_path, index=False)
    print(f"=========Processed New Batch===========")

    # return array of output file paths
    return [outfile_path]

def shutdown():
    outfile_path = os.path.join(
        job_output_folder,
        f"part_*.parquet",
    )

    # read all parquet files
    all_files = glob.glob(outfile_path)
    combined_df = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)

    consolidatedfile_path = os.path.join(
        job_output_folder,
        "consolidated_results.csv",
    )

    combined_df.to_csv(consolidatedfile_path, index=False)

def replace_general_value(feedback_group):
    """Replace the General in the category

    Args:
        feedback_group: string
    """
    if feedback_group == "General":
        return ""
    else:
        return f'{feedback_group}:'


def apply_data_manipulation(df_survey, user_prompt):
    """Apply data manipulation to the dataframe

    Args:
        df_survey: pd.DataFrame
        user_prompt: string
    """
    df_survey = df_survey.dropna(subset=['FEEDBACK'])
    for index, row in df_survey.iterrows():
        clean_feedback = row["FEEDBACK"].replace("\n", ". ").replace("\r", ". ")
        clean_feedback_with_group = replace_general_value(row["FEEDBACK_TYPE"]) + clean_feedback
        df_survey.loc[
            index, "USER_PROMPT"
        ] = user_prompt.format(clean_feedback_with_group)
        df_survey.loc[index, "COMPLETION"] = None

    return df_survey

system_prompt_for_theme_sentiment_generation = """
System message: As an intelligent AI assistant, your role is to critically analyze the customer reviews given by the COLES customers to generate review category, sentiments & competitor out of the review.
    These reviews are the feedback and suggestion given by the customer when they came to shop in stores.
    Here's how you'll do it:
    - Customer reviews will be presented in the format 'review_group: review'. The review_group is selected by the customer.
    - Your task is to analyze each review, and classify them under pre-defined categories.
    - You will be provided with categories. You have to choose the category which best describes the review.
    - You have to extract sentiment from the review. The sentiment can be "positive", "negative" or "neutral". All in lower case.
    - You have to analyze the reviews, and summarize it and tell the sentiment with respect to the Coles group.
    - You also have to list any other supermarkets which are Coles' competitors like "Woolworths", "ALDI", "Costco" or "IGA". If a competitor is not mentioned, list "NA".
    - Multiple category might apply to single review. In that case, Think step by step, generate the category, sentiment and competitor for each statement.
    - Be very careful while generating multiple category, sentiment and competitor from a single review.
    - Aim for the highest level of accuracy and precision in generating category and sentiments.
    - While analyzing sentiment, make sure to understand the semantic meaning. Think like a human and Try to understand sarcasm, humour while deciding sentiment.
    - Do not explain your reasoning, just make sure that you generate the output in given format.
    - If a review is very generic or does not fall under any pre-defined category, put it under 'General' category.
    - Make sure to not output any category or sentiment as blank.

    Here is the list of themes:

    Checkout Experience
    Customer Parking
    Product Freshness/Shelf Life
    Gift Card
    Insufficient Staff
    Loyalty program/Flybuys
    Own brand
    Packaging
    Product Availability
    Product Pricing
    Product Quality
    Product Variety
    Promotional Event
    Shopping Experience
    Sourcing
    Specials & Discounts
    Staff Behavior
    Store Cleanliness
    Store Congestion // Do not consider long checkout queue under this.
    Store Layout    // How the store is organized, poor visibility etc.
    Store Look and Feel // Store ambience, vibe and environment etc.
    Store Accessibility
    Survey Experience
    Sustainable Shopping
    Trolleys

    The DESIRED OUTPUT FORMAT is given below:
    The format:
    [{'review':[{'category':'theme','sentiment':'sentiment'},{'category':'theme','sentiment':'sentiment'}],'competitor':['competitor1','competitor2']}]
"""

user_prompt_for_theme_sentiment_generation = """
Analyze the reviews and prepare a dictonary of entries in above format by analyzing the reviews:
{}
response:"""

def generate_completion_with_retry(client, user_prompt, retry_count=3):
    completion = "Error"

    i = 0
    response = client.chat.completions.create(
        model="gpt-35-turbo-16k", # engine = "deployment_name".
        temperature=0.7,
        top_p=1,
        messages=[
            {"role": "system", "content": system_prompt_for_theme_sentiment_generation},
            {"role": "user", "content": user_prompt},
        ]
    )

    completion = response.choices[0].message.content

    # Retry loop.
    # for i in range(retry_count):
    #     try:
    #         response = client.chat.completions.create(
    #             model="gpt-35-turbo-16k", # engine = "deployment_name".
    #             temperature=0.7,
    #             top_p=1,
    #             messages=[
    #                 {"role": "system", "content": system_prompt_for_theme_sentiment_generation},
    #                 {"role": "user", "content": user_prompt},
    #             ]
    #         )

    #         completion = response.choices[0].message.content
    #         break
    #     except Exception as e:
    #         # Retry if status code is 429
    #         if e.status_code == 429:
    #             completion = "Ratelimit Error"
    #             # Sleep for 60 seconds, as the quota is getting reset.
    #             time.sleep(60)
    #             print(f"Ratelimit error, retrying for {i} time")
    #         else:
    #             # Print error and break the retry loop
    #             print(f"Error: {e}")
    #             i = retry_count
            
    # return the completion and retry count        
    return (completion, i)

def generate_themes_and_sentiments(df_survey):
    df_survey = apply_data_manipulation(
        df_survey, user_prompt_for_theme_sentiment_generation
    )

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    client = AzureOpenAI(
        api_version="2024-02-15-preview",
        azure_endpoint="https://INSTANCE.openai.azure.com/",
        azure_ad_token_provider=token_provider
    )

    retry_count = 0
    for index, row in df_survey.iterrows():
        start_time = time.time()
        (completion, retry) = generate_completion_with_retry(client, row["USER_PROMPT"])
        end_time = time.time()
        df_survey.loc[index, "COMPLETION"] = completion.replace("\n", "").replace("\r", "")
        df_survey.loc[index, "RETRY"] = retry
        df_survey.loc[index, "START_TIME"] = start_time
        df_survey.loc[index, "END_TIME"] = end_time
        print(f"{row['review_id']} - {retry}")
        retry_count += retry
        if retry_count >= 10:
            break

    remove_columns = ["USER_PROMPT"]
    df_survey.drop(columns=remove_columns, inplace=True)

    return df_survey

def process_batch(input_df):
    return generate_themes_and_sentiments(input_df)

        
