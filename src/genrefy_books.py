import google.generativeai as genai
import pandas as pd
import os
import json
import time
import logging
from tqdm import tqdm
from src.utils import gemini_helper

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-pro-preview-05-06" # Or "gemini-1.0-pro" or other suitable model
BATCH_SIZE = 500  # Number of books to process in a single API call (adjust based on token limits and performance)
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5 # Time to wait before retrying a failed API call

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def classify_book_batch_with_gemini(books_batch_data):
    """
    Sends a batch of books to Gemini for classification.

    Args:
        books_batch_data (list): A list of dictionaries, where each dictionary
                                 has "item_id", "title", and "author".

    Returns:
        list: A list of dictionaries with "item_id" and "genre", or None if an error occurs.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    generation_config = genai.types.GenerationConfig(
        # Only one candidate for straightforward classification
        candidate_count=1,
        # Ensure JSON output if model supports it (gemini-1.5-flash does)
        response_mime_type="application/json",
        temperature=0.1 # Lower temperature for more deterministic classification
    )

    books_json_string = json.dumps(books_batch_data, indent=2)
    prompt = gemini_helper.classifier_prompt(books_json_string=books_json_string)

    for attempt in range(MAX_RETRIES):
        try:
            logging.debug(f"Sending prompt for batch (attempt {attempt+1}/{MAX_RETRIES}):\n{prompt[:500]}...") # Log snippet
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )

            if response.parts:
                # Clean up the response text if it's wrapped in markdown ```json ... ```
                raw_json = response.text

                classified_data = json.loads(gemini_helper.clean_result_text(raw_json))
                # Basic validation
                if isinstance(classified_data, list) and all(
                    isinstance(item, dict) and "item_id" in item and "genre" in item
                    for item in classified_data
                ):
                    
                    for item in classified_data:
                        if item["genre"] not in gemini_helper.VALID_GENRES_SET:
                            logging.warning(f"Item ID {item['item_id']} received unexpected genre '{item['genre']}'. Setting to '{gemini_helper.FALLBACK_UNKNOWN}'.")
                            item["genre"] = gemini_helper.FALLBACK_UNKNOWN # Correct invalid genres
                    return classified_data
                else:
                    logging.error(f"Unexpected JSON structure from Gemini: {classified_data}")
                    return None # Or raise an error
            else:
                logging.warning(f"No parts in response for batch. Prompt feedback: {response.prompt_feedback}")
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     logging.error(f"Request blocked due to: {response.prompt_feedback.block_reason_message}")
                # If blocked, retrying might not help, consider failing the batch.
                # For now, we'll retry as per general retry logic.
                # return None # If blocked, likely won't succeed on retry with same content.

        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError for batch: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}")
        except Exception as e:
            logging.error(f"Error during Gemini API call for batch (attempt {attempt+1}): {e}")

        if attempt < MAX_RETRIES - 1:
            logging.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
            time.sleep(RETRY_DELAY_SECONDS)
        else:
            logging.error(f"Max retries reached for a batch. Skipping this batch.")
    return None

def classify_dataframe(df):
    """
    Classifies books in a DataFrame using Gemini.

    Args:
        df (pd.DataFrame): DataFrame with columns "Item ID", "Title", "Author".

    Returns:
        pd.DataFrame: Original DataFrame with an added "Genre" column.
    """
    if not all(col in df.columns for col in ["Item ID", "Title", "Author"]):
        raise ValueError("DataFrame must contain 'Item ID', 'Title', and 'Author' columns.")

    all_classified_books = []
    num_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(df), BATCH_SIZE), total=num_batches, desc="Classifying Batches"):
        batch_df = df.iloc[i:i+BATCH_SIZE]
        books_to_classify = []
        for _, row in batch_df.iterrows():
            books_to_classify.append({
                "item_id": str(row["Item ID"]), # Ensure item_id is string for JSON
                "title": row["Title"],
                "author": row["Author"]
            })

        logging.info(f"Processing batch {i//BATCH_SIZE + 1}/{num_batches} with {len(books_to_classify)} books.")
        classified_batch = classify_book_batch_with_gemini(books_to_classify)

        if classified_batch:
            all_classified_books.extend(classified_batch)
        else:
            # If a batch fails, mark its items as Unknown
            logging.warning(f"Batch {i//BATCH_SIZE + 1} failed. Marking items as '{gemini_helper.FALLBACK_UNKNOWN}'.")
            for book_input in books_to_classify:
                all_classified_books.append({"item_id": book_input["item_id"], "genre": gemini_helper.FALLBACK_UNKNOWN})
        
        # Optional: add a small delay between batches to be kind to the API
        time.sleep(1) # 1 second delay

    # Create a DataFrame from the results
    if not all_classified_books:
        logging.warning("No books were successfully classified.")
        df["Genre"] = gemini_helper.FALLBACK_UNKNOWN
        return df

    results_df = pd.DataFrame(all_classified_books)
    results_df["Item ID"] = results_df["item_id"] # Ensure column name matches for merge

    # Merge results back into the original DataFrame
    # Convert "Item ID" to string in original df for consistent merging if it's not already
    df_copy = df.copy()
    df_copy["Item ID"] = df_copy["Item ID"].astype(str)
    results_df["Item ID"] = results_df["Item ID"].astype(str)

    merged_df = pd.merge(df_copy, results_df[["Item ID", "genre"]], on="Item ID", how="left")
    merged_df.rename(columns={"genre": "Genre"}, inplace=True)
    merged_df["Genre"].fillna(gemini_helper.FALLBACK_UNKNOWN, inplace=True) # Fill any unmerged rows

    return merged_df