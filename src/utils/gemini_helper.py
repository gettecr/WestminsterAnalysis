# --- Genre Definitions ---
ALLOWED_GENRES = [
    "Romance", "Sci Fi", "Fantasy", "Historical Fiction", "Horror",
    "Biographies/Memoirs", "Mystery", "Western", "Christian Fiction", "Thriller"
]
FALLBACK_FICTION = "Literary Fiction"
FALLBACK_NON_FICTION = "Other"
FALLBACK_UNKNOWN = "Unknown"
VALID_GENRES_SET = set(ALLOWED_GENRES + [FALLBACK_FICTION, FALLBACK_NON_FICTION, FALLBACK_UNKNOWN])

# --- Prompt Template ---
def classifier_prompt(books_json_string):
  PROMPT_TEMPLATE = """
You are an expert book genre classifier.
Your task is to classify each book in the provided list into ONE of the following genres:
%(allowed_genres)s.

Follow these rules strictly:
1. If none of the listed genre names are applicable but the book is clearly a work of fiction, assign "%(fallback_fiction)s".
2. If the book is not a work of fiction (e.g., textbook, self-help, technical manual, cookbook), assign "%(fallback_non_fiction)s".
3. If you do not have enough information from the title and author to confidently classify, or if the item doesn't seem like a book, assign "%(fallback_unknown)s".
4. You MUST ONLY use the provided genre names or the specified fallback options. Do not invent new genres.
5. Respond with a JSON list of objects. Each object must have an "item_id" (matching the input) and a "genre" key.

Example input format (you will receive a list of books like this):
[
  {"item_id": "101", "title": "The Martian", "author": "Andy Weir"},
  {{"item_id": "102", "title": "Sapiens: A Brief History of Humankind", "author": "Yuval Noah Harari"}},
  {{"item_id": "103", "title": "The Da Vinci Code", "author": "Dan Brown"}},
  {{"item_id": "104", "title": "A Brief History of Time", "author": "Stephen Hawking"}},
  {{"item_id": "105", "title": "The Hobbit", "author": "J.R.R. Tolkien"}},
  {{"item_id": "106", "title": "Becoming", "author": "Michelle Obama"}},
  {{"item_id": "107", "title": "It", "author": "Stephen King"}},
  {{"item_id": "108", "title": "The Love Hypothesis", "author": "Ali Hazelwood"}},
  {{"item_id": "109", "title": "The Lincoln Highway", "author": "Amor Towles"}},
  {{"item_id": "110", "title": "Redeeming Love", "author": "Francine Rivers"}},
  {{"item_id": "111", "title": "Lonesome Dove", "author": "Larry McMurtry"}},
  {{"item_id": "112", "title": "The Mysterious Affair at Styles", "author": "Agatha Christie"}}
]

Example JSON output for the above:
[
  {{"item_id": "101", "genre": "Sci Fi"}},
  {{"item_id": "102", "genre": "Other"}},
  {{"item_id": "103", "genre": "Thriller"}},
  {{"item_id": "104", "genre": "Other"}},
  {{"item_id": "105", "genre": "Fantasy"}},
  {{"item_id": "106", "genre": "Biographies/Memoirs"}},
  {{"item_id": "107", "genre": "Horror"}},
  {{"item_id": "108", "genre": "Romance"}},
  {{"item_id": "109", "genre": "Historical Fiction"}},
  {{"item_id": "110", "genre": "Christian Fiction"}},
  {{"item_id": "111", "genre": "Western"}},
  {{"item_id": "112", "genre": "Mystery"}}
]

Classify the following books:
%(books_string)s
""" % {"allowed_genres":', '.join(ALLOWED_GENRES), 
      "fallback_fiction":FALLBACK_FICTION,
      "fallback_non_fiction":FALLBACK_NON_FICTION,
      "fallback_unknown":FALLBACK_UNKNOWN,
      "books_string":books_json_string}
  return PROMPT_TEMPLATE

def clean_result_text(raw_json: str):
  if raw_json.strip().startswith("```json"):
    raw_json = raw_json.strip()[7:-3].strip()
  elif raw_json.strip().startswith("```"): # Sometimes just ``` at start
    raw_json = raw_json.strip()[3:-3].strip()
  return raw_json
  