import csv
import re
import argparse
import sys
import io
import os

def parse_model_answers(md_content):
    """
    Parses the markdown content to extract model answers for each question.

    Args:
        md_content (str): The string content of the markdown file.

    Returns:
        tuple: A tuple containing:
            - list: A list of model names (tokenizers).
            - dict: A dictionary where keys are question numbers (str) and
                    values are dictionaries of model answers.
    """
    model_answers = {}
    model_names = []
    
    # Regex to extract the answer and confidence from a table cell.
    # It handles the checkmark/cross emoji, the answer text, and the confidence score in parentheses.
    answer_regex = re.compile(r"(\S+)\s(.*?)\s\((\d\.\d+)\)")

    in_results_table = False
    lines = md_content.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('| Q#'):
            # This is the header row of the detailed results table.
            in_results_table = True
            # Extract model names from the header. The first 3 columns are not models.
            headers = [h.strip() for h in line.split('|')]
            # Exclude Q#, Question, Correct Answer and the last empty part
            model_names = headers[4:-1] 
            continue

        if in_results_table and line.startswith('|---'):
            # This is the separator line, skip it.
            continue
            
        if in_results_table and line.startswith('|'):
            # This is a data row in the results table.
            parts = [p.strip() for p in line.split('|')]
            
            # The first part is empty, the second is the question number.
            question_num = parts[1]
            
            # The model answers start from the 5th part (index 4).
            answer_cells = parts[4:-1]
            
            model_answers[question_num] = {}
            
            for i, cell in enumerate(answer_cells):
                if i < len(model_names):
                    model_name = model_names[i]
                    match = answer_regex.search(cell)
                    
                    if match:
                        is_correct_symbol, answer, confidence = match.groups()
                        is_correct = '✅' in is_correct_symbol
                        model_answers[question_num][model_name] = {
                            'llm_answer': answer.strip(),
                            'is_correct': is_correct,
                            'confidence': float(confidence)
                        }
    
    return model_names, model_answers


def merge_data_and_write_files(questions_content, model_names, model_answers_data, file_prefix):
    """
    Merges the question data with the model answer data and writes the output
    to separate TSV files for each model.

    Args:
        questions_content (str): The string content of the questions TSV file.
        model_names (list): A list of the model names.
        model_answers_data (dict): The parsed model answer data.
        file_prefix (str): The prefix to use for the output filenames.
    """
    # Manually define the header for the input questions file as it has none.
    original_header = [
        'perturbed_question', 'opt1', 'opt2', 'opt3', 'opt4', 
        'affected_word', 'word_index', 'perturbation_type', 'perturbation', 'question_num'
    ]
    
    # Use io.StringIO to treat the string as a file.
    # This allows the csv.reader to correctly handle newlines within quoted fields.
    questions_file = io.StringIO(questions_content)
    reader = csv.reader(questions_file, delimiter='\t', quotechar='"')
    
    question_rows = []
    # Add a check for column count mismatch
    for i, row in enumerate(reader):
        if len(row) != len(original_header):
            print(f"[Warning] Line {i+1} in questions file has {len(row)} columns, expected {len(original_header)}. Skipping row.")
            print(f"   -> Content: {row}")
            continue
        question_rows.append(dict(zip(original_header, row)))

    
    # --- CHECK 1: Report number of rows found in the questions file ---
    print(f"\n[Check] Found and successfully parsed {len(question_rows)} rows in the questions file.")
    
    # Prepare a dictionary to hold the merged data for each model
    output_data = {model: [] for model in model_names}
    
    # Define the header for the new output files
    new_header = original_header + ['llm_answer', 'is_correct', 'confidence']

    # Iterate through each question using its index, as the order is the key.
    for idx, row in enumerate(question_rows):
        # The markdown file's Q# is 1-based, so we add 1 to the 0-based index.
        q_num_key = str(idx + 1)
        
        if q_num_key in model_answers_data:
            answers_for_q = model_answers_data[q_num_key]
            
            # For each model, create a new merged row
            for model_name in model_names:
                if model_name in answers_for_q:
                    model_answer_info = answers_for_q[model_name]
                    
                    # Create a copy of the original row and update it
                    new_row = row.copy()
                    new_row.update(model_answer_info)
                    
                    output_data[model_name].append(new_row)
        else:
            print(f"[Warning] No answer found for question at line {q_num_key}. Skipping row.")


    # --- NEW: Define output directory and create it if it doesn't exist ---
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Writing Output Files to '{output_dir}/' directory ---")

    # Write the merged data to separate files
    for model_name, data_rows in output_data.items():
        # Sanitize the model name for the filename
        sanitized_model_name = model_name.replace('/', '_')
        # --- NEW: Construct filename with prefix ---
        filename = f"{file_prefix}_{sanitized_model_name}.tsv"
        # --- NEW: Construct the full path including the directory ---
        output_path = os.path.join(output_dir, filename)
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=new_header, delimiter='\t')
                writer.writeheader()
                writer.writerows(data_rows)
            # --- CHECK 3: Report number of rows written to each file ---
            print(f"[Check] Successfully created '{output_path}' with {len(data_rows)} rows.")
        except IOError as e:
            print(f"Error writing file {output_path}: {e}")


def main():
    """
    Main function to parse command-line arguments and execute the script logic.
    """
    # Set up argument parser to accept input file paths
    parser = argparse.ArgumentParser(
        description="Merge perturbed questions with model answers and generate a TSV file per model."
    )
    parser.add_argument(
        "questions_file", 
        help="Path to the input TSV file (without a header) containing perturbed questions."
    )
    parser.add_argument(
        "answers_file", 
        help="Path to the input Markdown file containing model answers."
    )
    args = parser.parse_args()

    print("--- Starting Data Merging Process ---")

    # Read content from the provided file paths
    try:
        with open(args.questions_file, 'r', encoding='utf-8') as f:
            questions_tsv_content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{args.questions_file}' was not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.answers_file, 'r', encoding='utf-8') as f:
            model_answers_md_content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{args.answers_file}' was not found.", file=sys.stderr)
        sys.exit(1)

    # 1. Parse the model answers from the markdown content
    model_names, model_answers_data = parse_model_answers(model_answers_md_content)
    
    if not model_names:
        print("Could not find any model data to process in the markdown file.")
        return

    print(f"Found data for {len(model_names)} models: {', '.join(model_names)}")
    # --- CHECK 2: Report number of answer sets found in the markdown file ---
    print(f"[Check] Found {len(model_answers_data)} answer sets in the markdown file.")

    # --- NEW: Get the base name of the input file to use as a prefix ---
    file_prefix = os.path.splitext(os.path.basename(args.questions_file))[0]

    # 2. Merge the data and write the output files
    merge_data_and_write_files(questions_tsv_content, model_names, model_answers_data, file_prefix)
    
    print("\nProcess finished.")


if __name__ == "__main__":
    main()

