import os
import csv

def consolidate_summaries_to_csv(output_dir, csv_output_path):
    """
    Consolidates all .txt files from a specified directory into a single CSV file.
    The CSV file will have two columns: '파일명' (filename) and '파일 내용' (file content).

    Args:
        output_dir (str): The path to the directory containing the .txt summary files.
        csv_output_path (str): The path where the consolidated CSV file will be saved.
    """
    data_to_write = []
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' not found.")
        return

    # Iterate through all files in the output directory
    for filename in os.listdir(output_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(output_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                data_to_write.append({'파일명': filename, '파일 내용': content})
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    
    if not data_to_write:
        print(f"No .txt files found in '{output_dir}' to consolidate.")
        return

    # Write the consolidated data to a CSV file
    try:
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['파일명', '파일 내용']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(data_to_write)
        print(f"Successfully consolidated {len(data_to_write)} summary files to '{csv_output_path}'")
    except Exception as e:
        print(f"Error writing to CSV file '{csv_output_path}': {e}")

if __name__ == "__main__":
    # Define the directory where summary .txt files are located (relative to script)
    output_summaries_dir = "output"
    
    # Define the path for the output CSV file (relative to script)
    consolidated_csv_path = "consolidated_summaries_test.csv"

    consolidate_summaries_to_csv(output_summaries_dir, consolidated_csv_path)
