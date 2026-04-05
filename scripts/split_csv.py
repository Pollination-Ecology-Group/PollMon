"""
Usage: python scripts/split_csv.py [options]

Description:
    Splits a large CSV file into smaller chunks based on file size.

Parameters:
    --input, -i       : Path to the input CSV file (default: observations.csv)
    --output_dir, -o  : Directory to save the split files (default: devidedObservations)
    --size, -s        : Maximum size per file in GB (default: 2.0)
"""
import argparse
import os
import sys

def split_csv_file_into_smaller_chunks_based_on_size(path_to_input_csv_file, directory_for_output_files, maximum_file_size_in_gigabytes=2.0):
    if not os.path.isfile(path_to_input_csv_file):
        path_to_file_in_parent_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.basename(path_to_input_csv_file))
        if os.path.isfile(path_to_file_in_parent_directory):
            path_to_input_csv_file = path_to_file_in_parent_directory
        else:
            path_to_file_in_current_working_directory = os.path.join(os.getcwd(), os.path.basename(path_to_input_csv_file))
            if os.path.isfile(path_to_file_in_current_working_directory):
                path_to_input_csv_file = path_to_file_in_current_working_directory
            else:
                print(f"Error: Input file '{path_to_input_csv_file}' not found.")
                sys.exit(1)

    if not os.path.exists(directory_for_output_files):
        try:
            os.makedirs(directory_for_output_files)
            print(f"Created output directory: {directory_for_output_files}")
        except OSError as error_message:
            print(f"Error creating output directory: {error_message}")
            sys.exit(1)

    maximum_file_size_in_bytes = int(maximum_file_size_in_gigabytes * 1024 * 1024 * 1024)
    
    base_name_of_input_file = os.path.splitext(os.path.basename(path_to_input_csv_file))[0]
    counter_for_output_files = 1
    
    print(f"Splitting '{path_to_input_csv_file}' into chunks of ~{maximum_file_size_in_gigabytes} GB in '{directory_for_output_files}'...")

    try:
        with open(path_to_input_csv_file, 'r', encoding='utf-8', errors='replace') as file_handle_for_reading:
            header_row_content = file_handle_for_reading.readline()
            if not header_row_content:
                print("Error: Input file is empty.")
                sys.exit(1)
            
            size_of_header_row_in_bytes = len(header_row_content.encode('utf-8'))
            
            handle_for_current_output_file = None
            current_size_of_output_file_in_bytes = 0
            
            def function_to_create_new_split_file(file_number_suffix):
                path_to_new_output_file = os.path.join(directory_for_output_files, f"{base_name_of_input_file}_part_{file_number_suffix}.csv")
                file_handle_for_writing = open(path_to_new_output_file, 'w', encoding='utf-8', newline='')
                file_handle_for_writing.write(header_row_content)
                return file_handle_for_writing, size_of_header_row_in_bytes, path_to_new_output_file

            handle_for_current_output_file, current_size_of_output_file_in_bytes, path_to_current_output_file = function_to_create_new_split_file(counter_for_output_files)
            print(f"Writing to {path_to_current_output_file}...")

            for line_content in file_handle_for_reading:
                line_content_encoded_to_bytes = line_content.encode('utf-8')
                size_of_current_line_in_bytes = len(line_content_encoded_to_bytes)
                
                if current_size_of_output_file_in_bytes + size_of_current_line_in_bytes > maximum_file_size_in_bytes and current_size_of_output_file_in_bytes > size_of_header_row_in_bytes:
                    handle_for_current_output_file.close()
                    counter_for_output_files += 1
                    handle_for_current_output_file, current_size_of_output_file_in_bytes, path_to_current_output_file = function_to_create_new_split_file(counter_for_output_files)
                    print(f"Writing to {path_to_current_output_file}...")
                
                handle_for_current_output_file.write(line_content)
                current_size_of_output_file_in_bytes += size_of_current_line_in_bytes
                
            if handle_for_current_output_file:
                handle_for_current_output_file.close()
                
        print(f"Successfully split into {counter_for_output_files} files.")

    except Exception as error_message:
        print(f"An error occurred: {error_message}")
        sys.exit(1)

if __name__ == "__main__":
    argument_parser_instance = argparse.ArgumentParser(description="Split CSV file into chunks by size.")
    argument_parser_instance.add_argument("--input", "-i", default="observations.csv", help="Input CSV file path (default: observations.csv)")
    argument_parser_instance.add_argument("--output_dir", "-o", default="devidedObservations", help="Output directory (default: devidedObservations)")
    argument_parser_instance.add_argument("--size", "-s", type=float, default=2.0, help="Chunk size in GB (default: 2.0)")
    
    parsed_arguments = argument_parser_instance.parse_args()
    
    split_csv_file_into_smaller_chunks_based_on_size(parsed_arguments.input, parsed_arguments.output_dir, parsed_arguments.size)

