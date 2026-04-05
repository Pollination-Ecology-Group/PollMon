#!/usr/bin/env python3
"""
Usage: python scripts/extract_images.py --input <path_to_csv> [options]

Description:
    Downloads images from iNaturalist observations CSV based on target species.

Parameters:
    --input, -i      : Path to the input CSV file (Required)
    --limit, -l      : Max number of images to download (default: 4). Set to 0 for no limit.
    --start-row      : Row number to start processing from (default: 0)
"""
import argparse
import csv
import os
import sys
import time
import requests
import threading
import queue
import re
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO

DIRECTORY_WHERE_IMAGES_WILL_BE_SAVED = r"V:\PollinatorINaturalistData\extractedImages"

DICTIONARY_OF_TARGET_KEYWORDS_AND_FOLDER_NAMES = {
    "cryptocephalus": "cryptocephalus_species",
    "pieris brassicae": "pieris_brassicae",
    "aglais urticae": "aglais_urticae",
    "bombus terrestris": "bombus_terrestris",
    "bombus sylvarum": "bombus_sylvarum",
    "nitidulidae": "nitidulidae",
    "pieris rapae": "pieris_rapae",
    "bombus lapidarius": "bombus_lapidarius",
    "sphaerophoria scripta": "sphaerophoria_scripta",
    "melanostoma mellinum": "melanostoma_mellinum",
    "eristalis arbustorum": "eristalis_arbustorum",
    "helophilus pendulus": "helophilus_pendulus",
    "eristalis pertinax": "eristalis_pertinax",
    "helophilus trivittatus": "helophilus_trivittatus",
    "helophilus hybridus": "helophilus_hybridus",
    "anthaxia": "anthaxia_species",
    "myathropa florea": "myathropa_florea",
    "eristalis similis": "eristalis_similis",
    "sericomyia silentis": "sericomyia_silentis",
    "bombus pascuorum": "bombus_pascuorum",
    "apis mellifera": "apis_mellifera",
    "eristalis interruptus": "eristalis_interruptus",
    "chrysogaster solstitialis": "chrysogaster_solstitialis",
    "pyrophaena granditarsis": "pyrophaena_granditarsis",
    "eristalis intricaria": "eristalis_intricaria",
    "syritta pipiens": "syritta_pipiens",
    "eristalis tenax": "eristalis_tenax"
}

LIST_OF_COLUMNS_TO_CHECK_FOR_TAXONOMY = ["scientificName"]
NAME_OF_COLUMN_WITH_REFERENCE_URL = "references"
NAME_OF_COLUMN_WITH_UNIQUE_ID = "id"

MAXIMUM_ALLOWED_FILE_SIZE_IN_BYTES = 500 * 1024
NUMBER_OF_THREADS_FOR_API_REQUESTS = 2
NUMBER_OF_THREADS_FOR_DOWNLOADING_IMAGES = 20
SIZE_OF_BATCH_FOR_API_REQUESTS = 30
MAXIMUM_SIZE_OF_TASK_QUEUE = 50
PATH_TO_LOG_FILE_FOR_FAILURES = "failed_downloads.txt"
MAXIMUM_NUMBER_OF_IMAGES_ALLOWED_PER_SPECIES = 2000

global_dictionary_tracking_species_image_counts = {}
lock_for_synchronizing_species_counts = threading.Lock()

def create_http_session_with_retry_logic():
    session_object = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter_object = HTTPAdapter(max_retries=retry_strategy)
    session_object.mount("https://", adapter_object)
    session_object.mount("http://", adapter_object)
    return session_object

global_http_session_object = create_http_session_with_retry_logic()

LIST_OF_USER_AGENT_STRINGS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
]

def generate_request_parameters_with_random_user_agent(timeout_seconds=20):
    headers_dictionary = {
        'User-Agent': random.choice(LIST_OF_USER_AGENT_STRINGS)
    }
    parameters_dictionary = {
        'headers': headers_dictionary,
        'timeout': timeout_seconds
    }
    return parameters_dictionary

def extract_list_of_image_candidate_urls_from_photo_object(photo_data_object):
    list_of_candidate_urls = []
    
    if photo_data_object.get('original_url'): list_of_candidate_urls.append(photo_data_object['original_url'])
    if photo_data_object.get('large_url'): list_of_candidate_urls.append(photo_data_object['large_url'])
    if photo_data_object.get('medium_url'): list_of_candidate_urls.append(photo_data_object['medium_url'])
    
    base_url_string = photo_data_object.get('url')
    if base_url_string:
        for size_variant in ['original', 'large', 'medium', 'small']:
            new_url_string = re.sub(r'/(square|thumb|small|medium|large|original)\.([a-zA-Z0-9]+)(\?.*)?$', f'/{size_variant}.\\2\\3', base_url_string)
            if new_url_string != base_url_string and new_url_string not in list_of_candidate_urls:
                list_of_candidate_urls.append(new_url_string)
                
        if base_url_string not in list_of_candidate_urls:
            list_of_candidate_urls.append(base_url_string)
            
    set_of_seen_urls = set()
    list_of_unique_candidates = []
    for url_string in list_of_candidate_urls:
        if url_string not in set_of_seen_urls:
            list_of_unique_candidates.append(url_string)
            set_of_seen_urls.add(url_string)
            
    return list_of_unique_candidates

dictionary_of_runtime_statistics = {
    "count_of_rows_scanned": 0,
    "count_of_bytes_read": 0,
    "count_of_matches_found": 0,
    "count_of_urls_resolved": 0,
    "count_of_images_downloaded": 0,
    "count_of_images_failed": 0,
    "count_of_active_api_workers": 0,
    "count_of_active_downloaders": 0
}
lock_for_synchronizing_statistics = threading.Lock()
lock_for_synchronizing_failure_logging = threading.Lock()

def configure_logging_system():
    if not os.path.exists(DIRECTORY_WHERE_IMAGES_WILL_BE_SAVED):
        os.makedirs(DIRECTORY_WHERE_IMAGES_WILL_BE_SAVED)
    
    with open(PATH_TO_LOG_FILE_FOR_FAILURES, "w", encoding="utf-8") as file_handle:
        file_handle.write(f"Failure Log Started: {datetime.now().isoformat()}\n")

def initialize_species_image_counts_from_directory():
    print("Scanning output directory to count existing images...")
    
    for species_prefix in DICTIONARY_OF_TARGET_KEYWORDS_AND_FOLDER_NAMES.values():
        global_dictionary_tracking_species_image_counts[species_prefix] = 0
        
    if not os.path.exists(DIRECTORY_WHERE_IMAGES_WILL_BE_SAVED):
        return

    try:
        for species_prefix in DICTIONARY_OF_TARGET_KEYWORDS_AND_FOLDER_NAMES.values():
            full_path_to_species_directory = os.path.join(DIRECTORY_WHERE_IMAGES_WILL_BE_SAVED, species_prefix)
            if os.path.exists(full_path_to_species_directory):
                count_of_images_in_directory = len([filename for filename in os.listdir(full_path_to_species_directory) if filename.endswith('.jpg')])
                global_dictionary_tracking_species_image_counts[species_prefix] += count_of_images_in_directory

        list_of_files_in_root = os.listdir(DIRECTORY_WHERE_IMAGES_WILL_BE_SAVED)
        for filename in list_of_files_in_root:
            if os.path.isfile(os.path.join(DIRECTORY_WHERE_IMAGES_WILL_BE_SAVED, filename)):
                for species_prefix in global_dictionary_tracking_species_image_counts:
                    if filename.startswith(species_prefix):
                        global_dictionary_tracking_species_image_counts[species_prefix] += 1
                        break
    except Exception as error_message:
        print(f"Error scanning directory: {error_message}")

    print("Current Counts:")
    for species_prefix, current_count in global_dictionary_tracking_species_image_counts.items():
        status_string = "ACTIVE" if current_count < MAXIMUM_NUMBER_OF_IMAGES_ALLOWED_PER_SPECIES else "DONE (Limit Reached)"
        print(f"  {species_prefix}: {current_count} [{status_string}]")
    print("-" * 40)

def log_download_failure(url_string, reason_string):
    with lock_for_synchronizing_failure_logging:
        try:
            with open(PATH_TO_LOG_FILE_FOR_FAILURES, "a", encoding="utf-8") as file_handle:
                file_handle.write(f"{datetime.now().strftime('%H:%M:%S')} | {reason_string} | {url_string}\n")
        except Exception:
            pass

def find_index_of_column_in_header(header_row_list, column_name_string):
    column_name_lower = column_name_string.lower()
    for index, header_name in enumerate(header_row_list):
        if header_name.lower() == column_name_lower:
            return index
    return -1

def check_if_row_matches_target_species_and_limit_not_reached(csv_row_list, header_mapping_dictionary):
    for column_name in LIST_OF_COLUMNS_TO_CHECK_FOR_TAXONOMY:
        column_index = header_mapping_dictionary.get(column_name)
        if column_index is not None and column_index < len(csv_row_list):
            cell_value = csv_row_list[column_index].lower()
            for keyword_string, species_prefix in DICTIONARY_OF_TARGET_KEYWORDS_AND_FOLDER_NAMES.items():
                if keyword_string in cell_value:
                    with lock_for_synchronizing_species_counts:
                        if global_dictionary_tracking_species_image_counts.get(species_prefix, 0) >= MAXIMUM_NUMBER_OF_IMAGES_ALLOWED_PER_SPECIES:
                            return None
                    return species_prefix
    return None

def fetch_image_urls_for_batch_of_rows(batch_of_rows, header_mapping_dictionary):
    list_of_results = []
    
    dictionary_mapping_ids_to_rows = {}
    list_of_ids_to_fetch = []
    
    index_of_id_column = header_mapping_dictionary.get(NAME_OF_COLUMN_WITH_UNIQUE_ID)
    
    for row_data, species_prefix in batch_of_rows:
        observation_id = None
        if index_of_id_column is not None and index_of_id_column < len(row_data):
            observation_id = row_data[index_of_id_column]
        
        if not observation_id:
            index_of_reference_column = header_mapping_dictionary.get(NAME_OF_COLUMN_WITH_REFERENCE_URL)
            if index_of_reference_column is not None and index_of_reference_column < len(row_data):
                regex_match = re.search(r'/observations/(\d+)', row_data[index_of_reference_column])
                if regex_match:
                    observation_id = regex_match.group(1)
        
        if observation_id:
            observation_id = str(observation_id)
            if observation_id not in dictionary_mapping_ids_to_rows:
                dictionary_mapping_ids_to_rows[observation_id] = []
                list_of_ids_to_fetch.append(observation_id)
            dictionary_mapping_ids_to_rows[observation_id].append((row_data, species_prefix))
        else:
            with lock_for_synchronizing_statistics:
                dictionary_of_runtime_statistics["count_of_images_failed"] += 1
            log_download_failure("Unknown", "No Observation ID found in row")

    if not list_of_ids_to_fetch:
        return list_of_results

    try:
        time.sleep(random.uniform(1.0, 2.0))
        
        api_endpoint_url = "https://api.inaturalist.org/v1/observations"
        request_parameters = generate_request_parameters_with_random_user_agent(timeout_seconds=30)
        request_parameters['params'] = {
            'id': ",".join(list_of_ids_to_fetch),
            'per_page': len(list_of_ids_to_fetch),
            'only_id': 'false'
        }
        
        response_object = global_http_session_object.get(api_endpoint_url, **request_parameters)
        response_object.raise_for_status()
        
        json_response_data = response_object.json()
        
        set_of_fetched_ids = set()
        
        if 'results' in json_response_data:
            for result_item in json_response_data['results']:
                result_id_string = str(result_item['id'])
                set_of_fetched_ids.add(result_id_string)
                
                list_of_image_urls = []
                if result_item.get('photos'):
                    photo_data = result_item['photos'][0]
                    list_of_image_urls = extract_list_of_image_candidate_urls_from_photo_object(photo_data)
                
                if list_of_image_urls:
                    if result_id_string in dictionary_mapping_ids_to_rows:
                        for row_data, species_prefix in dictionary_mapping_ids_to_rows[result_id_string]:
                            list_of_results.append((list_of_image_urls, row_data, header_mapping_dictionary, species_prefix))
                else:
                    log_download_failure(f"ID: {result_id_string}", "Observation has no photos")
                    with lock_for_synchronizing_statistics:
                        dictionary_of_runtime_statistics["count_of_images_failed"] += len(dictionary_mapping_ids_to_rows.get(result_id_string, []))

        set_of_missing_ids = set(list_of_ids_to_fetch) - set_of_fetched_ids
        for missing_id in set_of_missing_ids:
            log_download_failure(f"ID: {missing_id}", "API did not return observation")
            with lock_for_synchronizing_statistics:
                dictionary_of_runtime_statistics["count_of_images_failed"] += len(dictionary_mapping_ids_to_rows.get(missing_id, []))

    except Exception as error_message:
        log_download_failure("Batch API", f"API Request Failed: {str(error_message)}")
        with lock_for_synchronizing_statistics:
            dictionary_of_runtime_statistics["count_of_images_failed"] += len(batch_of_rows)
            
    return list_of_results

def download_image_from_url_and_save_to_disk(list_of_candidate_urls, filename_string, species_prefix):
    for image_url in list_of_candidate_urls:
        try:
            request_parameters = generate_request_parameters_with_random_user_agent(timeout_seconds=30)
            response_object = global_http_session_object.get(image_url, **request_parameters)
            
            if response_object.status_code in [403, 404]:
                continue
                
            response_object.raise_for_status()

            image_object = Image.open(BytesIO(response_object.content))

            if image_object.mode in ("RGBA", "P"):
                image_object = image_object.convert("RGB")

            compression_quality = 95
            
            full_path_to_species_directory = os.path.join(DIRECTORY_WHERE_IMAGES_WILL_BE_SAVED, species_prefix)
            os.makedirs(full_path_to_species_directory, exist_ok=True)
            
            full_output_path = os.path.join(full_path_to_species_directory, filename_string)
            
            image_object.save(full_output_path, "JPEG", quality=compression_quality)
            current_file_size = os.path.getsize(full_output_path)
            
            while current_file_size > MAXIMUM_ALLOWED_FILE_SIZE_IN_BYTES and compression_quality > 10:
                compression_quality -= 10
                image_object.save(full_output_path, "JPEG", quality=compression_quality)
                current_file_size = os.path.getsize(full_output_path)
                
            if current_file_size > MAXIMUM_ALLOWED_FILE_SIZE_IN_BYTES:
                while current_file_size > MAXIMUM_ALLOWED_FILE_SIZE_IN_BYTES:
                    current_width, current_height = image_object.size
                    new_width = int(current_width * 0.8)
                    new_height = int(current_height * 0.8)
                    if new_width < 10 or new_height < 10:
                        break
                    image_object = image_object.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    image_object.save(full_output_path, "JPEG", quality=compression_quality)
                    current_file_size = os.path.getsize(full_output_path)

            return True
        except Exception:
            continue
            
    return False

def generate_unique_filename_from_row_data(row_data, header_mapping_dictionary, species_prefix):
    list_of_filename_parts = [species_prefix]
    
    index_of_id_column = header_mapping_dictionary.get(NAME_OF_COLUMN_WITH_UNIQUE_ID)
    if index_of_id_column is not None and index_of_id_column < len(row_data):
        list_of_filename_parts.append(str(row_data[index_of_id_column]))
    else:
        list_of_filename_parts.append(str(int(time.time() * 1000)))

    return "_".join(list_of_filename_parts) + ".jpg"

def worker_function_for_fetching_api_data(queue_for_api_tasks, queue_for_download_tasks):
    while True:
        task_item = queue_for_api_tasks.get()
        if task_item is None:
            break
        
        with lock_for_synchronizing_statistics:
            dictionary_of_runtime_statistics["count_of_active_api_workers"] += 1
            
        try:
            batch_of_rows, header_mapping_dictionary = task_item
            list_of_results = fetch_image_urls_for_batch_of_rows(batch_of_rows, header_mapping_dictionary)
            
            for result_item in list_of_results:
                with lock_for_synchronizing_statistics:
                    dictionary_of_runtime_statistics["count_of_urls_resolved"] += 1
                queue_for_download_tasks.put(result_item)
                
        except Exception as error_message:
            log_download_failure("API Worker", f"Unexpected Error: {str(error_message)}")
        finally:
            with lock_for_synchronizing_statistics:
                dictionary_of_runtime_statistics["count_of_active_api_workers"] -= 1
            queue_for_api_tasks.task_done()

def worker_function_for_downloading_images(queue_for_download_tasks):
    while True:
        task_item = queue_for_download_tasks.get()
        if task_item is None:
            break
            
        with lock_for_synchronizing_statistics:
            dictionary_of_runtime_statistics["count_of_active_downloaders"] += 1
            
        try:
            list_of_image_urls, row_data, header_mapping_dictionary, species_prefix = task_item
            filename_string = generate_unique_filename_from_row_data(row_data, header_mapping_dictionary, species_prefix)
            if download_image_from_url_and_save_to_disk(list_of_image_urls, filename_string, species_prefix):
                with lock_for_synchronizing_statistics:
                    dictionary_of_runtime_statistics["count_of_images_downloaded"] += 1
                
                with lock_for_synchronizing_species_counts:
                    global_dictionary_tracking_species_image_counts[species_prefix] = global_dictionary_tracking_species_image_counts.get(species_prefix, 0) + 1
                    if global_dictionary_tracking_species_image_counts[species_prefix] == MAXIMUM_NUMBER_OF_IMAGES_ALLOWED_PER_SPECIES:
                        print(f"\n[LIMIT REACHED] {species_prefix} has reached {MAXIMUM_NUMBER_OF_IMAGES_ALLOWED_PER_SPECIES} images. Stopping search for this species.")
            else:
                with lock_for_synchronizing_statistics:
                    dictionary_of_runtime_statistics["count_of_images_failed"] += 1
                log_download_failure(str(list_of_image_urls), "All candidates failed")
        except Exception as error_message:
            with lock_for_synchronizing_statistics:
                dictionary_of_runtime_statistics["count_of_images_failed"] += 1
            log_download_failure(str(task_item[0]), f"Downloader Error: {str(error_message)}")
        finally:
            with lock_for_synchronizing_statistics:
                dictionary_of_runtime_statistics["count_of_active_downloaders"] -= 1
            queue_for_download_tasks.task_done()

def function_to_monitor_progress_and_display_stats(event_to_stop_monitoring, total_file_size_in_bytes, start_time_seconds):
    while not event_to_stop_monitoring.is_set():
        current_time_seconds = time.time()
        elapsed_time_seconds = current_time_seconds - start_time_seconds
        
        with lock_for_synchronizing_statistics:
            stats_copy = dictionary_of_runtime_statistics.copy()
            
        percentage_complete = 0.0
        if total_file_size_in_bytes > 0:
            percentage_complete = (stats_copy.get("count_of_bytes_read", 0) / total_file_size_in_bytes) * 100
            
        sys.stdout.write(f"\rElapsed: {elapsed_time_seconds:.0f}s | Progress: {percentage_complete:.2f}% | Scanned: {stats_copy['count_of_rows_scanned']} | Found: {stats_copy['count_of_matches_found']} | URLs: {stats_copy['count_of_urls_resolved']} | DL: {stats_copy['count_of_images_downloaded']} | Fail: {stats_copy['count_of_images_failed']} | Threads: API:{stats_copy['count_of_active_api_workers']} D:{stats_copy['count_of_active_downloaders']}   ")
        sys.stdout.flush()
        time.sleep(1)

def process_input_csv_file(path_to_input_file, starting_row_number=0, maximum_downloads_limit=None):
    if not os.path.exists(path_to_input_file):
        print(f"File not found: {path_to_input_file}")
        return 0

    total_file_size_in_bytes = os.path.getsize(path_to_input_file)
    start_time_seconds = time.time()
    
    queue_for_api_tasks = queue.Queue(maxsize=MAXIMUM_SIZE_OF_TASK_QUEUE)
    queue_for_download_tasks = queue.Queue(maxsize=MAXIMUM_SIZE_OF_TASK_QUEUE)
    
    list_of_api_threads = []
    for _ in range(NUMBER_OF_THREADS_FOR_API_REQUESTS):
        thread_object = threading.Thread(target=worker_function_for_fetching_api_data, args=(queue_for_api_tasks, queue_for_download_tasks))
        thread_object.daemon = True
        thread_object.start()
        list_of_api_threads.append(thread_object)
        
    list_of_downloader_threads = []
    for _ in range(NUMBER_OF_THREADS_FOR_DOWNLOADING_IMAGES):
        thread_object = threading.Thread(target=worker_function_for_fetching_api_data, args=(queue_for_download_tasks,)) # Wait, this is wrong target
        # Correcting target to downloader_worker
        thread_object = threading.Thread(target=worker_function_for_downloading_images, args=(queue_for_download_tasks,))
        thread_object.daemon = True
        thread_object.start()
        list_of_downloader_threads.append(thread_object)
        
    event_to_stop_monitoring = threading.Event()
    monitor_thread_object = threading.Thread(target=function_to_monitor_progress_and_display_stats, args=(event_to_stop_monitoring, total_file_size_in_bytes, start_time_seconds))
    monitor_thread_object.daemon = True
    monitor_thread_object.start()

    print(f"Processing: {os.path.basename(path_to_input_file)}")
    print(f"API Workers: {NUMBER_OF_THREADS_FOR_API_REQUESTS} | Downloaders: {NUMBER_OF_THREADS_FOR_DOWNLOADING_IMAGES}")
    print("-" * 80)

    try:
        with open(path_to_input_file, 'r', encoding='utf-8', errors='replace') as file_handle:
            header_line_string = file_handle.readline()
            if not header_line_string:
                return 0
            header_row_list = next(csv.reader([header_line_string]))
            
            header_mapping_dictionary = {}
            for column_name in LIST_OF_COLUMNS_TO_CHECK_FOR_TAXONOMY + [NAME_OF_COLUMN_WITH_REFERENCE_URL, NAME_OF_COLUMN_WITH_UNIQUE_ID]:
                column_index = find_index_of_column_in_header(header_row_list, column_name)
                if column_index != -1:
                    header_mapping_dictionary[column_name] = column_index
            
            if NAME_OF_COLUMN_WITH_REFERENCE_URL not in header_mapping_dictionary:
                print("Error: Reference column missing.")
                return 0

            current_batch_of_rows = []

            for row_index, line_string in enumerate(file_handle, start=2):
                length_of_line_in_bytes = len(line_string.encode('utf-8'))
                with lock_for_synchronizing_statistics:
                    dictionary_of_runtime_statistics["count_of_rows_scanned"] += 1
                    dictionary_of_runtime_statistics["count_of_bytes_read"] += length_of_line_in_bytes
                
                if row_index < starting_row_number:
                    continue

                row_data = next(csv.reader([line_string]))
                
                matched_species_prefix = check_if_row_matches_target_species_and_limit_not_reached(row_data, header_mapping_dictionary)
                if matched_species_prefix:
                    with lock_for_synchronizing_statistics:
                        dictionary_of_runtime_statistics["count_of_matches_found"] += 1
                    
                    current_batch_of_rows.append((row_data, matched_species_prefix))
                    
                    if len(current_batch_of_rows) >= SIZE_OF_BATCH_FOR_API_REQUESTS:
                        queue_for_api_tasks.put((current_batch_of_rows, header_mapping_dictionary))
                        current_batch_of_rows = []
                
                if maximum_downloads_limit is not None:
                    with lock_for_synchronizing_statistics:
                        if dictionary_of_runtime_statistics["count_of_matches_found"] >= maximum_downloads_limit:
                            break
            
            if current_batch_of_rows:
                queue_for_api_tasks.put((current_batch_of_rows, header_mapping_dictionary))

        queue_for_api_tasks.join()
        
        for _ in range(NUMBER_OF_THREADS_FOR_API_REQUESTS):
            queue_for_api_tasks.put(None)
        for thread_object in list_of_api_threads:
            thread_object.join()
            
        queue_for_download_tasks.join()
        
        for _ in range(NUMBER_OF_THREADS_FOR_DOWNLOADING_IMAGES):
            queue_for_download_tasks.put(None)
        for thread_object in list_of_downloader_threads:
            thread_object.join()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        event_to_stop_monitoring.set()
        if 'monitor_thread_object' in locals() and monitor_thread_object.is_alive():
            monitor_thread_object.join()
        print("\nDone.")
        
    return dictionary_of_runtime_statistics["count_of_images_downloaded"]

def main_entry_point():
    argument_parser = argparse.ArgumentParser(description="Extract images from iNaturalist CSV data.")
    argument_parser.add_argument("--input", "-i", required=True, help="Path to the input CSV file.")
    argument_parser.add_argument("--limit", "-l", type=int, default=4, help="Max number of images to download (default: 4). Set to 0 for no limit.")
    argument_parser.add_argument("--start-row", type=int, default=0, help="Row number to start processing from (header is row 1).")
    
    parsed_arguments = argument_parser.parse_args()
    configure_logging_system()
    initialize_species_image_counts_from_directory()
    
    limit_value = parsed_arguments.limit if parsed_arguments.limit > 0 else None
    process_input_csv_file(parsed_arguments.input, parsed_arguments.start_row, limit_value)

if __name__ == "__main__":
    main_entry_point()

