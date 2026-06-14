import json
import multiprocessing
from multiprocessing import Lock
from transformers import pipeline
from queue import Empty

def producer(input_file, queue):
    with open(input_file, 'r') as file:
        for line in file:
            queue.put(line)
    for _ in range(num_consumers):  # Send end signal for each consumer
        queue.put(None)

def consumer(queue, output_file_path, device_id, print_lock,i):
    classifier = pipeline("zero-shot-classification",
                          model="sileod/deberta-v3-base-tasksource-nli", 
                          device=device_id)
    expected_keys = {'short_book_title', 'publication_date', 'url'}
    labels = ['football-related', 'not football-related']
    output_books_num = 0
    processed_books_num = 0

    while True:
        line = queue.get()
        if line is None:  # End signal
            break

        data = json.loads(line)
        processed_books_num += 1

        if 'meta' in data and isinstance(data['meta'], dict):
            meta_keys = set(data['meta'].keys())
            # Step1: Check with the title
            # if not meta_keys.issubset(expected_keys):
            #     prediction = classifier(data['meta']['title'], labels, multi_label=False)
            # else:
            #     prediction = classifier(data['meta']['short_book_title'], labels, multi_label=False)
            
            # Step2: Check with the text
            sample_text = data['text'][5000:7500]
            
            prediction = classifier(sample_text, labels, multi_label=False)
            
            if prediction['labels'][0] == "football-related" and prediction['scores'][0] >= 0.75 or prediction['labels'][1] == "football-related" and prediction['scores'][1] >= 0.75:
                output_books_num += 1
                with open(output_file_path, 'a') as output_file:
                    output_file.write(json.dumps(data) + '\n')
        with print_lock:
            print(f" Agent {i} - Processed: {processed_books_num}, Football-related: {output_books_num}", end='\r')
            
    with print_lock:
        print(f" Agent {i} finished processing. Total Processed: {processed_books_num}, Total Football-related: {output_books_num}")
        
if __name__ == '__main__':
    book_dir = "redpajama/football_207_text_0.65.jsonl"
    football_dir = "redpajama/football"
    
    print_lock = Lock()

    num_consumers = 3 # Adjust based on your requirement
    queue = multiprocessing.Queue(maxsize=50)  # Adjust the queue size as needed

    # Start producer process
    producer_process = multiprocessing.Process(target=producer, args=(book_dir, queue))
    producer_process.start()

    # Start consumer processes
    consumers = []
    for i in range(num_consumers):
        device_id = i % 3  # Use GPUs 0, 1, and 2 in a round-robin fashion
        output_file = f"{football_dir}_part_{i}.jsonl"
        consumer_process = multiprocessing.Process(target=consumer, args=(queue, output_file, device_id, print_lock,i))
        consumer_process.start()
        consumers.append(consumer_process)

    # Wait for all processes to complete
    producer_process.join()
    for consumer_process in consumers:
        consumer_process.join()

    # Optionally, merge the output files here
