import json
import csv

books_path = "redpajama/football_part_0.jsonl"

output_csv_path = "redpajama/football_test.csv"

with open(books_path, 'r') as file:
    # Create a CSV writer
    csv_writer = csv.writer(open(output_csv_path, 'w'))
    # Write the all data's meta title into the CSV file
    for line_number, line in enumerate(file, start=1):
        data = json.loads(line)
        if 'meta' in data and isinstance(data['meta'], dict):
            if 'title' in data['meta']:
                csv_writer.writerow([data['meta']['title']])
            else:
                csv_writer.writerow([data['meta']['short_book_title']])
        
        print("Processing Books", line_number, end='\r')
        
        
# def merge_files(input_files, output_file):
#     with open(output_file, 'w') as outfile:
#         for fname in input_files:
#             with open(fname, 'r') as infile:
#                 for line in infile:
#                     outfile.write(line)


# input_files = [f"football_part_{i}.jsonl" for i in range(6)] 
# output_file = "football_merged.jsonl"

# path = "redpajama/"
# input_files = [path + file for file in input_files]
# output_file = path + output_file   

# merge_files(input_files, output_file)
