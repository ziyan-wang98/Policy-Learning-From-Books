import os

# Path to parsed Wikipedia data.
wiki_path = 'text/'
output_file = 'wiki20230701.json'
# List all files under the path.
wiki_list = os.listdir(wiki_path)
# Iterate through text-file folders.
for per_file in wiki_list:
    if per_file == '.DS_Store':
        continue
    # File path.
    file_path = os.path.join(wiki_path, per_file)
    txt_list = os.listdir(file_path)
    # Iterate through each text file.
    for per_txt in txt_list:
        if per_txt == '.DS_Store':
            continue
        # Path to each text file.
        txt_path = os.path.join( wiki_path, per_file, per_txt )
        # cat file0 >> output_file appends file0 to the end of output_file.
        cms = 'cat {} >> {}'.format(txt_path, output_file)
        print (cms)
        os.system(cms)
