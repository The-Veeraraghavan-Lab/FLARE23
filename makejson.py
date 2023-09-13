import os
import json
import glob

path = 'inputs'

files = glob.glob(os.path.join(path, "*.nii.gz"))

# # Print the list of final subfolders
# print(final_subfolders)

listdir = []

for x in files:
        # Append the subfolder path to the list
        listdir.append({"image": '{}'.format(x)})
                        


json_dict = {"validation": listdir}

# Save the dictionary to a JSON file
with open("files2run.json", "w") as outfile:
    json.dump(json_dict, outfile, indent=4)  # Use indent for pretty formatting