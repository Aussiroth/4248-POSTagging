# python3.5 eval.py <output_file_absolute_path> <reference_file_absolute_path>
# make no changes in this file

import os
import sys


if __name__ == "__main__":
    out_file = sys.argv[1]
    reader = open(out_file)
    out_lines = reader.readlines()
    reader.close()

    ref_file = sys.argv[2]
    reader = open(ref_file)
    ref_lines = reader.readlines()
    reader.close()

    count = 0
    if len(out_lines) != len(ref_lines):
        print('Error: No. of lines in output file and reference file do not match.')
        exit(0)
    misclassified = {}
    total_tags = 0
    matched_tags = 0
    for i in range(0, len(out_lines)):
        cur_out_line = out_lines[i].strip()
        cur_out_tags = cur_out_line.split(' ')
        cur_ref_line = ref_lines[i].strip()
        cur_ref_tags = cur_ref_line.split(' ')
        total_tags += len(cur_ref_tags)

        for j in range(0, len(cur_ref_tags)):
            count += 1
            if cur_out_tags[j] == cur_ref_tags[j]:
                matched_tags += 1
            else:
                #print("Your output: " + cur_out_tags[j] + " --- Correct output: " + cur_ref_tags[j])
                cur_out_tags[j] = cur_out_tags[j].rsplit("/", 1)
                cur_ref_tags[j] = cur_ref_tags[j].rsplit("/", 1)
                if cur_ref_tags[j][1] not in misclassified:
                    misclassified[cur_ref_tags[j][1]] = {}
                if cur_out_tags[j][1] in misclassified[cur_ref_tags[j][1]]:
                    misclassified[cur_ref_tags[j][1]][cur_out_tags[j][1]] += 1
                else:
                    misclassified[cur_ref_tags[j][1]][cur_out_tags[j][1]] = 1

    print("Accuracy=", float(matched_tags) / total_tags)
    '''for k in misclassified:
        print(k)
        print(misclassified[k])
        print()'''
