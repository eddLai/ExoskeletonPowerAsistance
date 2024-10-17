def modify_trc_file(input_file, output_file):
    # Open the input file and output file
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        header = []
        # Read the header lines and write them unchanged to the output file
        for _ in range(5):
            line = infile.readline()
            header.append(line)
            outfile.write(line)
        
        # Process the remaining data lines from line 6 onwards
        for line in infile:
            # Split each line into columns
            columns = line.strip().split('\t')
            
            # Try to convert numeric columns (skip the first two columns: index and timestamp)
            for i in range(2, len(columns)):  # Start from the third column (index 2)
                try:
                    value = float(columns[i])
                    # Modify columns at indices 3n or 3n+2 (i.e., 0-based index 2, 4, 5, 7, ...)
                    if (i - 2) % 3 == 0 or (i - 2) % 3 == 2:
                        columns[i] = str(value * -1)
                except ValueError:
                    # If it's not a numeric value, skip modification
                    pass
            
            # Write the modified line to the output file
            outfile.write('\t'.join(columns) + '\n')

# Usage example
input_trc = 'data/Empty_project_filt_0-30 (1).trc'  # 原始 TRC 文件名
output_trc = 'output/walking_modified.trc'  # 修改後的 TRC 文件名

modify_trc_file(input_trc, output_trc)