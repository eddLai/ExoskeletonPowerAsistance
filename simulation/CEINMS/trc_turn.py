def modify_trc_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        header = []

        for _ in range(5):
            line = infile.readline()
            header.append(line)
            outfile.write(line)
        
        for line in infile:
            columns = line.strip().split('\t')
            
            for i in range(2, len(columns)): 
                try:
                    value = float(columns[i])
                    if (i - 2) % 3 == 0 or (i - 2) % 3 == 2:
                        columns[i] = str(value * -1)
                except ValueError:
                    pass
            
            outfile.write('\t'.join(columns) + '\n')

input_trc = 'data/Empty_project_filt_0-30 (1).trc'
output_trc = 'output/walking_modified.trc'

modify_trc_file(input_trc, output_trc)