import pandas as pd

def csv_to_sto(input_csv_path, output_sto_path):
    # Read the CSV file
    data = pd.read_csv(input_csv_path)

    # Select the first 8 columns (excluding any extra columns beyond the 8th)
    # Assuming the columns are in this specific order, including time as the first column
    columns_to_keep = ['gastroc_l.activation', 'gastroc_r.activation', 'tib_ant_l.activation',
                       'tib_ant_r.activation', 'hamstrings_l.activation', 'hamstrings_r.activation',
                       'rect_fem_l.activation', 'rect_fem_r.activation']  # First 8 columns
    
    # Keep only the specified columns
    data = data[columns_to_keep]

    # Open file to write as .sto format
    with open(output_sto_path, 'w') as sto_file:
        # Write .sto headers
        sto_file.write("sto file\n")
        sto_file.write("version=1\n")
        sto_file.write(f"nRows={data.shape[0]}\n")
        sto_file.write(f"nColumns={data.shape[1]}\n")
        sto_file.write("endheader\n")
        
        # Save the DataFrame to the file with tab-separated values
        data.to_csv(sto_file, sep='\t', index=False)

# Example usage
input_csv = 'your_input.csv'  # Replace with your CSV file path
output_sto = 'your_output.sto'  # Replace with your desired .sto output file path
csv_to_sto(input_csv, output_sto)