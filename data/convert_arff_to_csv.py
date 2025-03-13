import pandas as pd
from scipy.io import arff


def convert_arff_to_csv(input_arff, output_csv):
    # Load the ARFF file correctly
    data, meta = arff.loadarff(input_arff)

    # Convert the ARFF data into a DataFrame
    df = pd.DataFrame(data)

    # Decode byte strings to normal strings if necessary
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')

    # Save the DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f"Conversion complete! CSV saved to: {output_csv}")


if __name__ == "__main__":
    input_arff = "data/raw/bmt_dataset.arff"
    output_csv = "data/processed/bmt_dataset.csv"
    convert_arff_to_csv(input_arff, output_csv)
