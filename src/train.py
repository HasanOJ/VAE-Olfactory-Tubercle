import argparse
from data_processing import StatisticsCalculator

def main():
    parser = argparse.ArgumentParser(description='Train a model with specified test set.')
    parser.add_argument('--test_set', type=str, choices=['B01', 'B02', 'B05', 'B07', 'B20'], default='B20', help='Test set to use')
    parser.add_argument('--data_path', type=str, default='cell_data.h5', help='Path to the HDF5 dataset')
    args = parser.parse_args()

    # Initialize StatisticsCalculator with user-defined HDF5 file path
    stats_calculator = StatisticsCalculator(h5_file_path=args.data_path)
    global_min, global_max, global_mean, global_std = stats_calculator.calculate_statistics(args.test_set)
    
    print(f"Global Max: {global_max}")
    print(f"Global Min: {global_min}")
    print(f"Global Mean: {global_mean}")
    print(f"Global Std: {global_std}")

if __name__ == "__main__":
    main()
