def sum_values_from_file(file_path):
    total_sum = 0
    with open(file_path, 'r') as file:
        for line in file:
            try:
                number = float(line.strip())
                total_sum += number
            except ValueError:
                continue
    return total_sum/499


file_path = 'fig_Dmg/RFACNN_unseen/hausdorff.txt'
result = sum_values_from_file(file_path)
print(f"Total sum of values in the file: {result}")
