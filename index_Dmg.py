def read_dice_scores(file_path):
    """
    Reads dice scores from a file.
    :param file_path: Path to the file containing dice scores.
    :return: A list of dice scores.
    """
    with open(file_path, 'r') as file:
        scores = [float(line.strip()) for line in file.readlines()]
    return scores

def find_satisfying_indices(rfacnn_scores, rfaunet_scores, rfaatunet_scores):
    """
    Finds the indices where RFACNN < RFAUNet < RFAAttUNet.
    :param rfacnn_scores: List of dice scores for RFACNN.
    :param rfaunet_scores: List of dice scores for RFAUNet.
    :param rfaatunet_scores: List of dice scores for RFAAttUNet.
    :return: A list of indices where the condition is satisfied.
    """
    satisfying_indices = []
    for i in range(len(rfacnn_scores)):
        if rfacnn_scores[i] < rfaunet_scores[i] < rfaatunet_scores[i]:
            satisfying_indices.append(i+1)
    return satisfying_indices

if __name__ == "__main__":
    # File paths
    rfacnn_path = "fig_Dmg/RFACNN_unseen/dice.txt"
    rfaunet_path = "fig_Dmg/RFAUNet_unseen/dice.txt"
    rfaatunet_path = "fig_Dmg/RFAAttUNet_unseen/dice.txt"
    
    # Read the dice scores from the files
    rfacnn_scores = read_dice_scores(rfacnn_path)
    rfaunet_scores = read_dice_scores(rfaunet_path)
    rfaatunet_scores = read_dice_scores(rfaatunet_path)
    
    # Find the indices satisfying the condition
    satisfying_indices = find_satisfying_indices(rfacnn_scores, rfaunet_scores, rfaatunet_scores)
    
    # Output the indices
    print("Indices where RFACNN < RFAUNet < RFAAttUNet:", satisfying_indices)

