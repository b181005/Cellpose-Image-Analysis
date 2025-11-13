# Save this code as helper_functions.py

print("hey")

def analyze_data(data_list, analysis_type="mean"):
    """
    Performs a simple analysis on a list of numbers.

    Args:
        data_list (list): A list of numbers.
        analysis_type (str): The type of analysis ('mean' or 'sum').

    Returns:
        float: The result of the analysis.
    """
    if analysis_type == "mean":
        if not data_list:
            return 0
        result = sum(data_list) / len(data_list)
        print(f"Calculated the mean: {result}")
        return result
    elif analysis_type == "sum":
        result = sum(data_list)
        print(f"Calculated the sum: {result}")
        return result
    else:
        print(f"Error: Unknown analysis type '{analysis_type}'")
        return None