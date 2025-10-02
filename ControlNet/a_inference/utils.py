import json
import matplotlib.pyplot as plt
import re, os
from collections import defaultdict

def get_statistics(file_path, output_plot_dir):

    # Load the JSON data from the file
    # file_path = '/scratch/YOURNAME/project/ControlNet/a_inference/evaluate/result.json'
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Calculate the average SSIM and MSE
    total_ssim = sum(item["SSIM"] for item in data)
    total_mse = sum(item["MSE"] for item in data)
    num_elements = len(data)

    average_ssim = total_ssim / num_elements
    average_mse = total_mse / num_elements

    print(f"Average SSIM: {average_ssim}")
    print(f"Average MSE: {average_mse}")



    # hourly analysis: 

    # save one image:

    # file_path = '/scratch/YOURNAME/project/ControlNet/a_inference/evaluate_507/result.json'
    # with open(file_path, 'r') as f:
    #     data = json.load(f)

    # # Group SSIM and MSE values by hour
    # hourly_metrics = defaultdict(lambda: {'SSIM': [], 'MSE': []})

    # for item in data:
    #     # Extract hour from "generated_path"
    #     match = re.search(r'hour(\d+)_', item["generated_path"])
    #     if match:
    #         hour = int(match.group(1))
    #         hourly_metrics[hour]['SSIM'].append(item["SSIM"])
    #         hourly_metrics[hour]['MSE'].append(item["MSE"])

    # # Calculate average SSIM and MSE for each hour
    # average_ssim_by_hour = {hour: sum(values['SSIM']) / len(values['SSIM']) for hour, values in hourly_metrics.items()}
    # average_mse_by_hour = {hour: sum(values['MSE']) / len(values['MSE']) for hour, values in hourly_metrics.items()}

    # # Sort hours and prepare data for plotting
    # sorted_hours = sorted(average_ssim_by_hour.keys())
    # average_ssim_values = [average_ssim_by_hour[hour] for hour in sorted_hours]
    # average_mse_values = [average_mse_by_hour[hour] for hour in sorted_hours]

    # # Plotting SSIM and MSE over time
    # plt.figure(figsize=(10, 5))
    # plt.plot(sorted_hours, average_ssim_values, label='Average SSIM', marker='o')
    # plt.plot(sorted_hours, average_mse_values, label='Average MSE', marker='o')

    # # Labels and title
    # plt.xlabel('Time (Hour)')
    # plt.ylabel('Metrics')
    # plt.title('Average SSIM and MSE Over Time')
    # plt.legend()
    # plt.grid(True)

    # # Save the plot
    # output_plot_path = '/scratch/YOURNAME/project/ControlNet/a_inference/evaluate_507/a_analysis/average_metrics_over_time.png'
    # plt.savefig(output_plot_path)
    # # plt.show()



    # file_path = '/scratch/YOURNAME/project/ControlNet/a_inference/evaluate/result.json'
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Group SSIM and MSE values by hour
    hourly_metrics = defaultdict(lambda: {'SSIM': [], 'MSE': []})

    for item in data:
        # Extract hour from "generated_path"
        match = re.search(r'hour(\d+)_', item["generated_path"])
        if match:
            hour = int(match.group(1))
            hourly_metrics[hour]['SSIM'].append(item["SSIM"])
            hourly_metrics[hour]['MSE'].append(item["MSE"])

    # Calculate average SSIM and MSE for each hour
    average_ssim_by_hour = {hour: sum(values['SSIM']) / len(values['SSIM']) for hour, values in hourly_metrics.items()}
    average_mse_by_hour = {hour: sum(values['MSE']) / len(values['MSE']) for hour, values in hourly_metrics.items()}

    # Sort hours and prepare data for plotting
    sorted_hours = sorted(average_ssim_by_hour.keys())
    average_ssim_values = [average_ssim_by_hour[hour] for hour in sorted_hours]
    average_mse_values = [average_mse_by_hour[hour] for hour in sorted_hours]

    # Output directory for saving plots
    # output_plot_dir = '/scratch/YOURNAME/project/ControlNet/a_inference/evaluate/a_analysis'
    os.makedirs(output_plot_dir, exist_ok=True)

    # Plotting SSIM over time
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_hours, average_ssim_values, label='Average SSIM', marker='o', color='#4876b1')
    plt.xlabel('Time (Hour)')
    plt.ylabel('Average SSIM')
    plt.title('Average SSIM Over Time')
    plt.grid(True)
    output_ssim_path = os.path.join(output_plot_dir, 'average_ssim_over_time.png')
    plt.savefig(output_ssim_path)

    # Plotting MSE over time
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_hours, average_mse_values, label='Average MSE', marker='o', color='#e37e27')
    plt.xlabel('Time (Hour)')
    plt.ylabel('Average MSE')
    plt.title('Average MSE Over Time')
    plt.grid(True)
    output_mse_path = os.path.join(output_plot_dir, 'average_mse_over_time.png')
    plt.savefig(output_mse_path)
    print("evalutaing finished .....")