import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Function to download the csv file
def create_csv_download_link(dataframe, filename, title):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{title}</a>'
    return href

# Functions to download the seaborn plot
def save_seaborn_plot_as_png(plot, filename):
    # Get the parent figure of the Axes object
    fig = plot.get_figure()
    # Save the figure as a PNG file
    fig.savefig(filename, format='png')
    # Close the figure to release memory
    plt.close(fig)
    return filename


def create_png_download_link(filename, title):
    # Read the saved PNG file as bytes
    with open(filename, 'rb') as file:
        png_data = file.read()
    # Encode the PNG data as base64
    b64 = base64.b64encode(png_data).decode()
    # Generate the download link
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{title}</a>'
    return href