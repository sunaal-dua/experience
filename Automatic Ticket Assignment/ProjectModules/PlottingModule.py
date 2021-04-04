# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import subprocess
import sys
print("installing wordcloud python package...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "wordcloud"])
print("Done!")
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# count plot
def plot_count_plot(values, labels, xlabel="", ylabel="", title=''):
    """
    Plots a count plot. Width and height fixed at 20px and 7px respectively.
    
    values: pass count(integers) as an array/list/series
    labels: pass labels(string) corresponding to the count as an array/list
    xlabel: label for x-axis (string)
    ylabel: label for y-axis (string)
    title: title of the plot (string)
    """
    plt.figure(figsize=(20,7))
    plt.xticks(rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.bar(x=labels,height=values)
    plt.show()


# distribution plot
def distribution_plot(variable,title=''):
    """
    Plots a distribution plot. Width and height fixed at 7px and 5px respectively.
    
    variable: pass continous numerical values as an array/list/series
    title: title of the plot (string)
    """
    plt.figure(figsize=(7,5))
    sns.distplot(variable).set_title(title)
    plt.show()
    

# wordcloud plot
def create_wordclouds(f, x):
    """
    Plots a wordcloud plot using wordcloud library. Width and height fixed at 480px and 480px respectively. 
    This plot Will display top 50 words from the text
    
    f: text as an array/list/series
    x: group label
    """
    wordclouds_0=' '.join(map(str, f))
    wc = WordCloud(width=480, height=480, max_font_size=40, min_font_size=10, max_words=50).generate(wordclouds_0)
    plt.figure(figsize=(7,7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most common 50 words of {}".format(x))
    plt.margins(x=0, y=0)
    plt.show()