import pandas as pd
import numpy as np
import os

pd.set_option("display.max_columns", None)

# Leemos el csv
df = pd.read_csv("../data/1_raw/studient.csv", sep=";")