import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def tidal_model(t, a, b, c, d):
    return a * np.sin(b * (t - c)) + d

def load_csv(path):
    return pd.read_csv(path)

def fit_model(df):
    t = df['hora'].values
    h = df['altura'].values
    guess = [0.5, 0.5, 0, np.mean(h)]
    params, _ = curve_fit(tidal_model, t, h, p0=guess)
    return params

def simulate(params, resolution=0.01):
    a,b,c,d = params
    t = np.arange(0,24,resolution)
    h = tidal_model(t,a,b,c,d)
    return t,h

def plot(df, t, h):
    plt.scatter(df['hora'], df['altura'])
    plt.plot(t, h)
    plt.xlabel("Hora")
    plt.ylabel("Altura (m)")
    plt.title("Modelo Sinusoidal de Mareas - Pichilemu")
    plt.savefig("/mnt/data/modelo_mareas.png")
    plt.close()

