import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk
from tkinter.ttk import*
from tkinter import font
from PIL import ImageTk,Image
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd
from math import sqrt, log, exp, pi
from random import uniform

random_seed=36788765
np.random.seed(random_seed)

root = tk.Tk()
root.title("DAA - Exceptation Maximization Algorithm")
root.geometry("1000x500")
root.resizable(False, False)

panedwindow=tk.PanedWindow(root,orient=HORIZONTAL)
panedwindow.pack(fill=BOTH, expand=True) 

frame1 = tk.Frame(panedwindow, width=500, height=100, bg="white", relief=SUNKEN)

intro = tk.Label(frame1, text="Computing best Gaussian Model with\n Expectation-Maximization"
                 , font = ('times new roman', 18,'bold'), bg="white", fg="#2B68A3")

intro.place(x=50,y=90)

em = Image.open("em.jpg")
emImage = em.resize((500, 265))
test = ImageTk.PhotoImage(emImage)

img = tk.Label(frame1,image=test)
img.image=test

img.place(x=0,y=230)

frame2 = tk.Frame(panedwindow, width=500, height=100,relief=SUNKEN, bg="#F3F4F6")
panedwindow.add(frame1)  
panedwindow.add(frame2)

tk.Label(frame2, text="Mean1", font = ('times new roman', 14),bg="#F3F4F6").grid(row=0, padx=(150,20), pady=(110,15))
tk.Label(frame2, text="Standard1", font = ('times new roman', 14),bg="#F3F4F6").grid(row=1, padx=(150,20), pady=10)
tk.Label(frame2, text="Mean2", font = ('times new roman', 14),bg="#F3F4F6").grid(row=2, padx=(150,20), pady=10)
tk.Label(frame2, text="Standard2", font = ('times new roman', 14),bg="#F3F4F6").grid(row=3, padx=(150,20), pady=10)


e1 = tk.Entry(frame2, font = ('times new roman', 15, 'bold'), bd=1, width=8)
e1.grid(row=0, column=1, pady=(110,15))
e2 = tk.Entry(frame2, font = ('times new roman', 15, 'bold'), bd=1, width=8)
e2.grid(row=1, column=1)
e3 = tk.Entry(frame2, font = ('times new roman', 15, 'bold'), bd=1, width=8)
e3.grid(row=2, column=1)
e4 = tk.Entry(frame2, font = ('times new roman', 15, 'bold'), bd=1, width=8)
e4.grid(row=3, column=1)

compute = tk.Label(frame2, text="", font = ('times new roman', 12),fg='green',bg="#F3F4F6")
compute.place(x=10, y=410)

comp = tk.Label(frame2, text="", font = ('times new roman', 12),fg='green',bg="#F3F4F6")
comp.place(x=10, y=452)

c = tk.Label(frame2, text="", font = ('times new roman', 12),fg='green',bg="#F3F4F6")
c.place(x=10, y=475)

class Gaussian:

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, datum):
        u = (datum - self.mu) / abs(self.sigma)
        y = (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
        return y

    def __repr__(self):
        return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)

def test():
    while True:
        try:
            Mean1 = float(e1.get())
            Standard_dev1 = float(e2.get())
            Mean2 = float(e3.get())
            Standard_dev2 = float(e4.get())
            
            y1 = np.random.normal(Mean1, Standard_dev1, 1500)
            y2 = np.random.normal(Mean2, Standard_dev2, 750)
            data = np.append(y1, y2)
            Min_graph = min(data)
            Max_graph = max(data)
            x = np.linspace(Min_graph, Max_graph, 2000) 
            
            best_single = Gaussian(np.mean(data), np.std(data))
            
            comp['fg'] = 'green'
            comp['text']='Best single Gaussian: μ = {:.2}, σ = {:.2}'.format(best_single.mu, best_single.sigma)

            g_single = stats.norm(best_single.mu, best_single.sigma).pdf(x)
            sns.distplot(data, bins=20, kde=False, norm_hist=True)

            plt.plot(x, g_single, label='single gaussian')
            plt.legend()
            plt.show()

        except ValueError:
                compute['text']=""
                c['text']=""
                comp['fg']='red'
                comp['text']='You must give a float number!'
                break


def bestMixture():

 while True:
        try:
           
            Mean1 = float(e1.get())
            Standard_dev1 = float(e2.get())
            Mean2 = float(e3.get())
            Standard_dev2 = float(e4.get())
            
            y1 = np.random.normal(Mean1, Standard_dev1, 1500)
            y2 = np.random.normal(Mean2, Standard_dev2, 750)
            data = np.append(y1, y2)

            Min_graph = min(data)
            Max_graph = max(data)
            x = np.linspace(Min_graph, Max_graph, 2000) 

            class GaussianMixture_self:

                def __init__(self, data, mu_min=min(data), mu_max=max(data), sigma_min=1, sigma_max=1, mix=.5):
                    self.data = data
                    
                    self.one = Gaussian(uniform(mu_min, mu_max), 
                                        uniform(sigma_min, sigma_max))
                    self.two = Gaussian(uniform(mu_min, mu_max), 
                                        uniform(sigma_min, sigma_max))
                    
                    self.mix = mix

                def Estep(self):

                    self.loglike = 0.
                    for datum in self.data:  

                        wp1 = self.one.pdf(datum) * self.mix
                        wp2 = self.two.pdf(datum) * (1. - self.mix)

                        den = wp1 + wp2

                        wp1 /= den   
                        wp2 /= den     

                        self.loglike += log(den)

                        yield (wp1, wp2)

                def Mstep(self, weights):
                 
                    (left, rigt) = zip(*weights) 
                    one_den = sum(left)
                    two_den = sum(rigt)

                    self.one.mu = sum(w * d  for (w, d) in zip(left, data)) / one_den
                    self.two.mu = sum(w * d  for (w, d) in zip(rigt, data)) / two_den
                    
                    self.one.sigma = sqrt(sum(w * ((d - self.one.mu) ** 2)
                                              for (w, d) in zip(left, data)) / one_den)
                    self.two.sigma = sqrt(sum(w * ((d - self.two.mu) ** 2)
                                              for (w, d) in zip(rigt, data)) / two_den)
             
                    self.mix = one_den / len(data)

                    
                def iterate(self, N=1, verbose=False):

                    for i in range(1, N+1):
                        self.Mstep(self.Estep())
                        if verbose:
                            print('{0:2} {1}'.format(i, self))
                    self.Estep() 

                def pdf(self, x):
                    return (self.mix)*self.one.pdf(x) + (1-self.mix)*self.two.pdf(x)
                
                def __repr__(self):
                    return 'GaussianMixture({0}, {1}, mix={2.03})'.format(self.one, 
                                                                          self.two, 
                                                                          self.mix)

                def __str__(self):
                    return 'Mixture: {0}, {1}, mix={2:.03})'.format(self.one, 
                                                                    self.two, 
                                                                    self.mix)

            n_iterations = 300
            n_random_restarts = 4
            best_mix = None
            best_loglike = float('-inf')

            for _ in range(n_random_restarts):
                mix = GaussianMixture_self(data)
                for _ in range(n_iterations):
                    
                        mix.iterate()
                        if mix.loglike > best_loglike:
                            best_loglike = mix.loglike
                            best_mix = mix
                    
         
            compute['text']='Input Gaussian {:}: μ = {:.2}, σ = {:.2}'.format("1", Mean1, Standard_dev1)+'\nInput Gaussian {:}: μ = {:.2}, σ = {:.2}'.format("2", Mean2, Standard_dev2)
            comp['fg'] = 'green'
            comp['text']='Gaussian {:}: μ = {:.2}, σ = {:.2}, weight = {:.2}'.format("1", best_mix.one.mu, best_mix.one.sigma, best_mix.mix)
            c['text']='Gaussian {:}: μ = {:.2}, σ = {:.2}, weight = {:.2}'.format("2", best_mix.two.mu, best_mix.two.sigma, (1-best_mix.mix))

            sns.distplot(data, bins=20, kde=False, norm_hist=True);
            g_both = [best_mix.pdf(e) for e in x]
            plt.plot(x, g_both, label='gaussian mixture');
            g_left = [best_mix.one.pdf(e) * best_mix.mix for e in x]
            plt.plot(x, g_left, label='gaussian one');
            g_right = [best_mix.two.pdf(e) * (1-best_mix.mix) for e in x]
            plt.plot(x, g_right, label='gaussian two');
            plt.legend();
            plt.show()
            
        except ZeroDivisionError:
            compute['text']=""
            c['text']=""
            comp['fg']='red'
            comp['text']='You must give a different value from zero!'
            break
        
        except ValueError:
            compute['text']=""
            c['text']=""
            comp['fg']='red'
            comp['text']='You must give a float number!'
            break
      

st = Style()
st.configure('W.TButton', background='#4A8BCA', foreground='#295B8B', font=('Arial', 10 ))

graphButton = Button(frame2, text='Show', style='W.TButton', command=test).grid(row=5, column=1, sticky=E, pady=(20,4))
graphButton2 = Button(frame2, text='Best Model', style='W.TButton', command=bestMixture).grid(row=6, sticky=E, column=1, pady=4)
quitButton = Button(frame2, text='Quit', style='W.TButton', command=root.quit).grid(row=7, sticky=E, column=1, pady=4)



