# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:56:31 2017

@author: kulpatil
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
import scipy.stats as stats



class CA(object):
    def create_list_sum_of_categories(df, var, cat, var2):
        list1 = []
        for cat2 in range(int(df[var2].min()), int(df[var2].max())+1):
                list1.append( len(df[ (df[var] == cat) & (df[var2] == cat2) ]))   
        #return list1

    def chi_square_of_df_cols(df,col1,col2):
        ''' for each category of col1 create list with sums of each category of col2'''
        result_list = []
        for cat in range(int(df[col1].min()), int(df[col1].max())+1):
            result_list.append(create_list_sum_of_categories(df,col1,cat,col2)) 

    #return scs.chi2_contingency(result_list)
    
    def __init__(self, ct):
        self.rows = ct.index.values if hasattr(ct, 'index') else None
        self.cols = ct.columns.values if hasattr(ct, 'columns') else None
        
        # contingency table
        N = np.matrix(ct, dtype=float)
        #chisq=stats.chi2_contingency(ct,self.rows,self.cols)
        # correspondence matrix from contingency table
        P = N / N.sum()

        # row and column marginal totals of P as vectors
        r = P.sum(axis=1)
        c = P.sum(axis=0).T

        # diagonal matrices of row/column sums
        D_r_rsq = np.diag(1. / np.sqrt(r.A1))
        D_c_rsq = np.diag(1. / np.sqrt(c.A1))

        # the matrix of standarized residuals
        S = D_r_rsq * (P - r * c.T) * D_c_rsq

        # compute the SVD
        U, D_a, V = svd(S, full_matrices=False)
        D_a = np.asmatrix(np.diag(D_a))
        V = V.T

        # principal coordinates of rows
        F = D_r_rsq * U * D_a

        # principal coordinates of columns
        G = D_c_rsq * V * D_a

        # standard coordinates of rows
        X = D_r_rsq * U

        # standard coordinates of columns
        Y = D_c_rsq * V

        # the total variance of the data matrix
        inertia = sum([(P[i,j] - r[i,0] * c[j,0])**2 / (r[i,0] * c[j,0])
                       for i in range(N.shape[0])
                       for j in range(N.shape[1])])
        #print inertia
        self.F = F.A
        self.G = G.A
        self.X = X.A
        self.Y = Y.A
        self.inertia = inertia
        self.eigenvals = np.diag(D_a)**2
        #print self.eigenvals
                      
    def plot(self):
        """Plot the first and second dimensions."""
        labels=[]
        #fig_size[0] = 12
        #fig_size[1] = 9
        #plt.rcParams["figure.figsize"] = fig_size
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        xmin, xmax = None, None
        ymin, ymax = None, None
        if self.rows is not None:
            for i, t in enumerate(self.rows):
                x, y = self.F[i,0], self.F[i,1]
                print (x,'|',y,'|',t,'|',i)
                labels.append(t)
                plt.text(x, y, t, va='center', ha='center', color='r', fontsize=8)
                xmin = min(x, xmin if xmin else x)
                xmax = max(x, xmax if xmax else x)
                ymin = min(y, ymin if ymin else y)
                ymax = max(y, ymax if ymax else y)
        else:
            plt.plot(self.F[:, 0], self.F[:, 1], 'ro')
        import mpld3        
        if self.cols is not None:
            for i, t in enumerate(self.cols):
                x, y = self.G[i,0], self.G[i,1]
               # if (t)  in "Repair":
                print (x,'|',y,'|',t,'|',i)
                plt.text(x, y, t, va='center', ha='center', color='b', fontsize=14)
                xmin = min(x, xmin if xmin else x)
                xmax = max(x, xmax if xmax else x)
                ymin = min(y, ymin if ymin else y)
                ymax = max(y, ymax if ymax else y)
        else:
            plt.plot(self.G[:, 0], self.G[:, 1], 'bs')

        if xmin and xmax:
            pad = (xmax - xmin) * 0.1
            plt.xlim(xmin - pad, xmax + pad)
        if ymin and ymax:
            pad = (ymax - ymin) * 0.1
            plt.ylim(ymin - pad, ymax + pad)

        plt.grid()
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        #tooltip = mpld3.plugins.PointLabelTooltip(plt, labels=labels)
        #mpld3.plugins.connect(fig, tooltip)
        #mpld3.show()
        
    def scree_diagram(self, perc=True, *args, **kwargs):
        """Plot the scree diagram."""
        eigenvals = self.eigenvals
        xs = np.arange(1, eigenvals.size + 1, 1)
        ys = 100. * eigenvals / eigenvals.sum() if perc else eigenvals
        plt.plot(xs, ys, *args, **kwargs)
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalue' + (' [%]' if perc else ''))
        

def _test():
    import pandas as pd

    df = pd.io.parsers.read_csv('CADSx.csv')
    df=df.fillna(0)
    df = df.set_index('Brand')
    
    print (df.describe())
    print (df.head())
    #print(chi_square_of_df_cols(df))
    ca = CA(df)
    
    plt.figure(50)
    ca.plot()

    plt.figure(50)
    ca.scree_diagram()

    plt.show()
    return ca

if __name__ == '__main__':
   ca= _test()
   columns= ca.cols
   eigenval= ca.eigenvals
   rows= ca.rows
   
