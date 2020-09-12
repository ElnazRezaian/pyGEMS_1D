# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 01:11:37 2020

@author: ashis
"""
import re 
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def calc_RAE(truth,pred):
    
    RAE = np.mean(np.abs(truth-pred))/np.max(np.abs(truth))
    
    return RAE


def parseValue(expr):
	try:
		return eval(expr)
	except:
		return eval(re.sub("\s+", ",", expr))
	else:
		return expr



def parseLine(line):
	eq = line.find('=')
	if eq == -1: raise Exception()
	key = line[:eq].strip()
	value = line[eq+1:-1].strip()
	return key, parseValue(value)



def readInputFile(inputFile):

	readDict = {}
	with open(inputFile) as f:
		contents = f.readlines()

	for line in contents: 
		try:
			key, val = parseLine(line)
			readDict[key] = val

		except:
			pass 

	return readDict



class viz_params:
    
    def __init__(self,params_path):
        
        paramsDict = readInputFile(params_path)
        
        #Solutions File Path
        self.FOMPath = str(paramsDict["FOMPath"])
        self.PODPath = str(paramsDict["PODPath"])
        self.decPath = str(paramsDict["decPath"])
        self.encPath = str(paramsDict["encPath"])

        #Solutions to plot
        self.pod = bool(paramsDict["pod"])
        self.dec = bool(paramsDict["dec"])
        self.enc = bool(paramsDict["enc"]) 
        
        #FOM Parameters
        self.nx = int(paramsDict["nx"])
        self.dt = float(paramsDict["dt"])
        
        #Labels
        self.labs = list(paramsDict["labs"])
        
        #Line Styles
        self.lin_typ = list(paramsDict["lin_typ"])
        
        #Field Plots
        self.field_plots = bool(paramsDict["field_plots"])
        
        #time-instances to plot at
        self.fp_ts = list(paramsDict["fp_ts"])
        
        #Field Movies
        self.field_movies = bool(paramsDict["field_movies"])
        self.st = int(paramsDict["st"])
        self.end = int(paramsDict["end"])
        self.inter = int(paramsDict["inter"])
        
        #Error Plots
        self.error_plots = bool(paramsDict["error_plots"])
        
        #Error Movies
        self.error_movies = bool(paramsDict["error_movies"])
        
        
    def visualize(self,params):
        
        x = np.linspace(0,0.01,num=params.nx)
        file_path_figs = './Visualizations/'
        
        if not os.path.isdir(file_path_figs): os.mkdir(file_path_figs)
        #loading the solutions
        
        sol_FOM = np.load(params.FOMPath)
        
        
        if(params.pod):
            sol_POD = np.load(params.PODPath)
            
        if(params.dec):
            sol_dec = np.load(params.decPath)
            
        if(params.enc):
            sol_enc = np.load(params.encPath)
            
            
        if(params.field_plots):
            
            print("Generating Field Plots")
            if not os.path.isdir(file_path_figs+'Field Plots/'): os.mkdir(file_path_figs+'Field Plots/')
            
            for i in params.fp_ts:
                
                f, axs = plt.subplots(2)
                
                axs[0].plot(x,sol_FOM[:,0,i],params.lin_typ[0],label=params.labs[0])
                
                if(params.pod):
                    axs[0].plot(x,sol_POD[:,0,i],params.lin_typ[1],label=params.labs[1])
                    
                if(params.dec):
                    axs[0].plot(x,sol_dec[:,0,i],params.lin_typ[2],label=params.labs[2])
        
                if(params.enc):
                    axs[0].plot(x,sol_enc[:,0,i],params.lin_typ[3],label=params.labs[3])
        
                axs[0].set_xlabel('x')
                axs[0].set_ylabel('P (Pa)')
                axs[0].set_ylim([np.amin(sol_FOM[:,0,params.st:params.end]),np.amax(sol_FOM[:,0,params.st:params.end])])
                axs[0].set_xlim([0, 0.01])
                axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
                axs[0].legend(loc='upper left')
        
            
                axs[1].plot(x,sol_FOM[:,1,i],params.lin_typ[0],label=params.labs[0])
                
                if(params.pod):
                    axs[1].plot(x,sol_POD[:,1,i],params.lin_typ[1],label=params.labs[1])
                    
                if(params.dec):
                    axs[1].plot(x,sol_dec[:,1,i],params.lin_typ[2],label=params.labs[2])
        
                if(params.enc):
                    axs[1].plot(x,sol_enc[:,1,i],params.lin_typ[3],label=params.labs[3])
        
                axs[1].set_xlabel('x')
                axs[1].set_ylabel('U (m/s)')
                axs[1].set_ylim([np.amin(sol_FOM[:,1,params.st:params.end]),np.amax(sol_FOM[:,1,params.st:params.end])])
                axs[1].set_xlim([0, 0.01])
                axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            
            
                for ax in f.get_axes():
                    ax.label_outer()
                
                plt.show()
                f.savefig(file_path_figs+'Field Plots/P1_P2_'+str(i)+'.png')
                plt.close()
            
            
                f, axs = plt.subplots(2)
                
                axs[0].plot(x,sol_FOM[:,2,i],params.lin_typ[0],label=params.labs[0])
                
                if(params.pod):
                    axs[0].plot(x,sol_POD[:,2,i],params.lin_typ[1],label=params.labs[1])
                    
                if(params.dec):
                    axs[0].plot(x,sol_dec[:,2,i],params.lin_typ[2],label=params.labs[2])
        
                if(params.enc):
                    axs[0].plot(x,sol_enc[:,2,i],params.lin_typ[3],label=params.labs[3])
        
                axs[0].set_xlabel('x')
                axs[0].set_ylabel('T (K)')
                axs[0].set_ylim([np.amin(sol_FOM[:,2,params.st:params.end]),np.amax(sol_FOM[:,2,params.st:params.end])])
                axs[0].set_xlim([0, 0.01])
                axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
                axs[0].legend(loc='upper left')
        
            
                axs[1].plot(x,sol_FOM[:,3,i],params.lin_typ[0],label=params.labs[0])
                
                if(params.pod):
                    axs[1].plot(x,sol_POD[:,3,i],params.lin_typ[1],label=params.labs[1])
                    
                if(params.dec):
                    axs[1].plot(x,sol_dec[:,3,i],params.lin_typ[2],label=params.labs[2])
        
                if(params.enc):
                    axs[1].plot(x,sol_enc[:,3,i],params.lin_typ[3],label=params.labs[3])
        
                axs[1].set_xlabel('x')
                axs[1].set_ylabel('U (m/s)')
                axs[1].set_ylim([np.amin(sol_FOM[:,3,params.st:params.end]),np.amax(sol_FOM[:,3,params.st:params.end])])
                axs[1].set_xlim([0, 0.01])
                axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            
            
                for ax in f.get_axes():
                    ax.label_outer()
                
                plt.show()
                f.savefig(file_path_figs+'Field Plots/P3_P4_'+str(i)+'.png')
                plt.close()
        
            
        if(params.field_movies):
            
            print("Generating Field Movies")
            if not os.path.isdir(file_path_figs+'Field Movies/'): os.mkdir(file_path_figs+'Field Movies/')
            
            images=[]
            images1=[]
            
            for i in range(params.st,params.end):
            
                if((i % params.inter) != 0):
                    continue
        
        
                f, axs = plt.subplots(2)
                
                axs[0].plot(x,sol_FOM[:,0,i],params.lin_typ[0],label=params.labs[0])
                
                if(params.pod):
                    axs[0].plot(x,sol_POD[:,0,i],params.lin_typ[1],label=params.labs[1])
                    
                if(params.dec):
                    axs[0].plot(x,sol_dec[:,0,i],params.lin_typ[2],label=params.labs[2])
        
                if(params.enc):
                    axs[0].plot(x,sol_enc[:,0,i],params.lin_typ[3],label=params.labs[3])
        
                axs[0].set_xlabel('x')
                axs[0].set_ylabel('P (Pa)')
                axs[0].set_ylim([np.amin(sol_FOM[:,0,params.st:params.end]),np.amax(sol_FOM[:,0,params.st:params.end])])
                axs[0].set_xlim([0, 0.01])
                axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
                axs[0].legend(loc='upper left')
        
            
                axs[1].plot(x,sol_FOM[:,1,i],params.lin_typ[0],label=params.labs[0])
                
                if(params.pod):
                    axs[1].plot(x,sol_POD[:,1,i],params.lin_typ[1],label=params.labs[1])
                    
                if(params.dec):
                    axs[1].plot(x,sol_dec[:,1,i],params.lin_typ[2],label=params.labs[2])
        
                if(params.enc):
                    axs[1].plot(x,sol_enc[:,1,i],params.lin_typ[3],label=params.labs[3])
        
                axs[1].set_xlabel('x')
                axs[1].set_ylabel('U (m/s)')
                axs[1].set_ylim([np.amin(sol_FOM[:,1,params.st:params.end]),np.amax(sol_FOM[:,1,params.st:params.end])])
                axs[1].set_xlim([0, 0.01])
                axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            
            
                for ax in f.get_axes():
                    ax.label_outer()
        
                
                f.canvas.draw()
                image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
                images.append(image.reshape(f.canvas.get_width_height()[::-1] + (3,)))
                plt.close()
            
        
                f, axs = plt.subplots(2)
                
                axs[0].plot(x,sol_FOM[:,2,i],params.lin_typ[0],label=params.labs[0])
                
                if(params.pod):
                    axs[0].plot(x,sol_POD[:,2,i],params.lin_typ[1],label=params.labs[1])
                    
                if(params.dec):
                    axs[0].plot(x,sol_dec[:,2,i],params.lin_typ[2],label=params.labs[2])
        
                if(params.enc):
                    axs[0].plot(x,sol_enc[:,2,i],params.lin_typ[3],label=params.labs[3])
        
                axs[0].set_xlabel('x')
                axs[0].set_ylabel('T (K)')
                axs[0].set_ylim([np.amin(sol_FOM[:,2,params.st:params.end]),np.amax(sol_FOM[:,2,params.st:params.end])])
                axs[0].set_xlim([0, 0.01])
                axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
                axs[0].legend(loc='upper left')
        
            
                axs[1].plot(x,sol_FOM[:,3,i],params.lin_typ[0],label=params.labs[0])
                
                if(params.pod):
                    axs[1].plot(x,sol_POD[:,3,i],params.lin_typ[1],label=params.labs[1])
                    
                if(params.dec):
                    axs[1].plot(x,sol_dec[:,3,i],params.lin_typ[2],label=params.labs[2])
        
                if(params.enc):
                    axs[1].plot(x,sol_enc[:,3,i],params.lin_typ[3],label=params.labs[3])
        
                axs[1].set_xlabel('x')
                axs[1].set_ylabel('U (m/s)')
                axs[1].set_ylim([np.amin(sol_FOM[:,3,params.st:params.end]),np.amax(sol_FOM[:,3,params.st:params.end])])
                axs[1].set_xlim([0, 0.01])
                axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            
            
                for ax in f.get_axes():
                    ax.label_outer()
                
                f.canvas.draw()
                image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
                images1.append(image.reshape(f.canvas.get_width_height()[::-1] + (3,)))
            
                plt.close()
            
            kwargs_write = {'fps':10.0, 'quantizer':'nq'}
            imageio.mimsave(file_path_figs+'Field Movies/'+'P_1_2.gif', images, fps=100)
            
            kwargs_write = {'fps':1.0, 'quantizer':'nq'}
            imageio.mimsave(file_path_figs+'Field Movies/'+'P_3_4.gif', images1, fps=100)
        
        
        if(params.error_plots or params.error_movies):
            
            #calculate RAE
            if(params.dec):
                RAE_dec = np.zeros((params.end-params.st,4))
            if(params.pod):
                RAE_pod = np.zeros((params.end-params.st,4))
            if(params.enc):
                RAE_enc = np.zeros((params.end-params.st,4))
            
            for i in range(params.st,params.end):
                for j in range(4):
                    
                    if(params.dec):
                        RAE_dec[i-params.st,j] = calc_RAE(sol_dec[:,j,i],sol_FOM[:,j,i])
                    if(params.enc):
                        RAE_enc[i-params.st,j] = calc_RAE(sol_enc[:,j,i],sol_FOM[:,j,i])
                    if(params.pod):
                        RAE_pod[i-params.st,j] = calc_RAE(sol_POD[:,j,i],sol_FOM[:,j,i])
                        
        
        if(params.error_plots):
            
            print("Generating Error Plots")
            
            if not os.path.isdir(file_path_figs+'Error Plots/'): os.mkdir(file_path_figs+'Error Plots/')
            
            t = np.linspace(params.st,params.end,params.end-params.st)*((params.dt)/1e-6)
            labels = ['Pressure','Velocity','Temperature','Y1']
            for i in range(4):
                f,axs = plt.subplots(1)
                f.suptitle(labels[i])
                if(params.dec):
                    axs.plot(t,RAE_dec[:,i]*100,params.lin_typ[2],label=params.labs[2])
                if(params.enc):
                    axs.plot(t,RAE_enc[:,i]*100,params.lin_typ[3],label=params.labs[3])
                if(params.pod):
                    axs.plot(t,RAE_pod[:,i]*100,params.lin_typ[1],label=params.labs[1])
                
                
                axs.legend()
                axs.set_xlabel('Time ($\mu$s)')
                axs.set_ylabel('RAE(%)')
                axs.ticklabel_format(axis="y", style="sci", scilimits=(0,0))        
                plt.show()
                f.savefig(file_path_figs+'Error Plots/'+labels[i]+'.png')
        
        
        if(params.error_movies):
            
            print("Generating Error Movies")
            if not os.path.isdir(file_path_figs+'Error Plots/'): os.mkdir(file_path_figs+'Error Plots/')
            
            labels = ['Pressure','Velocity','Temperature','Y1']
            
            for prim_num in range(4):
                
                images=[]
                for i in range(params.st,params.end):
                    
                    if((i % params.inter) != 0):
                        continue
                    
                    f,axs = plt.subplots(1)
                    f.suptitle(labels[prim_num])
                    if(params.dec):
                        axs.plot(t,RAE_dec[:,prim_num]*100,params.lin_typ[2],label=params.labs[2])
                    if(params.enc):
                        axs.plot(t,RAE_enc[:,prim_num]*100,params.lin_typ[3],label=params.labs[3])
                    if(params.pod):
                        axs.plot(t,RAE_pod[:,prim_num]*100,params.lin_typ[1],label=params.labs[1])
                
                    axs.axvline(x=250,linestyle='--')
                    axs.axvline(x=i,linestyle='-')
                    
                    
                    axs.legend(loc=1)
                    axs.set_xlabel('Time($\mu$s)')
                    axs.set_ylabel('RAE(%)')
                    axs.set_frame_on(False)
                    axs.set_frame_on(True)
                    f.canvas.draw()
                    image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
                    images.append(image.reshape(f.canvas.get_width_height()[::-1] + (3,)))
                   
                kwargs_write = {'fps':1.0, 'quantizer':'nq'}
                imageio.mimsave(file_path_figs+'Error Plots/'+labels[prim_num]+'_err_mov.gif', images, fps=100)    
        print("Visualizations saved")
        
        
        