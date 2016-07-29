# coding: utf-8
'''
Created on 21/09/2015

@author: Jose Pedro Matos
'''
import numpy as np
import matplotlib.pyplot as plt
import time
from .domination import convexSorting
from .crowding import phenCrowdingNSGAII
from scipy.interpolate import interp1d
import warnings
#===============================================================================
# from .crowding import *
#===============================================================================

from celery import current_task

class OptAlgorithm(object):
    '''
    Abstract class for algorithm definition
    '''

    def __init__(self, data, targets, nVars, evalFun, errorObj, population=1000, epochs=100, lowerThreshold = None,
                 bands=[0.001, 0.01, 0.025, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.975, 0.99, 0.999],
                 bandWidth=0.025, minModels=5, displayEach=10, **kwargs):
        self.trained = False

        # handling of key arguments
        self.trainData=data
        self.targets=targets
        self.evalFun=evalFun
        self.errorObj=errorObj
        self.opt={}
        self.opt['nVars']=nVars
        self.opt['population']=population
        self.opt['epochs']=epochs
        self.opt['bands']=bands
        self.opt['bandWidth']=bandWidth
        self.opt['minModels']=minModels
        self.opt['lowerThreshold']=lowerThreshold
        self.opt['displayEach']=displayEach
        self.timing={}
        self.messaging={}
        self.normalize={}
        
        # addition of generic arguments to options
        self.opt.update(kwargs)
    
        # set needed variables (if missing)
        if not 'bounds' in self.opt:
            self.opt['bounds']=np.vstack((-5*np.ones(self.opt['nVars']),
                                              5*np.ones(self.opt['nVars'])))
        if not 'crowdingWindow' in self.opt:
            self.opt['crowdingWindow']=0.1
        if not 'crowdingFraction' in self.opt:
            self.opt['crowdingFraction']=1.0
        if not 'plotCases' in self.opt:
            self.opt['plotCases']=100
        if not 'plotLinks' in self.opt:
            self.opt['plotLinks']=False
        
        if self.opt['plotCases']>self.opt['population']:
            self.opt['plotCases']=self.opt['population']
    
        self.bandBounds = self.__establishBandBounds__()
    
    def __establishBandBounds__(self):
        # establish bounds
        bounds = np.zeros((2, len(self.opt['bands'])))
        for i0 in range(0, len(self.opt['bands'])):
            if i0>0:
                bounds[0,i0] = max((bounds[1,i0-1],
                                    (self.opt['bands'][i0]+self.opt['bands'][i0-1])/2,
                                    self.opt['bands'][i0]-self.opt['bandWidth']))
            else:
                bounds[0,i0] = max((0,
                                    self.opt['bands'][i0]-self.opt['bandWidth']))
            
            if i0<len(self.opt['bands'])-1:
                bounds[1,i0] = min(((self.opt['bands'][i0+1]+self.opt['bands'][i0])/2,
                                    self.opt['bands'][i0]+self.opt['bandWidth'],
                                    2*self.opt['bands'][i0]))
            else:
                bounds[1,i0] = min((1,
                                    self.opt['bands'][i0]+self.opt['bandWidth']))
        return bounds
    
    def predict(self, data=None, bands=None):
        # run simulations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if data!=None:
                simulations = self.evalFun(self.population, data=data)
            else:
                simulations = self.simulations
            if self.opt['lowerThreshold']!=None:
                simulations[simulations<self.opt['lowerThreshold']] = self.opt['lowerThreshold']
        
        # average results
        averaged = np.empty((simulations.shape[0], self.bandBounds.shape[1]))*np.nan
        for i0 in range(self.bandBounds.shape[1]):
            tmpValidNE = np.where(np.logical_and(self.fit[:,0]>=self.bandBounds[0,i0], self.fit[:,0]<=self.bandBounds[1,i0]))[0] 
            
            if len(tmpValidNE)>self.opt['minModels']:
                tmpValidEr = self.fit[tmpValidNE,1]
                sortIdxs = np.argsort(tmpValidEr)
                threshIdxs = np.where(tmpValidEr<=(np.median(tmpValidEr)+np.min(tmpValidEr))/2)[0]
                if len(threshIdxs)<self.opt['minModels']:
                    idxs = sortIdxs[:self.opt['minModels']]
                else:
                    idxs = threshIdxs
                averaged[:,i0] = np.mean(simulations[:,tmpValidNE[idxs]], axis=1)
        return averaged
    
    def processBands(self, simulations):
        bands = np.array(self.opt['bands'])
        
        for i0 in range(simulations.shape[0]):
            tmpBase = simulations[i0,:]
            tmp = ~np.isnan(tmpBase)
            tmpBase[tmp] = np.sort(tmpBase[tmp])
            if np.sum(tmp)>=2:
                tmp = tmpBase[-1]
                for i1 in range(len(tmpBase)-2, -1, -1):
                    if np.isnan(tmp):
                        tmp = tmpBase[i1]
                    else:
                        if ~np.isnan(tmpBase[i1]):
                            if np.round(tmp,6)<=np.round(tmpBase[i1],6):
                                tmpBase[i1] = np.nan
                            else:
                                tmp = tmpBase[i1]
                tmp = ~np.isnan(tmpBase)
                if np.sum(tmp)>1:
                    simulations[i0,:] = np.interp(bands, bands[tmp], tmpBase[tmp])
                else:
                    simulations[i0,:] = np.nan
            else:
                simulations[i0,:] = np.nan
        return simulations
    
    def start(self):
        tmpMessages=[]
        
        if not self.trained:
            self.__initPopulation__()
            
        self.simulations, self.fit=self.__eval__(self.population)
        self.frontLvls=self.__domination__(self.fit)[1]
    
        # compute
        for i0 in range(self.opt['epochs']):
            self.__iteration__()
            # output times
            tmpKeys=sorted(self.timing.keys())
            tmpToPrint=''
            for key in tmpKeys:
                if key == 'total':
                    tmpToPrint+=key + ': %.2e' % (self.timing['total']) + 's; '
                else:
                    if self.timing['total']!=0:
                        tmpToPrint+=key + ': %.2f' % (self.timing[key]/self.timing['total']*100) + '%; '
                    else:
                        tmpToPrint+=key + ': %.2f' % (0) + '%; '
            if (np.mod(i0+1, self.opt['displayEach'])==0 and i0!=0) or i0==self.opt['epochs']-1:
                print('Epoch %5u: ' % (i0) + tmpToPrint[:-2])
                tmpMessages.append('___Epoch %05u: ' % (i0) + tmpToPrint[:-2])
                
                # QQ plot
                tmpSimulations = self.processBands(self.predict())
                uniform, pValues = self.predictiveQQ(tmpSimulations)
                # metrics
                alpha, xi, piRel = self.metrics(uniform, pValues, tmpSimulations)
                print('    metrics: alpha:%f, xi:%f, pi:%f' % (alpha, xi, piRel))
                tmpMessages.append('______metrics: alpha:%f, xi:%f, pi:%f' % (alpha, xi, piRel))
                
                # output messages
                tmpKeys=sorted(self.messaging.keys())
                tmpToPrint=''
                for key in tmpKeys:
                        tmpToPrint+=key + ': ' + self.messaging[key] + '; '
                if tmpToPrint:
                    print('        msg: ' + tmpToPrint[:-2])
                    tmpMessages.append('______msg: ' + tmpToPrint[:-2])
                
                try:
                    current_task.update_state(state='PROGRESS',
                                              meta={'message': tmpMessages, 'title': None})
                except Exception:
                    pass
        self.trained = True
        
        # export results
        result = {'parameters': self.population, 'normalization': self.normalize,'fitness': self.fit}
        
        # plot
        if self.opt['plotResult']:
            self._plot()
            
        return result    
    
    def _plot(self):
        fig, plotAx = plt.subplots(nrows=2, figsize=(8, 10))
            # Pareto front
        plotAx[0].plot(self.rejectedFit[:,0], self.rejectedFit[:,1], 'ok', label='rejected')
        plotAx[0].plot(self.fit[:,0], self.fit[:,1], 'or', label='chosen')
        plotAx[0].plot(self.fit[self.frontLvls==0,0], self.fit[self.frontLvls==0,1], 'ob', label='Pareto')
        plotAx[0].set_xlim((0,1))
        self.__updateYLim__(self.fit[:,1], plotAx[0])
        plotAx[0].grid()
        plotAx[0].legend(fontsize=10, numpoints=1)
        
        # Fitness
        tmpSim = self._denormalize(y=self.simulations)[0]
        tmpTargets = self._denormalize(y=self.targets)[0]
        plotAx[1].plot(tmpTargets, 'ok', label='Observations')
        tmpBest=np.argmin(self.fit[:,1])
        plotAx[1].plot(tmpSim[:, tmpBest], '--r', label='Best')
        tmpPareto=np.where(self.frontLvls==0)[0]
        tmpSims2Use=tmpPareto[np.argsort(self.fit[tmpPareto,0])[np.linspace(0, len(tmpPareto)-1, self.opt['plotCases'], dtype=np.int32)]]
        for i0 in range(len(tmpSims2Use)):
            if i0 == 0:
                plotAx[1].plot(tmpSim[:, tmpSims2Use[i0]], '--k', alpha=0.5, label='Simulations')
            else:
                plotAx[1].plot(tmpSim[:, tmpSims2Use[i0]], '--k', alpha=0.5)

        plotAx[1].set_xlim((0, len(tmpTargets)))
        self.__updateYLim__(tmpTargets,  plotAx[1])
        plotAx[1].grid()
        plotAx[1].legend(fontsize=10, numpoints=1)
        plt.show(block=True)
    
    def __initPopulation__(self):
        self.population=np.random.uniform(0, 1, (self.opt['population'], self.opt['nVars']))
    
        tmpMin=np.tile(self.opt['bounds'][0,], (self.opt['population'], 1))
        tmpMax=np.tile(self.opt['bounds'][1,], (self.opt['population'], 1))
    
        self.population=self.population*(tmpMax-tmpMin)+tmpMin

    def __iteration__(self):
        start=time.time()
        
        # generate new candidates
        newPopulation=self.newPop(self.fit, self.population)
        
        # evaluate solutions
        newSimulations, newFit=self.__eval__(newPopulation)
        jointPopulation=np.vstack((self.population, newPopulation))
        jointFit=np.vstack((self.fit, newFit))
        jointSimulations=np.hstack((self.simulations, newSimulations))
        
        # domination
        jointFrontLvls=self.__domination__(jointFit)[1]
        
        # crowding
        jointCrowdDist=self.__crowding__(jointSimulations, jointFit, jointFrontLvls)
        
        # select the population 
        toKeepIdxs=self.selection(jointFit, jointFrontLvls, jointCrowdDist, jointPopulation)
        self.fit=jointFit[toKeepIdxs,]
        self.simulations=jointSimulations[:,toKeepIdxs]
        self.population=jointPopulation[toKeepIdxs,]
        
        self.frontLvls=jointFrontLvls[toKeepIdxs,]
        
        self.timing['total']=(time.time()-start)
        
    def __eval__(self, population):
        # evaluate function
        start=time.time()
        simulations=self.evalFun(population)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.opt['lowerThreshold']!=None:
                simulations[simulations<self.opt['lowerThreshold']] = self.opt['lowerThreshold']
        self.timing['run']=(time.time()-start)
        
        # compute errors
        start=time.time()
        self.errorObj.reshapeData(simulations)
        error, quantile=self.errorObj.compute(simulations)
        self.timing['eval']=(time.time()-start)
        
        # compute regularization
        if 'regFun' in self.opt:
            start=time.time()
            reg = self.opt['regFun'](population)
            error+=reg
            self.timing['reg']=(time.time()-start)
        
        # transform
        error = np.log10(error)
        
        fit=np.transpose(np.vstack((quantile, error)))
        return (simulations, fit)
    
    def __crowding__(self, simulations, fit, frontLvls):
        '''Phenotype crowding based on correlations with neighboring series [1 - high crowding; 0 - low crowding]'''
        
        #=======================================================================
        # # correl crowding
        # start=time.time()
        #     # order simulations by non-exceedance
        # tmpOrder=np.argsort(fit[:,0])
        # tmpOrderedSim=simulations[:,tmpOrder]
        # 
        # if 'crowding' in self.opt:
        #     # using openCL
        #     tmpError=tmpOrderedSim.T-np.tile(self.targets[np.newaxis], (tmpOrderedSim.shape[1], 1))
        #     tmpError=tmpError[:, np.random.choice(tmpError.shape[1], tmpError.shape[1]*self.opt['crowdingFraction'], replace=False)]
        #     
        #     self.opt['crowding'].reshapeData(tmpError)
        #     
        #     crowdDist=0.5*self.opt['crowding'].compute(tmpError, round(self.opt['crowdingWindow']*tmpError.shape[0]))
        # else:
        #     # using python
        #     tmpError=tmpOrderedSim-np.tile(self.targets[np.newaxis].T, (1, tmpOrderedSim.shape[1]))
        #     
        #     # compute distances
        #     crowdDist=np.zeros((tmpOrderedSim.shape[1], tmpOrderedSim.shape[1]))
        #     for i0 in range(crowdDist.shape[0]):
        #         for i1 in range(i0+1, min(crowdDist.shape[1], i0+round(self.opt['crowdingWindow']*crowdDist.shape[1]))):
        #             crowdDist[i0, i1]=0.5*(1+np.corrcoef(tmpError[:, i0],tmpError[:, i1])[0,1])
        #         crowdDist[i0, min(crowdDist.shape[1], i0+round(self.opt['crowdingWindow']*crowdDist.shape[1])):]=1
        #     crowdDist+=crowdDist.T
        #     crowdDist+=np.diag(np.ones_like(tmpOrder))
        #     
        #     # search for minima
        # crowdDist=np.amin(crowdDist, 1)
        # 
        #     # put crowding distances by the original order 
        # correlCrowd=np.zeros_like(crowdDist)
        # correlCrowd[tmpOrder]=crowdDist
        # 
        # self.timing['crowdingCorrel']=(time.time()-start)
        #=======================================================================
        
        # crowding NSGAII
        start=time.time()
        phenotype=phenCrowdingNSGAII(fit[:,0], fit[:,1], fronts=frontLvls)
        self.timing['crowdingNSGAII']=(time.time()-start)
        
        return np.vstack((phenotype, phenotype)).T
        
    def __domination__(self, fit):
        start=time.time()
        if ('forceNonExceedance' in self.opt):
            tmpMin=fit[np.argmin(fit[:,1]),0]
            tmpDiff=np.abs(fit[:,0]-tmpMin)
            fronts=convexSorting(fit[:,0], fit[:,1]+self.opt['forceNonExceedance']*tmpDiff)[0]
        else:
            fronts=convexSorting(fit[:,0], fit[:,1])[0]
            
        frontLvl=self.opt['population']*np.ones((fit.shape[0],))
        for i0 in range(len(fronts)):
            frontLvl[fronts[i0]]=i0
            
        self.timing['domination']=(time.time()-start)
        return (fronts, frontLvl)
    
    def newPop(self, fit, population):
        """abstract method"""
        pass
    
    def selection(self, fit, frontLvls, crowdDist):
        """abstract method"""
        pass
    
    def closestPareto(self, fit, population, frontLvls):
        """distance to the Pareto front"""
        inParetoFront=np.where(frontLvls==0)[0]
        
        tmp0=np.tile(fit[inParetoFront,0],(population.shape[0],1)) - np.tile(fit[:,0],(len(inParetoFront),1)).T
        tmp1=np.tile(fit[inParetoFront,1],(population.shape[0],1)) - np.tile(fit[:,1],(len(inParetoFront),1)).T
        tmp0=tmp0-np.min(tmp0)
        tmp1=tmp1-np.min(tmp1)
        tmp0Max=np.max(tmp0)
        tmp1Max=np.max(tmp1)
        if tmp0Max==0:
            tmp0[:]=0.0
        else:
            tmp0=tmp0/tmp0Max
        if tmp1Max==0:
            tmp1[:]=0.0
        else:
            tmp1=tmp1/tmp1Max
        dist=np.sqrt(np.square(tmp0)+np.square(tmp1))
        
        closest=inParetoFront[np.argsort(dist, axis=1)]
        distance=np.sort(dist, axis=1)
        
        return (closest, distance)
    
    def enforceBounds(self, newPopulation):
            
        boundedPopulation=np.max(np.dstack((newPopulation, np.tile(self.opt['bounds'][0,], (self.opt['population'],1)))), axis=2)
        boundedPopulation=np.min(np.dstack((boundedPopulation, np.tile(self.opt['bounds'][1,], (self.opt['population'],1)))), axis=2)
        changed=np.sum(np.abs(boundedPopulation),1)!=np.sum(np.abs(newPopulation),1)
        
        return (boundedPopulation, changed)
  
    def getBands(self):
        return self.opt['bands']
    
    def getPopulation(self):
        return self.population
    
    def setPopulation(self, population):
        self.trained = True
        self.population = population
    
    def __updateYLim__(self, data, axis):
        axis.set_ylim((np.floor(np.min(data)-0.25), np.ceil(np.max(data)+1)))
        
    def predictiveQQ(self, simulations, targets=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if targets==None:
                targets = self.targets
        bands = self.__toCustomLogSpace__(np.array(self.opt['bands'])[::-1])
        pValues = np.empty_like(targets)
        for i0 in range(pValues.shape[0]):
            sims, idxs = np.unique(simulations[i0,:],return_index=True)
            try:
                pValues[i0] = interp1d(sims, bands[idxs], kind='linear', assume_sorted=True)(targets[i0])
            except np.linalg.linalg.LinAlgError as ex:
                pValues[i0] = np.nan
            except ValueError as ex:
                # TODO: handle better extrapolations
                if targets[i0]<sims[0]:
                    pValues[i0] = bands[0]+(bands[0]-bands[1])/(sims[0]-sims[1])*(targets[i0]-sims[0])
                else:
                    pValues[i0] = bands[-1]+(bands[-1]-bands[-2])/(sims[-1]-sims[-2])*(targets[i0]-sims[-1])
        pValues = self.__fromCustomLogSpace__(pValues)
        pValues[pValues<0] = 0
        pValues[pValues>1] = 1
        
        pValues = np.sort(pValues[np.logical_not(np.isnan(pValues))])
        return (np.linspace(0,1, pValues.shape[0]), pValues)
    
    def __toCustomLogSpace__(self, x):
        y = np.empty_like(x)
        tmp = x<0.5
        y[tmp] = np.log(x[tmp])-np.log(0.5)
        tmp = np.logical_not(tmp)
        y[tmp] = -np.log(1-x[tmp])+np.log(0.5)
        return y
        
    def __fromCustomLogSpace__(self, y):
        x = np.empty_like(y)
        tmp = y<0
        x[tmp] = np.exp(y[tmp]+np.log(0.5))
        tmp = np.logical_not(tmp)
        x[tmp] = 1-np.exp(-y[tmp]+np.log(0.5))
        return x
    
    def metrics(self, uniform, pValues, simulations):
        # alpha
        alpha = 1-2*np.mean(np.abs(pValues-uniform));
        # xi
        xi = np.zeros_like(pValues)
        xi[np.logical_or(pValues==1, pValues==0)] = 1
        xi = 1-np.mean(xi)
        # piRel
        bandProb = self.bandBounds[1,:]-self.bandBounds[0,:]
        expected = np.sum(simulations * np.tile(bandProb,(simulations.shape[0],1)), axis=1)
        expected2 = np.sum(np.square(simulations) * np.tile(bandProb,(simulations.shape[0],1)), axis=1)
        piRel = np.mean(np.abs(expected)/np.sqrt(expected2-np.square(expected)))
        return (alpha, xi, piRel)