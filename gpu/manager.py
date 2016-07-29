# coding: utf-8
'''
Created on 24/02/2016

@author: Jose Pedro Matos
'''

import datetime as dt
import numpy as np
import pickle
import re

from celery import current_task

from gpu.ann import ann
from gpu.errorMetrics import errorMetrics
from gpu.psoAlt import PSOAlt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

def fSecond(ref, shift):
    return ref.astype(dt.datetime)+dt.timedelta(seconds=shift)

def fMinute(ref, shift):
    return ref.astype(dt.datetime)+dt.timedelta(minutes=shift)

def fHour(ref, shift):
    return ref.astype(dt.datetime)+dt.timedelta(hours=shift)

def fDay(ref, shift):
    return ref.astype(dt.datetime)+dt.timedelta(days=shift)

def fWeek(ref, shift):
    return ref.astype(dt.datetime)+dt.timedelta(days=7*shift)

def fMonth(ref, shift):
    return ref.replace(year=ref.year+np.int(ref.month+shift/12), month=np.mod(ref.month+shift,12))

def fYear(ref, shift):
    return ref.replace(year=ref.year+shift)

def fGenerateLeads(maxLead, number=None, maxGap=30):
    if number == None:
        number = np.ceil(np.sqrt(maxLead))
    tmp = np.unique(np.round(np.power(1.5, np.linspace(0, np.log(maxLead)/np.log(1.5), number+1))))
    leads = [1,]
    for i0 in range(1,len(tmp)):
        if (tmp[i0]-tmp[i0-1]>maxGap):
            leads.extend(np.linspace(tmp[i0-1], maxLead, np.ceil((maxLead-tmp[i0-1])/maxGap)))
            break
        else:
            leads.append(tmp[i0])
    return [int(i0) for i0 in leads]
    
class Data(object):
    __PERIOD__ = {'second': fSecond,
                  'minute': fMinute,
                  'hour': fHour,
                  'day': fDay,
                  'week': fWeek,
                  'month': fMonth,
                  'year': fYear,
                  }
    
    __CUSTOMFUNCTIONS__ = {('lead(', 's.fLead(l,'),
                           ('filter(','s.fmFilter('),
                           ('targets', 's.values'),
                           ('extra', 's.extra'),
                           ('cycle', 's.cycle(s.xPeriodic)[:,0], s.cycle(s.xPeriodic)[:,1]'),
                           ('sum(', 's.fSum(l,')
                           }
    
    def __init__(self, data, extra=None, refTime=dt.datetime(2000, 1, 1, 0, 0, 0), period='year', timeStepUnit='day', timeStepSize=1):
        self.refTime = refTime
        self.fPeriod = self.__PERIOD__[period]
        self.fTimeJump = self.__PERIOD__[timeStepUnit]
        self.period = period
        self.normalization = {}
        self.fData = None
        self.timeStepUnit = timeStepUnit
        self.timeStepSize = timeStepSize
    
        # join times from data and extra
        allDates = data[0]
        if extra != None:
            for e0 in extra:
                allDates = np.hstack((allDates, e0[0]))
        allDates = np.unique(allDates)
        
        # find a reference before records start
        firstRecord = np.min(allDates)
        dJump = -2
        while np.datetime64(self.fPeriod(self.refTime, dJump))>firstRecord:
            dJump*=2
        tmp = dJump
        dJump = 2
        while np.datetime64(self.fPeriod(self.refTime, tmp + dJump*2))<firstRecord:
            dJump *= 2
        dJump += tmp
        self.startReference = dJump
        
        # build straight time vector
        tmpDates = [firstRecord,]
        for i0 in range(1,len(allDates)):
            tmp = np.datetime64(self.fTimeJump(allDates[i0-1], timeStepSize))
            i1 = 1
            while tmp<allDates[i0]:
                tmpDates.append(tmp)
                i1 += 1
                tmp = np.datetime64(self.fTimeJump(allDates[i0-1], timeStepSize*i1))
            tmpDates.append(allDates[i0])
        self.dates = np.array(tmpDates, dtype='datetime64[s]')
        
        # add information derived from the reference and the period
        self.upDateGroupAndPeriod()
        
        # prepare target and extra arrays
        self.values = np.empty_like(self.dates, dtype=np.double)*np.nan
        self.values[self.__ismember__(data[0], self.dates)] = data[1]
        self.extra = []
        if extra != None:
            for i0 in range(len(extra)):
                self.extra.append(np.empty(self.dates.shape[0], dtype=np.double)*np.nan)
                self.extra[-1][self.__ismember__(extra[i0][0], self.dates)] = extra[i0][1]
        
        self.splitTra = np.ones_like(self.dates, dtype=np.bool)
        self.splitVal = np.zeros_like(self.dates, dtype=np.bool)
     
    def upDateGroupAndPeriod(self):
        self.splitGroup = np.empty(self.dates.shape, dtype=np.int)
        self.xPeriodic = np.empty(self.dates.shape)
        dJump = self.startReference
        for i0 in range(len(self.dates)):
            while np.datetime64(self.fPeriod(self.refTime, dJump+1))<self.dates[i0]:
                dJump+=1
            self.splitGroup[i0] = dJump
            self.xPeriodic[i0] = ((self.dates[i0]-np.datetime64(self.fPeriod(self.refTime, dJump))) / 
                                  (np.datetime64(self.fPeriod(self.refTime, dJump+1))-np.datetime64(self.fPeriod(self.refTime, dJump))))    
        self.xPeriodic = np.mod(self.xPeriodic, 1) + (self.xPeriodic==1).astype(float)
        
    def cycle(self, x):
        return np.transpose(np.array((np.sin(x*2*np.pi), np.cos(x*2*np.pi))))
    
    def split(self, valFraction):
        tmp = np.unique(self.splitGroup)
        np.random.shuffle(tmp)
        traSize = np.int(len(tmp)*(1-valFraction))
        self.idxVal = np.zeros_like(self.splitGroup, dtype=np.bool)
        for i0 in tmp[traSize:]:
            self.idxVal[np.where(self.splitGroup==i0)[0]]=True
        self.idxTra = np.logical_not(self.idxVal)
        self.splitTra = tmp[:traSize]
        self.splitVal = tmp[traSize:]
        
    def setSeasons(self, nSeasons, gamma=0.001, timeCoef=1):
        # build an ordered matrix for each season
        groups = np.unique(self.splitGroup)
        times = np.unique(self.xPeriodic)
        ordered = np.empty((len(groups), len(times)))*np.nan
            # prepare matrix
        for i0 in range(len(groups)):
            tmp = self.xPeriodic[self.splitGroup==groups[i0]] 
            ordered[i0, self.__ismember__(tmp, times)] = self.values[self.splitGroup==groups[i0]] 
            # fill missing with the average
        tmp = np.nanmean(ordered, axis=0)
        tmp[np.isnan(tmp)]=np.nanmean(tmp)
        for i0 in range(len(times)):
            ordered[np.isnan(ordered[:,i0]),i0] = tmp[i0]
        
        # focus on training
        tmp = np.zeros_like(groups, dtype=np.bool)
        for i0, g0 in enumerate(groups):
            if g0 in self.splitTra:
                tmp[i0] = g0
        ordered = ordered[tmp,:]
        
        # normalize
        ordered = (ordered-np.min(ordered))/(np.max(ordered)-np.min(ordered))

        # apply PCA
        pca = PCA(n_components=5)
        pca.fit(ordered)
        tmp = len(np.where(np.cumsum(pca.explained_variance_ratio_)<=0.8)[0]) + 1
        pcaData = pca.components_[:tmp,:]*np.transpose(np.tile(pca.explained_variance_ratio_[:tmp]/pca.explained_variance_ratio_[0],(ordered.shape[1],1)))
        
        # apply clustering
        toCluster = np.hstack((self.cycle(times)*timeCoef,np.transpose(pcaData)))
        kMeans = KMeans(n_clusters=nSeasons)
        kMeans.fit(toCluster)
        tmpSeasons = kMeans.predict(toCluster)
        
        # smooth clusters
        tmpIdxs = np.hstack((times, times[-1]+times, times[-1]*2+times))
        tmpClusters = np.hstack((tmpSeasons, tmpSeasons, tmpSeasons))
        tmpSmoothSeasons = np.zeros((nSeasons, len(times)))
        for i0 in range(nSeasons):
            centers = tmpIdxs[np.where(tmpClusters==i0)[0]]
            for c0 in centers:
                tmp = c0-tmpIdxs
                rbf=np.exp(-1/gamma*np.square(tmp));
                tmpSmoothSeasons[i0,:]=tmpSmoothSeasons[i0,:]+rbf[len(times):2*len(times)]
        tmp = 1/np.sum(tmpSmoothSeasons, axis=0)
        tmpSmoothSeasons = tmpSmoothSeasons*np.tile(tmp,(nSeasons,1))
        tmpSmoothSeasons = np.hstack((tmpSmoothSeasons[:,-2:], tmpSmoothSeasons, tmpSmoothSeasons[:,:1]))
        tmpSmoothTimes = np.hstack((times[-2:]-1, times, times[:1]+1))
        
        # prepare interpolation functions
        self.fSeasonCoef=[]
        self.fSeasonCoefData=[]
        for i0 in range(nSeasons):
            tmp = tmpSmoothSeasons[i0,:]
            tmp[tmp<0.001]=0
            tmp[tmp>0.999]=1
            self.fSeasonCoefData.append((tmpSmoothTimes, tmp))
            self.fSeasonCoef.append(interp1d(tmpSmoothTimes, tmp, kind='linear'))

    def __returnValid__(self, X, y):
        return ~np.isnan(np.sum(np.vstack((X,y)),axis=0))
    
    def getSeasonCoefs(self, season=0):
        return (self.fSeasonCoef[season](self.xPeriodic), np.where(self.fSeasonCoef[season](self.xPeriodic)!=0)[0])
    
    def getTraDates(self, season=0):
        tmpTarget0 = np.logical_and(self.__returnValid__(self.X, self.values), self.idxTra)
        tmpTarget1 = np.logical_and(tmpTarget0, self.fSeasonCoef[season](self.xPeriodic)!=0)
        return self.dates[tmpTarget1]
        
    def getValDates(self, season=0):
        tmpTarget0 = np.logical_and(self.__returnValid__(self.X, self.values), self.idxVal)
        tmpTarget1 = np.logical_and(tmpTarget0, self.fSeasonCoef[season](self.xPeriodic)!=0)
        return self.dates[tmpTarget1]
    
    def getTraX(self, season=0):
        tmpTarget0 = np.logical_and(self.__returnValid__(self.X, self.values), self.idxTra)
        tmpTarget1 = np.logical_and(tmpTarget0, self.fSeasonCoef[season](self.xPeriodic)!=0)
        return (self.X[:, tmpTarget1],
                self.fSeasonCoef[season](self.xPeriodic)[tmpTarget1],
                np.where(self.fSeasonCoef[season](self.xPeriodic)[tmpTarget0]!=0)[0])
    
    def getTraY(self, season=0):
        tmpTarget0 = np.logical_and(self.__returnValid__(self.X, self.values), self.idxTra)
        tmpTarget1 = np.logical_and(tmpTarget0, self.fSeasonCoef[season](self.xPeriodic)!=0)
        return (self.values[tmpTarget1],
                self.fSeasonCoef[season](self.xPeriodic)[tmpTarget1],
                np.where(self.fSeasonCoef[season](self.xPeriodic)[tmpTarget0]!=0)[0])
    
    def getValX(self, season=0):
        tmpTarget0 = np.logical_and(self.__returnValid__(self.X, self.values), self.idxVal)
        tmpTarget1 = np.logical_and(tmpTarget0, self.fSeasonCoef[season](self.xPeriodic)!=0)
        return (self.X[:, tmpTarget1],
                self.fSeasonCoef[season](self.xPeriodic)[tmpTarget1],
                np.where(self.fSeasonCoef[season](self.xPeriodic)[tmpTarget0]!=0)[0])

    def getValY(self, season=0):
        tmpTarget0 = np.logical_and(self.__returnValid__(self.X, self.values), self.idxVal)
        tmpTarget1 = np.logical_and(tmpTarget0, self.fSeasonCoef[season](self.xPeriodic)!=0)
        return (self.values[tmpTarget1],
                self.fSeasonCoef[season](self.xPeriodic)[tmpTarget1],
                np.where(self.fSeasonCoef[season](self.xPeriodic)[tmpTarget0]!=0)[0])

    def __ismember__(self, a, b):
        bind = {}
        for i, elt in enumerate(b):
            if elt not in bind:
                bind[elt] = i
        return [bind.get(itm, None) for itm in a]
    
    def fmFilter(self, x, beta=0.79352):
        tmp = np.array(x)
        for i0 in range(1,x.shape[0]):
            if not np.isinf(tmp[i0]):
                tmp0 = tmp[i0-1]+(1-beta)*(tmp[i0]-tmp[i0-1])
                if not np.isnan(tmp0):
                    tmp[i0] = tmp0
            else:
                tmp[i0] = tmp0
        return tmp
    
    def fLead(self, lead, x):
        tmp = np.empty_like(x)*np.nan
        if lead==0:
            return x
        if lead>0:
            tmp[lead:]=x[:-lead]
        else:
            tmp[:lead]=x[-lead:]
        return tmp
    
    def fSum(self, lead, x):
        if lead==0:
            return x
        if lead>0:
            x = np.cumsum(x)
            tmp = np.empty_like(x)*np.nan
            tmp[lead:]=x[lead:]-x[:-lead]
        elif lead<0:
            x = np.cumsum(x)
            tmp = np.empty_like(x)*np.nan
            tmp[:-lead]=x[:-lead]-x[lead:]
        return tmp
    
    def prepareInputs(self, lead):
        self.X = self.fData(self, lead)
    
    def prepareTargets(self, lead):
        self.values = self.fTarget(self, lead)
    
    def parseInputFunction(self, dataFunctionStr):
        allowed = [s0[0] for s0 in self.__CUSTOMFUNCTIONS__]
        toKeep = []
        i0 = 0
        for s0 in allowed:
            while s0 in dataFunctionStr:
                dataFunctionStr = dataFunctionStr.replace(s0, '#' + str(i0) + '#', 1)
                toKeep.append(s0)
                i0 += 1
        if len(re.findall('[a-zA-Z]', dataFunctionStr))>0:
            raise Exception('Some keywords in the data parsing function are not allowed. Code halted. The allowed keywords are: ' + str(allowed))
        dataFunctionStr = re.sub('[a-zA-Z]','', dataFunctionStr)
        
        for i0, s0 in enumerate(toKeep):
            dataFunctionStr = dataFunctionStr.replace('#' + str(i0) + '#', s0, 1)
        
        for s0 in self.__CUSTOMFUNCTIONS__:
            dataFunctionStr = dataFunctionStr.replace(s0[0], s0[1])
        self.fDataStr = 'lambda s, l: np.vstack((' + dataFunctionStr + ',))'
        self.fData = eval(self.fDataStr)
    
    def parseTargetFunction(self, targetFunctionStr):
        allowed = [s0[0] for s0 in self.__CUSTOMFUNCTIONS__]
        toKeep = []
        i0 = 0
        for s0 in allowed:
            while s0 in targetFunctionStr:
                targetFunctionStr = targetFunctionStr.replace(s0, '#' + str(i0) + '#', 1)
                toKeep.append(s0)
                i0 += 1
        if len(re.findall('[a-zA-Z]', targetFunctionStr))>0:
            raise Exception('Some keywords in the data parsing function are not allowed. Code halted. The allowed keywords are: ' + str(allowed))
        targetFunctionStr = re.sub('[a-zA-Z]','', targetFunctionStr)
        
        for i0, s0 in enumerate(toKeep):
            targetFunctionStr = targetFunctionStr.replace('#' + str(i0) + '#', s0, 1)
        
        for s0 in self.__CUSTOMFUNCTIONS__:
            targetFunctionStr = targetFunctionStr.replace(s0[0], s0[1])
        self.fTargetStr = 'lambda s, l: ' + targetFunctionStr
        self.fTarget = eval(self.fTargetStr)
    
    def setNormalization(self):
        # get valid training data
        tmpTarget0 = np.logical_and(self.__returnValid__(self.X, self.values), self.idxTra)
        X = self.X[:, tmpTarget0]
        y = self.values[tmpTarget0]
        # retrieve normalization constants
        self.normalization['X']={'mean': np.mean(X, axis=1, keepdims=True), 'std': np.std(X, axis=1, keepdims=True)}
        self.normalization['y']={'mean': np.mean(y, axis=0, keepdims=True), 'std': np.std(y, axis=0, keepdims=True)}
        # accounting for constant inputs
        tmp = self.normalization['X']['std']==0
        self.normalization['X']['std'][tmp] = 1
        tmp = self.normalization['y']['std']==0
        self.normalization['y']['std'][tmp] = 1
    
    def normalize(self, X=[], y=[]):
        result = []
        if len(X)!=0:
            result.append((X-self.__shapeToData(X, self.normalization['X']['mean']))/self.__shapeToData(X, self.normalization['X']['std']))
        if len(y)!=0:
            result.append((y-self.__shapeToData(y, self.normalization['y']['mean']))/self.__shapeToData(y, self.normalization['y']['std']))
        return result
        
    def denormalize(self, X=[], y=[]):
        result = list()
        if len(X)!=0:
            result.append((X*self.__shapeToData(X, self.normalization['X']['std']))+self.__shapeToData(X, self.normalization['X']['mean']))
        if len(y)!=0:
            result.append((y*self.__shapeToData(y, self.normalization['y']['std']))+self.__shapeToData(y, self.normalization['y']['mean']))
        return result
    
    def __shapeToData(self, data, x):
        tmp = x.shape 
        axis = np.where(np.array(tmp)==1)[0]
        if len(axis)==len(tmp):
            return np.tile(x, data.shape)
        if len(axis)>0:
            return np.repeat(x, data.shape[axis[0]], axis=axis[0])
        else:
            return None
      
    def getZeroThreshold(self):
        return (0-self.normalization['y']['mean'])/self.normalization['y']['std']
    
class Manager(object):
    #queue =     
    def __init__(self, data=None, dataFunction='cycle, lead(targets) ,lead(filter(targets))', targetFunction='targets', extra=None,
                 nodes=4, epochs = 250, population = 1000, regularization=0.05,
                 seasons=1, leads=(1, 5, 10,), refTime=dt.datetime(2000, 1, 1, 0, 0, 0), period='year', timeStepUnit='day', timeStepSize=1,
                 valFraction=0.3, displayEach=100, openClPlatform=0, openClDevice='GPU', **kwargs):

        if data!=None:
            # handling of arguments
            self.opt={}
            self.opt['seasons'] = seasons
            self.opt['leads'] = np.sort(leads)
            self.opt['period'] = period
            self.opt['nodes'] = nodes
            self.opt['regularization'] = regularization
            self.opt['valFraction'] = valFraction
            self.opt['population'] = population
            self.opt['epochs'] = epochs
            self.opt['timeStepUnit'] = timeStepUnit
            self.opt['timeStepSize'] = timeStepSize
            self.opt['refTime'] = refTime
            self.opt['period'] = period
            self.opt['displayEach'] = displayEach
            self.opt['openClPlatform'] = openClPlatform
            self.opt['openClDevice'] = openClDevice
            
            # addition of generic arguments to options
            self.opt.update(kwargs)
    
            # create training data
            self.data=Data(data, extra, refTime=self.opt['refTime'], period=self.opt['period'], timeStepUnit=self.opt['timeStepUnit'], timeStepSize=self.opt['timeStepSize'])
    
            # prepare target modifying function
            self.data.parseTargetFunction(targetFunction)
    
            # prepare data creation function
            self.data.parseInputFunction(dataFunction)
            
            # split in training and validation
            self.data.split(self.opt['valFraction'])
            
            # split in seasons
            # TODO: understand how the splitting depends on the input preparation
            self.data.setSeasons(self.opt['seasons'])
            
            # set normalization
            # TODO: fix for volumes, where the lead can greatly influence target scales
            self.data.tmpValues = self.data.values
            self.data.prepareTargets(365)
            self.data.prepareInputs(365)
            self.data.setNormalization()
            self.data.X = []
            self.data.values = self.data.tmpValues
    
    def train(self):
        self.__readyModels__=[]
        tmpCounter = 1
        for i0, lead in enumerate(self.opt['leads']):
            result=[]
            for season in range(self.opt['seasons']):
                try:
                    current_task.update_state(state='PROGRESS',
                                              meta={'title': 'Training new set: leadtime:%2u, Season:%2u' % (lead, season+1), 'message': None})
                except Exception:
                    pass
                
                self.data.tmpValues = self.data.values
                self.data.prepareTargets(lead)
                self.data.prepareInputs(lead)
                XTrain = self.data.getTraX(season=season)[0]
                yTrain = self.data.getTraY(season=season)[0]
                self.data.values = self.data.tmpValues
            
                XTrain, yTrain = self.data.normalize(XTrain, yTrain)
            
                # transpose for optimization
                XTrain = np.transpose(XTrain)
                yTrain = np.transpose(yTrain)
                
                # start ann object
                self.annToOptimize=ann(XTrain, nodes=self.opt['nodes'], openCL=True, workGroup=(64, 4), deviceType=self.opt['openClDevice'], platform=self.opt['openClPlatform'], verbose=0)
            
                # start error object
                self.errorOpenCL=errorMetrics(yTrain, stride=512, workGroup=(1, 16), deviceType=self.opt['openClDevice'], platform=self.opt['openClPlatform'], verbose=0)
           
                # prepare model
                print('Starting model ' + str(tmpCounter)  + '/' + str(self.opt['seasons']*len(self.opt['leads'])) + ' (lead:' + str(lead) + ', season:' + str(season) + ')')
                tmpCounter += 1
                model=PSOAlt(XTrain, yTrain, self.annToOptimize.getWeightLen(), evalFun=self.evalFun,
                     errorObj=self.errorOpenCL, regFun=self.regFun, lowerThreshold=self.data.getZeroThreshold(), displayEach=self.opt['displayEach'],
                     population=self.opt['population'], epochs=self.opt['epochs'], plotResult=False, bounds=6*np.vstack((-1*np.ones(self.annToOptimize.getWeightLen()),
                                                  1*np.ones(self.annToOptimize.getWeightLen()))))
                model.start()
                
                if False:
                    plt.figure()
                    plt.plot(self.data.getTraDates(season), self.data.getTraY(season)[0],'.r')
                    plt.plot(self.data.getTraDates(season),
                             self.processBands(
                                               model.predict(self.data.getTraX(season)[0]),
                                               np.array(model.opt['bands'])),
                             '.k')
                result.append(model)
            self.__readyModels__.append(result)
    
    def forecast(self, date, data, extra=None, num=999, leadTargets=None):
        # prepare input data
        if len(data)>0:
            X=Data(data, extra, refTime=self.opt['refTime'], period=self.opt['period'], timeStepUnit=self.opt['timeStepUnit'], timeStepSize=self.opt['timeStepSize'])
            X.fTarget = eval(self.data.fTargetStr)
            X.fData = eval(self.data.fDataStr)
            X.fSeasonCoef = self.data.fSeasonCoef
            X.normalization = self.data.normalization
        else:
            X = self.data
            
        # extend input data
        maxDate = X.fTimeJump(date, max(self.opt['leads']).astype(float))
        tmp = 0
        tmpDates = []
        while X.fTimeJump(X.dates[-1],tmp)<maxDate:
            tmp+=1
            tmpDates.append(np.datetime64(X.fTimeJump(X.dates[-1], tmp)))
        if len(tmpDates)>1:
            X.values = np.hstack((X.values, np.ones(tmp)*np.Inf))
            X.dates = np.hstack((X.dates, tmpDates))
            if X.extra!=None:
                for i0 in range(len(X.extra)):
                    X.extra[i0] = np.hstack((X.extra[i0], np.empty(tmp)*np.nan))
        
        # update data
        X.upDateGroupAndPeriod()
        
        # run models
        leadSimulations=[]
        for i0 in range(len(self.opt['leads'])):
            X.tmpValues = X.values
            X.prepareTargets(self.opt['leads'][i0])
            X.prepareInputs(self.opt['leads'][i0])
            X.split(1)
            wholePeriod = np.zeros((1,len(self.__readyModels__[i0][0].opt['bands'])))
            for season in range(self.opt['seasons']):
                toPredict, seasonCoefs = X.getValX(season)[0:2]
                refIdx = np.where(X.getValDates(season)==X.fTimeJump(date,self.opt['leads'][i0].astype(float)))[0]
                toPredict = X.normalize(toPredict[:,refIdx])[0]
                if toPredict.shape[1]!=0:
                    tmp = self.__readyModels__[i0][season].predict(np.transpose(toPredict))
                    tmp = X.denormalize(y = tmp)[0]
                    wholePeriod[0,:] = wholePeriod[0,:]+tmp*np.transpose(np.tile(seasonCoefs[refIdx], (tmp.shape[1],1)))
            leadSimulations.append(wholePeriod)
            X.values = X.tmpValues
        
        # sort bands and interpolate where needed
        bands = np.array(self.__readyModels__[0][0].getBands())
        for i0, s0 in enumerate(leadSimulations):
            s0 = self.__readyModels__[i0][0].processBands(s0)
        
        # choose target leads
        #=======================================================================
        # logLeads = np.log(self.opt['leads'])
        #=======================================================================
        if leadTargets==None:
            #===================================================================
            # leadTargets = np.unique(np.round(np.hstack((np.exp(np.linspace(np.min(logLeads), np.max(logLeads), num=num)), self.opt['leads']))))
            #===================================================================
            leadTargets = np.unique(np.round(np.hstack((np.linspace(np.min(self.opt['leads']), np.max(self.opt['leads']), num=num), self.opt['leads']))))
        

        # interpolate results in the lead space
        forecasts = np.empty((len(leadTargets)+1, leadSimulations[0].shape[1]))*np.nan
        #=======================================================================
        # forecasts[0,:] = X.values[np.where(X.dates==date)[0]]
        #=======================================================================
        forecasts[0,:] = X.fTarget(X, 0)[np.where(X.dates==date)[0]]
        for i0, l0 in enumerate(leadTargets):
            translation = l0-self.opt['leads']
            toCompute = np.where(translation==0)[0]
            if len(toCompute)==1:
                weights=[1,]
            else:
                toCompute = [np.where(translation==np.min(translation[translation>0]))[0][0],
                             np.where(translation==np.max(translation[translation<0]))[0][0]]
                translation = np.abs(translation[toCompute])
                weights = 1-translation/np.sum(translation)
            tmp = np.zeros_like(leadSimulations[0])
            for i1, c0 in enumerate(toCompute):
                tmp += leadSimulations[c0]*weights[i1]
            forecasts[i0+1, :] = tmp
        
        # associate days
        dates = [np.datetime64(date),]
        for l0 in leadTargets:
            dates.append(np.datetime64(X.fTimeJump(date,l0)))
        dates = np.array(dates)
        
        return {'dates': dates, 'simulations': forecasts, 'bands': bands}
    
    def hindcast(self, data, lead, extra=[], bands=None):
        # prepare input data
        if len(data)>0:
            X=Data(data, extra, refTime=self.opt['refTime'], period=self.opt['period'], timeStepUnit=self.opt['timeStepUnit'], timeStepSize=self.opt['timeStepSize'])
            X.fTarget = eval(self.data.fTargetStr)
            X.fData = self.data.fData
            X.fSeasonCoef = self.data.fSeasonCoef
            X.normalization = self.data.normalization
        else:
            X = self.data
        
        # choose relevant leads
        tmp=lead-self.opt['leads']
        tmpZeroIdx = np.where(tmp==0)[0]
        if len(tmpZeroIdx)>0:
            toCompute=[tmpZeroIdx[0],]
        else:
            toCompute=[np.where(tmp==np.min(tmp[tmp>0]))[0][0],
                       np.where(tmp==np.max(tmp[tmp<0]))[0][0]]
            
        # run models
        leadSimulations=[]
        for i0 in toCompute:
            X.tmpValues = X.values
            X.prepareTargets(self.opt['leads'][i0])
            X.prepareInputs(self.opt['leads'][i0])
            X.split(1)
            wholePeriod = np.zeros((X.X.shape[1],len(self.__readyModels__[i0][0].opt['bands'])))
            for season in range(self.opt['seasons']):
                toPredict, seasonCoefs, seasonIdxs = X.getValX(season)
                toPredict = X.normalize(toPredict)[0]
                
                tmp = self.__readyModels__[i0][season].predict(np.transpose(toPredict))
                tmp = X.denormalize(y = tmp)[0]
                wholePeriod[seasonIdxs,:] = wholePeriod[seasonIdxs,:]+tmp*np.transpose(np.tile(seasonCoefs, (tmp.shape[1],1)))
            leadSimulations.append(wholePeriod)
            X.values = X.tmpValues
            
        # sort bands and interpolate where needed
        bands = np.array(self.__readyModels__[0][0].getBands())
        for i0, s0 in enumerate(leadSimulations):
            s0 = self.__readyModels__[i0][0].processBands(s0)
                        
        # interpolate results in the lead space
        if len(leadSimulations)==1:
            leadSimulations = leadSimulations[0]
        else:
            translation = np.abs(lead-self.opt['leads'][toCompute])
            weights = 1-translation/np.sum(translation)
            tmp = np.zeros_like(leadSimulations[0])
            for i0, w0 in enumerate(weights):
                tmp += leadSimulations[i0]*w0
            leadSimulations = tmp
        leadSimulations = leadSimulations[np.logical_not(np.isnan(np.sum(leadSimulations, axis=1))),:]
        
        dates = X.getValDates(0)
        for season in range(1, self.opt['seasons']):
            dates = np.hstack((dates, X.getValDates(season)))
        return {'dates': np.unique(dates), 'simulations': leadSimulations, 'bands': bands}
    
    def evalFun(self, x, data=[]):
        # error
        self.annToOptimize.setWeights(x)
        if len(data)==0:
            tmp = self.annToOptimize.compute()
        else:
            tmp = self.annToOptimize.compute(data)
        return tmp

    def regFun(self, w):
        # regularization
        if len(w.shape)==1:
            w = np.expand_dims(w, 0)
        # reg = np.sqrt(np.sum(np.square(w[:, np.where(annToOptimize.getWeightsToRegularize())[0]]), axis=1)) #L2
        reg = np.sum(np.abs(w[:, np.where(self.annToOptimize.getWeightsToRegularize())[0]]), axis=1) # L1
        return self.opt['regularization']*reg
    
    def getTrainingDates(self):
        return {'dates': self.data.dates[self.data.idxTra], 'logical': self.data.idxTra}
    
    def getSeasons(self, num=100):
        seasons = np.empty((num, len(self.data.fSeasonCoef)))
        for i0, f0 in enumerate(self.data.fSeasonCoef):
            seasons[:,i0] = f0(np.linspace(0, 1, num))
        return seasons
    
    def getPerformance(self, lead, train=False):
        idx = np.where(self.opt['leads']==lead)[0]
        
        self.data.prepareInputs(lead)
        wholePeriod = np.zeros((self.data.X.shape[1],len(self.__readyModels__[idx][0].opt['bands'])))
        fullTargets = np.zeros(self.data.X.shape[1])
        valid = np.zeros(self.data.X.shape[1], dtype=np.bool)
        for season in range(self.opt['seasons']):
            if train:
                toPredict, seasonCoefs, seasonIdxs = self.data.getTraX(season)
                targets = self.data.getTraY(season)[0]
            else:
                toPredict, seasonCoefs, seasonIdxs = self.data.getValX(season)
                targets = self.data.getValY(season)[0]
            toPredict = self.data.normalize(toPredict)[0]
            
            tmp = self.__readyModels__[idx][season].predict(np.transpose(toPredict))
            tmp = self.data.denormalize(y = tmp)[0]
            wholePeriod[seasonIdxs,:] = wholePeriod[seasonIdxs,:]+tmp*np.transpose(np.tile(seasonCoefs, (tmp.shape[1],1)))
            fullTargets[seasonIdxs] = fullTargets[seasonIdxs]+targets*seasonCoefs
            valid[seasonIdxs] = 1
            
        tmpSimulations = self.__readyModels__[idx][0].processBands(wholePeriod[valid, :])
        uniform, pValues = self.__readyModels__[idx][0].predictiveQQ(tmpSimulations, targets=fullTargets[valid])
        alpha, xi, piRel = self.__readyModels__[idx][0].metrics(uniform, pValues, tmpSimulations)
        return {'pValues':pValues, 'uniform':uniform, 'alpha':alpha, 'xi':xi, 'pi':piRel}
    
    def save(self, filePath):
        self.data.fData = None
        self.data.fTarget = None
        self.data.fSeasonCoef = None
        
        toSave = self.__dict__
        toSave.pop('annToOptimize')
        toSave.pop('errorOpenCL')
        
        print(self.__readyModels__[0][0].__dict__.keys())
        for i0 in range(len(self.__readyModels__)):
            for i1 in range(len(self.__readyModels__[i0])):
                self.__readyModels__[i0][i1].__dict__.pop('simulations')
                self.__readyModels__[i0][i1].__dict__.pop('velocities')
                self.__readyModels__[i0][i1].__dict__.pop('jointVelocities')
                self.__readyModels__[i0][i1].__dict__.pop('trainData')
                self.__readyModels__[i0][i1].__dict__.pop('targets')
                self.__readyModels__[i0][i1].__dict__.pop('frontLvls')
                self.__readyModels__[i0][i1].__dict__.pop('gBestIdxs')
                self.__readyModels__[i0][i1].__dict__.pop('pBestIdxs')
        print(self.__readyModels__[0][0].__dict__.keys())
        
        with open(filePath, 'wb') as file:
            pickle.dump(toSave, file)
            
        self.data.fData = eval(self.data.fDataStr)
        self.data.fSeasonCoef = [interp1d(x0, y0, kind='linear') for x0, y0 in self.data.fSeasonCoefData]
        
        #=======================================================================
        # ['__readyModels__', 'annToOptimize']
        #=======================================================================
        
    def load(self, filePath):
        with open(filePath, 'rb') as file:
            tmp = pickle.load(file)
            self.__dict__.update(tmp)
        self.data.fData = eval(self.data.fDataStr)
        self.data.fSeasonCoef = [interp1d(x0, y0, kind='linear') for x0, y0 in self.data.fSeasonCoefData]
        
        # initialize openCL objects
        XTrain = np.transpose(self.data.getTraX(season=0)[0])
        yTrain = np.transpose(self.data.getTraY(season=0)[0])
        self.annToOptimize=ann(XTrain, nodes=self.opt['nodes'], openCL=True, workGroup=(64, 4), deviceType='ALL')
        self.errorOpenCL=errorMetrics(yTrain, stride=512, workGroup=(1, 16), deviceType='CPU', verbose=0)
        
        # bind evalFunction
        for l0 in self.__readyModels__:
            for s0 in l0:
                s0.evalFun=self.evalFun
