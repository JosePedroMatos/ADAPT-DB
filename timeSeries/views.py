from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound, JsonResponse
from django.conf import settings
from django_countries import countries
from timeSeries.models import Series, Value, Forecast, SatelliteData
from .decoder import decode
from gpu.manager import Manager, fGenerateLeads, fSecond, fMinute, fHour, fDay, fWeek, fMonth, fYear
from django.utils.safestring import mark_safe
from django.core.files.base import ContentFile
from django.core.urlresolvers import reverse
from celery import task
from celery.result import AsyncResult
from gpu.manager import Manager
from .satelliteData import TRMMSatelliteRainfall

import binascii
import json
import warnings
import os
import datetime
import pytz
import dateutil.parser

import numpy as np

def viewSeries(request, seriesList):
    
    #===========================================================================
    # updateSatelliteData('test3')
    #===========================================================================
    
    context = {'LANG': request.LANGUAGE_CODE,
               'LOCAL_JAVASCIPT': settings.LOCAL_JAVASCIPT,
               }
    seriesDict = {}
    errorsDict = {'missing': list(),
                  'noAccess': list(),
                  'noData': list(),
                  }
    
    seriesList=seriesList.split('/')
    
    # find recent dates
    recentDates = []
    for seriesName in seriesList:
        tmpResult = Series.objects.filter(name=seriesName)
        if len(tmpResult)>0:
            s0 = tmpResult[0]
            recentDates.append(Value.objects.filter(series=s0.id).order_by('date').last().date)
    recentDate = max(recentDates)
    recentDate = recentDate.replace(year=recentDate.year-2)
    
    # retrieve series' info
    for seriesName in seriesList:
        tmpResult = Series.objects.filter(name=seriesName)
        if len(tmpResult)>0:
            s0 = tmpResult[0]
            result = Value.objects.filter(series=s0.id).order_by('date')
            values = [{'x':obj.date.isoformat(), 'y':binascii.b2a_base64(obj.record).decode("utf-8")} for obj in result.filter(date__gte=recentDate)]
            
            forecasts = {}
            for f0 in Forecast.objects.filter(targetSeries=s0.id).filter(ready=True):
                forecasts[f0.name] = {}
                forecasts[f0.name]['urlForecast'] = '/timeSeries' + reverse('forecast', 'timeSeries.urls', kwargs={'forecastName': f0.name})
                forecasts[f0.name]['urlHindcast'] = '/timeSeries' + reverse('hindcast', 'timeSeries.urls', kwargs={'forecastName': f0.name})
                forecasts[f0.name]['description'] = f0.description
                forecasts[f0.name]['leadTime'] = f0.leadTime
                forecasts[f0.name]['seasons'] = f0.splitBySeason
            
            if s0.location.country in dict(countries):
                tmpCountry = dict(countries)[s0.location.country]
            else:
                tmpCountry = ''
            tmp = {'id': s0.id,
                'name': s0.name,
                'type': s0.type.name,
                'provider': s0.provider.name,
                'providerAbbreviation': s0.provider.abbreviation,
                'providerIcon': '/' + str(s0.provider.icon),
                'providerWebpage': s0.provider.website,
                'units': s0.type.units,
                'typeIcon': '/' + str(s0.type.icon),
                'lat': float(s0.location.lat),
                'lon': float(s0.location.lon),
                'location': s0.location.name,
                'quality': s0.quality,
                'timeStepUnits': dict(Series.TIME_STEP_PERIOD_TYPE)[s0.timeStepUnits],
                'timeStepPeriod': s0.timeStepPeriod,
                'encryptionKey': s0.encryptionKey,
                'metaEncrypted': s0.metaEncrypted,
                'river': s0.location.river,
                'country': tmpCountry,
                'catchment': s0.location.catchment,
                'values': values,
                'records': len(result),
                'forecasts': forecasts,
                }
               
            if len(result)==0:
                errorsDict['noData'].append(seriesName)
                tmp.update({'minDate': '',
                            'maxDate': ''})  
            else:
                tmp.update({'minDate': result.first().date.isoformat().split('T')[0],
                            'maxDate': result.last().date.isoformat().split('T')[0]})     
            seriesDict[str(s0)] = tmp
        else:
            errorsDict['missing'].append(seriesName)
        
    context['series'] = json.dumps(seriesDict)
    context['errors'] = json.dumps(errorsDict)
    
    # fields
    fields = (('Id', 'name'),
              ('Location', 'location'),
              ('River', 'river'),
              ('Catchment', 'catchment'),
              ('Type', 'type'),
              ('Units', 'units'),
              ('Time step', 'timeStepUnits'),
              ('Records', 'records'),
              ('From', 'minDate'),
              ('To', 'maxDate'),
              )
    context['fields'] = json.dumps(fields)
    
    return render(request, 'timeSeries/viewSeries.html', context)

def deleteTimeSeries(request, seriesName):
    series = Series.objects.filter(name=seriesName)
    Value.objects.filter(series=series).delete()
    # TODO: change this to a nice response.
    context = {'message': 'done!'}
    return HttpResponse(
                        json.dumps(context),
                        content_type="application/json"
                        )
    
def upload(request, seriesName):
    series = Series.objects.filter(name=seriesName)
    if series:
        provider = series[0].provider
        location = series[0].location
        type = series[0].type        
        
        result = Value.objects.filter(series=series[0].id).order_by('date')
        values = [{'x':obj.date.isoformat(), 'y':binascii.b2a_base64(obj.record).decode("utf-8")} for obj in result]
        
        context = {'LANG': request.LANGUAGE_CODE,
                   'series': series[0].name,
                   'encryptionKey': series[0].encryptionKey,
                   'metaEncrypted': series[0].metaEncrypted,
                   'timeStep': dict(Series.TIME_STEP_PERIOD_TYPE)[series[0].timeStepUnits],
                   'timeStepPeriod': series[0].timeStepPeriod,
                   'provider': str(provider),
                   'type': str(type),
                   'units': type.units,
                   'location': str(location),
                   'data': json.dumps(values),
                   'LOCAL_JAVASCIPT': settings.LOCAL_JAVASCIPT,
                   }
        return render(request, 'timeSeries/uploadValues.html', context)
    else:
        return HttpResponseNotFound('Series [' + seriesName + '] not found...')

def uploadTimeSeries(request, seriesName):
    # TODO: check provider pass
    
    # TODO: check duplicated values
    
    data = json.loads(request.POST.get('toUpload'))
    seriesObj = Series.objects.get(name=seriesName)
    
    warnings.filterwarnings('ignore', '.*Invalid utf8 character string.*',)
    
    toInput = list()
    for i0, d0 in enumerate(data):
        toInput.append(Value(series=seriesObj, date=d0['date'], record=binascii.a2b_base64(d0['value'])))
        if i0 % 1000==0:
            Value.objects.bulk_create(toInput)
            toInput = list()
    Value.objects.bulk_create(toInput)
    
    context = {'message': 'done!'}
    return HttpResponse(
                        json.dumps(context),
                        content_type="application/json"
                        )

def seriesData(request):
    context = {}
    if request.method == 'POST':
        seriesObj = Series.objects.get(name=request.POST.get('series'))
        provider = seriesObj.provider
        location = seriesObj.location
        type = seriesObj.type
        context = {'location': str(location),
                   'provider': str(provider),
                   'type': str(type),
                   'units': type.units,
                   'timeStepUnits': dict(Series.TIME_STEP_PERIOD_CHOICES)[seriesObj.timeStepUnits],
                   'timeStepPeriod': seriesObj.timeStepPeriod,
                   'metaEncrypted': seriesObj.metaEncrypted,
                   'encryptionKey': seriesObj.encryptionKey,
                   'name': seriesObj.name,
                   }
        return HttpResponse(
                            json.dumps(context),
                            content_type="application/json" 
        )

def getValues(request):
    context = {}
    if request.method == 'POST':
        s0 = Series.objects.get(name=request.POST.get('series'))
        dateFrom = datetime.datetime.strptime(request.POST.get('from'), "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo = pytz.utc)
        dateTo = datetime.datetime.strptime(request.POST.get('to'), "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo = pytz.utc)
        result = Value.objects.filter(series=s0.id).filter(date__gte=dateFrom).filter(date__lt=dateTo).order_by('date')
        values = [{'x':obj.date.isoformat(), 'y':binascii.b2a_base64(obj.record).decode("utf-8")} for obj in result]

        context = {'values': values}
    return JsonResponse(context)
    
def trainForecastBase(request, forecastName):
    forecast = Forecast.objects.filter(name=forecastName)
    if forecast:
        context = {'LANG': request.LANGUAGE_CODE,
                   'LOCAL_JAVASCIPT': settings.LOCAL_JAVASCIPT,
                   'info': mark_safe(json.dumps({'forecast': forecast[0].name,
                            'description': forecast[0].description,
                            'target': forecast[0].targetSeries.name,
                            'type': str(forecast[0].targetSeries.type),
                            'leadTime': forecast[0].leadTime,
                            'splitBySeason': forecast[0].splitBySeason,
                            'nodes': forecast[0].nodes,
                            'dataExpression': forecast[0].dataExpression,
                            'targetExpression': forecast[0].targetExpression,
                            'population': forecast[0].population,
                            'epochs': forecast[0].epochs,
                            })),
                   }
        
        return render(request, 'timeSeries/trainForecast.html', context)
    else:
        return HttpResponseNotFound('Forecast [' + forecastName + '] not found...')

@task(name='train')
def trainWrapper(info, data, extra):
    data = [np.array([np.datetime64(s0) for s0 in data[0]]), data[1]]
    if extra!=None:
        for i0 in range(len(extra)):
            extra[i0] = [np.array([np.datetime64(s0) for s0 in extra[i0][0]]), extra[i0][1]]
    
    man = Manager(data, extra=extra,
                  dataFunction=info['dataFunction'], targetFunction=info['targetFunction'], valFraction=0.6, 
                  nodes=info['nodes'], seasons=info['seasons'], population=info['population'],
                  epochs=info['epochs'], regularization=info['regularization'], refTime=dateutil.parser.parse(info['referenceDate']),
                  leads=fGenerateLeads(info['leadTime']), displayEach=100,
                  openClPlatform=settings.OPENCL_PLATFORM, openClDevice=settings.OPENCL_DEVICE,
                  )
    
    man.train()
    man.save(info['filePath'])
    Forecast.objects.filter(name=info['name']).update(ready=True)
    
    return 'done'

def fJumpDateFun (period):
    tmp = {'second': fSecond,
              'minute': fMinute,
              'hour': fHour,
              'day': fDay,
              'week': fWeek,
              'month': fMonth,
              'year': fYear,
              }
    return tmp[period]

def fGetForecastData(forecastName, periodJumpFun=None, referenceDate=None, periods=999, fromDate=None, toDate=None):
    forecast = Forecast.objects.filter(name=forecastName)
    if forecast:
        if fromDate!=None and toDate!=None:
            records = Value.objects.filter(series=forecast[0].targetSeries.id).filter(date__gte=fromDate).filter(date__lte=toDate).order_by('date')
        else:
            #### Modify according to the step of the series
            if referenceDate==None:
                referenceDate = Value.objects.filter(series=forecast[0].targetSeries.id).latest('date').date
            baseDate = periodJumpFun(referenceDate, -periods)
            records = Value.objects.filter(series=forecast[0].targetSeries.id).filter(date__gte=baseDate).order_by('date')
            
        values = decode([r0.record for r0 in records], forecast[0].targetSeries.encryptionKey)
        dates = [str(r0.date) for r0 in records]
        target = [dates, values]
        
        extra = []
        for s0 in forecast[0].extraSeries.filter().order_by('id'):
            if fromDate!=None and toDate!=None:
                records = Value.objects.filter(series=s0.id).filter(date__gte=fromDate).filter(date__lte=toDate).order_by('date')
            else:
                records = Value.objects.filter(series=s0.id).filter(date__gte=baseDate).order_by('date')
            values = decode([r0.record for r0 in records], s0.encryptionKey)
            dates = [str(r0.date) for r0 in records]
            extra.append([dates, values])
        if len(extra)==0:
            extra=None
            
        return (target, extra)
    else:
        return False

def trainForecastRun(request, forecastName):
    forecast = Forecast.objects.filter(name=forecastName)
    if forecast:
        if forecast[0].forecastFile.name != '' and os.path.isfile(forecast[0].forecastFile.path):
            os.remove(forecast[0].forecastFile.path)
        forecast[0].forecastFile.save(forecastName + '.gpu', ContentFile('dummy content'))
        
        info = {'leadTime': forecast[0].leadTime,
                'seasons': forecast[0].splitBySeason,
                'nodes': forecast[0].nodes,
                'dataFunction': forecast[0].dataExpression,
                'targetFunction': forecast[0].targetExpression,
                'population': forecast[0].population,
                'epochs': forecast[0].epochs,
                'regularization': float(forecast[0].regularize),
                'filePath': forecast[0].forecastFile.path,
                'name': forecast[0].name,
                'referenceDate': forecast[0].referenceDate.isoformat(),
                }

        target, extra = fGetForecastData(forecastName, fJumpDateFun(forecast[0].period))
        
        #=======================================================================
        # trainWrapper(info, target, extra)
        # context = {'job': 1}
        #=======================================================================
        
        job = trainWrapper.delay(info, target, extra)
        context = {'job': job.id}

        return JsonResponse(context)

def trainForecastProgress(request, forecastName, jobId):
    job = AsyncResult(jobId)
    data = job.result or job.state
    
    if data == 'done':
        print('done' + jobId)
        Forecast.objects.filter(name=forecastName).update(ready=True)
    
    context = {'progress': data}
    return JsonResponse(context)

def hindcast(request, forecastName):
    forecast = Forecast.objects.filter(name=forecastName)
    if forecast:
        if request.POST['lead'][0]=='null':
            lead = forecast[0].leadTime
        else:
            lead = float(request.POST['lead'])
        dateFrom = datetime.datetime.strptime(request.POST.get('from'), "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo = pytz.utc)
        dateTo = datetime.datetime.strptime(request.POST.get('to'), "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo = pytz.utc)
            
        fJumpFun = fJumpDateFun(forecast[0].period)      
        #=======================================================================
        # target, extra = fGetForecastData(forecastName, fJumpFun, periods=10)
        #=======================================================================
        target, extra = fGetForecastData(forecastName, fromDate=dateFrom, toDate=dateTo)
        
        target = [np.array([np.datetime64(s0) for s0 in target[0]]), target[1]]
        if extra!=None:
            for i0 in range(len(extra)):
                extra[i0] = [np.array([np.datetime64(s0) for s0 in extra[i0][0]]), extra[i0][1]]
        
        man = Manager(target, extra=extra)
        man.load(forecast[0].forecastFile.path)
        #=======================================================================
        # res = man.hindcast(data=target, lead=lead, extra=extra, bands=(0.01, 0.05, 0.25, 0.4, 0.5, 0.75, 0.95, 0.99))
        #=======================================================================
        res = man.hindcast(data=target, lead=lead, extra=extra)
        selectBands = (1,3,5,7,8,10,12,14)
        res['bands'] = res['bands'][selectBands,]
        res['simulations'] = res['simulations'][:,selectBands]
        
        trainingDates = man.data.dates[man.data.idxTra]
        trainingDates = trainingDates[np.logical_and(trainingDates>=np.datetime64(dateFrom), trainingDates<=np.datetime64(dateTo))]
        
    context = {'bands': res['bands'].tolist(),
                'dates': [str(d0) for d0 in res['dates']],
                'values': np.transpose(res['simulations']).tolist(),
                'timeStepUnits': dict(Series.TIME_STEP_PERIOD_TYPE)[forecast[0].targetSeries.timeStepUnits],
                'timeStepPeriod': forecast[0].targetSeries.timeStepPeriod,
                'trainingDates': [str(d0) for d0 in trainingDates],
                }
    
    return JsonResponse(context)

def forecast(request, forecastName):
    
    forecast = Forecast.objects.filter(name=forecastName)
    if forecast:
        reference = Value.objects.filter(series=forecast[0].targetSeries.id).latest('date').date
        if 'reference' in request.POST:
            tmp = datetime.datetime.strptime(request.POST.get('reference'), "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo = pytz.utc)
            if tmp<reference:
                reference = tmp

        fJumpFun = fJumpDateFun(forecast[0].period)      
        target, extra = fGetForecastData(forecastName, fJumpFun, reference, 2)
        
        target = [np.array([np.datetime64(s0) for s0 in target[0]]), target[1]]
        if extra!=None:
            for i0 in range(len(extra)):
                extra[i0] = [np.array([np.datetime64(s0) for s0 in extra[i0][0]]), extra[i0][1]]
        
        man = Manager(target, extra=extra)
        man.load(forecast[0].forecastFile.path)
        res = man.forecast(np.datetime64(reference), data=target, extra=extra)
        selectBands = (1,3,5,7,8,10,12,14)
        res['bands'] = res['bands'][selectBands,]
        res['simulations'] = res['simulations'][:,selectBands]
        
        context = {'bands': res['bands'].tolist(),
                   'dates': [str(d0) for d0 in res['dates']],
                   'values': np.transpose(res['simulations']).tolist(),
                   'timeStepUnits': dict(Series.TIME_STEP_PERIOD_TYPE)[forecast[0].targetSeries.timeStepUnits],
                   'timeStepPeriod': forecast[0].targetSeries.timeStepPeriod,
                   }
        
        return JsonResponse(context)


def storeSatellite(request, name):
    storeSatelliteData(name)
    
    return HttpResponse(
                        json.dumps({'done': True}),
                        content_type="application/json"
                        )

@task(name='storeSatelliteData')
def storeSatelliteData(name):
    # reviews all the history of the satellite product
    
    satelliteObj = SatelliteData.objects.get(name=name)
    
    satellite = satelliteObj.satellite
    dateIni = satelliteObj.startDate
    dateEnd = datetime.datetime.now()
    geometryFile = satelliteObj.geometry.path
    dataFolder = satelliteObj.dataFolder
    downloadFolder = os.path.join(settings.SATELLITE_DOWNLOAD, satellite)
    jsonGeometry = satelliteObj.jsonGeometry
    
    satelliteInstance = eval(satellite + '(dataFolder=dataFolder, downloadFolder=downloadFolder)')
    satelliteInstance.store(dateIni=dateIni, dateEnd=dateEnd, geometryFile=geometryFile, geometryStr=jsonGeometry)
    
    if len(satelliteObj.jsonGeometry)==0:
        satelliteObj.jsonGeometry = satelliteInstance.getGeometryInfo()
        satelliteObj.save()

@task(name='updateSatelliteData')
def updateSatelliteData(name):
    # only looks for recent data
    
    satelliteObj = SatelliteData.objects.get(name=name)
    
    satellite = satelliteObj.satellite
    geometryFile = satelliteObj.geometry.path
    dataFolder = satelliteObj.dataFolder
    downloadFolder = os.path.join(settings.SATELLITE_DOWNLOAD, satellite)
    jsonGeometry = satelliteObj.jsonGeometry
    
    satelliteInstance = eval(satellite + '(dataFolder=dataFolder, downloadFolder=downloadFolder)')
    satelliteInstance.update(geometryFile=geometryFile, geometryStr=jsonGeometry)
    
    if len(satelliteObj.jsonGeometry)==0:
        satelliteObj.jsonGeometry = satelliteInstance.getGeometryInfo()
        satelliteObj.save()
        
def getSatelliteData(request):
    # localhost:8000/timeSeries/satelliteGet/
    
    data = json.loads(request.POST.get('data'))
    
    name = data['name']
    info = data['info']
    datetimes = data['datetimes']
    
    satelliteObj = SatelliteData.objects.get(name=name)
    
    satellite = satelliteObj.satellite
    dataFolder = satelliteObj.dataFolder
    downloadFolder = os.path.join(settings.SATELLITE_DOWNLOAD, satellite)
    
    satelliteInstance = eval(satellite + '(dataFolder=dataFolder, downloadFolder=downloadFolder)')
    if info:
        data = satelliteInstance.getDataForJSON(dateIni=dateutil.parser.parse(datetimes[0]), dateEnd=dateutil.parser.parse(datetimes[-1]))
    else:
        data = satelliteInstance.getDataForJSON(dateIni=dateutil.parser.parse(datetimes[0]), dateEnd=dateutil.parser.parse(datetimes[-1]), returnInfo=False)
    data['name'] = name

    data['dates'] = [s0 + '.000Z' for s0 in data['dates']]
    
    for i0 in range(len(data['dates'])-1,-1,-1):
        if data['dates'][i0] not in datetimes:
            data['data'].pop(i0)
            data['dates'].pop(i0)
    
    return HttpResponse(
                        json.dumps(json.dumps(data)),
                        content_type="application/json"
                        )
    