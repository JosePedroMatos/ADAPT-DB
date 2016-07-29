from django.conf.urls import url
from timeSeries import views

app_name = 'timeSeries'

urlpatterns = [
    url(r'^select/series/(?P<seriesList>.+)/$', views.viewSeries, name='selectTimeSeries'),
    url(r'^deleteValues/series/(?P<seriesName>.+)$', views.deleteTimeSeries, name='deleteTimeSeries'),
    url(r'^upload/series/(?P<seriesName>.+)/uploadTimeSeries/$', views.uploadTimeSeries, name='uploadTimeSeries'),
    url(r'^upload/series/(?P<seriesName>.+)/$', views.upload, name='upload'),
    url(r'^trainForecast/(?P<forecastName>.+)/forecast/$', views.forecast, name='forecast'),
    url(r'^trainForecast/(?P<forecastName>.+)/hindcast/$', views.hindcast, name='hindcast'),
    url(r'^trainForecast/(?P<forecastName>.+)/train/$', views.trainForecastRun, name='train'),
    url(r'^trainForecast/(?P<forecastName>.+)/progress/(?P<jobId>.+)$', views.trainForecastProgress, name='progress'),
    url(r'^trainForecast/(?P<forecastName>.+)/$', views.trainForecastBase, name='trainMain'),
    url(r'^satelliteStore/(?P<name>.+)/$', views.storeSatellite, name='storeSatellite'),
    url(r'^satelliteGet/$', views.getSatelliteData, name='getSatellite'),
    url(r'^getValues/$', views.getValues, name='getValues'),
]

