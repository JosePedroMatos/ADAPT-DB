import os
import datetime
import decimal
from django.db import models
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from django.utils.html import format_html, mark_safe
from django_countries.fields import CountryField
from django.utils.crypto import get_random_string
from django.core.validators import MinValueValidator, MaxValueValidator, MaxLengthValidator
from decimal import Decimal
from django.db.models.signals import m2m_changed, pre_save
from django.core.exceptions import ValidationError    

# Functions
def seriesChanged(sender, **kwargs):
    if kwargs['instance'].series.count() > 5:
        raise ValidationError("You cannot choose more than 5 series.", code='invalid')

def targetChanged(sender, **kwargs):
    pass

def get_default_encryptionKey():
    # default encryptionKey
    return get_random_string()

# Views
class DataType(models.Model):
    # Table for storing time series' types of data
    
    abbreviation = models.CharField(max_length=16, null=False, blank=False)
    name = models.CharField(max_length=256, null=False, blank=False)
    units = models.CharField(max_length=64, null=False, blank=False)
    description = models.TextField(null=False, blank=False)
    observations = models.TextField(null=True, blank=True)
    
    staticDir = os.path.dirname(os.path.abspath(__file__))
    fs = FileSystemStorage(location=staticDir)
    icon = models.FileField(upload_to='static/timeSeries/seriesIcons/', storage=fs, null=False, blank=False, default='static/timeSeries/seriesIcons/charts.png')
    
    introducedBy = models.ForeignKey(User, null=False, blank=False, on_delete=models.PROTECT)
    created = models.DateTimeField(auto_now_add=True)
    
    def iconImage(self):
        return format_html('<img src="/{}"/>', mark_safe(self.icon)) 
    
    def iconImageSmall(self):
        return format_html('<img height="30" width="30" src="/{}"/>', mark_safe(self.icon)) 
    
    def __str__(self):
        return self.name + ' (' + self.units + ')'

class DataProvider(models.Model):
    # Table for storing data providers
    
    abbreviation = models.CharField(max_length=16, null=False, blank=False)
    name = models.CharField(max_length=256, null=False, blank=False)
    description = models.TextField(null=False, blank=False)
    email = models.EmailField(null=False, blank=False)
    website = models.URLField(null=True, blank=True)
    country = CountryField(blank_label='(select country)')
    
    staticDir = os.path.dirname(os.path.abspath(__file__))
    fs = FileSystemStorage(location=staticDir)
    icon = models.FileField(upload_to='static/timeSeries/providerIcons/', storage=fs, null=False, blank=False, default='static/timeSeries/providerIcons/noInfoIcon.png')

    introducedBy = models.ForeignKey(User, null=False, blank=False, on_delete=models.PROTECT)
    created = models.DateTimeField(auto_now_add=True)

    def iconImage(self):
        return format_html('<img height="30" src="/{}"/>', mark_safe(self.icon)) 
    
    def __str__(self):
        return self.abbreviation

class Location(models.Model):
    # Table for storing measurement locations
    
    name = models.CharField(max_length=256, null=False, blank=False)
    lat = models.DecimalField(decimal_places=5, max_digits=9, null=False, blank=False)
    lon = models.DecimalField(decimal_places=5, max_digits=9, null=False, blank=False)
    catchment = models.CharField(max_length=256, null=False, blank=True)
    river = models.CharField(max_length=256, null=False, blank=True)
    country = CountryField(blank_label='(select country)', null=False, blank=True)
    observations = models.TextField(null=True, blank=True)
    
    introducedBy = models.ForeignKey(User, null=False, blank=False, on_delete=models.PROTECT)
    created = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name + ' (lat: ' + str(self.lat) + ', lon: ' + str(self.lon) + ')'

class Series(models.Model):
    # Table for storing data series
    
    MINUTE = 'm'
    HOUR = 'h'
    DAY = 'd'
    WEEK = 'w'
    MONTH = 'M'
    YEAR = 'Y'
    TIME_STEP_PERIOD_CHOICES = (
        (MINUTE, 'minutes'),
        (HOUR, 'hours'),
        (DAY, 'days'),
        (WEEK, 'weeks'),
        (MONTH, 'months'),
        (YEAR, 'years'),
    )
    TIME_STEP_PERIOD_TYPE = (
        (MINUTE, 'minute'),
        (HOUR, 'hourly'),
        (DAY, 'daily'),
        (WEEK, 'weekly'),
        (MONTH, 'monthly'),
        (YEAR, 'yearly'),
    )
    
    name = models.CharField(max_length=255, null=False, blank=False, unique=True)
    location = models.ForeignKey(Location, null=False, blank=False, on_delete=models.PROTECT)
    provider = models.ForeignKey(DataProvider, null=False, blank=False, on_delete=models.PROTECT)
    type = models.ForeignKey(DataType, null=False, blank=False, on_delete=models.PROTECT)
    timeStepUnits = models.CharField(max_length=2, choices=TIME_STEP_PERIOD_CHOICES, default=DAY, null=False, blank=False)
    timeStepPeriod = models.IntegerField(default=1, null=False, blank=False)
    # TODO: Add the quality icon
    quality = models.SmallIntegerField(default=0, null=False, blank=False)
    importCodes = models.CharField(max_length=512, default=None, null=True, blank=False)
    
    metaEncrypted = models.BooleanField(default=False ,null=False, blank=False)
    metaEncryptionKey = models.CharField(max_length=255, default='', null=False, blank=True)
    
    encryptionKey = models.CharField(max_length=255, default=get_default_encryptionKey, null=False, blank=True)
    
    observations = models.TextField(null=True, blank=True)
    
    introducedBy = models.ForeignKey(User, null=False, blank=False, on_delete=models.PROTECT)
    created = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

    class Meta:
        verbose_name_plural = "Series"

class Value(models.Model):
    # table for storing Values
    
    series = models.ForeignKey(Series, null=False, blank=False, on_delete=models.PROTECT)
    date = models.DateTimeField(null=False, blank=False)
    record = models.BinaryField(null=True, blank=False)
    recordOpen = models.DecimalField(null=True, blank=False, decimal_places=5, max_digits=9)
    
class Forecast(models.Model):
    # table for storing forecasting objects and associate variables
    LINEAR = 'lin'
    TANSIG = 'tan'
    TYPE_CHOICES = (
        (LINEAR, 'Linear'),
        (TANSIG, 'Tansig'),
    )
    MSE = 'mse'
    MAE = 'mae'
    ERROR_CHOICES = (
        (MSE, 'Mean squared error'),
        (MAE, 'Mean absolute error'),
    )
    SECOND = 'second'
    MINUTE = 'minute'
    HOUR = 'hour'
    DAY = 'day'
    WEEK = 'week'
    MONTH = 'month'
    YEAR = 'year'
    PERIOD_CHOICES = (
        (SECOND, 'second'),
        (MINUTE, 'minute'),
        (HOUR, 'hour'),
        (DAY, 'day'),
        (WEEK, 'week'),
        (MONTH, 'month'),
        (YEAR, 'year'),
    )
    
    name = models.CharField(max_length=255, null=False, blank=False, unique=True)
    description = models.TextField(null=False, blank=False)
    extraSeries = models.ManyToManyField(Series, verbose_name='extra series', blank=True, related_name='+')
    targetSeries = models.ForeignKey(Series, verbose_name='target series', null=False, blank=False, related_name='forecast_targetSeries')
    staticDir = os.path.dirname(os.path.abspath(__file__))
    fs = FileSystemStorage(location=staticDir)
    forecastFile = models.FileField(upload_to='forecasts/', storage=fs, null=False, blank=False)
    period = models.CharField('period of the series', default = YEAR, max_length=6, choices=PERIOD_CHOICES, null=False, blank=False, help_text='The duration of a cycle (e.g. the hydrological year).')
    referenceDate = models.DateTimeField('reference date for the period', default=datetime.datetime(datetime.date.today().year, 10, 1, 0, 0, 0), null=False, help_text='The beginning of the hydrological year.')
    leadTime = models.PositiveIntegerField('lead time', default = 30, null=False, blank=False, help_text='How far into the future to extend the forecasts.')
    type = models.CharField('type of model', max_length=3, choices=TYPE_CHOICES, default=TANSIG, null=False, blank=False)
    regularize = models.DecimalField('regularize the model', validators=[MinValueValidator(Decimal('0'))], default=0.01, null=False, decimal_places=5, max_digits=7, help_text='Regularization constant to apply. Larger values will produce smoother forecasts.')
    splitBySeason = models.SmallIntegerField('number of seasons', default=3, validators=[MinValueValidator(Decimal('1')), MaxValueValidator(Decimal('6'))], null=False, help_text='Use different models for different seasons.')
    errorFunction = models.CharField('error function', max_length=3, choices=ERROR_CHOICES, default=MSE, null=False, blank=False, help_text='Choose the error function that evaluates how forecasts differ from observations.')
    allowNegative = models.BooleanField('allow negative values', default=False, null=False)
    ready = models.BooleanField('ready to forecast', default=False, null=False)
    nodes = models.SmallIntegerField('Number of nodes', default=4, validators=[MaxValueValidator(Decimal('10'))], null=False, help_text='Set the complexity of the model by choosing the number of hidden nodes in the artificial neural network. Better to keep it simple.')
    dataExpression = models.CharField('input expression', max_length=256, default='cycle, lead(filter(targets))', null=False, blank=False, help_text='The function that transforms inputs to the model.')
    targetExpression = models.CharField('output expression', max_length=256, default='targets', null=False, blank=False, help_text='The function that transforms targets from the original series.')
    population = models.SmallIntegerField('number of models', default=1000, validators=[MinValueValidator(Decimal('200')), MaxValueValidator(Decimal('4000'))], null=False, help_text='The number of models being simultaneously trained.')
    epochs = models.SmallIntegerField('epochs', default=200, validators=[MinValueValidator(Decimal('10')), MaxValueValidator(Decimal('1500'))], null=False, help_text='For how many iterations the models are trained.')
    
    introducedBy = models.ForeignKey(User, null=False, blank=False, on_delete=models.PROTECT)
    created = models.DateTimeField(auto_now_add=True)

class Colormap(models.Model):
    name = models.CharField(max_length=255, null=False, blank=False, unique=True)
    
    # Folder where data files should go (automatic)
    staticDir = os.path.dirname(os.path.abspath(__file__))
    fs = FileSystemStorage(location=staticDir)
    file = models.FileField(upload_to='static/timeSeries/colormaps/', storage=fs, null=False, blank=False)
    
    introducedBy = models.ForeignKey(User, null=False, blank=False, on_delete=models.PROTECT, verbose_name='Introduced by')
    created = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class SatelliteData(models.Model):
    TRMM3B42v7_3h  = ('TRMMSatelliteRainfall')  # name of the satellite data class
    PRODUCT_CHOICES = (
        (TRMM3B42v7_3h, 'TRMM 3B42 3h'),
    )

    name = models.CharField(max_length=255, null=False, blank=False, unique=True)
    satellite = models.CharField('Satellite product', max_length=255, default=TRMM3B42v7_3h, choices=PRODUCT_CHOICES, null=False, blank=False, help_text='Choice among supported satellite products.')
    observations = models.TextField(null=True, blank=True)
    startDate = models.DateTimeField('Start date', default=datetime.datetime(2010, 10, 1, 0, 0, 0), null=False, help_text='Start date of data collection.')
    
    # information to be filled automatically
    productSite = models.CharField(max_length=255, null=False, blank=False, unique=False, verbose_name='Product site')
    downloadSite = models.CharField(max_length=255, null=False, blank=False, unique=False, verbose_name='Product download site')
    description = models.TextField(null=False, blank=False)
    units = models.CharField(max_length=64, null=False, blank=False)
    timestep = models.CharField(max_length=255, null=False, blank=False, unique=False)
       
    # Folder where data files should go (automatic)
    staticDir = os.path.dirname(os.path.abspath(__file__))
    fs = FileSystemStorage(location=staticDir)
    dataFolder = models.CharField(max_length=255, null=False, blank=False, unique=False, default=os.path.join(fs.base_location, 'satelliteData/data/__unknonwn__'))
        
    # Geometry file (.geoJSON)
    staticDir = os.path.dirname(os.path.abspath(__file__))
    fs = FileSystemStorage(location=staticDir)
    geometry = models.FileField(upload_to='static/timeSeries/satelliteData/geometries/', storage=fs, null=False, blank=False)
    jsonGeometry = models.TextField(null=False, blank=True, default='')
    
    # Colormap
    colormap = models.ForeignKey(Colormap, null=True, blank=False, on_delete=models.PROTECT)
    
    introducedBy = models.ForeignKey(User, null=False, blank=False, on_delete=models.PROTECT, verbose_name='Introduced by')
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name_plural = "Satellite data"