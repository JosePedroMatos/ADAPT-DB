# -*- coding: utf-8 -*-
# Generated by Django 1.9.7 on 2016-07-07 15:39
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('timeSeries', '0006_satellitedata_startdate'),
    ]

    operations = [
        migrations.AddField(
            model_name='satellitedata',
            name='downloadFolder',
            field=models.CharField(default='/home/zepedro/Tethys/timeSeries/satelliteData/Downloads', max_length=255),
        ),
        migrations.AlterField(
            model_name='satellitedata',
            name='dataFolder',
            field=models.CharField(default='/home/zepedro/Tethys/timeSeries/satelliteData/Data', max_length=255),
        ),
    ]
