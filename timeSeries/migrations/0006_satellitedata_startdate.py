# -*- coding: utf-8 -*-
# Generated by Django 1.9.7 on 2016-07-07 15:24
from __future__ import unicode_literals

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('timeSeries', '0005_auto_20160707_1719'),
    ]

    operations = [
        migrations.AddField(
            model_name='satellitedata',
            name='startDate',
            field=models.DateTimeField(default=datetime.datetime(2010, 10, 1, 0, 0), help_text='Start date of data collection.', verbose_name='Start date'),
        ),
    ]
