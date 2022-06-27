# Generated by Django 4.0 on 2022-05-18 10:27

import datetime
from django.db import migrations, models
import django.db.models.deletion
from django.utils.timezone import utc


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Computers',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=110)),
                ('os', models.CharField(max_length=110)),
                ('status', models.CharField(max_length=110)),
                ('messages_status', models.CharField(max_length=110)),
            ],
        ),
        migrations.CreateModel(
            name='Messages',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('_from', models.CharField(max_length=110)),
                ('to', models.CharField(max_length=110)),
                ('time', models.DateTimeField(default=datetime.datetime(2022, 5, 18, 10, 27, 51, 136386, tzinfo=utc))),
                ('type', models.CharField(max_length=110)),
                ('read', models.BooleanField(default=False)),
                ('computer', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='record.computers')),
            ],
        ),
    ]
