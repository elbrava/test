# Generated by Django 4.0.4 on 2022-06-12 22:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('record', '0005_computers_active_computers_sid_alter_messages_to'),
    ]

    operations = [
        migrations.AlterField(
            model_name='computers',
            name='sid',
            field=models.CharField(default=None, max_length=110, unique=True),
        ),
        migrations.AlterField(
            model_name='computers',
            name='status',
            field=models.CharField(max_length=110, unique=True),
        ),
    ]
