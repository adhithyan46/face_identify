# Generated by Django 4.0.5 on 2023-03-22 10:31

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0007_hours'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='hours',
            name='department',
        ),
        migrations.RemoveField(
            model_name='hours',
            name='entry',
        ),
        migrations.RemoveField(
            model_name='hours',
            name='out',
        ),
    ]
