# Generated by Django 5.1.3 on 2024-12-12 04:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0011_detected_out_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='rep',
            name='name',
            field=models.CharField(default='Unknown', max_length=100),
        ),
        migrations.AddField(
            model_name='report',
            name='name',
            field=models.CharField(default='Unknown', max_length=100),
        ),
        migrations.AlterField(
            model_name='detected_in',
            name='name',
            field=models.CharField(blank=True, default='Unknown', max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='detected_out',
            name='name',
            field=models.CharField(blank=True, default='Unknown', max_length=100, null=True),
        ),
    ]
