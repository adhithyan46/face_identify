# Generated by Django 4.1.5 on 2023-02-08 09:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_rename_detected_detected_in_detected_out'),
    ]

    operations = [
        migrations.AddField(
            model_name='detected_in',
            name='photo',
            field=models.ImageField(default='app/facerec/detected/noimg.png', upload_to='detected/'),
        ),
        migrations.AddField(
            model_name='detected_out',
            name='photo',
            field=models.ImageField(default='app/facerec/detected/noimg.png', upload_to='detected/'),
        ),
    ]
