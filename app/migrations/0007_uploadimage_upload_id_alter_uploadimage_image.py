# Generated by Django 4.1.7 on 2023-05-15 09:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0006_remove_uploadimage_caption'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadimage',
            name='upload_id',
            field=models.IntegerField(default=1),
        ),
        migrations.AlterField(
            model_name='uploadimage',
            name='image',
            field=models.ImageField(upload_to='app/facerec/dataset/'),
        ),
    ]
